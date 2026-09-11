# The slow test set's cost was not data preparation

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-11. **Question:** [#331](https://github.com/dseinternational/vocabulary-growth/issues/331) proposed four ways to cut the `tests-slow` job after `notes/202609101310-slow-test-distribution.md` rebalanced it. **Evidence:** direct timing of each candidate on a 32-core Windows workstation, all figures measured the same day on the same machine. **Change:** one three-line fix in `diagnostics_var_names`, the synthetic frame for `test_prior_table_coverage`, a lazy build cache in `test_graph_equivalence`, and a PyTensor compiledir cache in CI.

**The issue's headline proposal was aimed at the wrong thing, and measuring it is what found the right one.** Item 1 assumed `test_prior_table_coverage.py`'s 591.7 s was its 106 real data preparations. Switching the file to the synthetic frame moved it from **6 m 01.8 s to 5 m 45.1 s** — 4.6%. The assumption was wrong, so the next question was where the time actually went.

## 1. Not `prepare`, not `priors`, not `build`

Timing the three engine stages in process, for all twenty-one registered models:

| stage                                              | total, 21 models |
| -------------------------------------------------- | ---------------: |
| `prepare` (the real DuckDB path)                   |            2.5 s |
| `priors`                                           |            0.7 s |
| `build`                                            |            3.3 s |
| **all three**                                      |        **6.6 s** |
| the synthetic path (`priors` + `build`, no DuckDB) |            3.6 s |

Six and a half seconds of pipeline against a six-minute file. The preparation stage is cheap because `tests/conftest.py` already silences the two things that made it expensive — the prior-distribution figures and the `describe_all` pass (`notes/202608241530`). What was left was a DuckDB query and some pandas masking.

## 2. It was `diagnostics_var_names`, at twelve times the cost of the graph it describes

```python
summary_var_names = [var.name for var in model.unobserved_RVs if var.size.eval() <= 2]
```

`Variable.eval()` compiles a PyTensor function for its own graph. One per unobserved RV, about 60 ms each, and the registered models carry 30 to 98 of them:

|                                        | total, 21 models |
| -------------------------------------- | ---------------: |
| build every graph                      |            6.4 s |
| ask each graph for its parameter names |       **75.8 s** |

The fix is to stop compiling once per variable. Most of these variables have a fully static type shape — every scalar prior does — so their element count is the product of it, needing no PyTensor at all; the rest are evaluated **together** in one compiled function. Over all twenty-one models and all eighty-five registered variants the two implementations return **byte-identical lists in byte-identical order, 106 of 106**, at 416.9 s against 97.7 s.

This is production code, not test code, so every fit pays it too — trivially, at six seconds each, which is why it was never noticed there.

## 3. The synthetic frame, kept for a different reason than the one proposed

It is worth 4.6% rather than the "roughly halves" the issue predicted, so on cost alone it would not be worth the risk. It is kept because it removes a dependency the contract does not have: the priors table is built from a model's **definition**, so checking it against a data-dependent parameter set imports the data into a graph-to-report contract — the same argument `test_graph_equivalence` has always made for itself. The file no longer needs the prepared DuckDB, so it no longer skips without one.

The risk the issue named is real and was checked rather than argued, over all 21 models and all 85 variants:

- **In no case does the synthetic frame report fewer parameters than the real one.** The check can only get stricter this way, never looser.
- **One case differs, in the safe direction.** `vg15/dse-native-only` leaves exactly one psi-informed study in the real pool, so its documented single-study branch drops `tau_psi` and `z_psi`; the synthetic frame's four cross-tab studies keep them.
- **No free variable crosses the `size <= 2` boundary** that separates the summary set from the gate set, so nothing moves between the two because one frame is smaller.
- **One case could not build synthetically**: `vg16/lag-same-form` reads `survey_vocab_max`, which the frame did not carry. Added, constant across the frame — the frame's children span two studies by construction, so a per-study ceiling would drop every lag and take `beta_lag` out of the graph, which is the one direction a stand-in must never move in. It moved no entry in `support/graph_baseline.json`.

## 4. `FAST_COMPILE`: rejected, and by a hard failure rather than a small saving

Item 2. Where it works it is exact — 2.1e-14 relative against the committed baseline, against a 1e-9 tolerance — and roughly halves the compile. It works on **six of twenty-one models**. The other fifteen raise:

```
NotImplementedError: Elemwise.perform (Python mode) does not support more than 32 operands.
```

`FAST_COMPILE` is `Mode("py", "fast_compile")`, and these graphs exceed what the Python linker will evaluate. `Mode(linker="cvm", optimizer="fast_compile")` was tried as the obvious escape and fails identically on the same fifteen. Even a working version would save about 16 s of the 31.9 s all twenty-one models spend compiling — small beside the 319 s §2 returned, and a mode that silently applies to six models and not the other fifteen is worse than none.

## 5. The compiledir cache: adopted, and the largest single saving of the four

Item 3, measured before any machinery: locally, the slow set at four workers, **6 m 41 s on a cold compiledir against 4 m 07 s on a warm one**. CI restores uv's package cache and nothing else, so every run has been the cold number — and on CI itself the cache is worth more than that, **7 m 25 s cold against 4 m 16 s warm** on the same commit, restoring 24 MB.

The key hashes `.python-version` and `uv.lock` and carries **no `restore-keys`**, which is the whole design. A stale numba function cache is not hypothetical here — on 2026-09-10 a pytensor/numba bump left `CPUDispatcher` entries producing a typing error that survived reruns until they were deleted by hand. `uv.lock` pins every package, so any version move gives a new key and a cold, correct compiledir; a loose prefix key would restore exactly the cache that failure mode needs. The cost is an occasional cold run after an unrelated dependency bump, which is the right way round.

**The per-worker `base_compiledir` mitigation the issue also proposed is not adopted, and the cache is why.** It was costed as free "on a cold CI cache" — but the cache is no longer cold, and per-worker directories would split it, since xdist does not assign the same tests to the same worker across runs. The condition it guards against is concurrent writes during parallel compilation; the local `loadgroup` runs have been clean, and if the race recurs the trade can be revisited against a measured cost rather than an assumed one.

## 6. The lazy build cache, and the open question it closes

`test_graph_equivalence`'s `built` fixture built all twenty-one graphs eagerly, so any selection paid for the whole registry — one model's five tests, or the two-build stability check that names two models. It now builds on first use. That closes the issue's open question about whether the module should keep its `xdist_group`: grouped it took 40.5 s on one worker and ungrouped 21.2 s across four, but ungrouped each worker rebuilt all twenty-one. A worker now builds only the models its own tests ask for, whichever way the tests are distributed. The mark stays, because the slow job's floor is total CPU and four partial copies of the registry is more of it than one.

## 7. What it bought

**On CI**, which is the figure that matters, reading the `tests-slow` job's pytest step:

| Run                                       |  pytest step |
| ----------------------------------------- | -----------: |
| main `789c213`, before `--dist loadgroup` |    10 m 23 s |
| main `82cd646`, after it                  |     9 m 33 s |
| this change, cold compiledir              |     7 m 25 s |
| this change, warm compiledir              | **4 m 16 s** |

**9 m 33 s → 4 m 16 s, 55%**, of which the code change is 9 m 33 s → 7 m 25 s and the compiledir cache the rest. `tests-fast` is unchanged at 1 m 32 s → 1 m 38 s, within the noise of a cold cache on its own first run.

**One correction to the previous note, while these are being recorded.** `notes/202609101310-slow-test-distribution.md` §3 reports "592 s → 207 s on four workers", noting "the 592 s from CI and the rest locally". The two halves are not comparable, and pairing them overstates what CI got: **on CI that change measured 10 m 23 s → 9 m 33 s, not 10 m 23 s → 3 m 27 s.** An `-n 4` run on a 32-core workstation is not an `ubuntu-26.04-arm` run. The rebalancing was real and worth having — it removed the one-file ceiling, without which none of the above would have shown — but its CI value was 8%, not 2.9x. Local figures below are labelled as such for that reason.

Locally, on a 32-core workstation, one day, same conditions:

| Measurement                                |     Before |          After | Ratio |
| ------------------------------------------ | ---------: | -------------: | ----: |
| `test_prior_table_coverage.py`, serial     | 6 m 01.8 s | **1 m 29.6 s** |  4.0x |
| `test_prior_table_coverage.py`, 32 workers |     39.8 s |     **21.9 s** |  1.8x |
| Slow set, 4 workers                        | 5 m 48.9 s | **4 m 07.3 s** |  1.4x |
| Slow set, 32 workers                       | 2 m 56.5 s |     2 m 36.7 s |  1.1x |
| Slow set, 4 workers, cold compiledir       |          — |     6 m 41.4 s |     — |

`test_prior_table_coverage.py` is no longer the slow set's largest cost. Its `--durations` entries have left the top fifteen entirely; the set's top costs are now `test_observation_deterministics`'s module fixture (87.8 s, a real fit) and `test_bivariate_re_holdout_mask`'s chain of sampling tests (~370 s of calls). Those are real sampling, not incidental cost, so the next lever on this job is more workers rather than less work — item 4, untouched here.

## 8. Item 4, and the other open question

**Sharding or a larger runner is not attempted.** The issue put it after 1 and 2 deliberately, and that ordering held: the work that did not need doing is now gone, and what remains is fits.

**Whether `tests-fast` should move to `--dist loadgroup` is answered "not without marks first".** `test_trend_gp_consolidation.py` carries a session-scoped cache of built models whose own docstring records the dependency — _"Under `--dist loadfile` this file is one worker's work, so the cache is never split across processes"_ — so the switch needs an `xdist_group` on it at minimum, and `test_centred_study_re.py` and `test_sex_shift_variant.py` want auditing for the same thing. The job is 3 m 31 s and has never been profiled, so there is no measured reason to make that change yet.
