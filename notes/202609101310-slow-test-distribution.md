# The slow test job was the cost of one file

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-10. **Question:** the study owner asked what could be done to speed up the `tests-slow` CI job, which took 10 m 20 s against `tests-fast`'s 3 m 31 s. **Evidence:** the job's own `--durations` output and a full per-test profile of the slow set (`pytest -m slow -n auto --dist loadfile --durations=0`). **Change:** `--dist loadgroup` with `xdist_group` marks on the three modules that need pinning, and the priors-table exemption check moved to the fast set.

**The job could not finish faster than one file.** `test_prior_table_coverage.py` accounts for 106 of the slow set's tests and **591.7 s of its 1,320 s of CPU — 45%**. Under `--dist loadfile` that whole module ran on one worker, serially, so the wall clock was pinned to it: 591.7 s of file against a 603 s run locally and 592 s on CI. The other twelve modules' 728 s of work fitted entirely inside that window on the other workers, which then idled.

## 1. Where the time is

Per-file totals, from the full local profile (32 workers, `loadfile`). Setup and teardown are attributed to the test that triggered them, so a module fixture's cost sits in its first test.

| Module                                  | Tests |    Seconds |
| --------------------------------------- | ----: | ---------: |
| `test_prior_table_coverage.py`          |   106 |  **591.7** |
| `test_bivariate_re_holdout_mask.py`     |     9 |      174.3 |
| `test_vg22_correlation_prior.py`        |     8 |      120.9 |
| `test_kappa_conditional_calibration.py` |    26 |      105.1 |
| `test_graph_equivalence.py`             |    28 |       83.3 |
| `test_subject_marginal_sampling.py`     |     1 |       66.6 |
| `test_observation_deterministics.py`    |     4 |       62.4 |
| `test_trivariate_joint_fallback.py`     |     6 |       51.2 |
| `test_review_graph_regressions.py`      |     7 |       41.8 |
| `test_joint_correlated_subject_re.py`   |     2 |       11.2 |
| `test_sex_shift_variant.py`             |     2 |       10.1 |
| `test_wave_forward_score.py`            |     3 |        0.9 |
| `test_exploratory_lifecycle.py`         |     1 |        0.9 |
| **Total**                               |   203 | **1320.3** |

(203 of the 301 tests are listed: the rest fall below the 0.005 s reporting threshold.)

**The cost is compilation, not sampling**, which is why cutting draw counts is not a lever. Three independent confirmations: the single heaviest test samples 8 draws in 1 chain and its own comment names nutpie's compilation of the quadrature graph as the cost; `support.synthetic_graphs.fixed_point_logp` calls `model.compile_logp()`, one PyTensor function compilation per test, which is what makes `test_graph_equivalence`'s log-probability case and `test_trivariate_joint_fallback` expensive; and `test_prior_table_coverage` runs the real `prepare` → `priors` → `build` for each of its 106 graph tests, so 106 full DuckDB loads and 106 full-size builds.

## 2. Why `loadfile` was the wrong flag for this set

`--dist loadfile` exists here for a real reason, recorded in the agent instructions: several modules have module-scoped fixtures that are themselves fits, and per-test distribution rebuilds them on every worker that draws one of their tests. But it was applied to the whole set, including the module that needs it least — `test_prior_table_coverage.py` has no fixtures at all, and each of its tests builds its own graph independently.

`--dist loadgroup` inverts the default: tests distribute individually unless they carry a `pytest.mark.xdist_group`, in which case a group travels to one worker. Auditing all thirteen slow modules for shared expensive state found exactly **three** that need the mark:

| Module                               | What is shared                                                       |
| ------------------------------------ | -------------------------------------------------------------------- |
| `test_graph_equivalence.py`          | `built` builds all twenty-one registered graphs; 107 tests read them |
| `test_observation_deterministics.py` | `two_fits` is two real nutpie fits of VG07, shared by four tests     |
| `test_subject_marginal_sampling.py`  | a module-scoped `conftest` fixture that builds a marginalised model  |

The third holds one test today, so its mark is a no-op and a statement of the invariant for the second one. `test_sex_shift_variant.py` also has module-scoped fixtures and was checked and left ungrouped: they are a module import and a catalogue lookup, not a fit. The rest build per test and share nothing.

## 3. What it bought

| Measurement                      | Before |     After | Ratio |
| -------------------------------- | -----: | --------: | ----: |
| Slow set, 4 workers (CI's count) |  592 s | **207 s** |  2.9x |
| Slow set, 32 workers             |  603 s | **118 s** |  5.1x |
| Slow set, tests selected         |    322 |       301 |       |
| Fast set, 32 workers             | 43.8 s |    43.8 s |       |
| Fast set, tests selected         |  2,153 |     2,154 |       |

All four run figures are observed, the 592 s from CI and the rest locally. The two "before" numbers being equal across 4 and 32 workers **is the finding**: under `loadfile` the wall clock was one file, so adding workers bought nothing. The "after" numbers scale with the worker count, which is what a balanced run looks like.

The floor is now total CPU divided by workers, so beyond this point the only levers are less total work or more workers, both in [#331](https://github.com/dseinternational/vocabulary-growth/issues/331). One caveat on the 1,320 s profile above: it was measured under 32-way contention, which inflates each test's own time — the 4-worker run does the same work in less total CPU, which is why 207 s beats the 330 s that profile would predict.

## 4. Twenty-one tests that were in the wrong set

`test_prior_table_coverage.py` carried a module-level `pytestmark = pytest.mark.slow`, which swept in `test_no_parameter_is_both_rendered_and_exempt` — a test whose own docstring says it "needs no graph". It was deselected from every pull request for nothing. The mark is now per test, on the two that build graphs, and the exemption check runs in the fast job. `test_observation_deterministics.py` had already made exactly this correction for itself under #273, with a comment recording that a module-level mark had stopped a struct contract running on any pull request.

While moving it: that test was parametrised over all twenty-one models and never read the model key, so it ran one assertion twenty-one times over `_PRIOR_SPECS`, which is global. The parametrisation is gone, which is where the fast set's 2,153 → 2,154 comes from rather than 2,153 → 2,174.

## 5. What is left, and one risk this raises

Four candidates are recorded in [#331](https://github.com/dseinternational/vocabulary-growth/issues/331): stop `test_prior_table_coverage` doing real data preparation (it wants parameter _names_, and `test_graph_equivalence` gets the same names from a synthetic frame); `FAST_COMPILE` for the compile-once-evaluate-once paths; caching the PyTensor compiledir in CI; and sharding the job or using a larger runner.

**Resolved 2026-09-11**, in `notes/202609111158-slow-test-cost-was-not-data-preparation.md`. The first candidate's premise was wrong: the preparation stage costs 6.6 s across all twenty-one models, and the file's time was `diagnostics_var_names` compiling one PyTensor function per unobserved RV — 75.8 s against 6.4 s to build every graph in the registry. Batching that, plus the compiledir cache, took CI's `tests-slow` pytest step from 9 m 33 s to 4 m 16 s. `FAST_COMPILE` was rejected: it fails outright on fifteen of twenty-one models. Sharding was not needed. **That note also corrects §3 above:** its "592 s → 207 s on four workers" pairs a CI before-figure with a local after-figure, and on CI the `loadgroup` change measured 10 m 23 s → 9 m 33 s.

The risk this change raises is worth stating plainly: 106 graph builds that used to compile serially on one worker now compile in parallel on four, which increases concurrent writes to the shared PyTensor compiledir. That is the condition behind the cold-cache race recorded on 2026-09-09, where the first parallel run after a pytensor/numba bump failed one test with a `CPUDispatcher` typing error and passed on the retry. Nothing here makes that race new, and the local run of all 301 tests under `loadgroup` was clean, but if it starts recurring the fix is a per-worker `base_compiledir` rather than a return to `loadfile`.
