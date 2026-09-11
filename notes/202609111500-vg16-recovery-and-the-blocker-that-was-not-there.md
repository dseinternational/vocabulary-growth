# VG16 parameter recovery, and the blocker that was not there

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-11. **Question:** [#297](https://github.com/dseinternational/vocabulary-growth/issues/297) (VG25) is deferred on one remaining item — the wave-sequential recovery simulator, [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) item 6 and [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 3.9 — and the study owner asked whether it could be progressed before the refit window. **Evidence:** the simulator's own build order, measured. **Change:** the ordering rule is now derived and guarded rather than asserted; VG16 becomes a recovery target; VG16 has parameter recovery for the first time.

**The blocker was a misdiagnosis, and measuring it is what showed that.** `recovery/spec.py` recorded VG16 as unsupported because "the cross-lag predictor is a function of earlier-wave comprehension, so forward simulation must proceed wave by wave; single-pass simulation would fit synthetic-lag data against real-lag truth". Three issues rest on that sentence. It is false for VG16.

## 1. Why it is false

The simulator has always drawn in **rounds, rebuilding the model between them** so the engine re-derives every row denominator from the simulated parent (`recovery/simulate.py`, the nesting machinery). The lag is derived in the same place — `prev_wave_lag_for_frame` is called inside the engine's _build_, off `analysis_df["understood"]` — so it is re-derived by that same rebuild.

And VG16's lag reads `understood`, drawn in **stage 0**, while entering `y_s_obs`, drawn in **stage 1**. So by the time the draw that uses the lag happens, the lag has already been recomputed from the simulated comprehension. Measured on the real frame, forcing the single pass and comparing the predictor at each build against the one the finished frame implies:

| build                                     |             `understood` differs from the finished frame | lag predictor differs from the finished frame |
| ----------------------------------------- | -------------------------------------------------------: | --------------------------------------------: |
| 0 (before any draw)                       |                                           1,702 of 1,708 |                                      575 rows |
| 1 (the build whose draw consumes the lag) | 448 — every one a row with no comprehension count at all |                                    **0 rows** |
| 2 (final coherence rebuild)               |                                                      448 |                                    **0 rows** |

Zero. The data were never going to be generated under one lag and fitted under another.

## 2. What replaces the assertion

Not a flag saying "this model needs the wave loop", which is how the wrong answer got written down in the first place. `spec.single_pass_is_sound` **derives** it: a predictor is declared with the column it reads and the nodes it enters, and one pass is sound exactly when the source is drawn in a strictly earlier stage than every consumer.

**The wave loop still exists, because the case is real — just not VG16's.** #297's proposed VG25 reads the prior wave's _signed_ share, and the joint engine draws `signed` in the **same** stage as the `spoken` it would shift. There a single pass would draw the consumer against a source still holding study data, and the derivation selects the wave loop automatically. That is the piece #297 actually needed, and it is built and tested.

**And the guard runs either way.** After simulation, every row's predictor as drawn is compared against the predictor the finished synthetic frame implies; a mismatch aborts. That is what makes the ordering rule evidence rather than a second assertion — had it existed in August it would have refused the claim this note corrects. `tests/test_recovery_wave_sequential.py` shows it failing, both on arrays and on a real simulation with the ordering forced unsound, because a guard never seen to fail is not evidence.

## 3. VG16 parameter recovery, for the first time

Three replicates, `test` tier, `--truth posterior`, roughly 7–12 minutes each.

**Only one is assessable.** The harness scores a replicate only if its fit's convergence is confirmed, and two of three missed the R-hat ≤ 1.01 gate:

| replicate | converged | max R-hat | min ESS | coverage (89%) | max abs z |
| --------- | --------- | --------: | ------: | -------------: | --------: |
| r01       | **yes**   |    1.0096 |     449 |          0.842 |      2.73 |
| r02       | no        |    1.0132 |     393 |              — |         — |
| r03       | no        |    1.0105 |     537 |              — |         — |

So the headline is one replicate, and the runbook's caution about what a handful of replicates can establish applies with force.

**On the assessed replicate, `beta_lag` missed low.** Truth 0.285, posterior median 0.201, sd 0.051, 89% ETI [0.121, 0.284] — the truth at the 94.8th percentile of the posterior, just outside, z = −1.64, a bias of −29%. Coverage over all 133 scored quantities was 0.842 against a nominal 0.89, which is the under-nominal coverage this engine already shows ([#225](https://github.com/dseinternational/vocabulary-growth/issues/225)) rather than anything specific to the lag.

## 4. The thing worth running more replicates for

Not a finding — two of the three did not converge and cannot be scored — but the reason to go on:

| replicate          | truth `beta_lag` | posterior median | 89% ETI        |
| ------------------ | ---------------: | ---------------: | -------------- |
| r01 (assessed)     |            0.285 |            0.201 | [0.121, 0.284] |
| r02 (not assessed) |        **0.045** |            0.195 | [0.107, 0.284] |
| r03 (not assessed) |            0.267 |            0.242 | [0.155, 0.328] |

The truths span 0.045 to 0.285; the posterior medians span 0.195 to 0.242. On three draws that is an impression, not a measurement, and one of the three is the only one that may be quoted. But it is the shape the VG16 review was worried about — that `beta_lag` may be reporting something other than the lag it was given — and r02 is by accident the `(beta ≈ 0)` cell #242 item 6 asks for. **The right next step is more replicates at a tier that converges**, which is now a thing that can simply be run.

## 5. What this leaves

- **#297 is unblocked on its stated dependency.** The wave-sequential simulator exists, is selected by derivation, and is guarded. What remains there is the four design decisions in the issue body and the registration itself, neither of which waits on this.
- **#289 task 3.9 / #242 item 6 are half done.** VG16 recovery runs and has produced its first evidence. The three designed cells — `(beta=0, rho≠0)`, `(beta≠0, rho=0)`, both nonzero — still need two things the harness does not have: a way to **set** a parameter in the truth draw rather than take what the posterior or prior offers, and the correlated-random-effects-plus-lag comparator, which is not a registered model.
- **A tier decision.** Two of three replicates missing the gate at `test` is itself a result: VG16 recovery needs a higher tier or the escalation rung, and that cost should be known before the designed cells are commissioned.
- Nothing here touches a model of record. Recovery fits land in their own directories, and no fit was refitted.
