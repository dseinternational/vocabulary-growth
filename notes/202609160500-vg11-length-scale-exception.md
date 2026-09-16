# VG11's GP length scale, and why the convergence exception now names it

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-16. **Decision, by the study owner, the same day:** record `ell`/`ell_unit` in VG11's hard-tier convergence exception, refit the nine publication models so every signature matches again, and leave the model itself alone. **What this note is:** the failure, what the slow direction is, and what was ruled out. **Nothing here changes a model definition or a prior.**

## 1. The failure

VG11 was fitted at `rep` on 2026-09-16 in pass 2 of the nine-model refit, from `main` at `82103df`. It was the only one of the nine that did not promote.

|                  |                                                                        |
| ---------------- | ---------------------------------------------------------------------- |
| convergence gate | **FAILED** (hard tier): max R-hat **1.0116**, two parameters over 1.01 |
| failing set      | `ell_unit` and `ell` — one quantity under two names                    |
| min ESS          | 888.9 (the gate's floor is 400)                                        |
| divergences      | 28 in 36,000 draws                                                     |
| energy BFMI      | 0.36–0.41 on all six chains (the gate's floor is 0.3)                  |
| wall time        | 1 h 57 m                                                               |
| retained at      | `failed/VG11-age-spoken-td-re-20260916T034535Z/`                       |

`ell` is `ell_unit` mapped onto the z scale by a fixed affine map, so the two carry the same R-hat by construction. The previous fit of record, on the same `rep` settings on 2026-09-08, had the same posterior and a lower R-hat: `ell` mean 1.915 against 1.917, SD 0.333 in both, R-hat 1.012 against 1.004, ESS 983 against 1,097.

**What did not fail is the point.** Every quantity the report reads converged with margin:

| quantity                   | points |  max R-hat | over 1.01 | min ESS |
| -------------------------- | -----: | ---------: | --------: | ------: |
| trajectory plot grid       |    500 |     1.0028 |         0 |   2,904 |
| trajectory query points    |      8 |     1.0025 |         0 |   2,982 |
| dispersion plot grid       |    500 |     1.0017 |         0 |   3,452 |
| dispersion query points    |      8 |     1.0016 |         0 |   3,457 |
| 16 HSGP basis coefficients |     16 |     1.0068 |         0 |     889 |
| 14,553 child effects       | 14,553 |     1.0011 |         0 |   9,982 |
| **`ell` / `ell_unit`**     |      1 | **1.0116** |     **1** | **983** |

## 2. The slow direction is the GP's amplitude/length-scale ridge

VG11's GP is orthogonalised against the constant and the linear trend, so what it models is curvature. Its length-scale prior admits 6 to 18 months on an 8–30 month domain — **0.55 to 1.64 times the domain's half-width**. Across that whole range an ExpQuad kernel, with the line already projected out, leaves nearly the same residual shape. The data can say how much curvature there is; they can barely say how it splits between amplitude and length scale.

Measured on the retained fit (`scripts/experiments/` has no runner for this; the measurements come from the trace directly):

| measurement                                  | value                            |
| -------------------------------------------- | -------------------------------- |
| `ell_unit` prior Beta(3,3) SD                | 0.189                            |
| `ell_unit` posterior SD                      | 0.165                            |
| prior-to-posterior contraction               | **0.125**                        |
| correlation of `ell_unit` with `eta`         | +0.32                            |
| `ell_unit` ESS as a share of draws           | 2.7% (983 of 36,000)             |
| curvature after removing the straight line   | 0.320 logits (5–95% 0.305–0.335) |
| total spread of the fitted curve across ages | 2.16 logits                      |

The curve is pinned to about five per cent; its decomposition is not. Chains drift slowly along the ridge, ESS falls near 1,000, and R-hat lands at 1.0116 — which is exactly where [`202608150900`](202608150900-rhat-gate-calibration.md) §3 said exceedances live: of 67,328 parameters scanned across 29 fits, the only ones over 1.01 sat below ESS 1,600, and that band's maximum was 1.0133.

The curvature itself is not simply the prior speaking: `eta`'s HalfNormal(0.5) prior has mean 0.40 and the posterior sits at 1.03 (SD 0.264), so the data ask for more amplitude than the prior offered. That sharpens, for VG11, the standing finding of [`202608231537`](202608231537-vg11-vg12-vg13-statistical-review.md) §6.2, which concluded for VG13's ten-month window that curvature was not identified under this graph: on VG11's 22-month window the curvature's **size** is identified and its **length scale** is not.

## 3. What was ruled out

- **The HSGP approximation.** The basis is m = 16 with boundary factor c = 5.24 (L = 9.673 in z, half-width 1.847), calibrated by the shared library from the _shortest_ admissible length scale. The posterior sits mid-range, nowhere near the basis's limits. Sizing is not implicated.
- **Model size.** [`202608150900`](202608150900-rhat-gate-calibration.md) §4 already killed the multiplicity hypothesis for this family: VG13 has nearly VG11's parameter count and nearly its minimum ESS and passes comfortably.
- **More sampling.** `rep-hightune`, which the July 2026 run used for VG11, VG12 and VG13, no longer exists: `dse_research_utils` 0.15.1 offers `dev`, `test`, `rep` and `rep-lite` only, and the parameters are module constants with no override. Raising them means a library change, which moves the recorded library version in the executable-code signature and so restales every fit anyway.
- **A different seed.** The seed is compared exactly by `_sampling_parameter_errors`, so a fit at another seed is not publishable. There is no seed to hunt for.
- **Fixing the length scale.** The engine supports it — `cfg_ell` accepts a float — and VG15's signed GP is the precedent, at contraction 0.033. That precedent went the other way: fixing it at the prior median was measured, changed nothing (a maximum median shift of 0.0023), and was therefore **not** adopted ([`202608060900`](202608060900-three-prior-conflicts.md) §5b). Here it would additionally need a new definition field with a `BACKFILL_DEFAULTS` entry, and would move VG11's amplitude and band. Rejected for the same reason as in VG15: it buys no measurable accuracy.

## 4. What changed

- **`CONVERGENCE_EXCEPTIONS["VG11"]`** now names `ell`, `ell_unit` and the 2026-08-15 basis coefficient, with the ceiling unchanged at 1.015. The coefficient clause is kept because a basis coefficient and the length scale are the same ridge seen from two sides; in this fit the coefficients reached only 1.0068.
- **Nothing else.** No definition, prior, engine or data rule moved. On a refit the same seed and the same graph give the same draws, so VG11's numbers are those already in the retained fit; the gate now accepts them and the report discloses why.
- **`tests/test_convergence_exception.py`** pins the new coverage and, more importantly, that it does not widen: the exception still closes on any additional failing parameter, on an ESS failure, on a worse R-hat, and on any other model.
- **The refit.** `fit_artifacts.py` is inside the hashed package, so this edit moves the executable-code signature and restales the eight fits promoted on 2026-09-15/16. All nine are being refitted from the merge commit. That re-run re-stamps provenance rather than changing numbers.

## 5. Not established

- **That the exception is permanent.** It is a disclosure, not a repair. The ridge is a property of a GP whose admissible length scales are all long relative to a 22-month window; a future model with a narrower length-scale prior, or a longer chain, would not need it.
- **That 28 divergences are benign.** They are a soft-tier caveat and are reported as one, alongside the exception. They were 15 in the previous fit and 16 in the August one, so the count has drifted upwards across refits on the same settings; nothing here explains that.
- **That the reported quantities are right because they converged.** Convergence is not accuracy. This note establishes only that the chains agree about what VG11 reports, and disagree slightly about a parameter it does not.
