# The comprehension reporting cap, rechecked on the enlarged pool: it stays at 72

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

**Date:** 2026-09-09. **Reruns:** the model-dependence test that returned `report_max_age_understood` from 84 to 72 months on 2026-08-22 ([`202608221200`](202608221200-reporting-source-by-quantity.md) §4), whose own terms for raising it again were _"not when the band is merely populated — when it can distinguish the child structures"_, measured as the VG19–VG20 gap at 72 and 84 months falling to the 0.15 interval widths that holds below 60. **Fits:** the 2026-09-08 `rep` fits of VG10, VG19 and VG20 from the full refit after the `us_03` ingestion, all on the frame `sha256:b27f64ea0640…` (1,708 rows, 943 children). **Script:** `scripts/experiments/model_dependence_of_reported_quantities.py`, unchanged in method; it now reads the output root from the environment rather than a hard-coded VM path.

## The result

**The rule does not fire.** The gap is where it was.

| quantity                       | worst VG19–VG20 gap, 2026-08-22 | worst VG19–VG20 gap, 2026-09-09 |
| ------------------------------ | ------------------------------: | ------------------------------: |
| understood, population curve   |                            0.14 |                        **0.12** |
| spoken, subject-marginal curve |                            0.02 |                        **0.03** |
| `q` at 72 months               |                            0.89 |                        **0.89** |
| spoken, population curve, 72   |                            0.58 |                        **0.58** |
| spoken, population curve, 84   |                            0.65 |                        **0.59** |

Gaps are the difference of posterior medians as a fraction of VG20's own 89% interval width at the same age. The two most-reported curves stay robust to the child-effect structure at every age. `q` at 72 months is VG19 0.87 against VG20 0.77, and the 0.10 between them is still nine tenths of the interval a reader would be shown. VG10 is still not the model that disagrees: its worst gap from VG20 is 0.16.

With the cap at 72 the posterior summaries stop there, so the check is at 72; 84 is outside the reported grid and cannot be read from the fit outputs. The spoken population curve, reported to 90, gives the same picture at 84 as before.

## Why nothing moved

| band         | rows | children | understood | spoken |
| ------------ | ---: | -------: | ---------: | -----: |
| 8–24 mo      |  535 |      389 |        447 |    340 |
| 24–36 mo     |  460 |      359 |        406 |    341 |
| 36–48 mo     |  299 |      227 |        211 |    286 |
| 48–60 mo     |  213 |      163 |         96 |    188 |
| 60–72 mo     |  164 |      131 |         64 |    142 |
| **72–84 mo** |   71 |       61 |     **23** |     66 |
| 84–120 mo    |   58 |       49 |         13 |     58 |

`us_03`'s 284 rows all landed below 36 months — 174 in the 8–24 band and 110 in 24–36 — and the older bands are within a row or two of where they were. The band the cap is about holds **23 comprehension observations across 61 children**, against 25 on 2026-08-22, and twenty-three observations cannot separate a constant child offset from a child rate any better than twenty-five could. This is the outcome the 2026-08-22 note predicted when it said the test was "a different and harder test than the one the 2026-08-13 raise passed": more children is not the same as more children seen later.

## What would change it

The same thing that would identify VG22's production-rate spread ([`202609091400`](202609091400-is-vg22-the-better-description.md)): comprehension observations on children above 72 months, and in particular repeat visits on children already in the pool. The two questions are one question — whether the pool's follow-up can identify how a child's trajectory bends — and both now point at data collection rather than modelling.

## Housekeeping

- VG19 and VG22 became development steps on 2026-09-09 ([`202609091600`](202609091600-model-roles-settled.md)), so VG19 is no longer refit-current by default. The next rerun of this check needs `-Models vg19,vg20` (and `vg10`, for the control column) refitted together, not a cycle's by-product.
- The 2026-08-22 note carries a dated pointer to this rerun in its new §8.
