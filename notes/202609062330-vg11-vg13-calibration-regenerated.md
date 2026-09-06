# The VG11-VG13 dispersion calibration, regenerated on the registered language scopes

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-06. Closes the second half of [#240](https://github.com/dseinternational/vocabulary-growth/issues/240)'s prior-calibration item and discharges the provenance caveat that has stood in `docs/models/PRIORS.md` since 2026-08-23.

## What was wrong, and what had already been fixed

The VG11-VG13 review ([202608231537](202608231537-vg11-vg12-vg13-statistical-review.md) §3.6) found that `scripts/kappa_conditional_calibration.py` built its frames by calling the loader **without** `definition.td_languages`, so it calibrated on the English-only default while the registered models fit English plus Romance. The frames differed by 12-15%: 16,235 rows against VG11's registered 18,500, 5,997 against VG12's 7,049, 5,406 against VG13's 6,356.

The tooling fix landed with the review's remediation: both `univariate_frame` and `bivariate_frames` now take the scope from the definition, and `tests/test_kappa_conditional_calibration.py::test_frames_use_the_registered_language_scope` pins all three row counts against the registered frames. What had **not** happened is the part that decides whether any prior needs to change: rerunning the estimator and comparing.

## The regenerated calibration

Run on 2026-09-06 with `uv run python scripts/kappa_conditional_calibration.py vg11-spoken vg12-understood vg13-understood vg13-q`. Every pool loaded exactly the registered row count, so the frames are the models' own.

| Pool                 |      n | Registered `k_min` / `e_young` / `e_old` | Regenerated                      | Largest move |
| -------------------- | -----: | ---------------------------------------- | -------------------------------- | ------------ |
| VG11 spoken          | 18,500 | 6 / 311 / 44                             | 6.08 / 283.1 / 44.4              | −0.13 SD     |
| VG12 understood      |  7,049 | 3 / 40 / 63                              | floor unidentified / 37.2 / 67.2 | +0.07 SD     |
| VG13 `kappa_u`       |  6,356 | 30 / 10 / 90                             | 27.9 / 12.1 / 82.6               | +0.21 SD     |
| VG13 `kappa_s` (`q`) |  6,241 | 3 / 33 / 27                              | floor unidentified / 34.7 / 30.9 | +0.19 SD     |

"Move" is on the log scale in units of that prior's own `s` (0.7 for VG11 and VG13's `q`, 0.9 for the two understood outcomes), because that is the scale the prior is stated on. **The largest is 0.21 prior standard deviations**, on VG13's young understood anchor.

**No registered prior changes.** A fifth of one prior SD is smaller than the rounding the settings already carry — VG11's `e_young` is registered at 311 against a fit of 283.1 and was itself a rounded 317 — and moving any of them would be a graph change requiring a refit of VG11, VG12, VG13, VG21 and VG23 to buy a shift the prior cannot resolve. The review predicted this ("the practical effect appears limited"), but it was a prediction from one recomputed number; it is now measured on all four pools.

Two structural statements in `PRIORS.md` were also re-checked on the corrected frames and both survive: VG12's conditional fit still puts **no** mass on a floor (`kappa_min` → 0 with an unbounded standard error, a rising curve that never reaches one inside the frame), and VG13's floor is still genuinely the young-age asymptote at 27.9 rather than a floor at 3.

## The loading diagnostic moved, and one claim with it

`--loading` refits with an age-varying subject loading, and is the measurement behind `PRIORS.md`'s account of what a rising `kappa` on the understood outcomes means. On the registered frames:

| Pool            |  Loading buys | LR (1 df) | Child scale over the anchor span |
| --------------- | ------------: | --------: | -------------------------------- |
| VG11 spoken     |   263.7 units |     527.4 | −23%                             |
| VG12 understood |   165.7 units |     331.4 | −44%                             |
| VG13 understood |   110.8 units |     221.6 | −33%                             |
| VG13 `q`        | **7.3 units** |  **14.6** | **+34%**                         |

The three affected pools' range becomes **110.8-263.7** units, against the 111-237 recorded from the English-only frames, and the sign reversal still holds: on both understood outcomes a constant-`tau` fit has `kappa` rising with age (VG12 37.2 → 67.2, VG13 40.0 → 110.5) and the age-varying loading turns it over (79.2 → 51.6, 75.3 → 61.4).

**`q`'s number is the one that changed materially: 0.8 units on the English-only frame, 7.3 here.** 7.3 log-likelihood units is a likelihood-ratio statistic of 14.6 on one degree of freedom, so "shows **no** loading drift at all" is no longer a defensible reading and `PRIORS.md` has been corrected. The argument it supports survives, for two reasons that are stronger than the significance test:

- **Magnitude.** 7.3 units against 110.8-263.7 is 15 to 36 times smaller, on the same children and the same design.
- **Sign.** The 810-item compression mechanism predicts the apparent child scale **shrinking** with age on understood, which is what all three affected pools show (−23% to −44%). `q`'s drift is **+34%** — the opposite direction, so it is not the compression signature at a reduced size; it is a different and much smaller effect.

`q`'s denominator is the child's own understood count, so the form's extent cancels, and that remains the reason the compression does not reach it. What the +34% is instead has not been established here.

## What is still on the English-only frames

Two measurements in `PRIORS.md` come from [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §21 rather than from the tooling, and the calibration script has no mode that reproduces them:

- the per-age-cell understood `kappa` profile quoted as 19.6, 21.0, 110.7 at 14, 15 and 16 months; and
- the rescoring measurement — scoring the identical rows out of each row's own form instead of 810 removes 84-96% of the drift.

Both are recorded in `PRIORS.md` as still carrying the earlier scope. Neither sets a prior; the first is quoted as the reason `s = 0.9` rather than 0.7 on the understood outcomes, which the regenerated fits do not disturb.

## Reproducing this

```bash
uv run python scripts/kappa_conditional_calibration.py vg11-spoken vg12-understood vg13-understood vg13-q
uv run python scripts/kappa_conditional_calibration.py --loading vg11-spoken vg12-understood vg13-understood vg13-q
```

Both are deterministic L-BFGS fits on the prepared DuckDB frame and take about a minute each. Related: [202608231537](202608231537-vg11-vg12-vg13-statistical-review.md), [202608231830](202608231830-vg11-vg13-immediate-remediation.md), [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md).
