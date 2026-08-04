# The Down syndrome understood-trajectory prior: anchor levels and the logit-linear mean

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Analysis and implementation note, 2026-08-04. **Implemented:** the understood trend anchors on the eight Down syndrome joint models (§7), and VG10 refitted against them (§10). **Proposed, not implemented:** the `eta_u` widening (§6) and the log-age mean form (§5), which is the only change that addresses the finding at its root. Every fit in §§1–9 is the `test`-config run of 2026-08-03 ([202608031341](202608031341-test-refit-after-data-and-prior-changes.md)) — 4 chains x 2,000 draws, not reporting quality; §10's refit is at the same configuration and is equally not reporting quality. **VG05, VG07, VG08, VG09, VG14, VG15 and VG16 remain stale** and must be refitted before anything from them is quoted.

> [!IMPORTANT]
> §10 refits VG10 and settles three things this note could only predict. The recalibration is **not over-committed**: both anchors move to prior CDF 0.51 and 0.66 with contraction unchanged at 0.89 and 0.90. It **did not move the answer**: the fitted population curve shifts by 0 to 2 words at every queried age, confirming §4's reading that the displacement was a prior-predictive failure and not a distorted posterior. And it **does nothing for `eta_u`**, which stays at prior CDF 0.880 with contraction 0.28 — §6's argument holds, and the log-age mean form (§5) remains the only proposal that addresses it.

## Summary

The prior predictive plot for words understood in VG10 (`prior_samples_u.png`) shows the population trajectory rising too slowly through the 20–60 month window. It does. Across 24–60 months the prior median population curve sits about 100 words below the fitted one, and 80% of prior mass falls below the frame's own median at those ages — 87% at 48 months. At 12 months the displacement reverses: the prior median is 47 words against a fitted and observed 16.

Two separate causes, and the second is the one that matters.

Within the 24–60 month window the displacement is almost entirely the **level of the two Beta anchors** — 94 of the 102-word mean gap, against 8 from the mean's shape. Those anchor priors are recalibrated in §7.

Across the whole range it is the **functional form of the mean**. `p_U(a)` is logit-linear in age between anchors at 24 and 84 months, and the fitted trajectory is strongly concave on that scale — its logit slope falls from +0.174 per month at 12–24 to +0.037 at 48–84. A straight line cannot follow that, so sliding it up to fix 24–60 pushes the 12-month end further out, which is exactly what the recalibration in §7 does. The residual after recalibration is structural, and the anchors are the wrong knob for it.

The measurable cost of the form error is carried by `eta_u`, the understood GP amplitude, which absorbs a systematic 1.18-logit correction the mean should be supplying. It sits at prior CDF 0.80–0.89 with contraction 0.29 in **all eight** Down syndrome joint models — the prior-limited signature this project has twice now treated as a defect ([202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §2). Fitting alternative mean forms to the fitted curve shows logit-linear in **log(age)** cuts the residual the GP must carry by a factor of four.

## 1. What was checked, and how

VG10's trace carries no `prior` group, so the prior was drawn afresh: the model was built through its own pipeline stages (`prepare_bivariate_re_data`, `configure_bivariate_priors`, `build_model_re`) and 4,000 draws of `p_u_plot` taken, with the output root redirected to a scratch directory so no model of record was touched. All figures below are on that draw, seed 20260804.

`p_u_plot` is the **population** curve — no study effects, no subject effects, no Beta-Binomial noise — while the blue scatter in the plot is individual administrations. The width of the orange fan is therefore not the diagnostic; the centring is. Two comparators are used for centring, and they agree:

- the frame's own median understood count in a ±3 month band, and
- the fitted population curve `p_u_query`, which reproduces the frame's medians closely (16 against 16.5 at 12 months, 113 against 132 at 24, 367 against 373 at 54).

Because subject effects are centred Normal on the logit scale and the logistic is monotone, the population curve is the median-child trajectory, so a median is the right comparator for it — the error corrected in [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §12, where a pool mean was read as if it were a median.

## 2. The displacement

| age | prior median | frame median | fitted | prior mass below the frame median |
| --- | ------------ | ------------ | ------ | --------------------------------- |
| 12  | 47           | 16.5         | 16     | 25%                               |
| 18  | 60           | 55           | 49     | 47%                               |
| 24  | 78           | 132          | 113    | 70%                               |
| 30  | 98           | 214          | 175    | 81%                               |
| 36  | 121          | 282          | 218    | 86%                               |
| 42  | 151          | 303          | 258    | 85%                               |
| 48  | 186          | 354          | 312    | 87%                               |
| 54  | 226          | 373          | 367    | 84%                               |
| 60  | 270          | 337          | 401    | 66%                               |
| 72  | 365          | 475          | 464    | 68%                               |
| 84  | 473          | 553          | 568    | 61%                               |

The sign flip at 12 months is the tell. A prior that were merely too low everywhere would be a level problem; one that is too high early and too low across the middle is a **slope** problem.

## 3. Where it comes from

The mean is

```text
slope     = (logit(p_hi) - logit(p_lo)) / (z_84 - z_24)
intercept = logit(p_lo) - slope * z_24
f_U(a)    = intercept + slope * z(a) + g_u(a)
```

with `p_lo ~ Beta(1, 7)` at 24 months, `p_hi ~ Beta(2, 1.5)` at 84 ([`gp_utils.trend_and_gp`](../src/vocab_growth/models/gp_utils.py:152)). Under VG10's anchoring the GP `g_u` is orthogonalised against `[1, z]` on the observed rows and pinned to zero at 54 months, so it contributes no level and no linear component and its prior median is ~0. **The prior median population curve is therefore the straight line through the two anchor medians**, and everything below follows from that.

Decomposing the prior-to-fitted gap against a logit-linear trend pinned at the _fitted_ anchor values separates the two causes:

| age | prior | line through fitted anchors | fitted | gap  | from anchor level | from mean shape |
| --- | ----- | --------------------------- | ------ | ---- | ----------------- | --------------- |
| 12  | 47    | 61                          | 16     | −31  | +13               | **−45**         |
| 24  | 78    | 110                         | 113    | +35  | +32               | +3              |
| 36  | 121   | 189                         | 218    | +97  | **+68**           | +28             |
| 48  | 186   | 302                         | 312    | +126 | **+115**          | +10             |
| 60  | 270   | 434                         | 401    | +131 | **+164**          | −33             |
| 72  | 365   | 560                         | 464    | +99  | +195              | **−95**         |
| 84  | 473   | 659                         | 568    | +95  | +185              | **−90**         |

Across 24–60 months the mean gap of +102 words is +94 anchor level and +8 shape. Outside that window shape dominates and reverses sign at both ends. The anchor medians are 76 words at 24 (fitted 110) and 475 at 84 (fitted 659).

## 4. The anchors are data-informed, but not prior-limited

The two anchors are strongly informed by the data despite the displacement:

| parameter       | posterior mean | prior CDF | contraction |
| --------------- | -------------- | --------- | ----------- |
| `p_slope_low_u` | 0.136          | 0.642     | **0.88**    |
| `p_slope_hi_u`  | 0.813          | 0.820     | **0.92**    |
| `eta_u`         | 0.943          | 0.884     | **0.29**    |

So this is a prior-predictive-check failure, not evidence that VG10's answer was distorted. The prior is off-centre; the likelihood overrules it and the posterior is 8–12 times narrower than the prior on both anchors. `eta_u` is the exception, and §6 takes it up.

One caveat on reading the anchors across the family. They only carry their nominal meaning — expected proportion at 24 and 84 months — in the models where the GP is orthogonalised against `[1, z]`. Elsewhere the mean and the GP are not separately identified, and the posterior anchors drift accordingly:

| model                    | 24 mo anchor | 84 mo anchor | `eta_u` | `eta_u` prior CDF |
| ------------------------ | ------------ | ------------ | ------- | ----------------- |
| VG05                     | 67           | 529          | 0.78    | 0.836             |
| VG07                     | 62           | 490          | 0.74    | 0.804             |
| VG08                     | 37           | 535          | 0.87    | 0.866             |
| VG09                     | 34           | 539          | 0.86    | 0.861             |
| **VG10** _(GP anchored)_ | **110**      | **659**      | 0.91    | 0.884             |
| VG14                     | 65           | 531          | 0.78    | 0.823             |
| **VG15** _(GP anchored)_ | **113**      | **663**      | 0.93    | 0.890             |
| **VG16** _(GP anchored)_ | **109**      | **658**      | 0.92    | 0.887             |

The three anchored models agree to within 4 words at 24 months and 5 at 84; the five unanchored ones scatter from 34 to 67. That is the ridge VG10 was built to remove, seen from a different direction, and it means the evidence about _where_ the anchors belong comes from VG10, VG15 and VG16 only. The prior, however, is shared by all eight, so the displacement in §2 applies to all eight.

## 5. The mean form is the root cause

Fitting each candidate two-parameter mean `logit(p) = a + b * t(age)` to the fitted population curve by least squares, and reading off what the GP is then left to carry:

| mean form                    | RMS residual | max residual |
| ---------------------------- | ------------ | ------------ |
| **age** (current)            | 0.438        | **1.184**    |
| **log(age)**                 | **0.139**    | **0.279**    |
| sqrt(age)                    | 0.289        | 0.749        |
| −1/age                       | 0.307        | 0.586        |
| log(age) + age (3-parameter) | 0.122        | 0.198        |

Log-age cuts the RMS residual by a factor of 3.2 and the maximum by 4.2, and a third parameter buys almost nothing beyond it. The residual pattern under the current form is a single systematic hump — −1.18 at 12 months, +0.27 to +0.49 across 24–54, back to −0.38 at 90 — which is precisely the shape of the displacement in §2.

The underlying fact is that the fitted logit slope decelerates by a factor of 4.7 across the range:

| window   | fitted logit slope per month |
| -------- | ---------------------------- |
| 12–24 mo | +0.174 (89% 0.162 to 0.187)  |
| 24–36 mo | +0.069                       |
| 36–48 mo | +0.045                       |
| 48–60 mo | +0.037                       |
| 60–84 mo | +0.037                       |

The mean has one slope for all of it.

**This is the change worth making, and it is not made here.** It alters the model graph, so by the convention of this project it is a new variant rather than an edit to a registered model, and it would want testing on one model before the family. It also changes what the anchors mean — they would interpolate in log-age — so §7's recalibration would need revisiting alongside it.

## 6. `eta_u` is carrying the form error, family-wide

`eta_u ~ HalfNormal(0.6)` has median 0.40. Every one of the eight Down syndrome joint models puts it between 0.74 and 0.93, at prior CDF 0.80–0.89 (§4 table), and VG10's contraction is 0.29 — the posterior is barely narrower than the prior and sits in its upper tail. That is the same signature this project has already acted on twice for `b_kappa_mag` and `eta_q`.

The temptation is to widen it, and it should be resisted for now. Widening `eta_u` from 0.6 to 1.0 moves the 24–60 month displacement from 79.8% to 77.9% — nothing — because the GP is orthogonalised and anchored, so more amplitude buys symmetric wiggle around the straight line rather than the systematic curvature the data want. A wider `eta_u` would relieve a real prior-data conflict without touching the thing causing it, and it interacts with sampler geometry in a family where `eta_q` was tightened for exactly that reason (PRIORS.md, and [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §5).

**Proposed, contingent on §5:** if the log-age mean is adopted, recheck `eta_u` before changing it — the strain should mostly disappear. If the current form is kept, `HalfNormal(0.9)` is the honest scale, and it should be tested for its effect on divergences and ESS before adoption.

## 7. Implemented: the understood anchors

Anchor levels are the wrong knob for a shape problem, but they are the right knob for the level, and the level is genuinely off. Five candidate pairs were drawn through the full prior predictive and scored by displacement from centred (50%) at each age, weighted by the comprehension rows each band holds:

| variant                             | row-weighted \|displacement − 50\| | worst |
| ----------------------------------- | ---------------------------------- | ----- |
| current `Beta(1,7)` / `Beta(2,1.5)` | 25.2                               | 36.8  |
| A `Beta(1.5,8)` / `Beta(2.5,1.5)`   | 21.8                               | 34.6  |
| B `Beta(1.5,9)` / `Beta(2.5,1.3)`   | 21.8                               | 31.6  |
| C `Beta(2,9)` / `Beta(2.5,1.2)`     | 18.6                               | 41.0  |
| **D `Beta(1.5,8)` / `Beta(3,1.3)`** | **19.7**                           | 33.5  |

**D is adopted**, on evidence rather than on the summary statistic:

- **24 months, `Beta(1, 7)` → `Beta(1.5, 8)`.** Median 76 → 108 words, 5–95% 17–306. The 21–26 month band is the densest in the frame (160 rows, 156 children) with a median of 132, and the three identified models put the anchor at 109–113. A median of 108 is centred on that evidence with the lower tail left wide.
- **84 months, `Beta(2, 1.5)` → `Beta(3, 1.3)`.** Median 475 → 592 words, 5–95% 260–780. This one is thinly evidenced — four administrations between 78 and 95 months, median 554 — against fitted anchors of 658–663. 592 sits between the two and the tails stay wide, deliberately.

Effect on the prior predictive:

| age                               | 12  | 18  | 24  | 30  | 36  | 42  | 48  | 54  | 60  | 72  | 84  |
| --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| % below frame median, current     | 25  | 47  | 70  | 81  | 86  | 85  | 87  | 84  | 66  | 68  | 61  |
| % below frame median, **adopted** | 16  | 36  | 59  | 72  | 77  | 71  | 72  | 64  | 41  | 48  | 43  |

Every band from 24 months up improves; 60–84 months lands close to centred. **The 12 and 18 month bands get worse** — 25% to 16% and 47% to 36% — for the reason §5 gives: lifting a straight line to fit its middle raises its backward extrapolation too. That is a known and accepted cost of this change, and the reason §5 rather than §7 is the fix.

Applied to all eight models carrying these anchors — **VG05, VG07, VG08, VG09, VG10, VG14, VG15, VG16** — in [`definitions.py`](../src/vocab_growth/models/definitions.py). Seven spell the anchors out; VG15 takes them from the `JointModelDefinition` dataclass defaults, which are changed with them. The prior is a claim about Down syndrome comprehension at two ages and does not depend on whether a given model identifies it, so leaving the unanchored five behind would split the family's prior for no reason.

Verified after the change by rebuilding VG10 from the registered definition and re-drawing the prior: the adopted row above is reproduced to within 1 percentage point at every age, and the prior's median words gained between 24 and 60 months moves from 162 to 235, against 205 in the frame. `ruff`, the 451-test suite, Prettier and CSpell all pass.

**VG02 is deliberately excluded.** The single-outcome Down syndrome understood model carries the same `Beta(1,7)` / `Beta(2,1.5)` pair at the same anchor ages, and probably has the same problem, but its prior predictive has not been measured on its own frame. It should be checked before it is moved.

## 8. Consequences

1. **The eight Down syndrome joint fits are stale.** Priors are part of the model graph; every one of them needs refitting before any figure is quoted. They were already stale against the reporting configuration — the current fits are `test` config from 2026-08-03. VG10 has since been refitted (§10); the other seven have not.
2. **PRIORS.md was wrong before this change and is updated by it.** Its anchor table listed the joint understood anchors as `Beta(1, 10)` and `Beta(1.1, 1.1)`, which is the pre-#135 state; the registry has read `Beta(1, 7)` / `Beta(2, 1.5)` since. Both rows are corrected and given the new values.
3. **The evidence class is scale calibration, not an independent anchor.** There is no independent Down syndrome comprehension norm in the library — Berglund et al. (2001) is production-only — which the model definitions already state. This recalibration is centred on the project's own frame and on the fitted anchors of three models, so it is the same weaker evidence class already accepted for this trajectory, made more accurate. It is not the posterior-derived double-dipping that #155 removed from the `q` anchors: the target is the prior's _location on the observable words scale at two fixed ages_, checkable against the frame directly, and the tails are deliberately left wider than the fits.

## 9. Open

1. **Fit a log-age mean variant** (§5) and compare against VG10 on the same frame. This is the substantive follow-up; everything else here is palliative, and §10 confirms it — the recalibration left `eta_u` exactly where it was.
2. **Recheck `eta_u`** after §5, or test `HalfNormal(0.9)` if the current form is kept (§6).
3. **Measure VG02's prior predictive** and move its anchors if it shows the same displacement (§7).
4. ~~**Refit the eight** and confirm the anchors land nearer prior CDF 0.5 with contraction unchanged.~~ Done for VG10 — §10, and the recalibration is not over-committed. **The remaining seven — VG05, VG07, VG08, VG09, VG14, VG15, VG16 — are still to refit.**
5. The five unanchored models' anchor posteriors scatter from 34 to 67 words at 24 months (§4). That is a known consequence of the un-orthogonalised GP, but it means their reported `intercept_u` / `slope_u` should not be read developmentally.
6. **`g_unit_u_hsgp_coeffs[2]` is now the model's only R-hat failure** (§10). The understood GP's low-order coefficients are where every remaining diagnostic failure sits, which is what §5 predicts should happen when the mean cannot supply the curvature.

## 10. VG10 refitted against the new anchors

`test` configuration, 4 chains x 2,000 draws, seed 47, no overrides — the same configuration as the 2026-08-03 run it is compared with, so the prior is the only thing that changed. 5m 22s, rendered.

### The recalibration is not over-committed

| parameter       | posterior median | prior CDF before | prior CDF after | contraction before | contraction after |
| --------------- | ---------------- | ---------------- | --------------- | ------------------ | ----------------- |
| `p_slope_low_u` | 110 → 111 words  | 0.642            | **0.511**       | 0.88               | 0.89              |
| `p_slope_hi_u`  | 659 → 660 words  | 0.820            | **0.658**       | 0.92               | 0.90              |
| `eta_u`         | 0.943 → 0.900    | 0.884            | **0.880**       | 0.29               | 0.28              |

Both anchors move toward the middle of their priors with contraction unchanged, which is the result open item 4 asked for: the prior moved to where the data already were rather than pulling the data toward it. Had contraction fallen, the new priors would have been doing work the likelihood was previously doing.

### It did not move the answer

The fitted population curve for words understood, before and after, at every queried age:

| age    | 12  | 18  | 24  | 30  | 36  | 42  | 48  | 54  | 60  | 66  | 72  | 78  | 84  | 90  |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before | 16  | 49  | 113 | 175 | 218 | 258 | 312 | 367 | 401 | 426 | 464 | 519 | 568 | 595 |
| after  | 16  | 50  | 113 | 175 | 219 | 259 | 313 | 368 | 402 | 427 | 466 | 520 | 570 | 597 |

Zero to two words, everywhere. This is the direct confirmation of §4: the anchors were data-informed at contraction 0.88–0.92 and the likelihood was already overruling the off-centre prior, so correcting the prior changes the prior predictive and nothing else. **No result previously reported from VG10 is revised by this change.**

### `eta_u` is untouched, as §6 predicted

Prior CDF 0.884 → 0.880, contraction 0.29 → 0.28. The anchor recalibration does not relieve the GP amplitude at all, because the strain does not come from the anchor levels — it comes from a mean that cannot express the trajectory's curvature. §5 remains the only proposal on the table that addresses it.

### Diagnostics

The convergence gate returns REVIEW, as it did before this change:

|                | before                                                                                          | after                           |
| -------------- | ----------------------------------------------------------------------------------------------- | ------------------------------- |
| gate           | not passed                                                                                      | not passed                      |
| divergences    | 2                                                                                               | 2                               |
| max R-hat      | 1.0177                                                                                          | **1.0110**                      |
| min ESS        | 387                                                                                             | 283                             |
| min BFMI       | 0.446                                                                                           | 0.453                           |
| R-hat failures | `kappa_min_s`, `kappa_excess_old_s`, `b_kappa_s`, `a_kappa_s`, `g_unit_u_hsgp_coeffs[2]`, `[5]` | **`g_unit_u_hsgp_coeffs[2]`**   |
| ESS failures   | `kappa_min_s`, `g_unit_u_hsgp_coeffs[2]`, `[3]`                                                 | `g_unit_u_hsgp_coeffs[2]`–`[5]` |

REVIEW is the pre-existing status for VG10 at `test` and is not introduced here. What did change is where the failures sit. Six R-hat failures become one, and the four spoken-dispersion parameters clear entirely — plausibly because a better-centred understood trend leaves less for the rest of the graph to compensate for, though nothing here establishes that and it was not predicted.

Every remaining failure, on both R-hat and ESS, is now a low-order coefficient of the understood GP. That is the same place §5 locates the problem, and it is the sharpest available evidence that the mean form rather than the anchor levels is what is left to fix. Min ESS falls from 387 to 283, which is the cost side of the same observation: the GP coefficients are being asked to do systematic work, and they mix poorly while doing it.
