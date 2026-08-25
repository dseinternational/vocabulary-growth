# Four sensitivity variants, and the gate that could not see the one thing VG16 measures

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Registered 2026-08-25, before the VM refit window, because a variant registered afterwards needs its own fit. VG16, VG21 and VG23 had **no** registered sensitivities at all; the registry now holds 71 entries against 67.

## The defect found while registering them

`sensitivity/compare.py` scored eight trajectory series — `Ey_understood`, `Ey_spoken`, `Ey`, `q`, `r`, `p_any`, `Ey_any`, `gap` — plus `psi` as a scalar for VG15. **`beta_lag` was in none of them.**

VG16 supplies no other reported number. Its whole reason for existing is the cross-lag coefficient, and its numerical headline is currently withdrawn pending [#242](https://github.com/dseinternational/vocabulary-growth/issues/242). So a VG16 variant could have halved the coefficient and still been scored **robust**, because the trajectories had not moved — a registered check that cannot fail in the direction that matters, which is the failure mode the retired `us01-ceiling-excluded` variants exhibited in mirror image.

`compare.load_beta_lag` now reads it. Two details worth recording:

- **It reads `diagnostics.csv`, not a `posterior_summary_*` file.** No engine persists `beta_lag` as a summary series, because it is a scalar on the model rather than a curve over ages. Every fit writes `diagnostics.csv`, so this also works on fits made before the function existed — including the current VG16 fit, from which it returns 0.199 [0.089, 0.311]. (That fit predates the lag correction in `91f07c2` and its estimate is withdrawn; the point here is that the loader works against real output.)
- **A scalar is comparable where the series are not.** `dse-native-only` keeps 265 of VG16's 1,431 rows, so its age grids will overlap the baseline's only partially — VG10's equivalent variant reports partial coverage on 43 of 395 rows rather than a verdict. `beta_lag` has no age grid, so it is comparable regardless. The variant most likely to return "coverage too low to judge" on trajectories will still return a clean verdict on the coefficient.

## What was registered, and what a "sensitive" verdict would mean

| variant                        | override                           | reads as                                                                     |
| ------------------------------ | ---------------------------------- | ---------------------------------------------------------------------------- |
| `("vg16", "conditional-only")` | `spoken_fallback = paired_only`    | the cross-lag is carried by rows that never observed the predictor's parent  |
| `("vg16", "dse-native-only")`  | `dse_native_only = True`           | the lag depends on counts scored against a denominator their form never used |
| `("vg21", "vague-anchors")`    | two high-anchor Betas widened      | VG21's conclusions turn on anchors recentred from its own data               |
| `("vg23", "eta-flat")`         | `subject_re_correlation_eta = 1.0` | `rho_uq` is held up by its prior rather than by the data                     |

**`conditional-only` is the sharpest.** VG16's estimand is a claim about children whose earlier comprehension was measured, yet 455 of 1,428 spoken rows enter through the fallback branch with no observed comprehension parent — 444 with no comprehension total and 11 where spoken exceeds understood. Dropping them leaves exactly the population the coefficient is supposed to describe. Measured: the row set is unchanged at 1,431 (the treatment changes which likelihood branch spoken rows take, not which rows load), so this is a pure likelihood contrast.

**`dse-native-only` acts on the regressor, not only on the outcome.** The lag predictor is the logit of `understood / 810`, so a short-form source enters the _predictor_ already deflated. Measured: 265 rows, 181 children, four studies (`ie_01`, `ie_02`, `uk_02`, `uk_06`) against the baseline's 1,431 rows, 767 children and fourteen studies. Study composition changes as well as size, so read it for the coefficient rather than for the trajectories.

**`vague-anchors` is a double-dipping gate.** VG21's two high anchors were recentred on **in-sample medians**, because no CDI comprehension norm exists above 18 months ([#228](https://github.com/dseinternational/vocabulary-growth/issues/228)). Anchors set from the data and then used to fit the data need the test that Target 8 applied to every other recalibrated anchor in the project, all seven of which passed. This is `("vg13", "window-22-vague-anchors")` expressed against VG21's own baseline: VG21 already carries the window, the anchor ages, the GP domain and `eta_q_sigma`, so only the two Betas move, and the widened values are copied unchanged so the two entries stay comparable. `p_slope_hi_q` is deliberately `Beta(1.3, 1.3)` rather than flatter — a median at 0.5 would sit against the Oxford CDI's own 418-item ceiling and confound the test rather than sharpen it.

**`eta-flat` checks a prior chosen for comparability.** VG23's `eta = 2` was matched to VG20's so the Down syndrome and typically-developing correlations are estimated under the same prior. That is the right choice and an informative one, which is exactly why it needs checking: `eta = 1` is the flat LKJ, uniform over correlation matrices. If `rho_uq` moves materially, the DS-versus-TD contrast is partly a statement about two priors.

## What is _not_ registered, and why it is a decision rather than an omission

[#242](https://github.com/dseinternational/vocabulary-growth/issues/242) asks for six. Three have no fields to override: a gap ceiling, leave-one-study-out, and zero-count boundary or continuity-correction handling. Adding them means new fields on `BivariateModelDefinition` — which is shared by **all twelve** bivariate models (VG05, VG07-VG10, VG13, VG16, VG19-VG23), so a field there marks every one of them stale. That is precisely what `spoken_fallback` did in [PR #256](https://github.com/dseinternational/vocabulary-growth/pull/256).

So they are cheap only _before_ a refit and expensive after. Left for the study owner as an explicit decision rather than registered unilaterally.

The coefficient-prior-scale variant on `beta_lag_sigma` is a different case and was deliberately held: the field already exists, so it can be added at any time without re-staling anything.

## Verification

Each variant builds a real PyMC graph, not merely a valid definition — the distinction the `single-admin` breakage established, where two variants sat broken for two days because nothing built them. `tests/test_sensitivity_overrides.py` adds three tests that build each against the prepared database and check the property the variant depends on: that VG16's two keep `beta_lag` in the graph and narrow the data as claimed, that VG21's moves two priors and nothing structural (same rows, same studies, same free-RV count), and that VG23's keeps `rho_uq_raw` while relaxing only its concentration.

The base-model map in `test_variants_are_single_factor_or_documented_pairs` is now resolved from `MODEL_REGISTRY` rather than from a literal that had to be edited by hand whenever a model gained its first variant — it failed here as a bare `KeyError: 'vg16'`, which reads as a broken variant rather than as a stale test.
