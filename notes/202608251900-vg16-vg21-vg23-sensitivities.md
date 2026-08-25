# Four sensitivity variants, and the gate that could not see the one thing VG16 measures

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Registered 2026-08-25, before the VM refit window, because a variant registered afterwards needs its own fit — and, for three of them, because the _fields_ they need mark all twelve bivariate models stale, which is free now and a second full refit later. VG16, VG21 and VG23 had **no** registered sensitivities at all; the registry now holds 74 entries against 67.

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

## The remaining three, and the fields they needed

[#242](https://github.com/dseinternational/vocabulary-growth/issues/242) asks for six. Three had no fields to override, and the study owner's decision was to **add the fields now** rather than pay for them twice: a field on `BivariateModelDefinition` marks all twelve bivariate models stale (VG05, VG07-VG10, VG13, VG16, VG19-VG23), which is free while every one of them is already in the refit set and a second full refit afterwards.

Three fields, all inert at their defaults:

| field                               | default                               | variant                      |
| ----------------------------------- | ------------------------------------- | ---------------------------- |
| `lag_max_gap_months: float \| None` | `None` — no ceiling                   | `("vg16", "lag-gap-12")`     |
| `exclude_studies: tuple[str, ...]`  | `()` — admit everything               | `("vg16", "no-us01")`        |
| `lag_zero_handling: str`            | `LAG_ZERO_CLIP` — the historical clip | `("vg16", "lag-continuity")` |

**The defaults reproduce the graph exactly.** Initial log-probability against the pre-change tree, to every printed digit: VG16 `-19670.400000`, VG10 `-19670.170000`, VG13 `-94810.830000`, VG20 `-19671.150000`, with matching free-RV counts. The only thing that moves is the fingerprint — the same standard [#256](https://github.com/dseinternational/vocabulary-growth/pull/256) held itself to.

Each variant was measured on the real frame rather than assumed:

| variant          |      rows |  lagged | studies | min predictor |
| ---------------- | --------: | ------: | ------: | ------------: |
| VG16 baseline    |     1,431 |     477 |      14 |        −9.210 |
| `lag-gap-12`     |     1,431 | **436** |      14 |        −9.210 |
| `no-us01`        | **1,218** | **341** |  **13** |        −9.210 |
| `lag-continuity` |     1,431 |     477 |      14 |    **−7.391** |

**`lag-gap-12`** tests the constancy assumption directly: `beta_lag` is one number for gaps of 1 to 28 months (median 6), and a prospective association measured over two years is not the same quantity as one measured over six. The ceiling drops the _lag_, not the row — the observation still enters both likelihoods and simply stops informing the coefficient, so the gap question is not confounded with a sample-size change. It is also applied after the source wave is chosen, so which wave is the source never depends on the ceiling; applying it during the walk would let a row just over the ceiling fall back to an _earlier_ source and acquire a longer gap than the one just rejected.

**`no-us01`** is the leave-one-study-out check with the most leverage: `us_01` supplies 136 of 477 lagged rows, 28.5% of the evidence for a coefficient reported as a property of children with Down syndrome rather than of a study. `it_01` is next at 106 and is one registry line away — the field takes any tuple, so a full sweep over the eight contributing studies needs no code. `exclude_studies` raises if it matches no rows, because a leave-one-out check that removes nothing cannot fail, which is the failure mode the retired `us01-ceiling-excluded` variants had.

**`lag-continuity`** replaces the clip on the predictor's boundary. Seven of the 477 lagged rows have a source of exactly zero understood words, and the clip puts them at `logit(1e-4) = -9.21` — a value fixed by the floor rather than by the data, and identical whether the source form had 810 items or 396. The `+0.5 / +1` correction puts them at −7.39, derived from the inventory instead. Seven rows is few, but they sit at the extreme of the predictor's range, which is where a regression coefficient takes its leverage; non-boundary sources move by under 0.002 logits, so this is a boundary treatment rather than a rescaling.

The coefficient-prior-scale variant on `beta_lag_sigma` is still deliberately held: the field already exists, so it can be added at any time without marking anything stale.

## Verification

Each variant builds a real PyMC graph, not merely a valid definition — the distinction the `single-admin` breakage established, where two variants sat broken for two days because nothing built them. `tests/test_sensitivity_overrides.py` adds three tests that build each against the prepared database and check the property the variant depends on: that VG16's two keep `beta_lag` in the graph and narrow the data as claimed, that VG21's moves two priors and nothing structural (same rows, same studies, same free-RV count), and that VG23's keeps `rho_uq_raw` while relaxing only its concentration.

The base-model map in `test_variants_are_single_factor_or_documented_pairs` is now resolved from `MODEL_REGISTRY` rather than from a literal that had to be edited by hand whenever a model gained its first variant — it failed here as a bare `KeyError: 'vg16'`, which reads as a broken variant rather than as a stale test.
