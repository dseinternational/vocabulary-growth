# Pre-refit changes: the uk_02 mask gap, sex as a covariate, VG26, #240's typically developing variants, and the spoken fallback kept

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-13. **Decisions, by the study owner, the same day:** add sex to the reporting models, accepting missing data while sex is sought for more Down syndrome studies ([#324](https://github.com/dseinternational/vocabulary-growth/issues/324)); fix the `uk_02` gap in the comprehension-below-production mask ([#236](https://github.com/dseinternational/vocabulary-growth/issues/236)); register VG26 = VG21 + `rho_uq` ([#240](https://github.com/dseinternational/vocabulary-growth/issues/240)); register the typically developing variants #240 needs; keep the spoken fallback as it is (#236 items 1–2). **Why now:** every one changes a frame, a graph or the executable-code signature, and the full refit is to run after the open issues are worked down, so each has to land before it or force another. **What this note is:** the record of what was built, what was measured on the way, and what each change does and does not do. Nothing here was fitted for record.

## 1. The mask gap

`mask_comprehension_below_production` compared `understood` against the recorded `produced` union and required that union to be present, so a row with no `produced` value could not be tested. The one such contradiction is a `uk_02` child at 48 months: 347 understood, 387 spoken, 254 signed, `produced` missing. A child who says 387 words produces at least 387.

The rule now compares against `max(produced, spoken)`, skipping missing values. Measured on the 2026-09-13 pool, it masks **exactly 11** comprehension counts where it masked 10: the same seven `ie_01`, two `uk_01` and one `it_01`, plus that `uk_02` row. No recorded `produced` falls below `spoken`, which is why nothing else moves.

**`signed` is not a term, though #236 proposed it.** `produced` is the modality union in some sources and the spoken count alone in others: `ie_02`, `uk_04`, `uk_05` and `uk_06` record it equal to `spoken`, and `signed` exceeds it on **132** of their rows. No administration has `understood < signed`, so including it would change nothing on today's pool; leaving it out keeps the rule's meaning the same in every source. The sum `spoken + signed` still flags 87 administrations, against 11.

**The fix had to be made twice.** The joint engine reads `uk_02`, `uk_07` and `es_01` from their own CSVs rather than from the merged view, so no loader rule reaches those rows — and the `uk_02` record is a marginal-only row of `uk_02`'s CSV. After the loader change the paired frames had moved and VG15's still carried the record. The joint frame builder now applies the same rule to each cross-tab source's marginal-only rows, against that source's own union column (`production`, `produced`, `spoken_or_gestured`). The four-cell rows need no check: their understood total is the cell sum.

Frames that moved: VG02, VG05, VG07–VG10, VG14, VG16, VG19, VG20, VG22 (the merged-view path) and VG15, VG24, VG25 (the cross-tab path). VG01, whose outcome is spoken, and every typically developing frame are unchanged.

## 2. Sex as a covariate

The design is #324's: a centred covariate estimated only where sex is recorded, girls `+1/2`, boys `-1/2`, a child of unrecorded sex `0`, with no row dropped. It is licensed by where missingness sits — in the Down syndrome pool exactly study-level, eight studies recording sex for every row and seven for none — and every model carrying it also carries study intercepts, which absorb the unrecorded studies' sex mix. The coefficients are identified from the recording studies, on the assumption that the effect is common across studies. The full statement is in `docs/models/README.md`, "Sex as a covariate".

**Which models.** The ones whose numbers are reported: VG15, VG20, VG24 and VG25 (through VG24); VG11, VG12, VG21 and VG23; and VG26 (through VG21). No development step. Prior `Normal(0, 0.5)`, `definitions._SEX_EFFECT_SIGMA`, the VG20 experiment's. **A consequence worth stating:** VG10 and VG13 are no longer exact nested nulls of VG20 and VG23. The tests that pinned the one-factor contrast now hold sex level and still find the correlation to be the only structural difference.

**Coverage on the prepared frames (measured).**

| frame                  | children girls / boys / unrecorded |                                              rows unrecorded |
| ---------------------- | ---------------------------------- | -----------------------------------------------------------: |
| VG20, VG15, VG24, VG25 | 259 / 300 / 384                    |                                                 711 of 1,708 |
| VG11                   | 6,657 / 7,118 / 778                | 1,449 of 18,500 (`Smith` 1,401, `Hoff` 44, `Kalashnikova` 4) |
| VG12                   | 2,766 / 3,053 / 0                  |                                                            0 |
| VG21, VG26             | 2,712 / 2,995 / 0                  |                                                            0 |
| VG23                   | 2,619 / 2,877 / 0                  |                                                            0 |

No child carries two sex values in either pool, and none has sex recorded on some rows and missing on others; the coder resolves sex per child anyway and refuses a conflict.

**What changed where.**

- **Definitions.** `sex_effect_sigma` and `sex_known_only` moved from the unregistered `BivariateSexShiftModelDefinition`, which no longer exists, onto `BivariateModelDefinition`; `sex_effect_sigma` also onto `UnivariateREModelDefinition` and `JointModelDefinition`. Classified GRAPH and DATA, with `BACKFILL_DEFAULTS` entries of `None` and `False` checked against the two readers in `models/sex_covariate.py`. The entries matter for the development steps that keep the defaults.
- **Loaders.** The Wordbank query selects `sex`, recoded to the Down syndrome view's `'M'`/`'F'`, as its last column. **Every frame that does not request it is byte-identical**: all twenty-two registered frames were rebuilt against a snapshot taken after the mask fix, and those carrying sex differ only by the column. The joint engine's cross-tab rows take their child's sex from the merged view by study and child; `uk_02`, `uk_07` and `es_01` come out fully covered and `nz_01`, which records none, at zero.
- **Graphs.** `beta_sex` on the univariate outcome logit; `beta_sex_u` and `beta_sex_q` on the bivariate understood and production-ratio logits; `beta_sex_u`, `beta_sex_q` and `beta_sex_sign` on the joint marginal logits **and the cross-tab compositions**. The last is deliberate and the opposite of what the child effects do: those are kept out of the Dirichlet-Multinomials because a free offset per child is co-identified with `psi` on thin rows, whereas a sex coefficient is one scalar on a covariate the data fix — the argument that lets VG25's lag into the cells. Measured on a synthetic build, moving `beta_sex_q` changes `pi_cells_obs` on every girl and boy row and on no row of unrecorded sex. Graph baselines were regenerated for the nine carrying models only, and every other model's entry is unchanged.
- **Predictions.** Population trajectories stay at contrast zero, the sex-balanced midpoint. The univariate and bivariate engines draw a new girl and a new boy from the **same** child-effect draw as the existing new child, so the two are paired: on a synthetic build their logits differ by exactly `beta_sex_u` in every draw. Their counts are drawn separately, so a predictive range still carries each count's own administration noise. The joint engine draws no new child at query ages, so its by-sex tables carry expected vocabulary only, as its pooled tables do.
- **Outputs.** `posterior_summary{,_u,_s,_sign}_by_sex.csv`, `posterior_summary_{q,r}_by_sex.csv`, `posterior_summary_sex_difference.csv` and `posterior_summary_sex_effect.csv`, written at fit time because `--render-only` does not rebuild summary tables. `report_cells.render_sex_section` renders them in a new "By sex" section on each carrying model's page, and the priors table gains a row per coefficient.
- **The non-random-effect bivariate engine** (VG05) refuses the field rather than ignoring it.
- **`vg20_sex_arm.py`** overrides VG20 directly; its `full` arm is now VG20 without the covariate.

**Not done.** Recovering sex from the providers for `us_03`, `uk_03`, `uk_04` and `ie_01`, which would take Down syndrome coverage from 59% to about 84%, is a request outside the code. When that data arrives the frames move again, so it should reach the database before a reporting refit that means to use it. `scripts/experiments/sex_effect_by_study.py`, which asks whether the effect is common across studies, is still to be run against a fit.

## 3. VG26

VG26 is `_as_definition_subclass(VG21, BivariateCorrelatedSubjectREModelDefinition, subject_re_correlation_eta=2.0)`, the change VG23 makes to VG13 applied to VG21's window, registered on the `bivariate_re` engine with a report template, recovery spec and every pinned test set updated. It is **unclassified**, so it is in the default refit scope (now nine models), and VG21 keeps its TD-reference role and VG13 its superseded one until VG26 has a fit. Its page carries #240's four checks, and reads VG23's `rho_uq` from the sibling output directory to set the 8–18 and 8–22 month values side by side.

## 4. #240's typically developing variants

Nineteen arms, none fitted:

| item                            | arm                  | on                           |
| ------------------------------- | -------------------- | ---------------------------- |
| 5, the 200-row threshold        | `no-study-threshold` | VG11, VG12, VG21, VG23, VG26 |
| 5, intercept-only study effects | `study-age-slopes`   | VG11, VG12, VG21, VG23, VG26 |
| 1, form-scale compression       | `a1-tau-age-varying` | VG11, VG12, VG21             |
| 6, GP amplitude                 | `eta-q-wide`         | VG21, VG23, VG26             |
| 6, VG13's debt                  | `single-admin`       | VG21, VG26                   |
| 6, VG13's debt                  | `vague-anchors`      | VG26                         |

Two needed engine work:

- **Per-study age slopes**, `study_age_slope_sigma`, on both random-effect engines: a zero-sum slope per study in logits per year from the GP anchor age, `tau_slope ~ HalfNormal(0.5)` (and `tau_u_slope`, `tau_q_slope` on the paired models), entering the observation logits only. A synthetic build confirms each row moves by its own study's slope times its years from the reference, and that the population grids are untouched, by value. Nested at zero: with every study slope at zero the graph is the intercept-only model's. BACKFILL `None`.
- **Proposal A1 on the univariate random-effect engine**, which the resolver already understood and the engine never implemented. Under VG11's and VG12's variance partition the young-anchor scale is the partition's own `tau_subject`, so the arm adds exactly one parameter, `log_tau_subject_ratio`, and holds dispersion flat (`kappa_excess_old` leaves the free set, `kappa_old = kappa_young`). **It is not nested at zero**, whatever `definitions.AgeVaryingSubjectScale` has said since VG10's arm: at `log_tau_subject_ratio = 0` the child scale is constant but dispersion is still flat, and VG11's and VG12's is not, so a ratio interval covering zero does not say the model of record is adequate. The comparison the arm supports is where the age variation sits, as the registry entry says. A1 is refused with singleton marginalisation and, as before, with a correlated child block — which is why VG23 and VG26 have no A1 arm.

Not registered: #240 item 2, held-out validation for the typically developing models, which is script work (`notes/202609131214`) rather than a variant.

## 5. The spoken fallback is kept

#236 items 1–2 asked to quantify or replace the `product_marginal` fallback. The 2026-09-06 arms answered the question they could: changing how the fallback is parameterised barely moves anything (`fallback-dispersion` 380/381 on VG14 and 87/88 on VG15 within the baseline 89% intervals; `marginal-moments` 88/88 on VG15 but 324/381 on VG14, sensitive on `Ey_understood` and the gap; #289 task 1.2), while removing the rows (`paired-only`) moves the trajectories a great deal and cannot separate "the fallback rows matter" from "less data". The exact marginalisation #236 contemplated targets the functional form, the part those arms find least consequential. **Decision: keep `product_marginal` as the default**, with the three arms remaining registered on VG10, VG14, VG15 and VG20 as the record of its cost. Nothing in the code changed for this.

## 6. What the refit inherits

- **Every fit of record is stale for publication**, as it already was on the signature; these changes add frame and definition staleness. The reporting models' 2026-09-08 fits now fail even render and consumer-script validation (definition or frame), so a remaining read of them — the VG15 → VG24 promotion review, for instance — needs `--allow-stale-fit`. VG01, VG03, VG04 and VG13 keep matching frames and definitions: VG01, VG03 and VG04 because `UnivariateModelDefinition` gained no field, VG13 through the three `BivariateModelDefinition` backfill entries.
- **The default scope is nine**: VG15, VG20, VG24, VG25, VG26, VG11, VG12, VG21, VG23. The runbook's lists are updated; `vg26` joins the serial TD pass.
- **By-sex numbers exist only after the refit.** The report pages carry the section now and say plainly when a fit has no covariate.

## 7. Checks run

- The full test suite, fast and slow, after the §8 and §9 fixes: **2,796 passed, 53 skipped**, no failures. All are parametrised cases that do not apply: 47 in `test_reporting_age_policy.py`, which runs over whatever fits of record sit under the output root and so moves with the machine, and 6 in `test_clamp_scope.py` for the univariate models that have no clamp field. New modules `tests/test_sex_covariate.py` and `tests/test_td_sensitivity_structures.py`; `tests/test_sex_shift_variant.py` removed with the class it pinned.
- Ruff and mypy clean; spellcheck and Prettier clean on the changed documents.
- Frame hashes for all twenty-two registered models, before and after, as in §2.
- `dev`-tier smoke fits through the whole pipeline, with render, into a scratch output root: see §8.

## 8. Smoke fits

Four `dev`-tier fits into a scratch output root (`D:/output/vocabulary-growth-smoke-20260913`), one per engine that gained code plus the univariate A1 arm, each through every pipeline stage and, for the three models, `--render`. `dev` chains are short, so every convergence gate read REVIEW and **none of the numbers below is a result**; they show the pieces run end to end and land where the exploratory work said they would.

| fit                       | wall time | gate                           | what it showed                                                                                                                                                                                        |
| ------------------------- | --------: | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VG20                      |  4 m 59 s | 0 divergences, max R-hat 1.09  | `beta_sex_u` +0.18 [+0.06, +0.31], `beta_sex_q` +0.37 [+0.17, +0.58]: girls ahead on both, the larger difference on production, as the #324 exploratory arm found                                     |
| VG15                      |  7 m 32 s | 2 divergences, max R-hat 1.06  | `beta_sex_u` +0.18 [+0.05, +0.33]; `beta_sex_q` +0.07 and `beta_sex_sign` +0.06, both intervals spanning zero; by-sex tables for all five quantities on each outcome's own age grid                   |
| VG12                      |  6 m 16 s | 3 divergences, max R-hat 1.06  | `beta_sex` +0.12 [+0.08, +0.15] on typically developing comprehension; new girl and new boy drawn from one child draw                                                                                 |
| VG12 `a1-tau-age-varying` |   5 m 4 s | 17 divergences, max R-hat 1.08 | the free set gains `log_tau_subject_ratio` (median −0.50) and loses `kappa_excess_old`; `kappa_old` equals `kappa_young`; the by-sex predictive runs through the A1 path; `beta_sex` unmoved at +0.12 |

The A1 arm moved `subject_variance_share` from 0.60 to 0.81. That is the kind of movement #240 item 1 asks about, and at `dev` tier with 17 divergences it is not evidence either way.

**What reading the rendered pages found**, all fixed in `report_cells.render_sex_section` and re-rendered:

- Rounding printed a lower bound of −0.3 words as `-0` and the youngest spoken gaps as `+0 to +0` beside a probability of 0.98. A row whose values are all under ten words now keeps one decimal, and a rounded zero prints as `0`.
- The new-child table listed every girl and then every boy. It is now one row per outcome and age with the two side by side.
- The closing callout said the population figures describe "a sex-balanced child in both populations" and described the Down syndrome pool's recording on every page, but each carrying model covers one population. It now states the fit's own coverage, read from the frame and only if that frame still hashes to the fitted one: by child, by study, and separately for studies that record no sex (absorbed by the study effect) and studies missing a few children (VG11's `Hoff` and `Kalashnikova`, where the unrecorded children sit beside coded ones and the assumption is that recording is unrelated to sex).
- The new-child caption said girls and boys "differ only by the sex coefficient". They share the child-effect draw but each count is drawn separately, so a range also carries administration noise; corrected here, on the page and in `docs/models/README.md`.
- The coefficient caption read "for comprehension" on every page, which is wrong for VG11, whose outcome is spoken. It now names what each coefficient present is on.

## 9. Second review pass

A second pass, at the study owner's request to review and double-check, ran the structures §8 had not exercised and set an independent read-only review against the engine diff.

**More smoke fits**, `dev` tier, same scratch root:

| fit                     | wall time | gate                           | what it showed                                                                                                                                                                                                                                                            |
| ----------------------- | --------: | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VG26                    | 10 m 50 s | 0 divergences, max R-hat 1.06  | first fit of the model; `rho_uq` 0.141 [0.111, 0.172], beside VG23's 0.127 [0.095, 0.159] from its 2026-09-08 fit; `beta_sex_u` +0.12, `beta_sex_q` +0.15; the page renders, including the window comparison                                                              |
| VG26 `single-admin`     |   7 m 4 s | 5 divergences, max R-hat 1.01  | no child effects and no correlation; by-sex tables written from the coefficients                                                                                                                                                                                          |
| VG21 `study-age-slopes` | 11 m 55 s | 0 divergences, max R-hat 1.05  | `tau_u_slope` 0.36, `tau_q_slope` 0.29 sampled beside the intercepts; `beta_sex` unmoved                                                                                                                                                                                  |
| VG11                    | 16 m 32 s | 25 divergences, max R-hat 1.07 | `beta_sex` +0.27 [+0.25, +0.31] on spoken, near the +0.33 [+0.29, +0.37] the 2026-09-04 regressions found on American English Wordbank without child effects; the page's coverage callout separates `Smith` from the 21 children missing within `Hoff` and `Kalashnikova` |
| VG11 `study-age-slopes` | 18 m 45 s | 8 divergences, max R-hat 1.15  | `tau_slope` 0.25 on the univariate engine; `beta_sex` unmoved                                                                                                                                                                                                             |
| VG25                    |   8 m 0 s | 2 divergences, max R-hat 1.12  | the joint engine with its sign lag, fitted after the cross-tab sex guard went in, which the frame passed; `beta_sex_u` +0.18, `beta_sex_q` +0.14, `beta_sex_sign` +0.02 spanning zero; by-sex tables and the coverage callout render                                      |

**Found and fixed.**

- **`scripts/kfold_loso.py` would have crashed on VG20.** It builds its own frame without `sex`, and VG20's build refuses one. The frame now carries the column, and the script says so when the models it compares disagree on the covariate: VG20 against VG22 for the VG20/VG22 decision's criterion 3 is no longer a one-factor contrast of child structures.
- **A1 is not nested at zero.** `definitions.AgeVaryingSubjectScale`, `gp_utils.build_subject_scale_of_z` and the registry said `log_tau_*_ratio = 0` reproduces the model of record. It reproduces the model of record with dispersion held flat, which on VG10, VG11, VG12 and VG21 is a different model; §4 is corrected to match.
- **A cross-tab child missing from the merged view took contrast zero silently.** The joint frame builder now refuses such a row in a study that records sex. Every `uk_02`, `uk_07` and `es_01` child is found on today's data, and VG15's frame hash is unchanged.
- **The bivariate by-sex tables wrote "new child" columns for fits with no child effects**, which the pooled tables leave out because they repeat the population columns. They now follow the pooled tables.
- **VG26's page found VG23 by a glob** that also matches VG23's sensitivity arms. It now reads VG23's registered directory, shows both fit dates, and says the VG23 value is not validated.
- **Documentation that the covariate made false**: the inventory still called VG10 and VG13 the exact nested nulls of VG20 and VG23, and the report's methods chapter said VG20 and VG10 "differ in nothing else" and did not mention sex at all. Both are corrected, and the methods chapter has a short section on the covariate (`@sec-sex-covariate`). The description of VG21's nine differing fields is made exact in the inventory and on VG26's definition.
- **Tests that asserted less than their names.** The study-slope test now compares the population grids by value; the A1 test checks the constant scale at a zero ratio; the by-sex predictive tests check the girl and boy are the same child as the existing new child, on both engines and both univariate paths, and that the new-child columns appear exactly when the pooled ones do.

**Not changed.** Every page rendered on this Windows machine carries a PyTensor "g++ not available" line in its Data section, the fits of record's included; it comes from the environment, not from this change. The report's model-family table still lists twenty models and omits VG24 and VG25 as well as VG26.
