# Model inventory

> [!NOTE]
> Drafted with assistance from Claude Code and OpenAI Codex. Revised by OpenAI Codex/GPT-6.

> [!WARNING]
> This study is in progress. A model's reporting role does not certify an existing fit. Validate its definition, data, convergence and provenance before using its output.

This page describes the registered models and their reporting roles. The [model definitions](../../src/vocab_growth/models/definitions.py) specify their assumptions; the [catalogue](../../src/vocab_growth/models/catalogue.py) selects the fitting engine and role. Individual model reports explain each structure in detail. The [code walkthrough](../tutorials/model-code-walkthrough.md) introduces the implementation.

## How the models are named

Model numbers record development order. A higher number does not by itself replace an earlier model. VG06 is retired. VG17 and VG18 are unregistered exploratory modules. The other models through VG26 are registered.

The tables use DS for children with Down syndrome and TD for typically developing children. A random intercept describes a persistent difference in level between children or studies. A random slope allows children to differ in their rate of change.

## Inventory

| Model                  | Population | Outcomes                      | Structure                                                                                             |
| ---------------------- | ---------- | ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| [VG01](vg01/index.qmd) | DS         | Spoken                        | Single-outcome baseline without study or child effects.                                               |
| [VG02](vg02/index.qmd) | DS         | Understood                    | Single-outcome baseline without study or child effects.                                               |
| [VG03](vg03/index.qmd) | TD         | Spoken                        | English-only counterpart to VG01.                                                                     |
| [VG04](vg04/index.qmd) | TD         | Understood                    | English-only counterpart to VG02.                                                                     |
| [VG05](vg05/index.qmd) | DS         | Understood and spoken         | Joint baseline; spoken vocabulary is a fraction of understood vocabulary.                             |
| [VG07](vg07/index.qmd) | DS         | Understood and spoken         | VG05 with study random intercepts.                                                                    |
| [VG08](vg08/index.qmd) | DS         | Understood and spoken         | VG07 with child random intercepts on comprehension.                                                   |
| [VG09](vg09/index.qmd) | DS         | Understood and spoken         | Adds child intercepts on the spoken share and two-anchor dispersion priors.                           |
| [VG10](vg10/index.qmd) | DS         | Understood and spoken         | VG09 with a per-draw Gaussian process anchor at 54 months.                                            |
| [VG11](vg11/index.qmd) | TD         | Spoken                        | Study and child intercepts; Gaussian process anchored at 19 months.                                   |
| [VG12](vg12/index.qmd) | TD         | Understood                    | Study and child intercepts; Gaussian process anchored at 19 months.                                   |
| [VG13](vg13/index.qmd) | TD         | Understood and spoken         | Joint model at 8–18 months with study and child intercepts.                                           |
| [VG14](vg14/index.qmd) | DS         | Understood, spoken and signed | Adds a signed share; assumes speech and signing are independent within understood words.              |
| [VG15](vg15/index.qmd) | DS         | Understood, spoken and signed | Adds the sign–speech association and study and child intercepts.                                      |
| [VG16](vg16/index.qmd) | DS         | Understood and spoken         | VG09 with earlier comprehension as a predictor of the later spoken share.                             |
| [VG19](vg19/index.qmd) | DS         | Understood and spoken         | VG10 with child slopes on comprehension and the spoken share.                                         |
| [VG20](vg20/index.qmd) | DS         | Understood and spoken         | VG10 with correlated child intercepts and a sex covariate.                                            |
| [VG21](vg21/index.qmd) | TD         | Understood and spoken         | Widens VG13's window to 8–22 months; includes sex.                                                    |
| [VG22](vg22/index.qmd) | DS         | Understood and spoken         | A rank-three factor model links child levels and rates across both outcomes.                          |
| [VG23](vg23/index.qmd) | TD         | Understood and spoken         | Correlated child intercepts at 8–18 months; includes sex.                                             |
| [VG24](vg24/index.qmd) | DS         | Understood, spoken and signed | VG15 with correlations among all three child intercepts.                                              |
| [VG25](vg25/index.qmd) | DS         | Understood, spoken and signed | VG24 with earlier signing relative to the child's own level as a predictor of the later spoken share. |
| [VG26](vg26/index.qmd) | TD         | Understood and spoken         | VG21 with correlated child intercepts at 8–22 months.                                                 |

VG11–VG13 and their TD extensions include English, Italian and Spanish (European). VG03 and VG04 remain English-only because they have no study effects to account for differences between datasets.

### Model roles

A model of record supplies a specified reported quantity. A TD reference supplies a comparison. Development steps retain earlier structures for evaluation but supply no headline numbers. A superseded model has lost its reporting role. An unclassified model retains full publication checks until the study owner assigns a role.

| Model                                                      | Role                                                                                                                                                        |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VG20                                                       | **Model of record** for DS understood and spoken trajectories and their child-level association.                                                            |
| VG15                                                       | **Model of record** for signing trajectories, the sign–speech association and total expressive vocabulary.                                                  |
| VG24                                                       | **Model of record** for the between-child correlation of signed and spoken shares, `rho_sign_q`. It does not replace VG15's trajectory role.                |
| VG11, VG12                                                 | **TD reference** for spoken and understood trajectories, respectively.                                                                                      |
| VG21                                                       | **TD reference** for comparisons at matched comprehension.                                                                                                  |
| VG23                                                       | **TD reference** for the correlation between child comprehension and spoken-share effects.                                                                  |
| VG01, VG02, VG03, VG04, VG05, VG07, VG08, VG09, VG10, VG14 | **Development steps** superseded for reporting by the models above.                                                                                         |
| VG16                                                       | **Development model**, limited to the comprehension cross-lag question. Its numerical headline remains withdrawn pending validation and sensitivity checks. |
| VG19, VG22                                                 | **Development steps** for child rates of change. Their rate magnitudes are not adopted for reporting.                                                       |
| VG13                                                       | **Superseded** by VG21 for matched-comprehension comparisons; retained as the narrower-window comparison for VG23.                                          |

VG25 and VG26 are unclassified candidates. Registration alone does not establish a reporting role.

These roles follow the [promotion record](../../notes/202608190900-vg20-promotion.md), [VG20/VG22 decision](../../notes/202609091200-vg20-vg22-gate-resolved.md) and [September role decisions](../../notes/202609091600-model-roles-settled.md). Update this table and the catalogue together when a role changes.

Publication validation requires models of record, TD references and unclassified models to pass. It reports and skips invalid development or superseded fits. Thus an obsolete development fit need not block publication of a valid reporting model.

### Limits on interpretation

VG16's corrected lag uses complete child-age waves and is independent of row order. Its coefficient still mixes persistent differences between children with change within a child. Parameter recovery is supported, but the [initial recovery check](../../notes/202609111500-vg16-recovery-and-the-blocker-that-was-not-there.md) was insufficient to clear the withdrawal. Sequential validation uses `scripts/wave_forward_score.py`.

VG19 and VG22 support further study of differences in growth rates. The existing follow-up did not reliably recover the size of the production-rate variation. The [VG22 assessment](../../notes/202609091400-is-vg22-the-better-description.md) explains why its added structure did not replace VG20.

VG25's lag enters the spoken marginal likelihood only. The earlier design also placed it in the cross-tabulations and produced two incompatible posterior modes. See the [revised design](../../notes/202609151930-vg25-lag-out-of-the-cells.md). Its lag is an association, not evidence that signing causes later speech. Standard leave-one-out scores for the outcomes used to construct the lag would reuse the held-out observation, so the model uses a separate forward-chaining validation procedure.

Signing reports must distinguish three quantities:

- `psi` describes the association between signing and speech across understood words within an administration, conditional on study and sex.
- `rho_sign_q` describes the association between children's persistent signed and spoken shares.
- `beta_sign_lag` describes the prospective association specified by VG25.

`psi` uses cross-tabulations from `uk_02`, `uk_07` and `es_01` within understood words, and `nz_01` within produced words. Their measurement differences and study-specific estimates matter. Child random effects enter the marginal likelihoods, not these cross-tabulations. Shared reporting tendencies may contribute to the fitted child correlations; the models cannot separate that contribution from vocabulary differences.

VG15 retains its legacy signed-dispersion prior and a sampled signed-GP length scale, following the [prior review](../../notes/202608060900-three-prior-conflicts.md). Weak information about the length scale limits claims about trajectory shape. Its signing peak is estimated under a prior between fixed outer reference ages. Evidence becomes sparse before the reporting limit, so a reporting cap is not a guarantee of adequate data at every age.

### Exploratory, unregistered prototypes

VG17 and VG18 live in `src/vocab_growth/models/exploratory/`. They are excluded from the registry and the standard fitting and publication workflow. Their output has an `exploratory_output.json` marker, lacks the validated fit lifecycle, and must not be published as a registered fit.

The earlier VG20 sex-shift experiment is now covered by the registered sex-covariate design. The experiment remains a dated comparison, not a separate reporting model.

### Registering a new model: everything that has to be updated

1. Add the statistical definition to `MODEL_REGISTRY` and a wrapper module that dispatches to its engine.
2. Add a `RegisteredModel` record in `catalogue.py`, including its role and reporting hooks. Dispatch tables derive from this record.
3. Decide how new definition fields affect existing fits. Use a subclass for model-specific structure. A field may be added to a shared class with a `BACKFILL_DEFAULTS` entry only when tests establish that its default reproduces the old behaviour. Without that evidence, existing fits become stale.
4. Create `docs/models/vgNN/index.qmd` before the first fit. The pipeline copies it even when rendering is not requested.
5. Declare replot support, or a justified exemption, in the catalogue. Review recovery support and its reasoned exclusions.
6. Update the model and graph checks. These include `test_model_definitions.py`, `test_clamp_scope.py`, `test_trend_gp_consolidation.py`, `test_subject_effect_plan.py`, `test_ds_joint_shared_priors.py` and `test_graph_equivalence.py`. Regenerate only the new model's graph baseline with `uv run python tests/support/regenerate_graph_baseline.py <model>`.
7. Update this inventory and the three identical agent-instruction files. Run the relevant fast and slow tests.

Keep provisional fits in a separate output root. An unclassified model's `dev` or `test` output can block publication sync because it is subject to full validation.

## Shared architecture

The [implementation note](../../notes/202609131044-model-review-implementation.md) describes the shared helpers. Bivariate study offsets sum to zero over all retained studies. The joint signing engine uses the studies involved in each outcome's likelihood. These define different reference populations.

Models express vocabulary on an 810-word reference scale. A paired spoken or signed likelihood instead conditions on the child's observed understood count. Missing comprehension uses the registered marginal fallback. That fallback matches the nested model's mean but generally not its variance; sensitivity variants assess this approximation.

Most count likelihoods are Beta-Binomial. Signing cross-tabulations use Dirichlet-Multinomial likelihoods. Age-dependent dispersion describes variation left after the model's child and study effects. Larger `kappa` means less residual spread. Do not compare its magnitude across models with different child structures or compare marginal comprehension dispersion directly with conditional production dispersion.

Mean trajectories combine an interpretable age trend with a smooth Gaussian process correction. Some models constrain that correction to zero at a reference age in every posterior draw. This helps separate the correction from the trend and random intercepts. See [PRIORS.md](PRIORS.md) for the prior forms and [REPORT_STYLE.md](REPORT_STYLE.md) for reporting requirements.

### Sex as a covariate

VG20, VG15, VG24, VG25, VG11, VG12, VG21, VG23 and VG26 include sex. Each coefficient has a `Normal(0, 0.5)` prior on the logit scale. A logit is the logarithm of the odds of the modelled proportion.

Girls are coded `+1/2`, boys `-1/2`, and children with unrecorded sex `0`. A child's conflicting recorded values are rejected. Each coefficient is a girl-minus-boy difference assumed constant across ages and studies. Unrecorded sex does not remove a row; predictions at zero are the model's midpoint on the logit scale, not an arithmetic average of girls' and boys' expected counts.

The univariate engine has `beta_sex`; the bivariate engine has `beta_sex_u` and `beta_sex_q`; the joint signing engine also has `beta_sex_sign`. Joint-engine coefficients enter both marginal and cross-tabulation likelihoods. Unlike VG25's fitted within-child lag baseline, sex is a fixed observed covariate.

The report reads each fit's sex coverage from its verified frame. A common sex effect and coding missing sex at the midpoint are assumptions, especially when missingness occurs within a study. The `sex_known_only` sensitivity restricts DS data to recorded sex; it is not the reporting default.

Fits write by-sex outcome and ratio tables, expected-count differences and coefficient summaries. Univariate and bivariate engines also produce new-child predictions by sex. The joint signing engine's by-sex tables contain expected counts only.

### Interval reporting convention

Reports use posterior medians, 50% inner intervals and 89% outer intervals. The default is an equal-tailed interval, with equal probability below and above its limits. Selected skewed quantities, including `psi`, `kappa` and milestone or peak ages, use highest-density intervals. The exact policy is in `vocab_growth.intervals`. An interval's width is not a decision threshold.

Summary tables use `*_ci50_lo` and `*_ci50_hi` for inner bounds, and `*_ci_lo` and `*_ci_hi` for outer bounds. Plot sidecars include the inner bounds when the figure draws them.

### Reporting ages: 6-monthly tables, whole-month companions

`ages_query` sets each model's canonical reporting ages. DS models generally use six-month steps and TD models three-month steps. The model definition and `vocab_growth.reporting_ages` determine the allowed range for each quantity.

| DS quantity                                                             | Upper reporting age, months | Reason                                                                           |
| ----------------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------- |
| Understood                                                              | 72                          | Shares the cap imposed by model dependence in the spoken share.                  |
| Spoken or signed share of understood words; total expressive vocabulary | 72                          | Depends on comprehension and, for signing quantities, the tighter component cap. |
| Signed count                                                            | 84                          | Reporting policy for the sparsely observed older signing ages.                   |
| Spoken count                                                            | 90                          | Upper canonical query age.                                                       |

VG04 and VG12 cap TD comprehension at 25 months. Joint TD models use their own shorter windows. The [cap decision](../../notes/202608221200-reporting-source-by-quantity.md) and [September check](../../notes/202609091700-comprehension-cap-check-rerun.md) explain why more rows alone are insufficient to extend the DS cap.

Caps apply during post-processing but remain part of the recorded definition. Changing one does not change posterior draws, yet it makes old summary tables incompatible. `--render-only` reads those existing tables; it does not regenerate them.

Whole-month companion tables use the nearest point on the plot grid. Read `grid_age_months`, `grid_offset_months` and the outcome-specific `n_obs` with a monthly estimate. These tables can extend beyond canonical reporting caps and include months with no observations. They support inspection, not an extension of the published evidence range. Use the canonical tables when exact agreement at a reporting age matters.

`Ey_*` describes expected counts for the stated target; `Y_*` and `P(Y<=k)` describe predictive counts and include the variation specified for that target. A credible interval for a reference-child mean is not the range expected among individual children. Joint signing tables that contain expected counts only cannot supply that predictive range.

### Structural decomposition (joint models)

At age `a`, `p_U(a)` is the proportion understood, `q(a)` the spoken share of understood words, and `r(a)` the signed share. Spoken and signed proportions of the full inventory are therefore `p_U(a) * q(a)` and `p_U(a) * r(a)`. Counts in the two modalities overlap; total expressive vocabulary is their union, not their sum.

The primary signing analyses mask `uk_01` signing because it records sign-only words. Its understood and spoken counts remain. `uk_06` records total signed words and is retained. See the individual [source records](../../data/readme.md).

## Model lineages

The inventory lists structural changes in development order. Three qualifications matter when using one model to assess another:

- VG08 to VG09 changes both the child structure and dispersion priors, so it does not isolate either change.
- VG10 and VG13 lack the sex covariates now carried by VG20 and VG23. Add the same covariate to the control when testing only the child correlation.
- Wider-window TD models use different observations. Their predictive scores are not direct same-data comparisons with narrower-window models.

VG06 was retired because Wordbank Words & Sentences comprehension values were production proxies. The loader now excludes those forms whenever comprehension is required. VG13's additional age restriction concerns data density and checklist ceilings, not removal of those proxies. VG21 supplies the wider 8–22-month matched-comprehension comparison; VG26 adds correlation but has no adopted replacement role.

VG14's independence assumption is a development baseline, not a general mathematical upper bound on total expressive vocabulary. VG15 estimates the overlap association. VG24 adds persistent child correlations, and VG25 adds the prospective signing term. In the joint engine's produced-only composition, the retained Dirichlet parameters sum to `conc * P(produced | understood)`, not `conc`.
