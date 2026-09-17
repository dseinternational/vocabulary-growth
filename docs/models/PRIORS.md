# Prior rationale and review

> [!NOTE]
> Drafted with assistance from OpenAI Codex/GPT-5 and Claude Code. Revised by OpenAI Codex/GPT-6.

Priors describe plausible parameter values before fitting. Some use external studies; others were calibrated on this project's data or chosen to stabilise estimation. Those are different sources of information and should be named wherever a prior matters to a conclusion.

This guide explains the prior families and their main limits. Exact current settings are in [definitions.py](../../src/vocab_growth/models/definitions.py), including shared constants and inherited values. For a completed fit, its manifest and rendered prior table record what it actually used. This guide does not maintain a second numerical specification.

## Where the priors live

The [model inventory](README.md) lists structures and reporting roles. Shared engines and `gp_utils.py` turn definitions into PyMC variables. `report_cells.render_priors_table()` displays the priors from a fit's record, and `tests/test_prior_table_coverage.py` checks coverage.

Use the registered definition when simulating or calibrating a prior. A loader's default language scope, age window or study filter may differ from the model's. The [TD calibration correction](../../notes/202609062330-vg11-vg13-calibration-regenerated.md) records a case where the estimator used an English-only frame for multilingual models.

## Prior families

### TD and DS prior differences

The populations share modelling tools, not identical priors. They have different age ranges, anchor levels and some dispersion and random-effect specifications. Even equal parameter values can imply different prior predictions under different age domains. Compare the resulting vocabulary counts and curves, not just distribution names.

### Anchor priors

The broad comprehension or spoken trajectory is specified by proportions at two reference ages, joined on the logit scale. Joint models also give the spoken share of comprehension, `q`, its own anchors. Multiplying a direct vocabulary proportion by 810 gives an expected count on the reference scale. Multiplying `q` by 810 does not give a spoken count; it must also be multiplied by the understood proportion.

An anchor parameter describes the trend component. A Gaussian process adds departures from that trend. A per-draw GP anchor makes the correction zero at one reference age, not at both trend anchors. In an unanchored model, the full-curve prior at a trend anchor is widened by the GP. Interpret prior plots as the combined model.

The DS comprehension anchors and the older DS spoken-share anchor were calibrated using the project's own data. They are regularisation, not independent developmental norms. The rationale and changes are recorded in the [comprehension review](../../notes/202608041216-ds-understood-trajectory-prior.md) and [spoken-share review](../../notes/202608041730-ds-spoken-q-trajectory-prior.md).

The older TD comprehension anchor also lacks a corresponding CDI comprehension norm. Words & Sentences records production; the loader excludes its production-proxy comprehension values. A reported normative median should not be treated as a prior on an individual child's score.

### Mean extrapolation above the high anchor

A logit-linear trend can approach near-total production when extended beyond the observed ages. The DS joint models use `CLAMP_Q_ONLY` to level the spoken-share trend above its upper anchor while leaving comprehension's trend unclamped. The transition is smooth; the high-anchor parameter is therefore not exactly the clamped mean evaluated at that age.

The GP can still depart from the levelled trend. Clamping does not establish that older-age predictions are supported by data. See the [clamp correction](../../notes/202608141200-clamp-q-only.md) and `gp_utils.trend_and_gp`.

### Reported age range for comprehension

Reporting caps are separate from prior specification. The [inventory](README.md#reporting-ages-6-monthly-tables-whole-month-companions) and `reporting_ages.py` state them. The DS comprehension and comprehension-conditioned quantities stop at 72 months; VG04 and VG12 stop at 25 months. A full model domain or an uncapped monthly table does not extend those published ranges.

### Signed ratio prior

Signing uses a three-anchor trend for the signed share of understood words. It allows a rise and fall, with a GP adding smooth departures. That shape is a modelling assumption, not proof that every child follows the same pattern.

VG14 fixes the middle reference age. VG15 and its joint-engine extensions estimate the middle-anchor position under `sign_peak_prior` between fixed outer ages. Despite the parameter name, it need not be the full curve's maximum. Do not describe its age as fixed. Sparse data and differences between studies limit its interpretation even when a posterior interval is available.

The [signing-shape review](../../notes/202606151700-vg14-signed-ratio-shape-and-p-any-bias.md) records the initial design. The [later prior decisions](../../notes/202608060900-three-prior-conflicts.md) explain the estimated peak and why the signed GP and legacy signed-dispersion block were retained. Literature on mental-age milestones motivates possible shapes but does not determine a chronological peak through a single developmental-age conversion.

### GP length-scale and amplitude priors

The length scale controls how quickly a smooth correction changes with age. The amplitude controls the size of its departures on the logit scale. `ell_unit` is mapped into the model's registered age-domain geometry; its raw value is not a length in months.

Read both with the HSGP basis and domain. A prior that looks broad in parameter space may still produce a restricted family of curves. Anchoring and removal of components aligned with the trend also affect the induced curve prior.

VG11 adopted `eta_sigma = 0.4` on 2026-09-16; `eta-wide` restores 0.5 as a sensitivity. This was a regularisation decision supported by an earlier comparison, not evidence that the amplitude was well identified. See [the decision record](../../notes/202609161440-vg11-eta-sigma-0.4.md).

### Age-varying dispersion priors

The usual concentration curve is:

```text
kappa(z) = kappa_min + exp(a_kappa + b_kappa * z)
```

Here `z` is standardised age. Larger `kappa` means less residual count variation at a given mean and denominator. It is not a direct measure of differences between children.

The legacy form puts priors on the floor, intercept and signed slope magnitude. Its prior meaning depends on age standardisation and it constrains the trend's direction. Some development models retain it for historical comparison; the signed block also retains it where the review found no need to change it.

The two-anchor form puts positive priors on the concentration above the floor at two fixed ages. The log excess is interpolated between those ages. Standardisation cancels from the interpolation weight, so the prior retains its age interpretation when the frame changes. Either age direction is possible. `kappa_young` and `kappa_old` are totals, including the floor; the components are not interchangeable with those totals.

Calibration must match the model's random effects. A fit without child effects assigns variation to the count likelihood that a hierarchical model can assign to persistent child differences. A marginal calibration therefore cannot be substituted for a conditional one. The [calibration record](../../notes/202608020829-kappa-and-eta-q-prior-recalibration.md) documents the estimator checks and failures as well as the adopted values.

### What a rising `kappa` on the understood outcomes means

Rising concentration means less residual spread at a fixed mean and denominator. It does not by itself show that children's vocabulary becomes more or less variable with age. Changes in the mean, checklist ceiling and child-effect structure also change observed spread.

The TD calibration found that an age-varying child loading changed the fitted dispersion trend. The [registered-frame recheck](../../notes/202609062330-vg11-vg13-calibration-regenerated.md) supports treating the variance split as model-dependent. Report total spread on the word-count scale when that is the question, and assess recovery of that quantity separately.

### Study and subject random-effect scale priors

Study effects describe persistent differences between datasets; child effects describe persistent differences between children. Their scale priors regularise different sources of variation. Several models use HalfNormal scale priors, but there is no single independent scale prior shared by every engine.

VG11 and VG12 parameterise total scatter and the share attributed to children through `SubjectVariancePartitionParams`. Correlated models use joint child-effect priors. VG19 adds child slopes, and VG22 uses a low-rank factor structure. Inspect the definition and its induced correlations before treating two models' priors as matched.

Repeated measurements help distinguish persistent differences from occasion-level variation. A prior or a reparameterisation cannot supply the information missing from a mostly single-visit dataset. See the [TD geometry investigation](../../notes/202608050900-td-hierarchical-geometry.md).

### VG15 association and four-cell concentration priors

The joint signing engine places a Normal prior on `log_psi` and exponentiates it. Independence is `psi = 1`. The registered centre is weakly positive and was motivated by the observed cross-tabulations, so it is data-informed. The study-level association scale controls how strongly the four sources are pooled and is weakly identified with so few studies.

`psi` is informed by `uk_02`, `uk_07`, `es_01` and `nz_01`, not `uk_02` alone. Their cell definitions differ. The likelihood uses study-adjusted margins and sex effects, while child intercepts remain outside the compositions. See the [association review](../../notes/202608121030-psi-heterogeneity-and-age-invariance.md) and the [source-construct correction](../../notes/202609021903-es01-gesture-construct-revisited.md).

`conc` controls Dirichlet-Multinomial variation among cells. The produced-only likelihood retains the original three Dirichlet parameters after conditioning; its total concentration is `conc * P(produced | understood)`. It is not the same concentration as the full four-cell likelihood.

## Evidence base: literature and normative data

The source assessments below were recorded in the prior review and corrected on 2026-09-04. They identify overlap and measurement limits. They are not a fresh literature search or proof that current priors have independent validation. Consult the source papers and dataset records when using them for a new analysis.

### Independence of candidate sources

Not every published cohort is independent of the fitted data. Where a prior is
anchored on a study whose participants are already in `vocab_data_merged.csv`,
it is regularisation, not independent prior evidence.

| Source                                                                           | Role for priors                                                                                                                   | Independent of training data?                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Wordbank by-child data (`wordbank_administration_data.csv`)                      | TD anchors, `q`, dispersion                                                                                                       | No. These are the TD training data. Normative tables can check scale, but their independence also needs to be established.                                                                                                                                                 |
| Berglund et al. (2001), n=330, Sweden                                            | DS _spoken_ anchors, growth shape, heterogeneity. The SECDI Words & Sentences form is production-only, so no comprehension anchor | Yes                                                                                                                                                                                                                                                                        |
| Galeote et al. (2008, 2011), Spain                                               | DS spoken-versus-TD comparison by mental age; symbolic-gesture `r`                                                                | **No**. The 186-child cohort _is_ `es_01`, supplied by the author (see `data/vocab_data_es_01.md`); treat as regularisation                                                                                                                                                |
| Næss et al. (2021), Norway                                                       | Qualitative corroboration only (receptive ahead of expressive; slower expressive growth than TD)                                  | Yes, but not on the CDI scale: 43 children tested at 6, 7 and 8 years with the BPVS-II and WPPSI-III Picture Naming, no parent report. Supplies no CDI anchor and no `q`                                                                                                   |
| Deckers et al. (2016, 2019), Netherlands                                         | Signed `r` corroboration; N-CDI validity in DS                                                                                    | Yes. 25–36 children aged 2;0–7;6, two waves 1.5 years apart, N-CDI Words & Sentences (702 words) with an added sign column. A production-only form; the 2019 receptive measure is a composite with the ROWPVT test, and no parent-report comprehension column is described |
| Kaat-van den Os et al. (2017), Netherlands                                       | Signed `r` corroboration; spurt heterogeneity; sign-to-speech modality shift                                                      | Yes. 26 children followed monthly from 18–24 months for 18 months on the Lexi questionnaire (263 words, a Language Development Survey adaptation, not a CDI); production only, no comprehension                                                                            |
| Miller et al. (1995); Mervis & Robinson (2000), US                               | DS anchors, parent-report validity                                                                                                | Yes                                                                                                                                                                                                                                                                        |
| Oliver & Buckley (1994), UK                                                      | DS low-age spoken anchor (10-word stage ~27 mo)                                                                                   | Yes, confirmed **not** to overlap `uk_01`                                                                                                                                                                                                                                  |
| Caselli et al. (1998); Zampini & D'Odorico (2013); Bello & Caselli (2014), Italy | DS trajectory, gesture, dispersion                                                                                                | **No**. Overlap the `it_01` Italian-CDI-DS cohort; treat as regularisation                                                                                                                                                                                                 |

Corrected on 2026-09-04 after reading the papers: the Norwegian and Dutch cohorts had been listed together as independent sources of DS anchors and `q`, but none of them reports a parent-report comprehension count at chronological age, and the Galeote cohort is `es_01`. Every DS source in the table that is both independent and on the CDI scale is therefore production-only, which is why the DS understood anchors below rest on the project's own data.

### Instrument scale

The model's 810-word scale is a common reference denominator. It is not an item-level crosswalk, and it does not make different checklists equivalent. A recommendation to compare common items does not, on its own, validate scoring every raw total against 810. The [methods chapter](../report/methods-data.qmd), dual-form comparisons and `dse-native-only` sensitivities describe the project's evidence and remaining assumptions.

### Recorded sensitivity results

The 2026-08-18 Target 8 checks used the definitions and data then available. All seven variants met the recorded convergence and trajectory-interval comparison rule:

| Model | Variant as fitted | Maximum absolute count difference |
| ----- | ----------------- | --------------------------------: |
| VG10  | `u-anchor-broad`  |                       0.549 words |
| VG10  | `eta-u-narrow`    |                       0.592 words |
| VG11  | `anchor-broad`    |                       0.055 words |
| VG11  | `eta-narrow`      |                       0.437 words |
| VG12  | `lo-anchor-broad` |                       0.360 words |
| VG12  | `hi-anchor-broad` |                       0.634 words |
| VG12  | `eta-narrow`      |                       7.038 words |

These are historical checks on reported trajectories, not proof that all parameters or later fits are insensitive to priors. VG11's old `eta-narrow` arm is now its default, with `eta-wide` as the reverse comparison. Updated data, definitions and target quantities require compatible new comparisons. The [run record](../../notes/202608180600-refit-run-record-and-storage-move.md) provides the historical context.

## Prior predictive audit

Use `scripts/prior_predictive_audit.py` to simulate from registered definitions. Inspect the combined prior on counts, not just one parameter at a time. Record the revision, frame, simulation settings and figures reviewed.

Check young-age floors, older-age ceilings, curve smoothness, spoken and signed shares, differences between children and studies, and signing-cell compositions. Compare each against the correct observation denominator and age support. Earlier audit verdicts do not certify a changed prior or model.

## Sensitivity targets

Use the registered variants in `src/vocab_growth/sensitivity/registry.py`. The main questions concern anchor levels, GP flexibility, dispersion, child and study scales, signing shape and association pooling, and lag construction. The sex covariate and child correlations also need comparisons appropriate to their reporting use.

Compare the quantities that motivate each model, together with their uncertainty. A cross-lag check must inspect the lag coefficient; agreement in population trajectories alone cannot validate it. A total-spread comparison must assess that spread, not infer recovery from its separate components. Follow the [recovery runbook](../runbooks/parameter-recovery.md) for designed-truth and cross-prior checks.

A prior-tail posterior or little reduction in uncertainty is a reason to investigate. Neither automatically proves an invalid model, and a narrow posterior does not by itself establish that the parameter is identified by the data.
