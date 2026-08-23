# Statistical review of VG11, VG12 and VG13

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

> [!IMPORTANT]
> Review record, 2026-08-23, against repository commit `0271871298f226591a2f90ff7ce4349dec856c47`. This review distinguishes confirmed implementation or reporting defects from statistical risks and missing validation. It compiled the three complete PyMC graphs and checked existing fit, sensitivity and recovery evidence, but did not rerun reporting-quality MCMC. Remediation is tracked in [#240](https://github.com/dseinternational/vocabulary-growth/issues/240). Later corrections to this evidence record should be appended as flagged updates rather than silently changing the conclusions.

## 1. Scope and method

The review traced VG11, VG12 and VG13 from their registered definitions through the typically-developing data loader, study and child coding, PyMC graph construction, Gaussian-process and dispersion utilities, posterior prediction, calibration, summaries, reports, sensitivity registry and recovery framework. It also audited the realised model frames, compiled each complete graph against prepared data, checked numerical invariants, ran focused tests and lint, and reconciled the results of three independent model-specific code reviews.

The question was assessed at three distinct levels:

1. **Implementation correctness:** whether the code represents the registered graph without shape, indexing, likelihood, constraint or prediction-construction errors.
2. **Statistical adequacy:** whether the registered graph is capable of identifying and supporting the quantities attributed to it.
3. **Validation and reporting correctness:** whether diagnostics, predictive checks, figures and prose test or describe the intended estimands.

## 2. Headline verdict

No fundamental PyMC implementation error was found. All three graphs compile with finite log density; observation, study and child dimensions align; probabilities and dispersion parameters remain in their valid domains; study effects satisfy their zero-sum constraint; and the Gaussian-process anchors are exact. VG13's nested likelihood is correctly factorised and preserves `spoken <= understood` in posterior prediction.

The models are not, however, fully statistically validated in their current reported form:

| model | implementation verdict                                                      | statistical and reporting verdict                                                                                                                             |
| ----- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VG11  | Registered likelihood and graph are coherent                                | Mean spoken trajectory may be usable with caveats; the child/residual variance split is misspecified or unvalidated, and the accepted fit remains provisional |
| VG12  | Registered likelihood and graph are coherent                                | Absolute child heterogeneity failed the available recovery trials; low BFMI, form-scale effects and older-age study confounding remain material risks         |
| VG13  | Nested understood and conditional-spoken graph is coherent over 8–18 months | The model contradicts the adopted 8–22-month specification, cannot support the report's no-curvature inference and lacks direct recovery evidence             |

The strongest common finding concerns heterogeneity: the combination of a fixed 810-word denominator and age-invariant child loadings makes the interpretation of `tau_subject` and `kappa(age)` dependent on the form scale and variance specification. Mean trajectories are less directly implicated than the variance decomposition. VG13's conditional production proportion `q = P(spoken | understood)` is better protected because its likelihood denominator is the observed number understood.

## 3. Shared findings

### 3.1 High statistical risk — the variance decomposition is strongly scale- and specification-dependent

VG11 and VG12 score one outcome against the common 810-word inventory and apply one constant child shift on the logit scale. VG13 applies separate age-invariant child effects to understood vocabulary and conditional `q`; only understood is scored against 810, while the conditional `q` likelihood uses observed understood vocabulary as its denominator. The common-denominator assumption is documented in [`data_utils.py`](../src/vocab_growth/data_utils.py#L631); the child effects are applied in [`common_univariate_re.py`](../src/vocab_growth/models/common_univariate_re.py#L444) and [`common_bivariate_re.py`](../src/vocab_growth/models/common_bivariate_re.py#L712). This implies parallel child trajectories on each latent logit scale while understood or spoken observations from substantially shorter CDI forms are treated as counts on the master inventory. The latter is defensible only to the extent that shorter instruments contain nested sets of easier words; it is not neutral rescaling.

The repository's own conditional calibration provides strong evidence against this variance structure:

| model/outcome        | log-likelihood gain from one age-varying child loading | constant-loading calibration result | age-varying-loading calibration result        | native-form rescoring removes |
| -------------------- | -----------------------------------------------------: | ----------------------------------- | --------------------------------------------- | ----------------------------: |
| VG11 spoken          |                                                 +237.1 | `tau` 1.056; `kappa` 317.5 to 50.5  | loading 1.304 to 1.028; `kappa` 516.9 to 48.0 |          84% of loading drift |
| VG12 understood      |                                                 +162.3 | `kappa` 43.0 to 66.4                | loading falls 44%; `kappa` 84.6 to 53.0       |                           96% |
| VG13 understood      |                                                 +111.3 | `kappa` 42.2 to 124.1               | loading falls 34%; `kappa` 79.0 to 63.5       |                           86% |
| VG13 conditional `q` |                                                   +0.8 | effectively unchanged               | effectively unchanged                         |                  not material |

The source analysis is in [`202608020829-kappa-and-eta-q-prior-recalibration.md`](202608020829-kappa-and-eta-q-prior-recalibration.md) and is summarised in [`docs/models/PRIORS.md`](../docs/models/PRIORS.md#L460). These diagnostics used the calibration loader's English-only default rather than the registered English-plus-Romance frames, and they are in-sample comparisons rather than out-of-sample model comparisons. They are strong evidence that the decomposition is sensitive to form extent and child-loading structure, but the exact magnitudes do not directly describe the current registered graphs and must be reproduced on their complete language scopes. They do not prove that a more flexible hierarchy will generalise better.

The evidence makes `tau_subject` a plausible age-averaged compromise and shows that `kappa(age)` may absorb both occasion-level variation and form-induced age variation. Total predictive spread can be better identified than its allocation between components, so predictive intervals require sensitivity testing rather than being assumed defective. Cross-model comparisons of persistent child variation and residual variation are not measurement-invariant without that work. The primary remedy to evaluate is an explicit form or item measurement model; the minimum sensitivity is an age- or form-varying child loading specified in advance and fitted on each registered frame.

### 3.2 Medium validation gap — current checks do not provide a uniform new-child or new-study target

The shared engine computes ordinary pointwise PSIS-LOO over administrations in [`common.py`](../src/vocab_growth/models/common.py#L1298). For repeated children, leaving out one administration retains information from that child's other administrations; for singleton children, their effect is effectively integrated from its prior, while study information remains. Pointwise LOO is therefore a mixed hierarchical target rather than a uniform assessment of a new child or a new study. The report explains this limitation only when Pareto diagnostics are poor in [`report_cells.py`](../src/vocab_growth/report_cells.py#L833), although the target distinction exists regardless of Pareto reliability.

Observed-row posterior-predictive calibration similarly includes fitted child effects and is read from `y_obs` in [`calibration.py`](../src/vocab_growth/calibration.py#L141). It is valid as in-sample conditional calibration, but it is not new-child calibration. The existing whole-child validation scripts do not support VG11, VG12 or VG13. Claims about out-of-sample or new-child performance therefore require leave-one-child-out and leave-one-study-out validation in which the held-out random effects are integrated out.

### 3.3 Medium — reports conflate three different estimands

First, the VG11 and VG12 median-trend bands are calculated from predictive counts that include a newly sampled child effect and Beta-Binomial residual variation in [`common_univariate_re.py`](../src/vocab_growth/models/common_univariate_re.py#L546). Their captions call them uncertainty intervals on the population mean and say they are not the range children occupy, which states nearly the opposite of what is plotted. VG13's posterior-predictive captions need more explicit estimand labels but do not make the same opposite claim, and its separate joint population-trajectory figure should not be conflated with those predictive plots.

Second, plot and query probabilities set child and study offsets to zero but are repeatedly called the "average child in the average study". Under the nonlinear inverse-logit link, `E[inverse_logit(f + delta)]` is not `inverse_logit(f + E[delta])`. A zero effect is a reference or median-type child in a logit-centred study, not the arithmetic response-scale mean. As an illustration rather than a fitted age-specific query, combining a probability of 0.0118 with a child scale near 1.04 gives a marginal mean near 0.019, so the distinction can be material.

Third, the shared headline renderer derives a variance-inflation row solely from `kappa` and labels it "Spread between children" in [`report_cells.py`](../src/vocab_growth/report_cells.py#L688). Once child random effects are present, `kappa` is residual Beta-Binomial variation conditional on child and study; `tau_subject` represents persistent between-child variation. Reports must distinguish the zero-effect latent curve, marginal population mean, predictive distribution for one new child, persistent child variance and conditional residual variance.

### 3.4 Medium — exact duplicate TD administrations enter every model

The project has a duplicate-removal helper whose documentation correctly explains that duplicates double-weight the likelihood and make a single administration appear longitudinal in [`data_utils.py`](../src/vocab_growth/data_utils.py#L468). It is used in the Down syndrome pipeline but not the typically-developing loader at [`data_utils.py`](../src/vocab_growth/data_utils.py#L1509).

The realised-frame audit found 22 excess exact copies of complete source records entering VG11, 3 entering VG12 and 2 entering VG13. The VG13 pairs are directly visible in [`wordbank_administration_data.csv`](../data/wordbank_administration_data.csv#L34684) and [`wordbank_administration_data.csv`](../data/wordbank_administration_data.csv#L90658). The numerical effect is small, but it perturbs exactly the child-versus-residual variance allocation that is already weakly identified. Deduplication must use complete source-administration identity before outcome-column projection, study filtering and child coding: applying the existing helper after projection can incorrectly collapse distinct same-child, same-age administrations that share the selected outcome. The number removed should be recorded and a loader regression test should enforce source-level uniqueness.

### 3.5 Medium design risk — the 200-row study threshold changes the study-population estimand

VG11 removes 5 of 15 eligible studies while removing only 315 of 18,837 rows; VG12 and VG13 each remove 3 of 9 studies while removing only 136 rows. The threshold is configured in the model definitions and applied in [`data_utils.py`](../src/vocab_growth/data_utils.py#L879). A hierarchical model can retain small groups through partial pooling, so deleting one-third of the study units changes the target population and may reduce information about between-study variance, although it could improve sampling geometry. The direction and magnitude of its effect on the fitted trajectory have not been quantified. If these exclusions represent a quality criterion rather than sample-size convenience, that rationale needs to be explicit and tested against a threshold-free sensitivity.

Age, study, language and instrument are also unevenly overlapped. In VG12, ages 19–24 are represented only by Caselli and Floccia and age 25 only by Floccia; VG13's retained studies cover markedly different age ranges. Because the hierarchy permits constant study intercepts but not study-specific age slopes, older-age shape can be partly confounded with study, language or form. Common-support reporting, threshold sensitivity and a random-slope sensitivity are warranted.

### 3.6 Medium — empirical prior calibration uses the wrong language scope

The registered VG11–VG13 models fit English plus Romance-language data, but both `univariate_frame()` and `bivariate_frames()` in [`kappa_conditional_calibration.py`](../scripts/kappa_conditional_calibration.py#L477) call the data loader without `definition.td_languages` and therefore receive the English-only default in [`data_utils.py`](../src/vocab_growth/data_utils.py#L1440). The calibration and registered frames contain 16,235 versus 18,522 rows for VG11, 5,997 versus 7,052 for VG12 and 5,406 versus 6,358 for VG13. For VG11 the scope change moves the young-age calibrated `kappa` from about 318 to 289. The prior is broad and the posterior moved towards the recalibrated magnitude, so the practical effect appears limited, but the claimed provenance is wrong. Calibration frames should be derived from the complete model definition and tested to match the model's rows, languages and studies.

## 4. VG11-specific findings

### 4.1 Medium — the accepted fit does not pass the ordinary strict convergence gate

The recorded fit has 16 divergences in 48,000 draws and one GP coefficient with R-hat 1.0125 and ESS 1,139. A deliberately narrow exception accepts that coefficient in [`fit_artifacts.py`](../src/vocab_growth/fit_artifacts.py#L374), and tests correctly prevent the exception from widening. Function-level grids mixed better and narrow GP-amplitude sensitivity was stable, so this is not evidence that headline curves are necessarily wrong. It does mean the fit remains provisional rather than strictly converged. The VG11 report also records that prior-predictive figures are unavailable.

### 4.2 Medium — direct recovery evidence is absent

The recovery framework supports VG11, but no completed VG11 recovery assessment was found. VG12 and the related DS models exhibit directional under-recovery of the child scale, which identifies a plausible failure mode but cannot establish VG11's bias. Multiple posterior-truth recovery replicates should score `tau_subject`, total variance, the subject variance share, both `kappa` anchors and trajectory grids before VG11's heterogeneity estimate is treated as quantitatively validated.

### 4.3 Low — GP orthogonality is overstated in the helper and its test

The GP helper projects out the constant and linear basis and then subtracts the anchor value from every row in [`gp_utils.py`](../src/vocab_growth/models/gp_utils.py#L679). That final shift restores a constant component unless the projected GP was already zero at the anchor. The result remains point-anchored and slope-orthogonal, and no resulting graph-identification failure was found, but the documented and tested claim of simultaneous full-basis orthogonality is mathematically inaccurate.

## 5. VG12-specific findings

### 5.1 High statistical risk — low energy exploration persists after tuning

Later tuning removed divergences, but minimum E-BFMI remained around 0.20–0.21. Low E-BFMI means Hamiltonian transitions explore the posterior's energy distribution inefficiently; it is a global sampling warning, not evidence that only interval tails are affected. The unresolved warning and GP-amplitude prior conflict are recorded in [`definitions.py`](../src/vocab_growth/models/definitions.py#L2694). Tail intervals and variance components require stability across seeds or parameterisations before headline use.

### 5.2 Medium — the child scale failed all available recovery trials

VG12 recovered `tau_subject` low in all 3 of 3 posterior-truth replicates by about 5.8%, with the truth outside every 89% interval, while total variance recovered better than its allocation between persistent child and residual components. The result is recorded in [`gp_utils.py`](../src/vocab_growth/models/gp_utils.py#L351). Three replicates are too few to estimate a correction, but they reject nominal interval coverage in those trials. Similar directional under-recovery occurs in related DS models, so this result does not by itself prove that the DS–TD contrast is biased; it does show that the absolute VG12 child scale is not calibrated.

### 5.3 Medium — key sensitivity results remain unreported

The high trajectory anchor is at 26 months, one month beyond the last admitted observation, and has no independent comprehension norm. Registered broad-anchor, GP-amplitude and single-administration sensitivities have not been reported in the model report. At minimum, the high-anchor, narrow-GP-amplitude and single-administration variants should be reported before VG12 is treated as the stable TD comprehension reference.

### 5.4 Low — the extrapolation warning is stale

The VG12 report says its last displayed query rows lie beyond observed data, but `report_max_age_understood=25` now removes rows beyond the observed support. The warning in [`docs/models/vg12/index.qmd`](../docs/models/vg12/index.qmd#L324) should be generated from the actual rendered table and support or removed.

## 6. VG13-specific findings

### 6.1 High — the canonical age window contradicts the adopted specification

VG13 remains capped at 18 months in [`definitions.py`](../src/vocab_growth/models/definitions.py#L2767), although the adjacent comment acknowledges that the original single-study rationale is obsolete and hundreds of admissible observations lie above the cap. The wrapper and report repeat the obsolete rationale in [`model_vg13.py`](../src/vocab_growth/models/model_vg13.py#L8) and [`docs/models/vg13/index.qmd`](../docs/models/vg13/index.qmd#L332).

The study-owner decision record [`202608211100-window-22-adopted.md`](202608211100-window-22-adopted.md) adopts the 8–22-month specification: it increases the frame from 6,358 rows and 5,496 children to 6,786 rows and 5,707 children, has an observed minimum BFMI of 0.273 versus 0.242 in the baseline fit, extends ceiling-safe matched-comprehension support from about 221 to approximately 320 words, and changes the recorded scientific conclusion from a narrowing gap to one indistinguishable from zero around 300–320 understood words. Because the variant also changes its domain, anchors and associated priors, the BFMI difference should not be attributed to the age window alone. The vague-anchor gate defined in advance passed in [`202608211545-window-22-prior-gate-passed.md`](202608211545-window-22-prior-gate-passed.md) and says VG21 registration should proceed. Current VG13 is internally valid within 8–18 months but should be retained only as the historical model after the adopted specification is promoted.

### 6.2 High — GP inertness cannot be interpreted as evidence of linearity

The shared GP length-scale prior spans roughly 6–18 months over VG13's ten-month domain. The helper first removes constant and linear components and then point-anchors the result, which restores a constant component while retaining slope orthogonality. Posterior contraction is essentially absent. A shorter-length-scale experiment produced 140 divergences while leaving the length scale largely prior-driven, as recorded in [`202608060900-three-prior-conflicts.md`](202608060900-three-prior-conflicts.md). The report's conclusion that GP inertness means the trajectories are close to straight is therefore unsupported: the current prior and basis severely suppress effective curvature capacity, so the model has little power to distinguish genuine linearity from suppressed curvature. The defensible conclusion is that curvature was not identified under this graph.

### 6.3 Medium — the BFMI prose overstates the reliability of central summaries

The current baseline has minimum BFMI about 0.242. The report says this degrades the tails rather than the centre and therefore leaves medians and means least affected in [`docs/models/vg13/index.qmd`](../docs/models/vg13/index.qmd#L276). E-BFMI is a global energy-exploration diagnostic and cannot certify that central estimates are unbiased. Emphasising medians is a reasonable precaution, but all posterior summaries remain subject to the energy warning until key quantities are stable across independent fits or parameterisations.

### 6.4 Medium validation gap — independent child effects on understood and `q` are untested

VG13 samples the child effects on understood vocabulary and conditional production independently. The corresponding independence assumption was rejected in the Down syndrome model family, and the repository identifies VG13 as a candidate for a correlated block, but no VG13-specific correlated fit was found. This is an untested structural assumption rather than a confirmed defect; shared-reporter variation would also complicate interpretation of any correlation.

### 6.5 Low — understood counts are cast before integrality is checked

VG13 converts understood values with `dtype=int` before its range checks in [`common_bivariate_re.py`](../src/vocab_growth/models/common_bivariate_re.py#L390). Current data are all integral, so the present fit is unaffected, but a future fractional value would be silently truncated. The raw finite, integral and range checks used for spoken counts should also be applied before casting understood counts.

## 7. Checks completed

The complete graphs built as follows:

| model | observations | retained studies | children | children with repeats | free random variables | initial compiled log density |
| ----- | -----------: | ---------------: | -------: | --------------------: | --------------------: | ---------------------------: |
| VG11  |       18,522 |               10 |   14,553 |                 1,947 |                    12 |                 -176,714.303 |
| VG12  |        7,052 |                6 |    5,819 |                 1,000 |                    12 |                  -51,311.452 |
| VG13  |        6,358 |                6 |    5,496 |                   830 |                    24 |                  -94,831.862 |

For all three models the audit verified observation, study and child indexing; exact zero-sum study effects; anchored and positive age-varying dispersion; valid probabilities; GP anchor values of zero; and coherent shared-new-child predictive shifts across ages. VG13 additionally had complete outcome pairing, no `spoken > understood` rows, no fallback marginal-spoken rows, exact `p_spoken = p_understood * q`, and coherent nested posterior prediction.

Focused tests covered model definitions, GP consolidation and anchoring, centred and zero-sum study effects, kappa calibration and parameterisation, variance partitioning, child identifier validation, nested likelihood construction, holdout masks, posterior summaries, sensitivity overrides and reporting ages. All selected tests passed; database-dependent tests that require a checkout-local prepared DuckDB skipped, so the real graphs were exercised separately against the available prepared database. Ruff passed for the reviewed source. No full reporting-quality posterior was sampled during this review, so existing diagnostic, sensitivity and recovery artefacts supply the posterior-level evidence above.

No current defect was found in array shapes, index alignment, Beta-Binomial parameter conversion, count bounds, prediction-grid slicing, subject namespacing, VG13 likelihood nesting, GP anchor values, study zero-sum scaling or model-registry dispatch.

## 8. Recommended actions

### Before relying on current reports

1. Correct the figure captions, zero-effect terminology, `kappa` headline label, VG12 extrapolation warning and VG13 BFMI/curvature wording.
2. Deduplicate the TD frame before filtering and coding, with a regression test and recorded removal count.
3. Pass the registered language scope through prior-calibration tooling and regenerate its evidence.
4. Promote and fit the adopted window-22 specification as VG21; retain VG13 as the historical 8–18-month model.

### Before making quantitative heterogeneity claims

5. Fit a form- or age-varying child-loading sensitivity specified in advance, preferably alongside an explicit form/item measurement model.
6. Implement leave-one-child-out and leave-one-study-out validation with held-out effects integrated out.
7. Expand reporting-quality recovery for VG11, VG12 and VG13, scoring total variance as well as its allocation between child and residual components.
8. Test simplification and alternative parameterisations for the VG11 divergence exception and low VG12/VG13 BFMI, recognising that existing attempts have not resolved the latter and that additional repeat measurement may be required; require stability across independent fits.
9. Test the 200-row study threshold, common-age support, study-specific slopes and a correlated understood/`q` child block for VG13.

## 9. Bottom line

VG11, VG12 and VG13 implement their registered likelihoods correctly, and VG13's nested probability construction is sound. Their strongest vulnerability is not a coding mistake in those likelihoods but a mismatch between the common 810-item measurement scale, the age-invariant child effects and the interpretations assigned to `tau_subject`, `kappa` and predictive intervals. VG11 and VG12 may support qualified mean-trajectory summaries, but their heterogeneity conclusions remain provisional. The present VG13 is a coherent historical 8–18-month model, not the canonical model implied by the project's adopted evidence.
