# VG16 statistical model review: code, estimand, prediction and validation

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

> [!IMPORTANT]
> Independent review completed 2026-08-23 against commit `0271871`. **VG16's current numerical result should not be quoted.** A row-order-dependent lag-construction defect changes which observations identify `beta_lag`; the current source definition has no compatible fitted posterior; the subject-marginal predictions omit the lag; and several report claims are stronger than the model or validation supports. Remediation is tracked in [#242](https://github.com/dseinternational/vocabulary-growth/issues/242).

## 1. Scope and conclusion

This review covered the VG16 wrapper and registered definition, the shared bivariate random-effects graph, data preparation and indexing, the nested understood/spoken likelihood, prior- and posterior-predictive paths, diagnostics and cross-validation, the model report, the two bespoke VG16 experiment scripts, the generic parameter-recovery harness and the current committed source data. It reconstructed the default VG16 frame in a temporary DuckDB rather than relying on historical notes, built the current PyMC graph on synthetic data, ran focused tests and independently checked the main code and statistical conclusions.

**Conclusion.** The observation-level likelihood is mostly a faithful implementation of the declared population-relative association model, and its dimensions, masks, random-effect indexing and nested likelihood are correct. It is not, however, a verified temporal cross-lag model in its current form. The lag helper has a live, material row-order defect; different-form rows for one child at the same recorded age have no explicit shared measurement model; the defining term is absent from subject-marginal prediction; the validation suite cannot recover or cross-validate the estimand sequentially; and the report overstates temporal direction, attenuation and evidence against a within-child association. Fixing the lag helper is necessary but not sufficient: the administration-wave estimand and the treatment of same-recorded-age forms must be decided before refitting.

## 2. Model actually fitted

For child $i$ in study $s$ at observation $t$, VG16 models expected understood proportion $p^U$ and the production ratio $q$ on the logit scale:

$$
\operatorname{logit}(p^U_{ist}) = f_U(a_{ist}) + b^U_s + b^U_i,
$$

$$
\operatorname{logit}(q_{ist}) = f_q(a_{ist}) + b^q_s + b^q_i + \beta_{\mathrm{lag}}x_{ist}.
$$

Let $t^-$ denote the most recent **strictly earlier recorded-age row with a usable understood count**; it can skip intervening visits whose understood count is missing. With the registered population-relative baseline,

$$
x_{ist} = \operatorname{logit}(U_{i,t^-}/810) - \{f_U(a_{i,t^-}) + b^U_s\}.
$$

The child's persistent understood effect $b^U_i$ is deliberately **not** subtracted, so the predictor mixes persistent between-child standing and temporary within-child deviation. Where both outcomes are observed and $S \le U$, spoken vocabulary is modelled conditionally as $S \mid U$ with $U$ trials and mean $q$; spoken-only rows and nesting violations use the documented marginal 810-trial approximation with mean $p^Uq$. The registered prior is $\beta_{\mathrm{lag}} \sim \operatorname{Normal}(0, 0.5)$, which is symmetric and does not encode a positive direction.

## 3. Implementation and measurement findings

### 3.1 P1: lag assignment is row-order and form-order dependent

`_compute_prev_wave_lag()` in `src/vocab_growth/models/common_bivariate_re.py` sorts by child and age but advances `last` and `last_age` immediately after each row. Consider one child with understood observed at age 12, followed by an understood-and-spoken row and a spoken-only row at age 24. If the understood row appears first, the age-12 source is assigned to that row and `last_age` immediately becomes 24, so the parallel age-24 spoken row receives no lag. If the spoken-only row appears first, both age-24 rows receive the age-12 source. The likelihood therefore changes under a permutation that carries no statistical information.

The defect is material in the reconstructed current frame:

| quantity                                                    | current helper | grouped retain-all interpretation |
| ----------------------------------------------------------- | -------------: | --------------------------------: |
| rows with a prior understood source                         |            412 |                               478 |
| rows with both a prior source and current spoken likelihood |            409 |                               475 |
| current-spoken rows assigned no lag by the helper           |              — |                            **66** |

The 66 omitted rows come from 46 children: 65 are `us_01` spoken-only WS rows paired with a same-age WG row, and one is from `uk_02`. They are 13.9% of the 475 spoken observations from 249 children that would receive a lag if every row were retained as a separate observation and sources were assigned by complete recorded-age group. The report is internally ambiguous: it first says every observation with a strictly earlier usable wave receives a lag, then says a duplicate same-age row receives zero. A form-selection, collapse or joint-measurement policy would produce different totals, so 475/249 is a code-specification candidate set rather than the definitive corrected analysis set. The current row-order-dependent mixture implements none of those policies consistently. The currently quoted 412 rows and 250 children are themselves not the coefficient's direct support: three lagged rows have no current spoken likelihood, leaving 409 rows and 248 children that can affect `beta_lag` before the defect is corrected.

The existing unit test checks `[12, 12, 24]`, which confirms that a same-age row is not used as a lag for another row at that same first wave. It does not check the failing `[12, 24, 24]` pattern or permutation invariance. The same row-wise algorithm is copied into `scripts/experiments/vg16_crosslag_quantification.py` and `scripts/experiments/vg16_within_lag_bias.py`, so those analyses inherit rather than detect the defect.

**Required correction.** Define and process a complete `(subject, age)` wave before advancing the prior-wave state. Every retained row at the current wave must use the same prior distinct-age source. If several understood measurements exist at the source wave, their selection, crosswalk or aggregation must be explicit. Reordering is not a correction.

### 3.2 P2: same-recorded-age forms expose an unresolved measurement-dependence assumption

The same-age defect exposes a broader issue. The current frame contains 100 child-age groups with two different-form rows, comprising 200 rows. The stored data establish the same child and recorded age in months, not necessarily simultaneous administrations. All 100 groups use different checklist ceilings; 89 of the 99 groups with two spoken values have different spoken counts, and all 10 groups with two understood values have different understood counts. The likelihood treats the two overlapping measurements as conditionally independent, with no form effect and no same-age residual correlation. Under grouped retain-all semantics, 67 such groups have an earlier understood source; for 66 the current masks are `(lag, no lag)`, while one receives the lag on both rows because of the opposite tie order.

These may contain distinct item information, so simply deleting one row is not automatically correct. Equally, treating overlapping forms completed by the same child at the same age as independent administrations can overstate information. The project must first define the estimand at an administration wave and then choose one defensible treatment: select one form under a pre-specified rule, equate and collapse the measurements, or model the paired forms jointly with their overlap and measurement error.

### 3.3 P1: subject-marginal predictions omit `beta_lag`

The fitted observation likelihood correctly adds `beta_lag*x_lag` to `q_obs`. The plot and query grids instead construct `q_plot` and `q_query` from the age trajectory alone, and `sample_posterior_predictive()` adds new child random effects without reading `use_cross_lag`, `beta_lag`, an earlier understood observation or a history distribution. As a result, `p_s_query_subject_marginal`, `y_s_plot`, `y_s_query`, spoken summary intervals, joint trajectories, spoken cumulative-distribution plots and the predictive understood-versus-spoken figure are all conditional on `x_lag=0`.

Population curves deliberately defined as zero-lag references are valid, and the subject-marginal quantities are valid predictions conditional on no usable lag history. The defect is the report's description: it calls them answers about where one unseen child would fall and coherent trajectories across age without disclosing that conditioning. They are not marginal draws over histories from VG16's sequential data-generating process. A full longitudinal prediction must specify a visit schedule, simulate earlier understood vocabulary, form the resulting lag and then simulate later $q$ and spoken vocabulary. Otherwise every affected artefact must be labelled as a zero-lag or no-usable-history conditional reference.

## 4. Statistical interpretation findings

### 4.1 P1: `beta_lag` does not isolate a temporal or within-child cross-lag

The population-relative predictor retains persistent understood standing. At the same time, the registered model draws the understood and production-ratio child effects independently, includes neither lagged $q$ nor a reciprocal path, and forces the lag association to zero before or without an observed prior understood wave. A stable receptive-to-conversion association can therefore be absent at a child's first measured wave and appear only after the study design happens to observe a prior assessment.

This makes `beta_lag` a history-dependent mixture of between-child and within-child association. It can proxy covariance between persistent receptive standing and persistent conversion ability that the independent random-effect distribution cannot express. It does **not** follow that the positive historical estimate is necessarily spurious or entirely caused by omitted covariance; that requires a combined model. It does follow that the report's statement that the data "establish the direction" is not supported by VG16 alone. The random-intercept cross-lag literature makes this separation central because conventional cross-lag coefficients can conflate stable between-person differences with within-person dynamics; see [the 2015 RI-CLPM paper](https://pubmed.ncbi.nlm.nih.gov/25822208/).

The defensible reading is narrower: conditional on VG16's structure and the observed measurement history, the posterior may favour a positive prospective association between prior receptive standing and later conversion. A wave-sequential comparison among VG16, VG20 and a correlated-random-effects-plus-lag model is necessary to separate persistent covariance from the lag under this structure, but it would not by itself establish a within-child dynamic, causal mechanism or causal direction; reciprocal dynamics and time-varying confounding would remain unaddressed.

### 4.2 P1: the fitted coefficient is not a proven lower bound

The report describes the historical fitted coefficient as a "floor" on the association with true standing. The supporting script defines reliability as `tau_subj_u**2 / var(x_lag)` and divides `beta_lag` by it. That is a classical errors-in-variables sensitivity calculation, not an identified correction for this model.

Its assumptions are not established here. The predictor is a bounded Beta-Binomial count put through a logit and boundary clip; its error is heteroscedastic and combines sampling variation, genuine occasion movement, trajectory-estimation uncertainty and checklist-form differences; the response model is nonlinear and multilevel; and persistent understood-to-conversion covariance is omitted. Bias towards zero is plausible under some data-generating processes but is not guaranteed. A separate within-baseline simulation produced upward bias for one nonzero truth; because that is a different parameterisation, it does not establish the direction of bias in the registered population-baseline estimator, but nor does it validate transferring a simple lower-bound argument across these joint estimators. The historical `0.383` disattenuated figure must therefore be presented, if retained at all, as a heuristic sensitivity under explicit classical-error assumptions, not as a corrected estimate or credible lower bound.

### 4.3 P1: failure to exclude zero is treated as evidence of a negligible within-child association

The off-record within-child fit is reported as `+0.103` with an 89% interval `[-0.085, 0.294]`. The report correctly says that this lower-tier fit is not quotable, but elsewhere says the within-child component "carries almost nothing", "has no memory" and that its failure to exclude zero is especially telling because one simulation showed upward bias. Those claims do not follow from the interval: it includes positive values larger than the historical population-relative headline of `+0.203`, and no substantively justified equivalence margin or region of practical equivalence was defined. Failure to establish a nonzero effect is not evidence that the effect is negligible.

A registered, reporting-quality within-child sensitivity is needed if the contrast is important. It should report the posterior probability that the coefficient lies within a pre-specified negligible-effect interval, alongside its full interval, rather than infer absence from non-significance.

### 4.4 P2: one coefficient averages over heterogeneous intervals and measurement regimes

Observed lag gaps range from 1 to 28 months, yet one constant coefficient is applied with no gap interaction or decay. The estimate is therefore a design-weighted association over this particular interval distribution, not a general earlier-to-later parameter. Longitudinal coefficients can depend on individually varying intervals; see [this continuous-time panel-model analysis](https://pubmed.ncbi.nlm.nih.gov/22420323/). At minimum the result needs restricted-gap and gap-interaction or decay sensitivities.

VG16 also maps every prior raw count to `logit(count/810)`. This is the project's documented difficulty-ordering harmonisation and is not shown by this review to be invalid. It is, however, untested for the lag coefficient. Among the 409 currently active lag rows, source-form ceilings are 396 for 90 rows, 408 for 107, 416 for 78, 674 for 52 and 810 for 82; 66 active rows from 50 children change form ceiling between source and target waves. Study intercepts cannot absorb within-study form transitions. A DSE-native restriction is supported by the engine but registered only for VG10 and VG15, not VG16. It would leave only 81 active lag rows from 75 children in two studies, so a difference could reflect study composition as well as measurement scale. Same-form-only, DSE-native and preferably form-effect or crosswalk-uncertainty sensitivities are useful checks, but none alone identifies a scale effect.

Six lag sources have understood count zero and are clipped to a fixed logit of `-9.210`; no boundary-treatment sensitivity exists. Of the 409 current likelihood-supporting rows, 47 use the documented marginal spoken fallback rather than the conditional $S\mid U$ branch. Neither fact proves that the estimate is biased, but both identify inexpensive sensitivity checks.

### 4.5 P2: identification relies on an untested available-case assumption

The lag is active only where an earlier usable understood count and a later spoken count are observed. The helper skips intervening visits with missing understood data, and understood missingness is partly structural because forms measure different outcomes. Of the 47 currently active marginal-fallback rows, 45 lack current understood. VG16 conditions on these available observations and has no model for outcome-observation or form-assignment processes. This is not automatically biased, but interpretation requires the observation process to be ignorable after the modelled study, age and child structure. Support counts and sensitivities should therefore distinguish structural form missingness from other missingness and test whether results change under complete-pair and form-restricted analyses.

### 4.6 P2: the symmetric coefficient prior is not predictively calibrated

`Normal(0, 0.5)` avoids preferring a positive or negative coefficient, but symmetry does not make an interval excluding zero purely a data result. The VG16 report itself records that the shared prior-predictive figures do not isolate the cross-lag and that no beta-specific prior-versus-posterior or effect-scale calibration artefact exists. Before interpreting posterior exclusion of zero, the prior should be translated into changes in $q$ over the empirical `x_lag` range and checked through a beta-specific prior predictive; a prior-scale sensitivity should accompany the corrected fit.

## 5. Validation and diagnostic findings

### 5.1 P2: ordinary understood PSIS-LOO leaks the held-out outcome through the lag predictor

VG16 converts earlier understood counts to fixed NumPy covariates before constructing the PyMC graph. Generic diagnostics then compute separate pointwise LOO values for `y_u_obs` and `y_s_obs`. Leaving an earlier understood likelihood term out does not remove that exact count from later spoken likelihood terms, where it remains inside `x_lag`. The understood score is therefore not genuine leave-one-understood-out prediction. Spoken LOO is at best prediction conditional on the same child's observed understood history and, for conditional rows, current understood count; that may be useful, but it is a different estimand from unconditional new-child prediction.

Pareto-$k$ diagnostics test the stability of importance reweighting and cannot detect this outcome leakage. The generic report wording that LOO predicts an unseen observation is consequently wrong for VG16. The suggested alternatives do not currently repair the gap: `scripts/loso_compare.py` supports VG07–VG09, and `scripts/kfold_loso.py` supports VG07–VG10, VG19 and VG20 but omits VG16. VG16 needs an explicitly defined grouped forward-chaining or held-out-child sequential score; until then, understood LOO should be suppressed or prominently caveated.

### 5.2 P2: end-to-end recovery of the headline estimand is absent

The generic recovery harness correctly refuses VG16 because its design matrix depends on an earlier simulated outcome and therefore requires wave-sequential simulation. The bespoke experiments are informative but do not establish positive-truth recovery or interval coverage for the registered population-relative Bayesian model under correlated child traits. The lag-indexing reproduction in those scripts also shares the confirmed row-order defect.

The minimum recovery matrix should include `(beta=0, rho_uq!=0)`, `(beta!=0, rho_uq=0)` and both nonzero, using the real gap, form, missingness and parallel-wave structure. Comparing VG16 with VG20 plus a lag in those scenarios would directly test whether independent child effects load persistent covariance onto `beta_lag` and whether the intended temporal component is recoverable.

### 5.3 P2: the pair plot cannot show the trade-offs the report assigns to it

The generic diagnostic path caps a pair plot at the square root of ArviZ's 40-subplot limit, so only six scalar variables are selected in model order. A current synthetic VG16 graph places `beta_lag` after those six; the child and study scales discussed alongside it are also not all present. The report nevertheless says this plot is where trade-offs between `beta_lag` and the scales would appear. A dedicated, deliberately ordered VG16 pair plot is required if that diagnostic claim is retained.

## 6. Report and definition drift

| statement                                                                                                    | correction                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Earlier models make $q$ a function of age alone, so same-age children necessarily convert equally.           | VG09 and VG10 already include study and child random effects on $q$. They lack a structured understood-standing-to-$q$ association, not conversion heterogeneity.                                                                                       |
| The production-rate-by-understood plot illustrates the between-child result.                                 | Its formula plots lag-free population $p_U(a)$ against lag-free population $q(a)$ and merely re-expresses age; it contains no `beta_lag`. A separately refitted zero-lag model could move the other posterior parameters and hence the numerical curve. |
| The pair plot displays competition between `beta_lag` and the random-effect scales.                          | The generated pair plot omits `beta_lag` under the current variable order and subplot cap.                                                                                                                                                              |
| 412 observations and 250 children identify the coefficient.                                                  | Those counts describe any lag source. Only 409 rows and 248 children currently have a lag and a current spoken likelihood; the grouped retain-all candidate set would be 475 rows from 249 children before resolving same-age forms.                    |
| The understood cap is the high anchor above which its mean is levelled.                                      | The current understood reporting cap is 72 months, the high anchor is 84 months, and only the $q$ mean is clamped.                                                                                                                                      |
| The population baseline is bias-robust because the within baseline produced a negative short-panel artefact. | The report later withdraws that mechanism and attributes the negative development-tier result to non-convergence. The definition banner and comments are stale.                                                                                         |

The historical reporting-quality fit at commit `d041e7f` was recorded as clean and gave `beta_lag=+0.203`, 89% interval `[0.093, 0.316]`. That is evidence about the earlier implementation, not a verified result for the current source. The shared dispersion definition changed on 2026-08-19 and the repository explicitly records VG16 as awaiting refit; the 2026-08-22 reporting-cap change then invalidated the fit fingerprint, although that cap alone cannot move the posterior. No compatible local trace was available during this review. `docs/models/README.md` and later prose must not continue to hard-code the old coefficient after the correction.

## 7. Components verified as correct

The review found no defect in the following components:

- The VG16 wrapper, registry dispatch and use of the shared bivariate random-effects engine are correct.
- Apart from naming and the active population-relative lag, VG16 matches VG10's anchored mean, GP, random-effect and dispersion structure as intended.
- Observation, outcome, study, subject, GP-grid and query dimensions are internally consistent.
- Subject identifiers are namespaced by study, so lag sources cannot cross studies through reused source identifiers.
- When forming the population-relative baseline, the code subtracts the source child's understood effect from their expected source logit while retaining the population trajectory and study effect; `x_lag` consequently retains the child's persistent understood standing as intended.
- Non-centred child effects and sum-zero study effects are implemented consistently.
- Missing-outcome masks and separate understood/spoken likelihood indices are coherent.
- The nested spoken likelihood correctly uses $S\mid U$ where paired observations satisfy $S\le U$ and applies the documented marginal fallback elsewhere.
- `beta_lag` enters the observed spoken likelihood, parameter summaries and convergence gate, and its prior is symmetric around zero.

These checks establish that the model mostly does what its code says. They do not rescue the lag-row defect or validate the stronger scientific interpretation.

## 8. Verification performed and limits

The reconstructed default frame matched the project's current expected dimensions: 1,431 rows, 987 understood observations, 1,428 spoken observations, 767 children and 14 studies. Focused tests covered cross-lag helpers, model definitions, recovery registration, K-fold support, reporting cells, degenerate LOO handling and holdout masks: 104 tests passed. Ruff passed on the relevant engine, wrapper, tests, recovery and cross-validation code. The current PyMC graph was built on synthetic data to confirm that `beta_lag` is a free scalar and that the current dispersion blocks are in use.

No reporting-quality sampling run was attempted: the review's purpose was to inspect correctness, the current compatible output is absent, and a full fit before correcting the confirmed defect would produce another invalid headline. Consequently this note does not claim that the corrected coefficient is positive, null or negative, and it does not infer the direction or magnitude by which the defect will change it.

## 9. Remediation gates

### Gate A — define and correct the data unit

1. Decide what constitutes one administration wave and how different-form rows at the same recorded child-age are selected, mapped to a common scale, collapsed or modelled jointly.
2. Compute prior sources by complete subject-age groups, not rows, and apply one source consistently to every retained current-wave likelihood row.
3. Correct both experiment scripts and add `[12, 24, 24]`, permutation-invariance and multiple-source-form tests.
4. Persist lag-source rows, current-spoken support, contributing children, gaps, forms and conditional/marginal branch counts as fit artefacts.

### Gate B — define the estimand and predictions

1. State whether the target is a history-conditioned prospective association, a within-child deviation effect or a persistent between-child association.
2. Fit a correlated-random-effects-plus-lag comparison rather than forcing those components into separate models, while stating that this separates model components rather than establishing a causal or within-child direction.
3. Implement wave-sequential subject-marginal prediction or relabel all grid outputs as zero-lag references.
4. Remove the claims of established temporal direction, a proven attenuated floor and a negligible within-child component unless purpose-built evidence supports them.

### Gate C — demonstrate recoverability and robustness

1. Add wave-sequential recovery across zero and nonzero lag/correlation combinations.
2. Implement an explicitly conditioned forward or held-out-child validation score and suppress misleading understood LOO.
3. Run gap, leave-one-study-out, same-form/DSE-native, conditional-only and zero-count boundary or continuity-correction sensitivities.
4. Audit the available-case assumption, distinguishing structural form missingness from other missingness and adding complete-pair or form-restricted checks.
5. Add a beta-specific prior predictive and coefficient-prior-scale sensitivity.
6. Assess whether the conclusion changes materially with study, form transition, lag interval or fallback likelihood branch.

### Gate D — refit and report

1. Fit the corrected current definition at reporting quality and pass all convergence, lifecycle and provenance gates.
2. Regenerate prediction, calibration and diagnostic artefacts under their stated conditioning history.
3. Update every hard-coded coefficient, support count and model-description claim from the compatible fit.
4. Do not publish or present a numerical estimate as the current VG16 result until Gates A–D are complete; historical and diagnostic estimates may be quoted only with their provenance and caveats.

## 10. Disposition

VG20 remains the model of record for Down syndrome joint trajectories. VG16 can remain a single-purpose development model for the prior-understood association, but its existing headline is withdrawn pending [#242](https://github.com/dseinternational/vocabulary-growth/issues/242). The previous positive posterior is useful evidence that the corrected model deserves refitting; it is not a substitute for that refit and should not be used to predict the corrected result.
