> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

# VG14 and VG15 statistical-model review

**Date:** 23 August 2026  
**Reviewed revision:** `0271871298f226591a2f90ff7ce4349dec856c47`  
**Tracking issue:** [#238](https://github.com/dseinternational/vocabulary-growth/issues/238)

## Purpose and conclusion

This note records a read-only review of the current VG14 and VG15 code, statistical specification, data classification, derived estimands, diagnostics, tests and report text. It distinguishes confirmed implementation defects from modelling assumptions whose numerical consequences require sensitivity analysis.

Both model graphs build successfully on freshly prepared current data and have finite initial log densities. Their principal probability calculations are implemented coherently: the Beta-Binomial parameterisation is correct, cell mappings and masks align, the VG15 Plackett root is algebraically correct, and the cell likelihood avoids double-counting the suppressed spoken and signed marginals.

VG14 is nevertheless not adequate as a standalone inferential model. It omits study and child effects, assumes conditional independence between speaking and signing, combines conditional and marginal likelihoods under common dispersion curves, and imports dispersion priors calibrated for a random-effects graph it does not contain. The repository is correct to treat VG14 as superseded by VG15.

VG15 is structurally stronger, but several corrections and additional checks are needed before its headline association estimates should be treated as publication-ready. In particular, repeated cell observations from the same child enter the composition likelihood without child dependence, the observations identifying the association parameter are omitted from quantitative LOO and calibration, and several derived quantities are reported incorrectly.

## Review method

The review traced each model from its registered definition through data preparation, PyMC graph construction, likelihood factorisation, posterior extraction, diagnostics, report artefacts and Quarto interpretation. The current source data were prepared in a temporary output location so the actual model frames and observation masks could be inspected without changing the repository.

Focused tests covering model definitions, nested likelihoods, four-cell construction, GP anchoring, reporting ages, trend consolidation and recovery specification passed. Some reporting tests pass because they encode the obsolete reporting-age behaviour identified below rather than the canonical policy.

The actual current graphs built with the following dimensions and finite initial log densities:

| Model | Analysis rows | Initial log density |
|---|---:|---:|
| VG14 | 1,431 | -22,220.496 |
| VG15 | 1,431 | -26,074.348 |

No current compatible reporting-quality trace was present. Consequently, this review verifies the current graph, algebra, priors, masks, data classification and initial execution, but it cannot confirm current-definition convergence, effective sample sizes, posterior predictive coverage or the numerical magnitude of the identified specification risks.

## Findings

### 1. P1: comprehension-conditioned outputs exceed the declared 72-month evidence cap

The canonical policy in [`reporting_ages.py`](../src/vocab_growth/reporting_ages.py) classifies `q`, `r` and `p_any` as ratios conditioned on understood vocabulary and assigns them the understood-vocabulary cap of 72 months. VG14 and VG15 both declare understood = 72 months and signed counts = 84 months.

VG14 nevertheless trims `posterior_summary_r` and `posterior_summary_p_any` with the signed-count cap and passes the signed cap to the signed-ratio and sign-speech crossover plots. Its modality plot computes the `p_any` cap from spoken and signed limits only, again producing 84 months.

VG15 makes the same error for the `r` and `p_any` summary tables. Its plot stage correctly takes the earlier of the understood and signed limits, so a current fit would produce tables through 84 months beside figures ending at 72 months.

The downstream DS-versus-TD expressive comparison independently hardcodes 84 months while describing the quantities as comprehension-capped. The VG15 report uses a nearest-row lookup for requested ages, so once profiles are correctly capped it could silently print the 72-month value under an 84-month label. Tests and documentation currently preserve the obsolete policy instead of detecting the inconsistency.

This is post-processing only and does not alter a fitted posterior. If 84 months is scientifically intended, the canonical policy should be changed explicitly; otherwise all summaries, plots, scripts, tests and report prose should use the 72-month ratio-of-understood limit.

### 2. P1: VG15 composition observations omit repeated-child dependence

VG15 constructs child effects for understood, spoken and signed marginal trajectories. The four-cell within-understood and three-cell within-produced Dirichlet-Multinomial likelihoods deliberately use population-plus-study `q` and `r` trajectories without child shifts.

The current association data contain:

| Source | Administrations | Children |
|---|---:|---:|
| `es_01` | 185 | 185 |
| `uk_02` | 56 | 28 |
| `uk_07` | 82 | 30 |
| `nz_01` | 111 | 33 |

The 249 longitudinal administrations in `uk_02`, `uk_07` and `nz_01` come from 91 children and are conditionally independent between visits in the cell likelihood. The Dirichlet-Multinomial concentration accounts for extra variability among cells within one administration; it does not model serial correlation between a child's visits.

The omission does not by itself prove a biased point estimate. Its defensible consequences are that uncertainty in `psi` and study-specific associations may be too narrow, children with more visits receive greater weight, and unmodelled child modality propensities may alter the association estimand. Odds ratios are non-collapsible, meaning a population-level odds ratio can differ from a child-conditioned odds ratio even without conventional confounding.

The present estimand is therefore a population-plus-study composition association, not a within-child association adjusted for child modality propensities. Report statements that all repeated measurements are clustered, or that `psi` has empirically separated from child signing heterogeneity, overstate what the graph establishes: the two quantities are prevented from competing in the cell likelihood rather than jointly identified there.

A child-shifted cell model, one-administration-per-child analysis, cluster bootstrap or child-level held-out sensitivity is required to measure the numerical effect.

### 3. P1/P2: the reported signed-ratio peak is a tent-knot location, not the fitted curve maximum

`peak_unit_sign` locates the middle knot of a piecewise-linear tent. The full signed-ratio latent adds a GP departure after constructing that tent. The GP value and derivative are not constrained at the knot, and the three anchor heights are sampled independently rather than ordered.

A two-million-draw prior simulation found a 5.52% probability that the middle anchor was no higher than at least one outer anchor, before adding the GP. No reporting code calculates the global maximum of the complete `r(a)` curve per posterior draw. Calling `peak_unit_sign` the age at which the fitted signed ratio peaks is therefore incorrect for VG15. VG14 has the same conceptual problem: its middle knot is fixed, but its complete curve can peak elsewhere after adding the GP.

The existing parameter can be reported as the tent-knot or parametric-apex location. If the scientific estimand is the full curve peak, calculate the global argmax of `r(a)` per posterior draw over a declared support and report boundary censoring and multiple-peak diagnostics. If genuine unimodality is required, the anchor ordering and GP departure need shape constraints.

Peak-age summaries also currently use a mean and equal-tailed interval in places where project policy requires a median and HDI.

### 4. P2: VG14 dispersion priors target a different random-effects graph

VG14 has no study or child random effects, but imports `_DS_JOINT_UNDERSTOOD_KAPPA_RE` and `_DS_JOINT_Q_KAPPA_RE`. Those components are explicitly calibrated for models with subject random effects on both outcomes. A conditional residual dispersion after modelling child heterogeneity is not the same estimand as marginal dispersion in a graph without that heterogeneity.

Applying the committed calibration procedure to VG14's exact current frame produced approximately:

| Dispersion concentration | No-effects calibration | Current component target |
|---|---:|---:|
| Understood at 18 months | 13.7 | 92.6 |
| Understood at 72 months | 3.23 | 14.0 |
| Conditional spoken ratio at 18 months | 7.1 | 18.2 |
| Conditional spoken ratio at 72 months | 1.67 | 8.4 |

In a Beta-Binomial model, larger `kappa` means less unexplained variation. The current priors therefore favour substantially narrower individual predictive variation than a calibration matching VG14's graph. The no-effects targets lie near the extreme lower tail of the imported priors, so their nominal breadth does not make the mismatch negligible.

The conditional spoken calibration is not a complete replacement target because VG14 uses the same curve for both conditional and fallback marginal observations. The understood mismatch is direct. A current trace is unavailable, so the extent to which the likelihood overcomes the prior cannot be quantified.

### 5. P2: VG15 understood study centring includes an uninformed study

`nz_01` supplies no understood observations, and its within-produced likelihood algebraically cancels the understood trajectory. Nevertheless, `delta_u` is zero-summed over every retained study under an incorrect assertion that every study informs understood vocabulary.

The prior-only `nz_01` understood offset therefore counterbalances offsets from the 13 studies that do inform understood vocabulary. A common shift among informed-study offsets can be absorbed by this uninformed nuisance coordinate under its prior. This makes the population understood level partly dependent on an unobserved constraint component.

The likely numerical effect is limited because one of fourteen studies is affected, but this is a genuine identification defect. The signing and `psi` blocks already implement the correct informed-study-only construction and provide the pattern to follow.

### 6. P2: VG15 milestone algorithms report initial states and boundaries as transitions

The milestone helper reports the grid `argmax` as a reached peak even when the maximum occurs at the final reporting age, where the peak is right-censored. Its `draws_reaching` indicator is based only on whether a finite grid maximum exists, so it cannot warn about boundary censoring.

The helper also returns the first age at which a condition is true rather than requiring a false-to-true transition. A trajectory where speech-only exceeds sign-only from the first eligible age is therefore labelled as an overtake, and a trajectory that was never majority sign-only is labelled as falling below half. A current unit test explicitly encodes the first behaviour, and the downstream comparison script duplicates the algorithm.

Crossings should require an actual transition. Grid-boundary maxima should be classified as censored rather than reached. Peak and milestone ages should use the project's HDI policy.

### 7. P2: observations identifying VG15 association are absent from quantitative validation

VG15 LOO and calibration include only the marginal understood, spoken and signed likelihoods. They exclude `cells_obs` and `nz_prod_cells_obs`, which identify the headline association parameter. Marginal-count LOO therefore does not validate `psi`, cell composition or honest new-child association performance.

The within-understood composition PPC aggregates all three contributing sources after discarding their labels, but the output filename, title and axis refer to `uk_02`. With heterogeneous source associations, aggregate agreement can hide offsetting source-specific errors. The report text itself records aggregate cell discrepancies and weaker reproduction of the `nz_01` both cell, although those reported magnitudes should not be treated as current-definition evidence without a compatible trace.

No current subject-LOSO implementation supports VG15, despite publication-facing text saying that the honest new-child calculation is run separately. Parameter recovery is wired for the cell outcomes, but recovery under the assumed model cannot detect real-data misspecification or omitted repeated-child dependence.

Required additions are pointwise cell-likelihood calibration, per-source age-banded PPCs, child-level held-out assessment, multiple current-graph reporting-quality recovery replicates, and a sensitivity allowing separate concentration parameters for the four-cell and three-cell likelihoods.

### 8. P2: VG14 is a composite rather than coherent longitudinal joint model

The current VG14 frame has 1,431 administrations from 767 children. There are 335 repeatedly observed children contributing 999 rows. VG14 discards child identity and has no study effects, so repeated records are treated independently and age trends may absorb study composition, instrument and signing-practice differences. Beta-Binomial overdispersion does not represent correlation between visits from the same child.

The nested likelihood uses conditional outcome-out-of-understood observations when possible and falls back to marginal outcome-out-of-810 observations when understood is unavailable or recorded nesting is violated. Current counts are 973 conditional and 455 fallback spoken observations, with 11 spoken-greater-than-understood violations, and 569 conditional plus 117 fallback signed observations.

The same dispersion curve is used for a conditional ratio and for its fallback marginal outcome. These have different generative variance meanings: if understood and the conditional production ratio each vary, their product is not generally Beta-distributed with the same `kappa`. The result is a pragmatic composite likelihood rather than exact marginalisation of one joint model.

VG14 also factorises spoken and signed production and derives `p_any` under conditional independence. The arithmetic is correct under that assumption, but the cell data motivating VG15 show that independence is not an adequate substantive model. VG14 should remain a documented historical development step rather than an inferential alternative to VG15.

### 9. P2: VG14 `p_any` validation is incomplete and not like-for-like

The VG14 validation reads only the `uk_02` four-cell source. Current support also includes `uk_07` and `es_01` within-understood cells and `nz_01` within-produced cells, with materially different descriptive associations by source. A `uk_02` check is therefore not a general validation of conditional independence.

The observed gap is averaged over the empirical age distribution of `uk_02`, while the model gap is averaged over an equally spaced model grid. Differences can arise from age weighting alone. The comparison also uses pointwise posterior medians and supplies no posterior interval. A defensible comparison would evaluate every posterior draw at the observed ages or use one declared standard age distribution for both quantities.

### 10. P3: validation, efficiency and documentation hardening

- VG14 converts understood counts to integer before checking integrality, so a future non-integral count could be silently truncated. Current values are integral.
- Signing anchors are checked for sorted order but not strict inequalities; duplicate anchors can produce division by zero.
- `sign_peak_prior` is indexed without checking length, finiteness or positive Beta parameters.
- VG15 samples many prior-only child-effect coordinates. These integrate out exactly and do not bias inference, but they increase NUTS dimension, trace size and convergence-gate burden.
- The VG15 model wrapper and several methods sections still describe an older `uk_02`-only scalar-association model, state that the signed peak is fixed, or imply validation that is not implemented.
- Zero random effects describe a conditional reference child and reference study, not an outcome-scale population average because the inverse-logit transformation is nonlinear. Report wording should avoid calling that curve an average child without qualification.

## Components verified as correct

- VG14 and VG15 registry and dispatch are consistent.
- Beta-Binomial parameters use `alpha = p * kappa` and `beta = (1 - p) * kappa` with the intended row-specific denominators.
- Current cell order and source mappings are consistent through loaders, model coordinates, likelihood construction, posterior extraction and recovery simulation.
- Four-cell records suppress duplicate spoken and signed marginals while retaining the understood total. The marginal-understood-times-conditional-composition factorisation is coherent and avoids double-counting.
- Cell sums, count bounds, masks and coordinate alignments are checked before model construction.
- The rationalised Plackett root is correct, continuous at `psi = 1`, respects the Frechet bounds and reconstructs the requested margins.
- The VG15 `p_any` union and independence-bound formulas are correct.
- Study indexing and posterior study-name ordering use the same sorted order.
- The GP anchor and contrast-orthogonalisation implementation does not create the suspected intercept-identifiability defect. After anchoring, the construction is more precisely described as contrast-orthogonal or an oblique complement, but the centred contrasts and reference-age anchor remain identified.
- VG14 posterior prediction coherently draws understood first and conditions spoken and signed predictive counts on the same understood draw.
- The shared sampling path and convergence gate include every free-variable element.

## Recommended sequence

1. Correct the reporting-age policy implementation, exact-age report lookup, peak labelling or derivation, milestone transitions and related tests and documentation. These changes do not require refitting when the necessary posterior arrays are available.
2. Correct VG15 understood-study centring and refit.
3. Add a repeated-child composition sensitivity, cell-likelihood calibration and per-source PPCs before interpreting `psi` uncertainty as publication-ready.
4. Run a current reporting-quality VG15 fit, several recovery replicates and honest child-level held-out validation.
5. Recalibrate VG14 dispersion only if VG14 is retained for substantive use; otherwise reinforce its superseded status and avoid presenting its derived union as an inferential result.

Implementation and validation work is tracked in [issue #238](https://github.com/dseinternational/vocabulary-growth/issues/238).
