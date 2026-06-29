# Prior rationale and review notes

<!-- cspell:words conc -->

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

> [!WARNING]
> This is a working document for issue 89, last reviewed on 2026-06-29. It
> records the current prior inventory, first-pass interpretation, and review
> questions. It is not yet the final prior rationale for the technical report.

## Purpose

This document reviews the priors used across the `vocab_growth` model family and
records why they are currently considered plausible, useful, or in need of
further sensitivity checking.

The goal is not to make every prior broad. The goal is to make each prior:

- interpretable on the observable vocabulary scale;
- explicit about whether it is developmental, computational, or data-informed;
- checked through prior predictive simulation;
- tested for sensitivity where the data are sparse or the parameter is weakly
  identified.

The fuller publication-ready discussion will live in the technical report. This
file is the working review ledger.

## Where the priors live

The model-specific prior choices are defined in
[`src/vocab_growth/models/definitions.py`](../../src/vocab_growth/models/definitions.py).
The common engines turn those definitions into PyMC variables:

- univariate models: [`common.py`](../../src/vocab_growth/models/common.py);
- univariate study-random-effect models:
  [`common_univariate_re.py`](../../src/vocab_growth/models/common_univariate_re.py);
- bivariate models: [`common_bivariate.py`](../../src/vocab_growth/models/common_bivariate.py);
- bivariate random-effect models:
  [`common_bivariate_re.py`](../../src/vocab_growth/models/common_bivariate_re.py);
- trivariate signing model:
  [`common_trivariate.py`](../../src/vocab_growth/models/common_trivariate.py);
- joint sign/speech model:
  [`common_joint_modality.py`](../../src/vocab_growth/models/common_joint_modality.py).

The current model list and lineage are maintained in
[`docs/models/README.md`](README.md). That inventory is the source of truth for
which models this review must cover.

## Model coverage

| Model | Population | Outcomes                     | Prior features to review                                                                                    |
| ----- | ---------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| VG01  | DS         | spoken                       | Single-outcome spoken anchors, GP, kappa.                                                                   |
| VG02  | DS         | understood                   | Single-outcome understood anchors, GP, kappa.                                                               |
| VG03  | TD         | spoken                       | TD spoken anchors, GP, kappa, subsampling.                                                                  |
| VG04  | TD         | understood                   | TD understood anchors, GP, kappa, subsampling.                                                              |
| VG05  | DS         | understood + spoken          | Understood anchors, production-ratio `q` anchors, GP, kappa.                                                |
| VG06  | TD         | understood + spoken          | TD understood anchors, `q` anchors, GP, kappa, subsampling.                                                 |
| VG07  | DS         | understood + spoken          | VG05 plus study random-effect scales.                                                                       |
| VG08  | DS         | understood + spoken          | VG07 plus subject random effects on understood.                                                             |
| VG09  | DS         | understood + spoken          | VG08 plus subject random effects on `q`; diagnostic ridge motivates VG10.                                   |
| VG10  | DS         | understood + spoken          | VG09 plus posterior-informed `q` anchors and GP anchoring.                                                  |
| VG11  | TD         | spoken                       | VG03 plus study random effects, full TD data, GP anchoring.                                                 |
| VG12  | TD         | understood                   | VG04 plus study random effects, full TD data, GP anchoring.                                                 |
| VG13  | TD         | understood + spoken          | Young TD bivariate model, study random effects, GP anchoring.                                               |
| VG14  | DS         | understood + spoken + signed | Adds signed ratio `r`, sign GP, sign kappa, signing-data decisions.                                         |
| VG15  | DS         | understood + spoken + signed | VG14 plus `psi`, Dirichlet-Multinomial concentration, study and subject random effects, VG10 stabilisation. |

## Prior families

### TD and DS prior differences

The TD and DS models do not use fundamentally different prior systems. Most of
the model machinery is shared:

- GP length-scale and amplitude priors are the same for TD and DS, except for the
  DS-only signing models.
- Beta-Binomial `kappa` priors are the same.
- Study and subject random-effect scale priors are the same where those effects
  exist.
- Baseline `q(a) = P(speak | understood)` priors are the same for TD and DS
  bivariate models, except for the posterior-informed VG10/VG15 `q` anchors.

The main TD/DS differences are concentrated in the anchor ages and in a few
anchor distributions:

| Prior area            | DS                                                                                      | TD                                                                                | Interpretation                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Anchor ages           | Usually 24 and 84 months.                                                               | Usually 12 and 26 months; VG13 uses 10 and 16 months.                             | Priors are placed over different developmental windows.                                                            |
| Spoken low anchor     | `Beta(1, 15)` at 24 months in VG01.                                                     | `Beta(1, 15)` at 12 months in VG03/VG11.                                          | Same distributional shape, different age.                                                                          |
| Understood low anchor | `Beta(1, 10)` at 24 months in DS understood and joint models.                           | `Beta(1, 20)` at 12 months in VG04/VG06/VG12; `Beta(1, 15)` at 10 months in VG13. | TD early-understanding priors are more tightly concentrated near the floor.                                        |
| High anchor           | Usually `Beta(1.1, 1.1)` at 84 months.                                                  | Usually `Beta(1.5, 1.1)` at 26 months; VG13 uses `Beta(2, 2)` at 16 months.       | TD high-age priors lean more toward larger vocabulary by the older anchor; DS high-age priors remain much broader. |
| Baseline `q` anchors  | `Beta(1, 1.5)` low and `Beta(2, 1.2)` high in VG05-VG09 and VG14; tighter in VG10/VG15. | Same broad defaults in VG06/VG13.                                                 | Mostly shared, except for posterior-informed DS stabilisation models.                                              |
| Signing priors        | DS-only in VG14/VG15.                                                                   | Not modelled.                                                                     | There is no TD signing counterpart.                                                                                |

Review notes:

- DS/TD comparisons are not prior-symmetric at the anchor level. The asymmetry is
  mainly developmental: the priors are anchored at different ages because the
  observed developmental windows differ.
- The strongest substantive asymmetry is that TD high-age anchors are mildly
  optimistic by 26 months, while DS high-age anchors remain very broad at 84
  months.
- This should be stated plainly in the technical report so readers do not
  mistake the shared machinery for fully identical prior assumptions.

### Anchor priors

For univariate trajectories, the linear trend is anchored by expected vocabulary
proportions at two ages. For joint models, the understood trajectory `p_U(a)` is
anchored in the same way, and the spoken production ratio
`q(a) = P(speak | understood)` has its own pair of anchors.

The anchors are probabilities. For direct vocabulary trajectories, multiplying
by 800 gives the expected number of words out of the common reference inventory.
For `q`, the anchor is a fraction of understood words, so it should not be read
as a direct word count without also considering `p_U(a)`.

| Prior use                              | Models                                  | Distribution     | Observable interpretation                                              |
| -------------------------------------- | --------------------------------------- | ---------------- | ---------------------------------------------------------------------- |
| Low-age spoken anchor                  | VG01, VG03, VG11                        | `Beta(1, 15)`    | Median 0.045, 5-95% 0.003-0.181, or about 36 words median out of 800.  |
| Low-age DS understood anchor           | VG02, VG05, VG07-VG10, VG14, VG15       | `Beta(1, 10)`    | Median 0.067, 5-95% 0.005-0.259, or about 54 words median out of 800.  |
| Low-age TD understood anchor           | VG04, VG06, VG12                        | `Beta(1, 20)`    | Median 0.034, 5-95% 0.003-0.139, or about 27 words median out of 800.  |
| Low-age young-TD understood anchor     | VG13                                    | `Beta(1, 15)`    | Median 0.045, 5-95% 0.003-0.181, or about 36 words median out of 800.  |
| High-age DS single/U anchor            | VG01, VG02, VG05, VG07-VG10, VG14, VG15 | `Beta(1.1, 1.1)` | Median 0.500, 5-95% 0.060-0.940, or about 400 words median out of 800. |
| High-age TD single/U anchor            | VG03, VG04, VG06, VG11, VG12            | `Beta(1.5, 1.1)` | Median 0.599, 5-95% 0.126-0.955, or about 479 words median out of 800. |
| High-age young-TD understood anchor    | VG13                                    | `Beta(2, 2)`     | Median 0.500, 5-95% 0.135-0.865, or about 400 words median out of 800. |
| Baseline low-age `q` anchor            | VG05-VG09, VG13, VG14                   | `Beta(1, 1.5)`   | Median 0.370, 5-95% 0.034-0.864 of understood words.                   |
| Baseline high-age `q` anchor           | VG05-VG09, VG13, VG14                   | `Beta(2, 1.2)`   | Median 0.654, 5-95% 0.197-0.956 of understood words.                   |
| Posterior-informed low-age `q` anchor  | VG10, VG15                              | `Beta(3, 22)`    | Median 0.110, 5-95% 0.035-0.240 of understood words.                   |
| Posterior-informed high-age `q` anchor | VG10, VG15                              | `Beta(20, 4)`    | Median 0.843, 5-95% 0.696-0.938 of understood words.                   |

Review notes:

- The low-age direct trajectory anchors encode strong floor expectations, which
  are scientifically plausible but should be checked against prior predictive
  counts at the youngest queried ages.
- The high-age DS `Beta(1.1, 1.1)` anchor is deliberately broad. It prevents the
  prior from declaring either low or high later vocabulary impossible, but it can
  interact with the GP and random effects in sparse age regions.
- The baseline `q` anchor priors are very broad. They are weak regularisation,
  not strong developmental knowledge.
- VG10 and VG15 use tighter `q` anchors informed by the VG07 posterior. These
  should be labelled as posterior-informed regularisation from overlapping data,
  not independent prior evidence. The rationale is documented in
  [`notes/202605131500-vg09-structural-options.md`](../../notes/202605131500-vg09-structural-options.md).

### Signed ratio prior

VG14 and VG15 model signing as
`r(a) = P(sign | understood)`. The current signed mean is intercept-only:

```text
intercept_sign ~ Normal(logit(0.15), 0.75)
```

On the probability scale this gives a median signed ratio of about 0.15, with a
5-95% interval of about 0.05-0.38 before adding the signed GP. If all 800 words
were understood, this would correspond to roughly 120 signed words at the
intercept level, with a 5-95% range of about 39-302 words.

Review notes:

- Earlier signed-anchor specifications were too restrictive or produced
  implausible extrapolation below the signing data floor. The current
  intercept-only specification is a structural response to that failure.
- The sign GP, not a monotone signed slope, carries the rise-then-fall pattern.
- The signed prior is partly informed by earlier VG14/VG15 model criticism and
  refits. It should be labelled as a correction from the prior-review workflow,
  not as independent external evidence.
- The rationale and sensitivity history are documented in
  [`notes/202606151700-vg14-signed-ratio-shape-and-p-any-bias.md`](../../notes/202606151700-vg14-signed-ratio-shape-and-p-any-bias.md).

### GP length-scale and amplitude priors

The HSGP priors use a unit length-scale parameter mapped onto a length-scale in
months:

```text
ell_unit ~ Beta(alpha, beta)
ell_months = ell_low + (ell_high - ell_low) * ell_unit
```

The common range is 6-18 months.

| Use                                 | Distribution                 | Observable interpretation                                          |
| ----------------------------------- | ---------------------------- | ------------------------------------------------------------------ |
| Standard U, spoken, and `q` smooths | `ell_unit ~ Beta(3, 3)`      | Median length-scale about 12 months; 5-95% about 8.3-15.7 months.  |
| Signed-ratio smooth                 | `ell_unit_sign ~ Beta(2, 5)` | Median length-scale about 9.2 months; 5-95% about 6.8-13.0 months. |
| Standard GP amplitude               | `eta ~ HalfNormal(0.4)`      | Median logit-scale deviation about 0.27; 95% about 0.78.           |
| Signed GP amplitude                 | `eta_sign ~ HalfNormal(1.0)` | Median logit-scale deviation about 0.67; 95% about 1.96.           |

Review notes:

- The standard length-scale prior encodes smooth developmental departures rather
  than rapid month-to-month oscillation.
- The signed length-scale is shorter and its amplitude is larger because signing
  needs to express a hump that can rise and fall over the observed age window.
- The signed GP prior is a key sensitivity target because signed data are sparse
  and age coverage is uneven.
- The HSGP basis settings should be reviewed together with the length-scale
  prior. A length-scale prior can look defensible while the basis approximation
  still constrains the realised functions.

### Age-varying dispersion priors

The Beta-Binomial concentration is age-varying:

```text
kappa(z) = kappa_min + exp(a_kappa + b_kappa * z)
b_kappa = -b_kappa_mag
```

where `z` is standardised age. The shared default is:

```text
kappa_min ~ LogNormal(log(5), 0.6)
a_kappa ~ Normal(log(8), 1.0)
b_kappa_mag ~ HalfNormal(0.3)
```

First-pass prior simulation gives:

| Standardised age | Median kappa | 5-95% kappa | Median rho = `1 / (kappa + 1)` |
| ---------------- | -----------: | ----------: | -----------------------------: |
| `z = -1`         |         16.6 |    6.0-60.6 |                          0.057 |
| `z = 0`          |         14.4 |    5.4-48.1 |                          0.065 |
| `z = +1`         |         12.7 |    4.9-40.4 |                          0.073 |

Review notes:

- Smaller `kappa` means more overdispersion relative to a Binomial at the same
  mean. The prior allows substantial extra-binomial heterogeneity.
- The sign of `b_kappa` encodes increasing heterogeneity with age, with a later
  plateau at `kappa_min`.
- This structure should be checked carefully near floor and ceiling regions,
  where `kappa` can be weakly identified even if predictions look reasonable.
- Alternative `kappa` priors are a sensitivity target for main reporting models.

### Study and subject random-effect scale priors

Study and subject random intercepts use non-centred Normal effects with
HalfNormal scale priors. The common scale prior is:

```text
tau ~ HalfNormal(0.5)
```

On the logit scale, a one-standard-deviation shift has prior median about 0.34.
As an odds multiplier, `exp(tau)` has prior median about 1.40 and a 95th
percentile about 2.67.

Review notes:

- This prior is regularising but not tiny. It allows meaningful study and child
  differences.
- Later DS models often estimate subject-level scales above this prior's centre,
  especially for signing. That suggests real heterogeneity rather than purely
  prior-driven variation, but the posterior-vs-prior comparison should be
  documented in the technical report.
- Subject random effects for sparse modalities remain a sensitivity target,
  especially `tau_subj_sign` in VG15.

### VG15 association and four-cell concentration priors

VG15 introduces a scalar Plackett association between signing and speaking within
understood words:

```text
log_psi ~ Normal(0.3, 0.5)
psi = exp(log_psi)
```

This prior has median `psi` about 1.35, 5-95% about 0.59-3.07, and about 72.5%
prior probability above independence (`psi = 1`).

VG15 also uses:

```text
log_conc ~ Normal(3.0, 1.0)
conc = exp(log_conc)
```

This gives median concentration about 20, with a 5-95% interval about 3.9-104.

Review notes:

- The `psi` prior is weakly positive, not neutral. That is consistent with the
  uk_02 four-cell data motivating VG15, but it should be explicitly labelled as
  data-informed regularisation rather than independent prior evidence.
- Because `psi` is identified primarily from a small uk_02 cross-tabulation, a
  neutral prior such as `log_psi ~ Normal(0, 0.5)` or a broader alternative
  should be included in sensitivity checks.
- The current VG15 engine deliberately feeds the four-cell likelihood
  population-plus-study marginals, not subject-shifted marginals, so `psi`
  remains a population-conditioned association. The rationale is documented in
  [`notes/202606171200-vg15-subject-re-stabilisation.md`](../../notes/202606171200-vg15-subject-re-stabilisation.md).

## Prior predictive review status

Generated model reports already include prior predictive plots for many models,
but this review has not yet completed a consistent cross-model prior predictive
audit.

The audit should record, for each model family:

- whether young-age floor behaviour is plausible;
- whether old-age ceiling behaviour is plausible;
- whether trajectories are smooth without being too rigid;
- whether `q(a)` and `r(a)` remain plausible over the full query range;
- whether random effects imply realistic between-study and between-child
  heterogeneity on the probability scale;
- whether simulated counts have plausible spread before seeing the data;
- whether VG15 four-cell simulations imply plausible sign-only, speak-only,
  both, and neither compositions.

Current local fitted output is available for VG01-VG09 and VG11. The review
should either regenerate or retrieve corresponding output for VG10, VG12, VG13,
VG14, and VG15 before making final claims.

## Sensitivity targets

The following sensitivity checks should be prioritised before the technical
report makes robustness claims:

| Target                                   | Why it matters                                                                    | Suggested alternatives                                                                                                             |
| ---------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| VG10/VG15 posterior-informed `q` anchors | These are intentionally tighter and informed by VG07.                             | Baseline broad `q` anchors; slightly wider posterior-informed anchors.                                                             |
| Signed GP amplitude and length-scale     | Signing data are sparse and the hump is GP-driven.                                | Wider/narrower `eta_sign`; standard `ell_unit_sign ~ Beta(3, 3)`; shorter length-scale alternative.                                |
| Signed intercept prior                   | The signed level was previously prior-dominated under another parameterisation.   | Wider `Normal(logit(0.15), 1.0)`; shifted medians such as 0.10 and 0.20.                                                           |
| Kappa priors                             | Dispersion can dominate predictive uncertainty, especially near floor or ceiling. | Broader `kappa_min`; flatter age trend; non-monotone or constant-kappa comparison where feasible.                                  |
| Random-effect scales                     | Study and subject effects can trade off with global age curves.                   | Wider `tau` prior; narrower `tau` prior; study-only or no-subject variants where already supported by flags.                       |
| VG15 `psi`                               | Identified from sparse four-cell data and prior is weakly positive.               | Neutral `log_psi ~ Normal(0, 0.5)`; broader `Normal(0, 1)`; stronger positive prior only as an explicit data-informed sensitivity. |
| VG15 concentration                       | Controls four-cell overdispersion.                                                | Broader `log_conc`; lower/higher median concentration.                                                                             |

Sensitivity summaries should compare headline quantities, not only raw
parameters:

- expected words understood, spoken, signed, and total expressive at query ages;
- `q(a)` and `r(a)` at clinically relevant ages;
- VG15 `psi` and `P(psi > 1)`;
- four-cell sign/speech composition;
- uncertainty intervals for the above.

## Provisional conclusions

The current prior set is coherent with the model architecture, but several priors
are not neutral defaults and need explicit labelling.

- The anchor priors encode developmental floor expectations at young ages and
  broad uncertainty at older ages.
- The baseline `q` anchors are deliberately broad.
- VG10 and VG15 tighten `q` using earlier posterior information to stabilise a
  weakly identified trajectory decomposition.
- The signed-ratio prior is the result of an explicit prior-predictive failure
  and correction: the current intercept-only mean avoids a misleading monotone
  signed slope, while the GP carries the signing hump.
- The shared kappa prior encodes substantial extra-binomial heterogeneity and a
  monotone increase in heterogeneity with age.
- Random-effect scale priors allow meaningful study and subject differences and
  should be interpreted on the logit and probability scales.
- VG15 `psi` is weakly positively regularised and must be tested against neutral
  alternatives.

No final robustness conclusion should be made until the prior predictive audit
and sensitivity checks above are complete.
