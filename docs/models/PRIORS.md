# Prior rationale and review notes

<!-- cspell:words conc -->

> [!NOTE]
> Drafted by LLM-based AI tools (OpenAI Codex/GPT-5; "Evidence base" section and
> prior–norm comparison by Claude Code/Opus 4.8).

> [!WARNING]
> This is a working document for issue 89, last reviewed on 2026-07-01. It
> records the current prior inventory, first-pass interpretation, review
> questions, and a first pass at the external evidence base. It is not yet the
> final prior rationale for the technical report.

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

| Model            | Population | Outcomes                     | Prior features to review                                                                                           |
| ---------------- | ---------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| VG01             | DS         | spoken                       | Single-outcome spoken anchors, GP, kappa.                                                                          |
| VG02             | DS         | understood                   | Single-outcome understood anchors, GP, kappa.                                                                      |
| VG03             | TD         | spoken                       | TD spoken anchors, GP, kappa, subsampling.                                                                         |
| VG04             | TD         | understood                   | TD understood anchors, GP, kappa, subsampling.                                                                     |
| VG05             | DS         | understood + spoken          | Understood anchors, production-ratio `q` anchors, GP, kappa.                                                       |
| VG06 _(retired)_ | TD         | understood + spoken          | TD understood anchors, `q` anchors, GP, kappa, subsampling; retained here only as historical prior context.        |
| VG07             | DS         | understood + spoken          | VG05 plus study random-effect scales.                                                                              |
| VG08             | DS         | understood + spoken          | VG07 plus subject random effects on understood.                                                                    |
| VG09             | DS         | understood + spoken          | VG08 plus subject random effects on `q`; diagnostic ridge motivates VG10.                                          |
| VG10             | DS         | understood + spoken          | VG09 plus posterior-informed `q` anchors and GP anchoring.                                                         |
| VG11             | TD         | spoken                       | VG03 plus study random effects, full TD data, GP anchoring.                                                        |
| VG12             | TD         | understood                   | VG04 plus study random effects, full TD data, GP anchoring.                                                        |
| VG13             | TD         | understood + spoken          | Young TD bivariate model, study random effects, GP anchoring.                                                      |
| VG14             | DS         | understood + spoken + signed | Adds signed ratio `r`, sign GP, sign kappa, signing-data decisions.                                                |
| VG15             | DS         | understood + spoken + signed | VG14 plus `psi`, Dirichlet-Multinomial concentration, study and subject random effects, VG10 stabilisation.        |
| VG16             | DS         | understood + spoken          | VG09 plus prior-understood cross-lag coefficient `beta_lag`; uses the same main prior families plus the lag prior. |

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

| Prior area            | DS                                                                                      | TD                                                                            | Interpretation                                                                                                     |
| --------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Anchor ages           | Usually 24 and 84 months.                                                               | Usually 12 and 26 months; VG13 uses 10 and 16 months.                         | Priors are placed over different developmental windows.                                                            |
| Spoken low anchor     | `Beta(1, 25)` at 24 months in VG01.                                                     | `Beta(1, 30)` at 12 months in VG03/VG11.                                      | Both concentrate near the floor after the young-age prior-predictive recalibration.                                |
| Understood low anchor | `Beta(1, 7)` at 24 months in VG02; `Beta(1, 10)` in the DS joint models.                | `Beta(1.2, 8)` at 12 months in VG04/VG12; `Beta(1, 15)` at 10 months in VG13. | The TD understood low anchor was recalibrated up off the floor (TD comprehension is already substantial young).    |
| High anchor           | Usually `Beta(1.1, 1.1)` at 84 months.                                                  | Usually `Beta(1.3, 1.3)` at 26 months; VG13 uses `Beta(2, 6)` at 16 months.   | The TD high anchors were softened toward the middle from a more optimistic prior; DS high-age priors remain broad. |
| Baseline `q` anchors  | `Beta(1, 1.5)` low and `Beta(2, 1.2)` high in VG05-VG09 and VG14; tighter in VG10/VG15. | `Beta(1, 10)` low and `Beta(2, 7)` high in VG13.                              | VG13's young-TD production ratio is far below the DS window, so it no longer inherits the DS defaults.             |
| Signing priors        | DS-only in VG14/VG15.                                                                   | Not modelled.                                                                 | There is no TD signing counterpart.                                                                                |

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
by 810 gives the expected number of words out of the common reference inventory.
For `q`, the anchor is a fraction of understood words, so it should not be read
as a direct word count without also considering `p_U(a)`.

| Prior use                              | Models                      | Distribution     | Observable interpretation                                              |
| -------------------------------------- | --------------------------- | ---------------- | ---------------------------------------------------------------------- |
| Low-age DS spoken anchor               | VG01                        | `Beta(1, 25)`    | Median 0.027, 5-95% 0.002-0.113, or about 22 words median out of 810.  |
| Low-age TD spoken anchor               | VG03, VG11                  | `Beta(1, 30)`    | Median 0.023, 5-95% 0.002-0.095, or about 19 words median out of 810.  |
| Low-age DS understood anchor (single)  | VG02                        | `Beta(1, 7)`     | Median 0.094, 5-95% 0.007-0.348, or about 76 words median out of 810.  |
| Low-age DS understood anchor (joint)   | VG05, VG07-VG10, VG14, VG15 | `Beta(1, 10)`    | Median 0.067, 5-95% 0.005-0.259, or about 54 words median out of 810.  |
| Low-age TD understood anchor           | VG04, VG12                  | `Beta(1.2, 8)`   | Median 0.104, 5-95% 0.011-0.341, or about 84 words median out of 810.  |
| Low-age young-TD understood anchor     | VG13                        | `Beta(1, 15)`    | Median 0.045, 5-95% 0.003-0.181, or about 36 words median out of 810.  |
| High-age DS single anchor (VG01/VG02)  | VG01, VG02                  | `Beta(2, 1.5)`   | Median 0.586, 5-95% 0.168-0.924, or about 475 words median out of 810. |
| High-age DS understood anchor (joint)  | VG05, VG07-VG10, VG14, VG15 | `Beta(1.1, 1.1)` | Median 0.500, 5-95% 0.060-0.940, or about 405 words median out of 810. |
| High-age TD single/U anchor            | VG03, VG04, VG11, VG12      | `Beta(1.3, 1.3)` | Median 0.500, 5-95% 0.079-0.921, or about 405 words median out of 810. |
| High-age young-TD understood anchor    | VG13                        | `Beta(2, 6)`     | Median 0.228, 5-95% 0.053-0.521, or about 185 words median out of 810. |
| Baseline low-age `q` anchor            | VG05-VG09, VG14             | `Beta(1, 1.5)`   | Median 0.370, 5-95% 0.034-0.864 of understood words.                   |
| Young-TD low-age `q` anchor            | VG13                        | `Beta(1, 10)`    | Median 0.067, 5-95% 0.005-0.259 of understood words.                   |
| Baseline high-age `q` anchor           | VG05-VG09, VG14             | `Beta(2, 1.2)`   | Median 0.654, 5-95% 0.197-0.956 of understood words.                   |
| Young-TD high-age `q` anchor           | VG13                        | `Beta(2, 7)`     | Median 0.201, 5-95% 0.046-0.471 of understood words.                   |
| Posterior-informed low-age `q` anchor  | VG10, VG15                  | `Beta(3, 22)`    | Median 0.110, 5-95% 0.035-0.240 of understood words.                   |
| Posterior-informed high-age `q` anchor | VG10, VG15                  | `Beta(20, 4)`    | Median 0.843, 5-95% 0.696-0.938 of understood words.                   |

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
5-95% interval of about 0.05-0.38 before adding the signed GP. If all 810 words
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

## Evidence base: literature and normative data

This section records the external evidence that can anchor or challenge the
priors above, and — critically — separates _independent_ evidence from
_regularisation_ drawn from data that overlap the training set (issue 89,
step 3).

### Independence of candidate sources

Not every published cohort is independent of the fitted data. Where a prior is
anchored on a study whose participants are already in `vocab_data_merged.csv`,
it is regularisation, not independent prior evidence.

| Source                                                                                                                      | Role for priors                                 | Independent of training data?                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Wordbank by-child data (`wordbank_administration_data.csv`)                                                                 | TD anchors, `q`, dispersion                     | **No** — it _is_ the TD training data. Use the published normative percentiles as the non-circular check, not the rows. |
| Berglund et al. (2001), n=330, Sweden                                                                                       | DS anchors, growth shape, heterogeneity         | Yes                                                                                                                     |
| Næss et al. (2021), Norway; Galeote et al. (2008), Spain; Deckers et al. (2016), Kaat-van den Os et al. (2017), Netherlands | DS anchors, `q`, signed `r`                     | Yes                                                                                                                     |
| Miller et al. (1995); Mervis & Robinson (2000), US                                                                          | DS anchors, parent-report validity              | Yes                                                                                                                     |
| Oliver & Buckley (1994), UK                                                                                                 | DS low-age spoken anchor (10-word stage ~27 mo) | Yes — confirmed **not** to overlap `uk_01`                                                                              |
| Caselli et al. (1998); Zampini & D'Odorico (2013); Bello & Caselli (2014), Italy                                            | DS trajectory, gesture, dispersion              | **No** — overlap the `it_01` Italian-CDI-DS cohort; treat as regularisation                                             |

### Instrument scale (Fenson et al., 2007, via Hutchins, 2013)

The MB-CDI forms have different item totals, which the models fold onto the
common 810-item reference scale:

- **Words & Gestures (WG)** — 396-item checklist, _separate_ comprehension and
  production columns; infant form (~8–18 months).
- **Words & Sentences (WS)** — 680-item checklist, _production only_; toddler
  form (~16–30 months). This is why the TD loader keeps WS as a spoken-only
  observation and excludes WS "comprehension".

Two consequences for the priors:

1. Because WG carries the only CDI comprehension data, an _understood_
   proportion derived from WG cannot exceed 396/810 = **0.489** on the model
   scale. A high-age understood anchor near 0.5 would sit against the WG ceiling and implicitly assume near-total WG comprehension; VG13's 16-month understood anchor was recalibrated to `Beta(2, 6)` (median 0.228, about 185 words) partly for this reason, keeping it clear of the ceiling.
2. Fenson et al. caution that percentile ranks are unstable at ages where a
   skill is just emerging ("small differences in raw scores can have
   dramatically different effects on percentile ranks"). The youngest-age
   anchors should be re-centred toward the norms but **not** tightened.

### TD anchor priors vs Wordbank norms

Wordbank US-English, typically-developing, monolingual, cross-sectional deciles
(downloaded 2026-07-01) translated onto the model's 810-item scale. The final
column flags where the prior _centre_ departs from the normative median at the
anchor age:

| Anchor (models)                       | Prior            | Prior median (words/810) | Wordbank median | Prior ÷ empirical                   |
| ------------------------------------- | ---------------- | -----------------------: | --------------: | ----------------------------------- |
| Spoken low @12mo (VG03/VG11)          | `Beta(1, 30)`    |               0.023 (19) |      0.013 (11) | 1.8× high                           |
| Understood low @12mo (VG04/VG12)      | `Beta(1.2, 8)`   |               0.104 (84) |      0.104 (84) | 1.0× (matches)                      |
| Spoken high @26mo (VG03/VG11)         | `Beta(1.3, 1.3)` |              0.500 (405) |     0.436 (353) | 1.1× (broad, covers)                |
| Young-TD understood low @10mo (VG13)  | `Beta(1, 15)`    |               0.045 (36) |      0.062 (50) | 0.73×                               |
| Young-TD understood high @16mo (VG13) | `Beta(2, 6)`     |              0.228 (185) |     0.222 (180) | 1.0× (matches)                      |
| Understood high @26mo (VG04/VG12)     | `Beta(1.3, 1.3)` |              0.500 (405) |               — | no CDI norm (WS is production-only) |

Every prior's 5–95% band still covers the empirical median, so none is
inconsistent with the norms — but the low-age and VG13 high anchors now sit close to the normative medians after the young-age recalibration (#135/#138/#140); the 26-month understood anchor still cannot be anchored to CDI norms (WS is production-only) and remains a sensitivity target. Source data: Wordbank vocabulary norm tables,
<https://wordbank.stanford.edu/data/?name=vocab_norms>.

### Production ratio `q(a)` from norms

Because WG reports comprehension and production at the same ages, an empirical
TD `q(a) = P(speak | understood)` can be read off as the ratio of median
production to median comprehension (indicative — a ratio of population medians,
not a within-child median):

| Age (months) | Empirical TD `q(a)` |
| ------------ | ------------------: |
| 10           |                0.12 |
| 12           |                0.13 |
| 16           |                0.19 |
| 18           |                0.26 |

The baseline `q` anchors (`Beta(1, 1.5)` / `Beta(2, 1.2)`, medians 0.37 / 0.65)
sit roughly 3× above this at young ages. VG13 no longer inherits these — it now uses `Beta(1, 10)` / `Beta(2, 7)` (medians 0.067 / 0.201), matching the empirical TD ratio above. The **posterior-informed** VG10/VG15
low-age `q` anchor, `Beta(3, 22)` (median 0.110), matches the independent TD
`q(10–12 mo) ≈ 0.12` almost exactly. This upgrades the VG10/VG15 tightening from
purely internal regularisation (from the VG07 posterior) to a choice
_corroborated by independent TD norms_.

### DS anchor priors vs independent cohorts

The DS anchors (24 and 84 months) can be checked against the independent DS CDI
cohorts — those not overlapping the training data. Only expressive (spoken)
vocabulary can be anchored this way: the usable cohorts report production, and DS
comprehension at chronological age has no independent source here (Berglund's
form is production-only, Galeote et al. (2008) is mental-age-based, and the
Italian comprehension cohorts overlap `it_01`).

Berglund et al. (2001) — 330 DS children on a 710-item Swedish CDI — give a full
spoken trajectory by chronological age (their Table 3), translated onto the
model's 810-item scale:

| Age (months) | Berglund DS spoken (approx. median words / 810) | Notes                            |
| ------------ | ----------------------------------------------: | -------------------------------- |
| 12           |                                      ~0 (0.000) | 12% have ≥1 word                 |
| 24           |                                     ~10 (0.013) | 53% pass 10 words, 3% pass 50    |
| 36           |                                     ~30 (0.045) | mean 36 words (range 0–165)      |
| 48           |                                     ~50 (0.063) | 54% pass 50 words; max child 668 |
| 60           |                                     ~65 (0.081) | 73% pass 50 words                |

Comparison with the DS spoken prior (VG01, `Beta(1, 25)` at 24 months, median 0.027, about 22 words):

- The prior now places about 22 words at **24 months**, which Berglund observes around 30 months; at 24 months the independent median is ~10 words, so the DS spoken-low prior is ~2x high — recalibrated much closer to the cohort, in the same direction as the TD spoken-low anchor.
- The **84-month high anchor** (`Beta(2, 1.5)`, median 0.586, about 475 words) is
  **beyond the range of every independent DS CDI cohort** (Berglund tops out at
  60 months; CDIs are young-child instruments). It is deliberately broad and can
  only be checked against the project's own older DS data — i.e. it is
  regularisation, not independently anchored. This mirrors the un-anchored TD
  understood high anchor at 26 months.
- The DS **understood** low anchor (`Beta(1, 10)` at 24 months, median 0.067 ≈ 54
  words) has **no independent chronological-age comprehension source** in the
  current library. It is directionally sensible (understood > spoken at 24
  months) but its level rests on the project's own DS comprehension data — a gap
  worth filling.

Milestone timing corroborates the shape: the 50-word level is reached by ~25% of
DS children at age 3, ~50% at age 4, and ~75% at age 5 (Berglund et al., 2001;
consistent with Næss et al., 2021). Galeote et al. (2008) add that, matched on
mental age, DS spoken vocabulary is comparable to TD while gesture use is
superior — evidence for the signed ratio `r(a)` rather than a chronological-age
anchor.

### Dispersion (`kappa`)

Fitting a Beta-Binomial (n = 810, matching the model likelihood) per age to the
by-child Wordbank TD data (English variants, `typically_developing`,
`health_conditions` null — the loader's filter) gives the empirical dispersion.
`kappa` is the concentration; `rho = 1 / (kappa + 1)` is the intra-child
overdispersion.

| Outcome / form  | Age span | Empirical `kappa` | Empirical `rho` | Age trend                      |
| --------------- | -------- | ----------------: | --------------: | ------------------------------ |
| Understood (WG) | 8–18 mo  |              6–14 |       0.07–0.13 | ~flat                          |
| Spoken (WG)     | 8–18 mo  |             10–36 |       0.03–0.09 | `kappa` falls with age         |
| Spoken (WS)     | 16–30 mo |              3–14 |       0.07–0.26 | `kappa` falls steeply with age |

Against the shared prior (`kappa` median ~13–17, 5–95% ~5–60; `b_kappa < 0`):

- **Direction confirmed.** For the spoken/production outcome (the primary one)
  `kappa` clearly falls with age (WG spoken slope −0.13/month; WS spoken
  −0.09/month) — dispersion rises with age, exactly the sign the prior encodes.
  Comprehension is roughly flat. Independently, Zampini & D'Odorico (2013) report
  DS vocabulary variability _increasing_ from 36 months, the same direction.
- **Level slightly too tight at the high-dispersion end.** The prior's central
  `kappa ≈ 14` is a reasonable mid-range value, but the empirical range is wider.
  At older toddler ages (WS 24–30 months) `kappa` falls to ~3–4 (`rho ≈
0.21–0.26`), below the prior's ~5 lower 5–95% bound. Part of this is a ceiling
  artefact (WS counts pile toward the 680-item form limit), and the model's GP
  mean and study random effects absorb some spread that these raw per-age fits do
  not — so the fitted-model `kappa` would sit somewhat higher. Even so, a broader
  `kappa_min` allowance is worth a sensitivity check at older ages.

### Methodological endorsement of the 810-item design

Laudańska et al. (2026), systematically reviewing CDI use across
neurodevelopmental and genetic conditions, recommend exactly the harmonisation
this project adopts: proportion-based scoring on a common overlapping item set
to compare across CDI forms and languages. Their pooled DS expressive-vocabulary
age trend and cohort catalogue provide a meta-analytic DS anchor, with the
caveat that clusters mix forms and languages. A useful cross-anchor: DS
expressive vocabulary at ages 3–4 is comparable to TD at 16–20 months (Berglund
et al., 2001), which via the Wordbank WS norms pins the DS spoken trajectory
through the preschool years.

## Prior predictive audit

Prior-predictive output was regenerated for one representative of each model
family — VG11 and VG12 (typically-developing univariate with study random
effects), VG10 (Down syndrome bivariate understood + spoken, study + subject
random effects and a GP anchor), VG13 (typically-developing bivariate, young
8-18 month window), VG14 (trivariate signing) and VG15 (joint sign/speech) —
using `scripts/prior_predictive_audit.py`, which builds each model and draws
from the prior predictive only (no posterior sampling). The `prior_samples_*`,
`prior_predictions` and `prior_predictive_checks` plots in each model's output
directory were reviewed against the checklist below.

| Check                           | Finding                                                                                                                                                                                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Young-age floor                 | Plausible. Every trajectory family places prior-predictive mass near zero at the youngest ages (spoken and signed at 8-12 months, understood a little higher); no prior draw forces a high count at young ages.                                                                 |
| Old-age ceiling                 | Plausible. Understood and spoken curves approach the 810-word ceiling only gradually and only for the fastest draws; the bulk of the prior mass stays well below saturation across the query range, so the ceiling is reachable but not imposed.                                |
| Smoothness                      | Appropriate. The HSGP produces smooth curves with individual-draw wiggle, admitting both near-linear and gently curved trajectories without high-frequency oscillation.                                                                                                         |
| `q(a)` (speak given understood) | Plausible. The prior band is a smooth 0-to-1 sigmoid rising from about 0.05 at the youngest ages toward about 0.9 by ~100 months, with no mass piling implausibly at the bounds.                                                                                                |
| `r(a)` (sign given understood)  | Deliberately broad. The intercept-only mean plus GP spans roughly 0-1 with most mass low-to-mid and a visible narrowing ("waist") at the 54-month GP anchor — the intended weakly-informative signed prior (data set the level, the GP carries the hump); no piling at 0 or 1.  |
| Random-effect heterogeneity     | Plausible. At the observation level the study/subject random effects widen the prior-predictive cloud enough to cover the observed between-study and between-child spread without implying implausible extremes on the probability scale.                                       |
| Simulated count spread          | Plausible. The prior-predictive count clouds bracket the observed counts for every outcome (understood, spoken, signed) before the data are seen — neither too narrow (which would fight the data) nor degenerate at 0 or 810.                                                  |
| VG15 signing / four-cell        | Plausible. Signed counts stay low with a broad, hump-capable upper tail (matching the sparse signing data); the `log_psi ~ Normal(0.3, 0.5)` association prior spans the independence reference `psi = 1`, so the four-cell composition is not prior-forced toward association. |

**Conclusion.** The priors pass the prior-predictive audit: they encode the
developmental floor and a reachable-but-not-imposed ceiling, keep the production
and signed ratios in plausible ranges, and generate count spreads that bracket
the observed data without dominating it. The signed-ratio prior is the broadest
by design and the association prior is weakly positive but spans independence.
No prior required revision on prior-predictive grounds. Evidence: each model's
`prior_samples_*.png` under `output/models/<model>-<config>/`, regenerated by
`scripts/prior_predictive_audit.py`.

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
- Checked against independent Wordbank normative deciles, the TD anchor priors
  are broad enough to cover the norms and, after the young-age recalibration (#135/#138/#140), their centres now track the normative medians; the independent TD `q(a)` curve corroborates both VG13's recalibrated `q` anchors and the VG10/VG15 `q`-anchor tightening.
  See "Evidence base: literature and normative data" above.
- The independent DS cohorts anchor only DS _spoken_ vocabulary and only to ~60
  months (Berglund et al., 2001): the DS spoken-low prior is ~2x high at 24 months, the DS understood-low anchor has no independent chronological-age
  source, and the 84-month high anchor is beyond all independent CDI data.
- A per-age Beta-Binomial fit to the Wordbank by-child data confirms the sign of
  the `kappa` age-trend (dispersion rises with age for production) but shows the
  prior is slightly tight at the high-dispersion (older-age) end.

No final robustness conclusion should be made until the prior predictive audit
and sensitivity checks above are complete.
