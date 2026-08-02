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
| VG10             | DS         | understood + spoken          | VG09 plus per-draw GP anchoring at a reference age (Option D).                                                     |
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
- Baseline `q(a) = P(speak | understood)` priors are the same across the DS joint
  models (VG05, VG07-VG10, VG14-VG16); young-TD VG13 uses lower anchors.

The main TD/DS differences are concentrated in the anchor ages and in a few
anchor distributions:

| Prior area            | DS                                                                                               | TD                                                                            | Interpretation                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Anchor ages           | Usually 24 and 84 months.                                                                        | Usually 12 and 26 months; VG13 uses 10 and 16 months.                         | Priors are placed over different developmental windows.                                                            |
| Spoken low anchor     | `Beta(1, 25)` at 24 months in VG01.                                                              | `Beta(1, 30)` at 12 months in VG03/VG11.                                      | Both concentrate near the floor after the young-age prior-predictive recalibration.                                |
| Understood low anchor | `Beta(1, 7)` at 24 months in VG02; `Beta(1, 10)` in the DS joint models.                         | `Beta(1.2, 8)` at 12 months in VG04/VG12; `Beta(1, 15)` at 10 months in VG13. | The TD understood low anchor was recalibrated up off the floor (TD comprehension is already substantial young).    |
| High anchor           | Usually `Beta(1.1, 1.1)` at 84 months.                                                           | Usually `Beta(1.3, 1.3)` at 26 months; VG13 uses `Beta(2, 6)` at 16 months.   | The TD high anchors were softened toward the middle from a more optimistic prior; DS high-age priors remain broad. |
| Baseline `q` anchors  | `Beta(2, 12)` low and `Beta(3, 2)` high across the DS joint models (VG05, VG07-VG10, VG14-VG16). | `Beta(1, 10)` low and `Beta(2, 7)` high in VG13.                              | Weakly-informative; VG13's young-TD production ratio sits lower still.                                             |
| Signing priors        | DS-only in VG14/VG15.                                                                            | Not modelled.                                                                 | There is no TD signing counterpart.                                                                                |

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

| Prior use                             | Models                      | Distribution     | Observable interpretation                                              |
| ------------------------------------- | --------------------------- | ---------------- | ---------------------------------------------------------------------- |
| Low-age DS spoken anchor              | VG01                        | `Beta(1, 25)`    | Median 0.027, 5-95% 0.002-0.113, or about 22 words median out of 810.  |
| Low-age TD spoken anchor              | VG03, VG11                  | `Beta(1, 30)`    | Median 0.023, 5-95% 0.002-0.095, or about 19 words median out of 810.  |
| Low-age DS understood anchor (single) | VG02                        | `Beta(1, 7)`     | Median 0.094, 5-95% 0.007-0.348, or about 76 words median out of 810.  |
| Low-age DS understood anchor (joint)  | VG05, VG07-VG10, VG14, VG15 | `Beta(1, 10)`    | Median 0.067, 5-95% 0.005-0.259, or about 54 words median out of 810.  |
| Low-age TD understood anchor          | VG04, VG12                  | `Beta(1.2, 8)`   | Median 0.104, 5-95% 0.011-0.341, or about 84 words median out of 810.  |
| Low-age young-TD understood anchor    | VG13                        | `Beta(1, 15)`    | Median 0.045, 5-95% 0.003-0.181, or about 36 words median out of 810.  |
| High-age DS single anchor (VG01/VG02) | VG01, VG02                  | `Beta(2, 1.5)`   | Median 0.586, 5-95% 0.168-0.924, or about 475 words median out of 810. |
| High-age DS understood anchor (joint) | VG05, VG07-VG10, VG14, VG15 | `Beta(1.1, 1.1)` | Median 0.500, 5-95% 0.060-0.940, or about 405 words median out of 810. |
| High-age TD single/U anchor           | VG03, VG04, VG11, VG12      | `Beta(1.3, 1.3)` | Median 0.500, 5-95% 0.079-0.921, or about 405 words median out of 810. |
| High-age young-TD understood anchor   | VG13                        | `Beta(2, 6)`     | Median 0.228, 5-95% 0.053-0.521, or about 185 words median out of 810. |
| DS-joint low-age `q` anchor           | VG05, VG07-VG10, VG14-VG16  | `Beta(2, 12)`    | Median 0.126, 5-95% 0.028-0.316 of understood words.                   |
| Young-TD low-age `q` anchor           | VG13                        | `Beta(1, 10)`    | Median 0.067, 5-95% 0.005-0.259 of understood words.                   |
| DS-joint high-age `q` anchor          | VG05, VG07-VG10, VG14-VG16  | `Beta(3, 2)`     | Median 0.614, 5-95% 0.249-0.902 of understood words.                   |
| Young-TD high-age `q` anchor          | VG13                        | `Beta(2, 7)`     | Median 0.201, 5-95% 0.046-0.471 of understood words.                   |

Review notes:

- The low-age direct trajectory anchors encode strong floor expectations, which
  are scientifically plausible but should be checked against prior predictive
  counts at the youngest queried ages.
- The high-age DS `Beta(1.1, 1.1)` anchor is deliberately broad. It prevents the
  prior from declaring either low or high later vocabulary impossible, but it can
  interact with the GP and random effects in sparse age regions.
- The DS-joint `q` anchors are weakly-informative and encode only the
  developmental direction (few understood words spoken early, a majority by school
  age). `q_low` is centred at the independent TD `q(~12mo) ≈ 0.12`; `q_high` has no
  independent DS source, so it is deliberately broad (5-95% ~0.25-0.90) and lets
  the data set the 84-month level.
- These replace the earlier `Beta(3, 22)` / `Beta(20, 4)` anchors, which were read
  off the VG07 posterior and then propagated across the DS-joint family — using a
  model's own posterior (fit to the same DS data) to set its prior. That
  prior-data double-dipping is removed; the `Beta(20, 4)` high anchor in particular
  was the tightest prior in the family with no independent basis. The history is
  documented in
  [`notes/202605131500-vg09-structural-options.md`](../../notes/202605131500-vg09-structural-options.md).

### Signed ratio prior

VG14 and VG15 model signing as `r(a) = P(sign | understood)`, whose developmental
trajectory is a **hump** — near zero at young ages, peaking in the preschool years,
then receding as words move into speech. The signed mean is therefore a
**three-anchor "tent"**: Beta priors on `r` at a young, a peak and an old reference
age (`sign_anchor_ages = (15, 36, 96)` months), joined by two logit-linear segments
meeting at the peak anchor and clamped flat beyond the outer anchors (see
[`gp_utils.tent_and_gp`](../../src/vocab_growth/models/gp_utils.py)).

| Anchor | Age   | Distribution  | `r` median | 5-95%        |
| ------ | ----- | ------------- | ---------- | ------------ |
| young  | 15 mo | `Beta(2, 20)` | 0.08       | [0.02, 0.21] |
| peak   | 36 mo | `Beta(3, 4)`  | 0.42       | [0.15, 0.72] |
| old    | 96 mo | `Beta(2, 16)` | 0.11       | [0.02, 0.26] |

Because the peak sits at the middle anchor age by construction, the full
prior-predictive `r(a)` median is a **hill** — rising to ~0.42 at ~36 mo, declining
to ~0.11 — and the implied words-signed median is a gentle hill (peaking ~55 words at
~54-60 mo) rather than the monotonic rise an intercept-only mean produced. The GP
(`eta_sign ~ HalfNormal(0.4)`) now only carries smooth departures.

Review notes:

- **Why a hump, not an intercept or a slope.** An intercept-only mean gave a _flat_
  prior-median `r`, so words signed = understood × `r` rose monotonically (reviewer
  pushback: the median should be hill-shaped). A free monotone _slope_ extrapolated
  to a spurious ~58% signed at 12 mo. The three-anchor tent gives the hill directly
  and, being concave, sends `r` low at _both_ the young and old ends — avoiding the
  young-extrapolation failure.
- **Anchor ages/levels are independent, not data-fit.** Signing peaks around _mental_
  age ~17 months (Miller 1992 via Clibbens: signed vocabulary ~2× spoken there,
  declining by MA ~26 mo), which at a DS developmental quotient ~0.5 is chronological
  ~34 mo — hence the ~36-month peak anchor. The inverted-U shape is confirmed by
  Zampini (parabolic gesture trajectory). DS children retain signs _longer_ than TD
  (Te Kaat-van den Os review) and `uk_06` has real 60-115 mo signers, so the old
  anchor stays modest (~0.11), not near-zero. The peak _level_ is kept broad because
  the peak _age_ is only weakly identifiable from the data.
- Since the mean now carries the hump, `eta_sign` reverts to the standard ~0.4 (it
  was inflated to ~1.0 only to force a hump out of a flat mean). VG15 additionally
  anchors the signed GP at 54 mo, so the tent supplies the hump and the GP deviates
  around it.
- Independence: Miller (US) / Clibbens (UK) are independent of the training data;
  Zampini (Italian) overlaps `it_01`, so it is cited for the _shape_ only. Shape and
  sensitivity history in
  [`notes/202606151700-vg14-signed-ratio-shape-and-p-any-bias.md`](../../notes/202606151700-vg14-signed-ratio-shape-and-p-any-bias.md).

### GP length-scale and amplitude priors

The HSGP priors use a unit length-scale parameter mapped onto a length-scale in
months:

```text
ell_unit ~ Beta(alpha, beta)
ell_months = ell_low + (ell_high - ell_low) * ell_unit
```

The common range is 6-18 months.

| Use                                 | Distribution                 | Observable interpretation                                                                              |
| ----------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| Standard U, spoken, and `q` smooths | `ell_unit ~ Beta(3, 3)`      | Median length-scale about 12 months; 5-95% about 8.3-15.7 months.                                      |
| Signed-ratio smooth                 | `ell_unit_sign ~ Beta(2, 5)` | Median length-scale about 9.2 months; 5-95% about 6.8-13.0 months.                                     |
| Standard GP amplitude               | `eta ~ HalfNormal(0.4)`      | Median logit-scale deviation about 0.27; 95% about 0.78.                                               |
| `q`-ratio GP amplitude              | `eta_q ~ HalfNormal(0.20)`   | Median logit-scale deviation about 0.13; 95% about 0.39. Tightened from 0.4 (see below).               |
| Signed GP amplitude                 | `eta_sign ~ HalfNormal(0.4)` | Median about 0.27; 95% about 0.78 (reverted to standard — the three-anchor mean now carries the hump). |

Review notes:

- The standard length-scale prior encodes smooth developmental departures rather
  than rapid month-to-month oscillation.
- The `q`-ratio GP amplitude `eta_q` was tightened from 0.4 to 0.20. Broadening the
  `q` anchors (removing the double-dipping) surfaced a weakly-identified `q`
  slope/intercept ridge in the subject-RE-on-`q` models (VG09/VG10/VG15/VG16);
  tightening `eta_q` curbs the GP-vs-linear-trend competition and restores mixing
  (VG10 `test` min ESS 120 → 450, divergences 6 → 2) without re-introducing any
  data-informed prior. It is a smoothness assumption on `q`, not a data-tuned value.
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
```

where `z` is standardised age. Every model uses that curve; they differ in how `(a_kappa, b_kappa)` are given priors.

#### Legacy form — intercept and slope (VG05, VG07-VG10, VG14-VG16)

```text
b_kappa = -b_kappa_mag

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

- Smaller `kappa` means more overdispersion relative to a Binomial at the same mean. The prior allows substantial extra-binomial heterogeneity.
- The sign of `b_kappa` encodes increasing heterogeneity with age, with a later plateau at `kappa_min`.
- This structure should be checked carefully near floor and ceiling regions, where `kappa` can be weakly identified even if predictions look reasonable.
- Alternative `kappa` priors are a sensitivity target for main reporting models.
- **This form has three known weaknesses**, all of which the two-anchor form below removes and none of which are repaired by re-tuning the three numbers above. `a_kappa` is the age term at `z = 0`, so its prior describes the pool's _mean age_ and silently changes meaning when the pool is resampled or filtered. `b_kappa_mag` is a slope per unit standardised age, so one shared prior is about 3.5x tighter on the Down syndrome pool (age sd ~21 months) than on the typically-developing pool (~6 months). And `b_kappa_mag >= 0` forces dispersion to fall with age, which typically-developing comprehension rejects — its dispersion is flat to slightly rising.

#### Two-anchor form (VG01-VG04, VG11, VG12, VG13)

The same curve, with the age term `exp(a_kappa + b_kappa * z)` given priors at two reference **ages in months** and `(a_kappa, b_kappa)` solved for so the curve passes through both:

```text
kappa_min           ~ LogNormal(log(k_min),   s_min)
kappa_excess_young  ~ LogNormal(log(e_young), s)     at anchor_ages[0]
kappa_excess_old    ~ LogNormal(log(e_old),   s)     at anchor_ages[1]

b_kappa = (log kappa_excess_old - log kappa_excess_young) / (z_old - z_young)
a_kappa =  log kappa_excess_young - b_kappa * z_young
```

`a_kappa` and `b_kappa` remain in the trace as derived quantities under the same names, so a migrated model's dispersion posterior stays comparable with the fits that preceded it. `kappa_young` and `kappa_old` carry _total_ kappa at the anchors — floor plus excess — which is the quantity a per-age empirical estimate can be checked against. The joint engines take one form per outcome, suffixed `_u` and `_s`, so a model may anchor one and leave the other on the legacy form.

| Model          | Outcome               | Anchors (months) | `k_min` | `e_young` | `e_old` | `s` | Calibration |
| -------------- | --------------------- | ---------------- | ------: | --------: | ------: | --: | ----------- |
| VG01           | spoken                | 18, 36           |       3 |        45 |     4.0 | 0.7 | marginal    |
| VG02           | understood            | 18, 36           |       3 |        11 |     3.2 | 0.8 | marginal    |
| VG03           | spoken                | 12, 20           |       3 |        30 |     3.0 | 0.7 | marginal    |
| VG04           | understood            | 12, 18           |       3 |       7.6 |     7.2 | 0.7 | marginal    |
| VG11           | spoken                | 12, 20           |       6 |       311 |      44 | 0.7 | conditional |
| VG12           | understood            | 12, 20           |       3 |        40 |      63 | 0.9 | conditional |
| VG13 `kappa_u` | understood            | 12, 17           |      30 |        10 |      90 | 0.9 | conditional |
| VG13 `kappa_s` | q = spoken/understood | 12, 17           |       3 |        33 |      27 | 0.7 | conditional |

**The two calibrations answer different questions and are not interchangeable.** A _marginal_ calibration estimates how much counts vary at an age, full stop; it is the right target only for a model with no grouping structure, which is why VG01-VG04 use it. A model with study and subject random intercepts has already removed most of that variation before its likelihood runs, so its `kappa` describes what is left once a child's own level is known — a much smaller residual. Substituting one for the other is a large error, not a rounding one: on VG11 the marginal number was 30 at 12 months where the conditional estimate is 317, and the fit went to 312 with the prior at CDF 1.000.

**VG04 and VG12 are the cleanest demonstration**, being the same outcome and population under the two specifications. VG04 carries no random effects and its dispersion is 11.8 at 12 months; VG12 carries study and subject intercepts and its is 43.0. Fit VG04's own rows _conditionally_ and they give 42.8; fit VG12's _marginally_ and they give 11.0. The gap is the specification, not the data.

Both sets of estimates come from `scripts/kappa_conditional_calibration.py`, which fits a saturated per-age mean alongside whichever effects the model carries — study effects and a quadrature-integrated subject effect for the random-effect models, neither for VG01-VG04. Each pool declares its own grouping and the estimator mirrors it. The `--recover` and `--mean-sweep` modes are what establish that a given pool can be calibrated at all; both must be run before adding one.

`s = 0.9` on the two understood outcomes rather than 0.7 is not generic caution. Typically-developing understood `kappa` per age cell runs 19.6, 21.0, 110.7 at 14, 15 and 16 months, so the fitted rise is a two-parameter summary of a jagged profile and should not be stated more confidently than that. What produces the jaggedness is now known — see "What a rising `kappa` on the understood outcomes means" below — and it is a further reason to keep these two anchors wide.

Prior simulation on each model's own age grid:

| Model | 8 mo           | 12 mo         | 18 mo        | 24 mo         | 36 mo       | oldest        |
| ----- | -------------- | ------------- | ------------ | ------------- | ----------- | ------------- |
| VG01  | 177 [30, 1150] | 105 [24, 491] | 49 [18, 147] | 24 [11, 52]   | 7.9 [3, 20] | 3.1 [0.8, 12] |
| VG03  | 99 [19, 586]   | 34 [13, 99]   | 9.2 [4, 20]  | 4.6 [1.5, 15] | —           | 3.6 [1.0, 13] |

Median and 5-95%. The upper tails at the youngest ages look alarming next to the legacy table above but are not: at `n = 810`, `kappa = 200` still gives 2.2x the binomial standard deviation, so it is _not_ near-binomial, and the observed dispersion at those ages is genuinely in that range — the Down syndrome 18-month cell estimates 64 (profile interval [35, 107]) and the typically-developing 12-month cell 89 ([66, 119]).

Anchors are placed where the age term is roughly an order of magnitude above the floor and where it has fallen back to it. Both priors then sit inside the data, so the prior between them is an interpolation of two checked values rather than an extrapolation from an intercept and a slope whose tails compound as `exp(2b)`.

Review notes:

- The prior on `kappa` at any given age is **exactly invariant** to the pool's age standardisation: the interpolation weight is `(age - young) / (old - young)` in months, and the standardisation cancels. Resampling or a study filter cannot move it.
- `kappa_min` is carried over from the legacy recalibration unchanged for the spoken and ratio outcomes. The anchored form leans on it harder — beyond the old anchor the floor alone sets the level — so its ~8% of prior mass below `kappa = 1` now shows at old ages. Tightening `kappa_min_sigma` is a candidate follow-up.
- **The floor is not always a floor.** With `b_kappa > 0` the exponential term vanishes at young ages instead of old ones, so `kappa_min` becomes the _young_-age asymptote. That is why VG13's is 30 rather than 3: a third of its 8-18 month frame sits below the young anchor, and the 8-11 month cells estimate 23-32. VG12's conditional fit puts no mass on a floor at all (it goes to zero with an unbounded standard error, a rising curve never reaching one inside the frame), so it keeps the weak default and the anchors carry the level.
- The sign of `b_kappa` is unconstrained, and this is what the comprehension models needed: their fitted `kappa` _rises_ with age, which `b_kappa_mag >= 0` cannot represent at any setting. For the spoken models the anchors put only about 1% of prior mass on a rising trajectory — correctly, since spoken dispersion demonstrably falls. On what the rise does and does not mean, see immediately below.
- Dropping `kappa_min` entirely and using a pure log-linear `kappa` was tested and rejected: it costs 10 to 168 log-likelihood units against the floored form across the six pools.
- **The Down syndrome joint frame is calibrated as a lower bound, not a point estimate.** Its 671 comprehension rows are the whole Down syndrome comprehension dataset — every model in that population loads the same 1,218 unfiltered rows, so there is nothing to pool in — and no configuration of spline flexibility, age window or anchor pair recovers a known `kappa` to within 30%. But the failure is a one-directional, monotone downward bias rather than scatter: holding `tau` fixed and varying only the truth, `kappa`(24) recovers at −2% when the truth is 12, −4% at 41, −26% at 82 and −36% at 163, because a large `kappa` is near-binomial and the optimum slides down the flat ridge. VG09, VG10, VG15 and VG16 therefore take medians equal to each estimate divided by the bias measured at it, with `sigma = 1.0` — wider than anywhere else in the family. Their previous `HalfNormal(0.3)` slope prior was not defensible on any reading: all eight Down syndrome joint models put `b_kappa_mag_u` at prior CDF 0.993-0.9999, well mixed, and five of the eight have _negative_ contraction on the spoken slope — the posterior wider than the prior.
- **VG05, VG07, VG08 and VG14 stay on the legacy form deliberately.** The calibration has to match the specification, and theirs differ: VG05 carries no random effects, VG07 only study ones, and VG08 a subject effect on understood but not on `q`. All three are steps in the VG05 → VG07 → VG08 → VG09 → VG10 lineage, which exists to isolate what each random effect does, so changing a prior partway along would confound it. VG14's frame is the signing subset.

### What a rising `kappa` on the understood outcomes means

VG12's and VG13's dispersion priors rise with age, and the two-anchor form exists partly so they can. That is a correct description of the models' `kappa` parameter and **not** a finding that comprehension becomes more variable as children get older. On the instrument's own scale it becomes less so.

The cause is the 810-item reference scale interacting with a subject intercept whose scale is fixed in age. Comprehension is collected only on WG (396 items) and Oxford CDI (418), and those are the _easiest_ items, so as children work up a form the modelled proportion `y / 810` compresses: by 16-18 months the mean row sits at about half its form's extent. The apparent between-child spread on the logit scale therefore falls with age, a constant `tau_subject` cannot follow it, and `kappa(age)` — the only age-varying spread parameter in the likelihood — absorbs the residue. Where the observed spread crosses below `tau`, `kappa` runs away, which is what produces the 110.7 at 16 months in the per-cell profile above.

Three measurements pin it down, all in section 21 of [`notes/202608020829-kappa-and-eta-q-prior-recalibration.md`](../../notes/202608020829-kappa-and-eta-q-prior-recalibration.md) and reproducible with `scripts/kappa_conditional_calibration.py --loading`:

- Letting the subject loading vary with age costs one parameter and buys 111-237 log-likelihood units on the three affected pools, and reverses the sign of the fitted `kappa` trend on both understood outcomes.
- `q` — the same children, the same design, a mean profile within 10% of understood's — shows **no** loading drift at all, worth 0.8 units. Its denominator is the child's own understood count, so the form's extent cancels.
- Rescoring the identical rows out of each row's own form instead of 810 removes 84-96% of the drift.

Two consequences. For the priors, none: the calibration must mirror the model's own structure, the registered models carry a constant `tau_subject`, and so a `kappa` prior fitted under that assumption is the right one for them. For reporting, `kappa` on the understood outcomes is a compound of observation-level dispersion and a subject scale the model holds fixed, and should not be quoted as a statement about children. The 810-item scale itself is not in question — it is the harmonisation this project deliberately adopts (see "Instrument scale" below) — only an untraced consequence of it.

See `notes/202608020829-kappa-and-eta-q-prior-recalibration.md` for the calibration, the estimator correction behind it, and the forms that were rejected.

### Study and subject random-effect scale priors

Study and subject random intercepts use non-centred Normal effects with
HalfNormal scale priors. The two levels take different scales:

```text
study scales   tau_u, tau_q, tau_sign                      ~ HalfNormal(0.5)
subject scales tau_subject, tau_subj_u, tau_subj_q, ...    ~ HalfNormal(1.5)
```

On the logit scale `HalfNormal(0.5)` has median 0.34 and `HalfNormal(1.5)` median
1.01, with 5-95% of 0.09 to 2.94. As an odds multiplier, `exp(tau)` at the
subject scale has prior median about 2.75.

**The subject scales were `HalfNormal(0.5)` until the recalibration** and were the
family's largest remaining prior-data conflict: all fourteen subject-scale
parameters in the registry sat at prior CDF 0.86 to 0.994, none below. The
conditional dispersion estimator
(`scripts/kappa_conditional_calibration.py`) reports `tau` alongside `kappa` for
every pool, because separating the two is what it exists to do, so a calibration
had been available since the dispersion work and had simply not been read off
it. It puts the subject scale at 0.74-0.77 on the typically-developing frames,
0.85 on Down syndrome understood, and 1.12-1.15 on the two production ratios.
`HalfNormal(1.5)` lands every one of those, and every current posterior, between
prior CDF 0.38 and 0.64.

Two details of that estimate are worth recording:

- **It agrees with the posteriors to three significant figures** on all four
  typically-developing parameters — 1.056 against 1.060 for VG11, 0.736 against
  0.735 for VG12, 0.770 against 0.768 and 1.119 against 1.117 for VG13 — and to
  within 3% on the five Down syndrome understood ones. A quadrature-integrated
  maximum-likelihood GLMM and a Hamiltonian sampler with an HSGP mean reaching
  the same number is independent corroboration of both.
- The four that differ are all the Down syndrome ratio (estimate 1.147 against
  posteriors 1.25-1.38). Its recovery check independently measures an 8% downward
  bias on that pool, which accounts for VG15's gap exactly and about half of the
  others'.

Review notes:

- The family stays HalfNormal rather than moving to the LogNormal the `kappa`
  anchors use. A scale prior with mass at zero lets a subject effect the data do
  not support shrink away, and that is worth keeping even where the effect is
  overwhelming. Widening the scale removes the conflict without giving it up.
- **The study scales are unchanged and need no change**: their posteriors sit at
  prior CDF 0.43 to 0.82 across every model carrying them. That the two levels
  shared one default was the accident; only one level was mis-set. The estimator
  fits study effects as fixed, so it offers no opinion on the study scale either.
- `tau_subj_sign` (VG15) has no calibration of its own — nothing estimates a
  signing subject scale — so it inherits the family setting. Its posterior at
  1.082 was in the same tail as the rest and is now at prior CDF 0.53. Subject
  random effects for sparse modalities remain a sensitivity target.
- The `tau-wide` / `tau-narrow` sensitivity variants now bracket 1.5 for the
  subject scales (3.0 and 0.75) and still bracket 0.5 for the study ones.

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

The DS-joint low-age `q` anchor, `Beta(2, 12)` (median 0.126), is centred on this
independent TD `q(10–12 mo) ≈ 0.12` — an independent corroboration of the _level_,
not a value read from the fitted DS data. VG13 uses lower `Beta(1, 10)` / `Beta(2, 7)`
(medians 0.067 / 0.201), matching the younger empirical TD ratio above. The high-age
DS `q` anchor, `Beta(3, 2)` (median 0.61), has no independent source and is
deliberately broad (5-95% ~0.25-0.90), so the DS data — not the prior — set the
84-month level.

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

> [!IMPORTANT]
> The per-age slopes in the table above are estimated by regressing `log kappa` on standardised age. That is **not** the model's parameter: because `kappa` flattens onto `kappa_min`, the log-slope of total `kappa` is shallower than `b_kappa_mag`, and the regression estimates it low — by roughly a factor of two. Fitting the model's own three-parameter form to the same cells gives `b_kappa_mag` of **2.78** for Down syndrome spoken and **1.71-1.78** for typically-developing spoken, against the 1.38 and 0.77 the log-linear regression reported. The direction below is unaffected; the magnitude is. See `notes/202608020829-kappa-and-eta-q-prior-recalibration.md` section 17.

> [!IMPORTANT]
> Every figure in this section is **marginal** — no random effects — and so applies only to VG01 and VG03. For a model with study and subject random intercepts the same data give a `kappa` three to ten times larger, because the random effects absorb the between-child spread that these per-age fits leave in the residual. The conditional estimates are in the two-anchor table above and in section 19 of the note; do not read this section's numbers across to VG11, VG12 or VG13.

Against the shared prior (`kappa` median ~13–17, 5–95% ~5–60; `b_kappa < 0`):

- **Direction confirmed.** For the spoken/production outcome (the primary one)
  `kappa` clearly falls with age — dispersion rises with age, exactly the sign the
  prior encodes. Comprehension is roughly flat, and on the typically-developing
  random-effects frame very slightly rising, which the shared prior's
  `b_kappa_mag >= 0` cannot represent at all. Independently, Zampini & D'Odorico
  (2013) report DS vocabulary variability _increasing_ from 36 months, the same
  direction.
- **The floor is real and is about 3.** Three independent pools (DS spoken, and
  the two typically-developing spoken frames) put `kappa_min` at 3.08–3.54,
  against a shared prior centred at 5 whose 5th percentile was 1.86. Dropping the
  floor and using a pure log-linear `kappa` was tested and costs 10–168
  log-likelihood units, so the plateau is a genuine feature rather than a
  parameterisation convenience. Part of the old-age level is a ceiling artefact
  (WS counts pile toward the 680-item form limit), and the model's GP mean and
  study random effects absorb some spread that these raw per-age fits do not.
- **The age slope was out by a factor of five, not "slightly too tight".** The
  shared `HalfNormal(0.3)` has a median of 0.20 and a 95th percentile of 0.59
  against corrected empirical values of 1.7–2.8. Widening it to `HalfNormal(0.75)`
  moved VG03 from prior CDF 1.00 and contraction 0.18 to 0.93 and 0.82 but did not
  go far enough, and widening a third time is not viable: the intercept and slope
  tails compound as `exp(2b)`, so at `HalfNormal(1.5)` about 30% of prior mass puts
  `kappa` above 200 at `z = -2`. This is what the two-anchor form above resolves,
  and why VG01, VG03, VG11, VG12 and VG13 have migrated to it. The remaining
  models still carry the mis-scaled shared default and are the outstanding work.
- **The random-effect models needed a second correction on top of that one.**
  Re-parameterising fixes the shape of the prior but not what it is a prior
  _about_: a marginally-calibrated `kappa` transplanted into a model with subject
  random intercepts is a prior for the wrong quantity, and VG11 showed the size of
  that error at a factor of ten. VG11, VG12 and VG13 are now calibrated
  conditionally. VG09, VG10 and VG16 are not, because their frame cannot support
  it, so their dispersion priors remain the weakest part of this specification —
  VG12's and VG13's posteriors sat at less than half their conditional estimate
  before the change, and nothing rules out the same being true of them.

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

| Check                           | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Young-age floor                 | Plausible. Every trajectory family places prior-predictive mass near zero at the youngest ages (spoken and signed at 8-12 months, understood a little higher); no prior draw forces a high count at young ages.                                                                                                                                                                                                                                |
| Old-age ceiling                 | Plausible. Understood and spoken curves approach the 810-word ceiling only gradually and only for the fastest draws; the bulk of the prior mass stays well below saturation across the query range, so the ceiling is reachable but not imposed.                                                                                                                                                                                               |
| Smoothness                      | Appropriate. The HSGP produces smooth curves with individual-draw wiggle, admitting both near-linear and gently curved trajectories without high-frequency oscillation.                                                                                                                                                                                                                                                                        |
| `q(a)` (speak given understood) | Plausible. The prior band is a smooth 0-to-1 sigmoid rising from about 0.05 at the youngest ages toward about 0.9 by ~100 months, with no mass piling implausibly at the bounds.                                                                                                                                                                                                                                                               |
| `r(a)` (sign given understood)  | Re-specified as a three-anchor hump after this audit. The audited intercept-only mean gave a flat median (words signed = understood x r therefore rose monotonically); the signed mean is now a tent through young / peak / old anchors (`r` ~0.08 / ~0.42 / ~0.11 at 15 / 36 / 96 mo), so the prior median is a hill peaking ~0.42 at ~36 mo and the words-signed median is a gentle hill. The 54-month GP-anchor "waist" (VG15) is retained. |
| Random-effect heterogeneity     | Plausible. At the observation level the study/subject random effects widen the prior-predictive cloud enough to cover the observed between-study and between-child spread without implying implausible extremes on the probability scale.                                                                                                                                                                                                      |
| Simulated count spread          | Plausible. The prior-predictive count clouds bracket the observed counts for every outcome (understood, spoken, signed) before the data are seen — neither too narrow (which would fight the data) nor degenerate at 0 or 810.                                                                                                                                                                                                                 |
| VG15 signing / four-cell        | Plausible. Signed counts stay low with a broad, hump-capable upper tail (matching the sparse signing data); the `log_psi ~ Normal(0.3, 0.5)` association prior spans the independence reference `psi = 1`, so the four-cell composition is not prior-forced toward association.                                                                                                                                                                |

**Conclusion.** The priors encode the developmental floor and a
reachable-but-not-imposed ceiling, keep the production ratio in a plausible range,
and generate count spreads that bracket the observed data without dominating it;
the association prior is weakly positive but spans independence. The one prior
since revised on prior-predictive grounds is the **signed ratio**: this audit's
`r(a)` had a flat, floor-hugging median (~14% of mass below 0.05), and it was
re-specified as a three-anchor hump (a tent through young / peak / old anchors) so
the prior median is a hill — see the "Signed ratio prior" section. Evidence: each
model's `prior_samples_*.png` under `output/models/<model>-<config>/`, regenerated
by `scripts/prior_predictive_audit.py`.

## Sensitivity targets

The following sensitivity checks should be prioritised before the technical
report makes robustness claims:

| Target                                                                                                            | Why it matters                                                                                                                                                                                                                                       | Suggested alternatives                                                                                                                                                                                                                 |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DS-joint `q` anchors (esp. `q_high`)                                                                              | Weakly-informative, broadened off the VG07-posterior values; `q_high` has no independent DS source.                                                                                                                                                  | The former `Beta(3, 22)` / `Beta(20, 4)` as a revert check; narrower/wider `q_high` such as `Beta(4, 2)` or `Beta(2, 1.5)`.                                                                                                            |
| Signed GP amplitude and length-scale                                                                              | Signing data are sparse and the hump is GP-driven.                                                                                                                                                                                                   | Wider/narrower `eta_sign`; standard `ell_unit_sign ~ Beta(3, 3)`; shorter length-scale alternative.                                                                                                                                    |
| Signed hump anchors (peak / old)                                                                                  | Three-anchor tent; the peak _age_ is only weakly identifiable, and the old anchor is the words-fall (Miller) vs plateau (uk_06) knob.                                                                                                                | Lower/higher peak level (`Beta(2, 6)` / `Beta(4, 3)`); higher old anchor (`Beta(2, 8)`); shifted peak age (`sign_anchor_ages`).                                                                                                        |
| Kappa priors                                                                                                      | Dispersion can dominate predictive uncertainty, especially near floor or ceiling.                                                                                                                                                                    | Broader `kappa_min`; flatter age trend; non-monotone or constant-kappa comparison where feasible.                                                                                                                                      |
| Random-effect scales                                                                                              | Study and subject effects can trade off with global age curves.                                                                                                                                                                                      | Wider `tau` prior; narrower `tau` prior; study-only or no-subject variants where already supported by flags.                                                                                                                           |
| VG15 `psi`                                                                                                        | Identified from sparse four-cell data and prior is weakly positive.                                                                                                                                                                                  | Neutral `log_psi ~ Normal(0, 0.5)`; broader `Normal(0, 1)`; stronger positive prior only as an explicit data-informed sensitivity.                                                                                                     |
| VG15 concentration                                                                                                | Controls four-cell overdispersion.                                                                                                                                                                                                                   | Broader `log_conc`; lower/higher median concentration.                                                                                                                                                                                 |
| Young-age trajectory anchors (`p_slope_*`, `eta`) on VG10 (DS understood), VG11 (TD spoken), VG12 (TD understood) | Re-centred toward the young-age empirical/normative band (#135/#138/#140/#142). The DS understood anchors and the 26-month TD understood high anchor (VG04/VG12) have **no** independent norm, so their re-centring is data-informed regularisation. | Revert each anchor to its pre-recalibration vague prior and un-widen `eta`. Registered as `u-anchor-broad` / `eta-u-narrow` (vg10), `anchor-broad` / `eta-narrow` (vg11), `lo-anchor-broad` / `hi-anchor-broad` / `eta-narrow` (vg12). |

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
- The DS-joint `q` anchors are weakly-informative, broadened off the
  VG07-posterior-derived `Beta(3, 22)` / `Beta(20, 4)` to remove prior-data
  double-dipping; `q_high` is deliberately broad as it has no independent DS source.
- The signed-ratio prior is a three-anchor hump (a tent through young / peak / old
  reference ages), so its prior median is a hill — replacing the intercept-only mean
  (flat median) and avoiding the monotone-slope young-extrapolation failure. The
  anchor ages/levels come from the independent DS sign literature (peak ~mental age
  17 mo ≈ chronological ~36 mo; Miller/Clibbens, Zampini, Te Kaat-van den Os), not
  the in-sample data; `eta_sign` reverts to the standard ~0.4 since the mean now
  carries the hump.
- The shared kappa prior encodes substantial extra-binomial heterogeneity and a
  monotone increase in heterogeneity with age.
- Random-effect scale priors allow meaningful study and subject differences and
  should be interpreted on the logit and probability scales.
- VG15 `psi` is weakly positively regularised and must be tested against neutral
  alternatives.
- Checked against independent Wordbank normative deciles, the TD anchor priors
  are broad enough to cover the norms and, after the young-age recalibration (#135/#138/#140/#142), their centres now track the normative medians; the independent TD `q(a)` curve corroborates both VG13's recalibrated `q` anchors and the VG10/VG15 `q`-anchor tightening. Each anchor's code comment in `definitions.py` now cites the external norm as its basis where one exists and demotes the in-sample statistic to corroboration.
  See "Evidence base: literature and normative data" above.
- Where an anchor has _no_ independent norm — the DS understood anchors (VG02/VG10) and the 26-month TD understood high anchor (VG04/VG12, WS is production-only) — the re-centring is data-informed regularisation rather than external anchoring, and is now a registered sensitivity target (Target 8: `u-anchor-broad`/`hi-anchor-broad` etc.) so the young-age conclusions can be shown not to hinge on it.
- The independent DS cohorts anchor only DS _spoken_ vocabulary and only to ~60
  months (Berglund et al., 2001): the DS spoken-low prior is ~2x high at 24 months, the DS understood-low anchor has no independent chronological-age
  source, and the 84-month high anchor is beyond all independent CDI data.
- A per-age Beta-Binomial fit to the Wordbank by-child data confirms the sign of
  the `kappa` age-trend (dispersion rises with age for production) but shows the
  prior is slightly tight at the high-dispersion (older-age) end.

No final robustness conclusion should be made until the prior predictive audit
and sensitivity checks above are complete.
