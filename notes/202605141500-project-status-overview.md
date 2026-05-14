---
title: "Vocabulary growth — project status overview"
subtitle: "Through VG09B (anchored)"
date: "2026-05-14"
format:
  pdf:
    documentclass: article
    papersize: a4
    fontsize: 11pt
    mainfont: "Source Sans 3"
    sansfont: "Source Sans 3"
    monofont: "Monaspace Neon"
    monofontoptions: "Scale=0.8125"
    linestretch: 1.25
    geometry:
      - top=25mm
      - left=25mm
      - right=25mm
      - bottom=25mm
    colorlinks: true
    link-citations: true
    toc: true
    toc-depth: 2
    number-sections: true
    number-depth: 3
  html:
    toc: true
    toc-depth: 2
    number-sections: true
    number-depth: 3
---

::: {.callout-warning}
This document was drafted with assistance from an AI model (Claude,
Anthropic) and should be independently verified before being relied on
for research conclusions.
:::

## Executive summary

### What the project is for

Families, teachers, speech-and-language therapists and clinicians
who support children with Down syndrome (DS) routinely have to
answer practical questions about vocabulary development: _"By what
age would a typical child with DS understand 100 words? By what age
would they say 100 words? My 3-year-old understands a lot but says
very little — is that within the normal range?"_ The clinical and
educational decisions that follow — when to refer, what to expect,
how to set goals, whether a child's progress is unusual — depend on
having answers grounded in the data, expressed as probability
distributions rather than single numbers, and queryable at any age
or comprehension level.

This project produces those answers. Its goal is a set of
_interpretable, defensible, queryable_ statistics that can inform
expectations, intervention and teaching practice for children with
DS between roughly 12 months and 8 years of age:

- how many words a typical child with DS _understands_ at each age,
- how many a typical child with DS _says_ at each age, and
- how those two grow together — captured through the _production
  ratio_ `q = p_S / p_U`, the fraction of understood words that are
  also produced.

The corresponding numbers are also estimated for typically
developing (TD) children, mostly as a reference point — the
qualitative _DS-vs-TD comparison_ is the part that matters most for
clinical communication.

### Pooling the international evidence

No single study of DS vocabulary development is large enough to
support stable, age-resolved estimates across the 12-to-90-month
range. The project's distinctive contribution is to **bring together
data from multiple international studies into one coherent
statistical analysis**: 964 age-valid rows from 510 unique DS
subjects across **10 study labels** spanning the United Kingdom,
Ireland, Italy, and the United States, harmonised into a single
pooled dataset. For the current joint understood + spoken models,
950 of those rows carry at least one of the two modelled outcomes.

That pooling is what unlocks the analysis. It also creates the
central methodological problem the project has had to solve: studies
differ systematically in _which children they recruit_, _which CDI
form they administer_, and _at what ages they collect data_. Naïve
pooling produces population trajectories that track which studies
contributed data at each age rather than how children actually
develop. The whole modelling iteration documented below is, in
effect, the story of identifying and absorbing those between-study
and between-subject confounds while keeping the developmental signal
intact.

A TD reference is held in parallel: a reproducible 10% Wordbank
sample (1,655 observations) is fitted with the same model family,
so the DS-vs-TD contrast is methodologically apples-to-apples.

### Why the answers are returned as probability distributions

The product is a family of Bayesian statistical models. Inference
returns a full posterior probability distribution over every
quantity of interest, rather than a single number with a confidence
interval. That probabilistic framing is what supports statements of
the form _"there is an 80% chance a typical 3-year-old with Down
syndrome understands between X and Y words"_ — phrasing most readers
already assume a confidence interval supports, but only Bayesian
intervals actually do.

The same posterior also supports _predictive_ distributions for an
individual unseen child, _milestone_ distributions ("by what age
will the typical child with DS reach 100 spoken words?"), and
_matched-comprehension_ distributions ("at 200 understood words,
what fraction does the typical DS child speak?"). All three framings
appear in this report.

### Headline scientific picture (stable across the model iterations)

- DS vocabulary growth continues across the whole sampled age range
  (≈ 12–90 months); the typical DS trajectory is monotone and rising.
- Spoken production lags comprehension persistently — the gap is
  larger and longer-lasting in DS than in TD development, and it
  _grows_ with comprehension level through the early and middle
  range before closing in the upper age range.
- Between-subject heterogeneity on the production-ratio
  `q = p_S / p_U` is the single largest source of variation in the
  model family ($\tau^{\text{subj}}_q \approx 1.19$ on the logit scale under VG09B),
  substantially larger than between-study variation.

### Where the modelling currently stands

The modelling has iterated through nine numbered specifications —
from a univariate "age → spoken" baseline (VG01) to a joint
understood + spoken DS model with study and per-subject random
effects on both trajectories (VG09) — and a tenth, **VG09B**, that
resolves a structural identifiability problem in VG09 by anchoring
the Gaussian-process correction to zero at a reference age.

**VG09B is the current candidate to replace VG09 as the headline
DS joint model:** its diagnostics are clean ($\hat R \le 1.01$,
$\text{ESS} \ge 400$ across all reported parameters), it preserves VG09's
variance partition, and the parameterisation removes the GP–intercept
ridge that produced VG09's marginal $\hat R$ and ESS warnings. The
cost is that the mid-age production-ratio trajectory shifts downward
by roughly 5–10 percentage points relative to VG09 — a change in
the direction supported by the structural argument, not a reporting
accident.

Open methodological questions remain: the non-monotone dip in `q`
beyond ~72 months under VG09 and VG09B (absent in VG07), how much
upper-age interpretation is shaped by finite checklist ceilings, and
whether the GP anchor should be applied symmetrically to the rest of
the model family (VG05–VG08, and the univariate VG01–VG04). A
separate strand of work — bringing signed/gestured vocabulary into
the model family — is also scoped below. These are all reflected in
[Next steps](#next-steps).

## Data and methods

### Pooled dataset

The DS data source currently contains 964 age-valid rows from 510
unique subject IDs across 10 international study labels in the
DuckDB `vocab_combined` view. The contributing studies span the
United Kingdom, Ireland, Italy, and the United States, and were
collected over roughly three decades. They use different CDI /
MacArthur forms and different study protocols, so the data
preparation pipeline (`scripts/prepare_data.py`) harmonises columns
into a common schema before merging into a DuckDB database; all
models load from that view via
`vocab_growth.data_utils.load_combined_data()`.

For the bivariate DS models, 950 rows have at least one of
`understood` or `spoken` observed (704 understood rows, 949 spoken
rows, 703 complete understood + spoken pairs). 288 of the 510 DS
subjects have ≥ 2 bivariate-analysis rows, contributing 728 of the
950 rows (maximum 8 on a single subject). That strong within-subject
repeated-measures structure — a direct consequence of pooling
longitudinal cohorts — motivates the move to subject random
intercepts in VG08 and VG09.

The TD reference uses a reproducible 10% Wordbank sample (1,655
observations), fitted with the same model family so the DS-vs-TD
contrasts are methodologically consistent.

Raw study data sit in `data/` as one CSV per study;
`scripts/prepare_data.py` merges them into a DuckDB database with a
unified `vocab_combined` view, which all models load via
`vocab_growth.data_utils.load_combined_data()`.

### Modelling approach

All models share the same statistical shape:

- a smooth (but flexible) mean trajectory over age, composed of a
  linear trend plus a Hilbert-space Gaussian-process (HSGP) correction
  on the logit scale;
- a Beta-Binomial likelihood with age-varying dispersion — Beta-Binomial
  rather than Binomial because individual children at the same age
  show much more variability than a Binomial can absorb;
- (in the joint models) a coupled production-ratio component so that
  `p_S ≤ p_U` is guaranteed by construction;
- (in VG07 onward) random intercepts on the logit scale to absorb
  systematic study- and subject-level shifts that would otherwise
  contaminate the population trajectory.

The outcomes are parent-reported counts from finite CDI /
MacArthur-style vocabulary checklists. Model probabilities therefore
refer to the probability that a _checklist word_ is understood or
spoken, and expected counts are expected words within the assessed
inventory, not total lexical knowledge. As children approach the
upper end of a form, additional words outside the checklist are
unobserved, so apparent flattening, compressed uncertainty, or very
high production ratios may partly reflect instrument ceiling rather
than developmental saturation.

Inference is by Hamiltonian Monte Carlo (PyMC + nutpie / NUTS). The
_reporting_ sampling configuration is 6 chains × (6,000 tune +
6,000 draw) at `target_accept = 0.95`. The project's convergence
threshold is $\hat R \le 1.01$ and $\text{ESS} \ge 400$ for every reported parameter;
across the rep run on 12 May 2026, 252,000 draws of VG01–VG07
produced one divergence (in VG06) and met the threshold in every
model.

## Model family

| ID    | Outcome                     | Population | Random effects                                      |
| ----- | --------------------------- | ---------- | --------------------------------------------------- |
| VG01  | Spoken                      | DS         | —                                                   |
| VG02  | Understood                  | DS         | —                                                   |
| VG03  | Spoken                      | TD         | —                                                   |
| VG04  | Understood                  | TD         | —                                                   |
| VG05  | Understood + spoken (joint) | DS         | —                                                   |
| VG06  | Understood + spoken (joint) | TD         | —                                                   |
| VG07  | Understood + spoken (joint) | DS         | Study                                               |
| VG08  | Understood + spoken (joint) | DS         | Study + subject (on U)                              |
| VG09  | Understood + spoken (joint) | DS         | Study + subject (on U and q)                        |
| VG09B | Understood + spoken (joint) | DS         | As VG09, with anchored GP and tighter anchor priors |

VG09B is the experimental variant that implements "Option A + D" from
the structural-options note (`notes/202605131500-vg09-structural-options.md`):

- **Option A (tighter q-anchor priors):**
  `p_slope_low_q ~ Beta(3, 22)` (mean ≈ 0.12, sd ≈ 0.06);
  `p_slope_hi_q ~ Beta(20, 4)` (mean ≈ 0.83, sd ≈ 0.08).
- **Option D (per-draw GP anchor):** `g_u` and `g_q` are
  reparameterised as `eta · (g_unit − g_unit(a_ref))` with
  `a_ref = 54` months. Every posterior draw of the GP passes through
  zero at the reference age, so the linear trend uniquely defines the
  level there and the GP can only describe deviations from linearity.

All other components (study REs, subject REs on `u` and `q`,
dispersion priors, Beta-Binomial likelihood) are identical to VG09.

## Iteration through the family — what each step taught us

This section is the project's intellectual narrative. Each model in
the sequence was motivated by a specific weakness in its predecessor.

### VG01–VG06 — baseline trajectories

The univariate DS models (VG01 spoken, VG02 understood) and their TD
counterparts (VG03, VG04) established that the data is informative
enough at every age to identify a smooth growth trajectory, and that
the Beta-Binomial dispersion is well-behaved across the age range.
The joint models (VG05 DS, VG06 TD) added a coupled production ratio,
which enforces `p_S ≤ p_U` and makes the comprehension–production gap
a first-class parameter rather than a derived quantity.

The DS joint model VG05 already showed the qualitative
DS-vs-TD picture that survived all the subsequent refinements:

- DS comprehension and production both grow across the whole age
  range, but production lags comprehension by years rather than
  months;
- the production ratio `q` rises from near zero to near one over a
  much wider comprehension range than in TD — the gap is _structurally
  larger_ in DS, not just delayed.

VG05 also exposed a problem: an apparent dip in _understood_ words
between roughly 40 and 60 months that the GP fitted in spite of being
developmentally implausible.

### The Simpson's-paradox finding — VG07 fixes the artefact

Investigation (`notes/202604121055-understood-ds-decline.md`) traced
the 40–60 month understood-vocabulary dip to a _compositional shift_
in contributing studies: higher-scoring studies (1, 2, 6, 7) stop
contributing understood-word data after ~50 months, and lower-scoring
studies (3, 5) continue. The pooled-mixture trajectory dips even
though no individual study's trajectory does. This is a textbook
Simpson's paradox.

**VG07** introduced study-level random intercepts on both the
understood logit and the production-ratio logit, with priors
$\tau_U, \tau_q \sim \text{HalfNormal}(0.5)$. The between-study SD
posteriors are substantial ($\tau_U \approx 0.50$, $\tau_q \approx 0.66$
on the logit scale) and
well-identified (ESS > 7,000). With those study shifts absorbed, the
population-level understood trajectory becomes monotone-increasing
across the whole age range. The leftward "hook" in
understood-vs-spoken plots disappears.

This is the project's single most important methodological finding to
date: _what looked like a developmental decline was driven by which
studies contributed data at which ages._ It motivates the
preference for VG07 over VG05 for any DS reporting that touches the
40–70 month understood range.

### VG08 — subject random intercepts on understood

VG07 still treated each observation as if it came from an independent
child. But 288 of 510 DS subjects have ≥ 2 bivariate-analysis
observations, contributing 728 of the 950 rows used by the joint DS
models. Longitudinal recruitment tends to over-represent
higher-engaging families, so those multi-observation subjects sit
above the population mean — and contribute disproportionately to the
likelihood.

VG08 added a non-centred subject random intercept on the understood
logit. The variance partition split cleanly:

- between-study SD on understood ($\tau_U$): essentially unchanged
  (~0.51) — confirming VG07's study REs were _not_ secretly absorbing
  within-subject correlation;
- between-subject SD on understood ($\tau^{\text{subj}}_U \approx 0.78$):
  the largest single component of variation revealed so far;
- residual Beta-Binomial dispersion at mid-age dropped from
  `exp(1.75) ≈ 5.8` to `exp(2.93) ≈ 18.6` — i.e. about one-third of
  what VG07 had been estimating.

The trajectory shifted in clinically meaningful ways: lower at 12–18
months, steeper through the middle of the age range, and showing a
production-ratio _plateau_ around `q ≈ 0.78–0.84` from ~60 months on,
rather than VG07's continued climb toward 1.0.

### VG09 — subject random intercepts on the production ratio too

If between-subject variability justifies a subject RE on understood,
the same logic applies to how much of each child's understood
vocabulary they actually produce. VG09 added a parallel subject RE on
the production-ratio logit.

This exposed the largest variance component in the entire model
family: $\tau^{\text{subj}}_q \approx 1.20$ on the logit scale, equivalent on the
count scale to a typical DS child's `q` sitting anywhere in roughly a
five-fold range around the population median at matched comprehension.
VG08 had been absorbing this into the spoken-side Beta-Binomial
dispersion; with VG09, residual overdispersion on spoken vocabulary
collapsed substantially — $\exp(a_{\kappa_S})$ rose from ~6.4 (VG08) to ~27.6
(VG09), an order-of-magnitude reduction.

Three leave-one-subject-out (LOSO) comparisons established that VG09
generalises better than its predecessors. The gold standard — a
K = 5 LOSO refit comparison — ranks VG09 > VG08 > VG07 with
overwhelming statistical significance:

| Pair         | elpd_diff (better − worse) |  dSE | diff / dSE |
| ------------ | -------------------------: | ---: | ---------: |
| VG09 vs VG07 |                 **+339.0** | 34.6 |   **9.79** |
| VG09 vs VG08 |                     +109.4 | 19.3 |       5.68 |
| VG08 vs VG07 |                     +229.6 | 26.1 |       8.80 |

(Source: `output/comparisons/kfold_loso_compare.csv`. n = 510 subjects;
15 refits at `test` config, 41 minutes wall time total.)

VG09 wins both descriptively (cleanest variance partition) and
predictively (best K-fold LOSO).

### VG09 diagnostic issue — and the structural response

At reporting-quality sampling, VG09 returned five parameters with
marginal $\hat R$ above 1.01 (max 1.020) and one with $\text{ess}_{\text{tail}} < 400$
(min 358). Tightening the sampler (`target_accept` = 0.99,
`tune` = 8,000) doubled wall time and _did not improve_ the diagnostic:
this is a posterior-geometry issue, not a step-size issue.

The structural diagnosis (`notes/202605131400-vg09-sampler-diagnostics.md`
and `notes/202605131500-vg09-structural-options.md`): VG09's `q`
trajectory has three components that each carry a global level — the
linear trend (`intercept_q + slope_q · a_z`), the HSGP correction
(`eta_q · g_unit_q`), and the random-intercept families. The data
identify their _sum_, not their decomposition, so the posterior is
ridge-shaped along that direction. The HSGP is zero-mean _in prior
expectation_ (averaged over draws) but each individual draw of the GP
function carries an arbitrary constant that competes with
`intercept_q`.

VG09B implements the smallest credible structural fix: tighter anchor
priors and a per-draw GP zero anchor at `a_ref = 54` months. The
remainder of this report focuses on what VG09B does and does not
change.

## VG09B — detailed results

### Diagnostics: every flagged parameter cleared

The VG09B A+D variant — tighter q-anchor priors plus a GP anchor at
`a_ref = 54` months — cleared the reported VG09 diagnostic flags.
Bulk ESS roughly doubled on the previously flagged parameters, and
$\hat R$ dropped to $\le 1.009$ on every reported parameter (threshold:
$\hat R \le 1.01$, $\text{ESS} \ge 400$). This is consistent with the
structural diagnosis that VG09 had a GP-level intercept redundancy,
but the diagnostic improvement should be attributed to the combined
A+D change rather than to the GP anchor in isolation.

| Parameter       | VG09 $\hat R$ | VG09 $\text{ESS}_{\text{bulk}}$ | VG09B $\hat R$ | VG09B $\text{ESS}_{\text{bulk}}$ |
| --------------- | ------------- | ------------------------------- | -------------- | -------------------------------- |
| `slope_q`       | 1.020         | 430                             | **1.008**      | **1,244**                        |
| `p_slope_low_q` | 1.013         | 1,008                           | **1.007**      | **1,610**                        |
| `p_slope_hi_q`  | 1.012         | 431                             | **1.006**      | **1,221**                        |
| `eta_q`         | 1.010         | 483                             | **1.009**      | **1,207**                        |
| `p_slope_hi_u`  | 1.014         | 628                             | **1.007**      | **1,338**                        |
| `intercept_u`   | 1.012         | 759                             | **1.007**      | **1,225**                        |
| `intercept_q`   | 1.005         | 799                             | **1.004**      | **2,152**                        |

Summary: _$\hat R \le 1.01$ and $\text{ESS} \ge 400$ across reported parameters._

### Variance partition is preserved

The point of VG09B was to fix VG09's parameterisation _without_
changing the substantive variance partition. The structural variance
components are essentially identical to VG09:

| Component (logit-SD)                                 | VG07 | VG08 | VG09 | **VG09B** |
| ---------------------------------------------------- | ---: | ---: | ---: | --------: |
| Between-study, understood ($\tau_U$)                 | 0.50 | 0.51 | 0.52 |  **0.52** |
| Between-study, q ($\tau_q$)                          | 0.66 | 0.74 | 0.94 |  **0.99** |
| Between-subject, understood ($\tau^{\text{subj}}_U$) |    — | 0.78 | 0.84 |  **0.84** |
| Between-subject, q ($\tau^{\text{subj}}_q$)          |    — |    — | 1.20 |  **1.19** |
| Understood BB dispersion $a_{\kappa_U}$ (log)        | 1.75 | 2.93 | 3.10 |  **3.10** |
| Spoken BB dispersion $a_{\kappa_S}$ (log)            | 1.40 | 1.86 | 3.32 |  **3.31** |

Between-subject variation on `q` is still the single largest
component in the family, an order of magnitude larger than the
Beta-Binomial nugget on either outcome. VG09B inherits that finding
intact.

### Posterior summary (selected parameters)

| Parameter              |  Mean |   SD | 90 % HDI       | $\text{ESS}_{\text{bulk}}$ | $\hat R$ |
| ---------------------- | ----: | ---: | -------------- | -------------------------: | -------: |
| $\text{intercept}_U$   | −1.15 | 0.22 | [−1.52, −0.80] |                      1,225 |    1.007 |
| $\text{slope}_U$       |  1.13 | 0.20 | [ 0.80, 1.45]  |                        930 |    1.007 |
| $\eta_U$               |  0.58 | 0.16 | [ 0.32, 0.83]  |                      3,524 |    1.002 |
| $\text{intercept}_q$   | −0.23 | 0.31 | [−0.73, 0.27]  |                      2,152 |    1.004 |
| $\text{slope}_q$       |  1.53 | 0.31 | [ 1.03, 1.99]  |                      1,244 |    1.008 |
| $\eta_q$               |  0.94 | 0.24 | [ 0.56, 1.35]  |                      1,207 |    1.009 |
| $\tau_U$               |  0.52 | 0.14 | [ 0.31, 0.73]  |                     10,507 |    1.000 |
| $\tau_q$               |  0.99 | 0.21 | [ 0.67, 1.32]  |                      6,900 |    1.000 |
| $\tau^{\text{subj}}_U$ |  0.84 | 0.04 | [ 0.77, 0.91]  |                      5,831 |    1.000 |
| $\tau^{\text{subj}}_q$ |  1.19 | 0.09 | [ 1.05, 1.33]  |                      6,354 |    1.000 |
| $a_{\kappa_U}$         |  3.10 | 0.11 | [ 2.93, 3.28]  |                     13,068 |    1.001 |
| $a_{\kappa_S}$         |  3.31 | 0.14 | [ 3.09, 3.53]  |                      8,730 |    1.001 |

(Source: `output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/diagnostics.csv`.)

The reparameterisation shifted `intercept_q` from −1.17 (VG09) to
−0.23 (VG09B) — an upward shift of about 0.94 logit units — and
`p_slope_hi_q` from 0.80 to 0.93. $\tau^{\text{subj}}_q$ and `eta_q` are
essentially unchanged. The shift is therefore entirely between the GP
and the linear trend, _not_ between the GP and the random effects:
exactly what the structural argument predicted.

### Trajectories: typical (median) DS child under VG09B

Population-level trajectory — study REs and subject REs set to zero,
i.e. expected within-subject change as a typical DS child ages.
Vocabulary counts here are model expectations on a 800-word total
(`Ey_median`); 90 % HDIs (`Ey_hdi_lo`–`Ey_hdi_hi`) reflect the
uncertainty in the typical DS trajectory, not individual-child spread.
At older ages, especially beyond ~72 months, these should be read as
within-checklist estimates. A child may continue learning words
outside the assessed inventory even when the checklist-based
understood or spoken count is approaching the form ceiling.

| Age (mo) | Understood | 90 % HDI   | Spoken | 90 % HDI       | q (median) | 90 % HDI       |
| -------: | ---------: | ---------- | -----: | -------------- | ---------- | -------------- |
|       12 |         20 | [13, 27]   |    0.1 | [0.04, 0.26]   | 0.007      | [0.002, 0.013] |
|       18 |         43 | [32, 57]   |    0.9 | [0.40, 1.56]   | 0.021      | [0.010, 0.034] |
|       24 |         87 | [64, 111]  |    4.2 | [1.99, 6.67]   | 0.048      | [0.025, 0.074] |
|       30 |        140 | [105, 174] |   12.3 | [6.25, 19.00]  | 0.089      | [0.050, 0.132] |
|       36 |        183 | [140, 225] |   33.2 | [18.0, 48.6]   | 0.182      | [0.112, 0.260] |
|       42 |        218 | [170, 266] |   78.4 | [49.5, 109.2]  | 0.363      | [0.248, 0.481] |
|       48 |        255 | [201, 308] |  134.8 | [93.7, 175.8]  | 0.534      | [0.408, 0.654] |
|       54 |        298 | [241, 357] |  192.1 | [145.3, 244.0] | 0.651      | [0.535, 0.759] |
|       60 |        344 | [281, 408] |  258.0 | [201.2, 314.1] | 0.757      | [0.648, 0.851] |
|       66 |        392 | [324, 462] |  320.9 | [258.7, 383.0] | 0.829      | [0.722, 0.922] |
|       72 |        435 | [361, 510] |  362.1 | [294.2, 428.3] | 0.844      | [0.721, 0.949] |
|       78 |        473 | [389, 555] |  380.3 | [303.8, 456.1] | 0.820      | [0.661, 0.962] |
|       84 |        506 | [417, 599] |  398.1 | [307.3, 482.0] | 0.804      | [0.621, 0.984] |
|       90 |        536 | [440, 632] |  432.3 | [339.1, 521.7] | 0.825      | [0.645, 0.996] |

(Source: `output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/posterior_summary_u.csv`,
`posterior_summary_s.csv`, `posterior_summary_q.csv`.)

![VG09B joint trajectory: median understood and spoken vocabulary by
age for a typical child with DS, with 90 % HDI bands for the typical
DS trajectory.
](../output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/joint_trajectory_hdi.png){#fig-vg09b-joint
fig-align="center" width=85%}

![VG09B production ratio `q = p_S / p_U` against age (DS).
](../output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/production_rate.png){#fig-vg09b-q
fig-align="center" width=85%}

### Production rate against words understood

For families and clinicians the comparison that matters most is at
_matched comprehension_: for a given vocabulary that a child
understands, what fraction do they typically produce? VG09B's curve
is steeper than VG07's in the middle of the comprehension range —
i.e. production catches up over a narrower comprehension window once
it starts catching up at all — but starts later. This sharpens the
already-substantial DS–TD divergence above ~150 understood words.

Here too, `q = p_S / p_U` means "the fraction of understood checklist
words that are also spoken". Near the checklist ceiling, receptive
vocabulary can no longer increase within the instrument even if a
child continues to learn words outside it, so upper-range `q`
estimates should be interpreted as within-inventory production
ratios rather than ratios over the child's full vocabulary.

![VG09B: production rate against words understood (DS).
](../output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/production_rate_by_understood.png){#fig-vg09b-q-vs-u
fig-align="center" width=85%}

### The production lag: DS (VG09B) vs TD (VG06)

The chronological version of the same comparison sets the
DS production-ratio trajectory under VG09B against the TD reference
under VG06. The contrast is stark: a typical TD child crosses
`q = 0.5` (speaking half the words they understand) at around 17
months and reaches `q = 0.9` by 23 months. A typical DS child under
VG09B crosses `q = 0.5` at ~47 months — a chronological lag of
about two-and-a-half years — and does not reach `q = 0.9` at all
within the sampled age range. The DS curve peaks at ~0.84 at 72
months and dips back to ~0.82–0.83.

![Production ratio `q` against age — DS (VG09B) vs TD (VG06). Bands
are 90% HDI for the typical DS and typical TD trajectories
respectively. Horizontal dashed lines mark `q = 0.5` and `q = 0.9`.
](../output/comparisons/ds_td_q_by_age_vg09b.png){#fig-ds-td-q-age
fig-align="center" width=85%}

Approximate milestone crossings (median trajectory):

| Milestone  | TD (VG06)   | DS (VG09B)  | DS lag        |
| ---------- | ----------- | ----------- | ------------- |
| `q = 0.25` | 13.6 months | 38.2 months | ~25 months    |
| `q = 0.50` | 16.7 months | 46.8 months | ~30 months    |
| `q = 0.75` | 19.7 months | 59.6 months | ~40 months    |
| `q = 0.90` | 22.7 months | not reached | beyond sample |

A complementary matched-comprehension view using VG09B (the headline
DS joint model) instead of VG09 keeps the qualitative DS-vs-TD story
but shifts the DS curve down at low–middle comprehension. The DS
crossings move to higher comprehension values relative to the
VG09-based numbers in the meeting-review document.

![Production ratio `q` against words understood — DS (VG09B) vs TD
(VG06). The DS curve sits well to the right of TD's, indicating that
DS children accumulate substantially more receptive vocabulary
before their spoken vocabulary catches up.
](../output/comparisons/ds_td_q_vs_understood_vg09b.png){#fig-ds-td-q-u
fig-align="center" width=85%}

Both views show the same finding in different framings: the
comprehension–production asymmetry in DS is not a simple
chronological delay — it is _structurally larger and longer-lasting_
than the TD asymmetry, on both age and comprehension axes.

### Three-way q comparison: VG07, VG09, VG09B

The single most important plot in this report shows how the
production-ratio trajectory changes across the three competing DS
models. VG07 is the relevant baseline because it has no subject REs
on `q` and therefore is not subject to the GP/intercept/RE
redundancy that motivated VG09B.

![Production ratio `q` against age, DS: VG07 (study REs only) vs VG09
(study + subject REs on U and q, unanchored GP) vs VG09B (as VG09,
anchored GP and tighter priors).
](../output/comparisons/vg07_vg09_vg09b_q_by_age.png){#fig-q-three-way
fig-align="center" width=85%}

| Age (mo) | VG07 q | VG09 q | **VG09B q** |
| -------: | -----: | -----: | ----------: |
|       12 |  0.041 |  0.011 |   **0.007** |
|       18 |  0.085 |  0.033 |   **0.021** |
|       24 |  0.148 |  0.073 |   **0.048** |
|       30 |  0.211 |  0.133 |   **0.089** |
|       36 |  0.300 |  0.260 |   **0.182** |
|       42 |  0.443 |  0.474 |   **0.363** |
|       48 |  0.599 |  0.657 |   **0.534** |
|       54 |  0.711 |  0.769 |   **0.651** |
|       60 |  0.782 |  0.846 |   **0.757** |
|       66 |  0.833 |  0.890 |   **0.829** |
|       72 |  0.871 |  0.898 |   **0.844** |
|       78 |  0.904 |  0.882 |   **0.820** |
|       84 |  0.934 |  0.870 |   **0.804** |
|       90 |  0.958 |  0.882 |   **0.825** |

(Source: `output/comparisons/vg07_vg09_vg09b_q_by_age.csv`.)

Four patterns are worth flagging:

1. **VG07 is monotone-increasing across the whole age range** (0.04 at
   12 months to 0.96 at 90 months), and that monotonicity is the shape
   expected from published CDI data for DS.
2. **VG09 and VG09B both peak then dip.** VG09 peaks at ~0.90 at 72
   months and dips to ~0.88 at 90 months; VG09B peaks at ~0.84 at 72
   months and dips to ~0.82 at 90 months. The non-monotone tail
   appears after adding subject REs on `q`: where data are sparse the
   population mean can be pulled toward the prior, and the
   inverse-logit / Jensen interaction with the wide subject-RE
   distribution can bend the population curve downward. The dip is
   shared by VG09 and VG09B — i.e. it is not caused by the
   GP–intercept ridge that VG09B was designed to fix. VG07 is the
   monotone comparator.
3. **At mid-ages (36–66 mo) VG09B sits roughly 10 percentage points
   below VG07 and 10–13 percentage points below VG09.** At 48 months,
   VG07 / VG09 / VG09B give 0.60 / 0.66 / 0.53. The VG09 → VG09B shift
   is consistent with the A+D structural fix: in VG09 the unanchored
   `g_q` carried ~1.5 logit units of constant level at the reference
   age that the data could not distinguish from
   `intercept_q + slope_q · a_z(a_ref)`. Once the GP is forced through
   zero at 54 months, the linear trend uniquely defines the level
   there and the GP is restricted to genuine deviations from
   linearity; the tighter q-anchor priors also contribute to the
   changed trajectory.
4. **At very young ages (12–30 mo) VG07 is substantially higher than
   either VG09 or VG09B.** This is a side-effect of the subject REs on
   `q`: with a wide subject-level distribution in logit-`q` space, the
   population-level latent `q` at the lower edge of the data range
   shrinks toward zero via the same Jensen effect that produces the
   upper-tail dip.

There are _two distinct shifts_ hiding inside the three-way
comparison. **VG07 → VG09** is the effect of adding subject REs on
`q`; that change shows up most at the young end (shrinkage to zero
at 12–30 mo) and at the upper tail (the plateau / dip after 72 mo).
**VG09 → VG09B** is the parameterisation fix; that change shows up
most at mid-ages (the ~5–10 pp downward shift) because that is where
the GP-level redundancy was being absorbed. VG09B is _not_ "VG09
made to look like VG07" — it is "VG09 with the latent GP-level
identifiability resolved", and the remaining difference between
VG09B and VG07 is the genuine effect of adding subject REs on `q`.

### Implications for reported numbers

If VG09B replaces VG09 as the headline DS joint model, several
quoted numbers in the meeting-review document
(`notes/202605120945-meeting-project-review.md`) will move:

- The typical DS (median) trajectory under VG09B is very close to
  VG09 for understood vocabulary and lower for spoken vocabulary at
  older ages (Ey at 84 months: 506 vs 509 understood, 398 vs 434
  spoken). It is also slightly lower at very young ages.
- The matched-comprehension `q` table will shift downward by
  approximately 5–10 pp through the comprehension range that
  corresponds to mid-ages — i.e. the DS-vs-TD divergence becomes
  _larger_ under VG09B than under VG09 in that range, sharpening the
  qualitative story already in the meeting note.
- The K-fold LOSO numbers in this report are the VG09 numbers and
  have not been re-run against VG09B. The structural argument
  predicts they will not change much (the variance partition is
  preserved) but the empirical check is outstanding.

## Extending the model family: signed vocabulary

A natural next step is to bring _signed_ vocabulary into the model
family. Children with Down syndrome frequently use signing or
gesture as a bridge to spoken production, and parents and educators
routinely ask whether signing is _adding to_ a child's expressive
repertoire or _replacing_ the words they would otherwise produce. A
joint model that estimates the probability of a word being signed
(as well as understood and spoken) at each age would let us answer
that question directly. The problem is that the signing data we have
is not semantically uniform across studies — the most important
finding of this section is that _"signed"_ means something different
in different rows of the merged dataset.

### What the data supports — and the semantic heterogeneity

The DuckDB `vocab_combined` view used by the model code contains 414
age-valid rows with a non-null `signed` count, from 236 unique
subjects across five study labels:

| Study label | Signed rows | Unique subjects | Age range (mo) | Rows with signed > 0 | Rows with understood + spoken |
| ----------- | ----------: | --------------: | -------------- | -------------------: | ----------------------------: |
| `uk_01`     |         218 |             133 | 15–115         |                   63 |                            29 |
| `uk_02`     |          95 |              58 | 19–56          |                   95 |                            89 |
| `uk_04`     |          44 |              18 | 18–45          |                   35 |                            44 |
| `uk_05`     |          46 |              16 | 17–36          |                   37 |                            46 |
| `uk_06`     |          11 |              11 | 60–115         |                   11 |                            11 |

The previous 101-row count sometimes quoted from the flattened
`data/vocab_data_merged.csv` is not the right basis for modelling:
that CSV merge drops the UK 01 and UK 02 signing columns before
concatenation, leaving only Studies 6, 7 and 9. The DuckDB view is
the authoritative source for fitted models.

Of the 414 signed rows in the DuckDB view, 219 rows from 122 subjects
also carry both `understood` and `spoken` counts, so a joint
understood + spoken + signed analysis is mechanically feasible on
that subset.

The aggregate picture (ignoring the semantic issue for a moment)
for those 219 complete rows is:

| Age band (mo) | Rows | Rows with signed > 0 | Median signed | Median spoken | Median understood |
| ------------- | ---: | -------------------: | ------------: | ------------: | ----------------: |
| 0–24          |   42 |                   21 |           0.5 |             2 |                60 |
| 24–36         |   90 |                   80 |          48.5 |            16 |             180.5 |
| 36–48         |   49 |                   36 |           102 |            79 |               299 |
| 48–60         |   26 |                   26 |           209 |           284 |               465 |
| 60–72         |    5 |                    5 |           387 |           399 |               654 |
| 72–84         |    3 |                    3 |            23 |           400 |               576 |
| 84–120        |    4 |                    4 |            66 |           391 |               616 |

Two row-level patterns:

- In the 24–36-month window — where the bulk of the signing data
  sits — signed counts exceed spoken counts on the same row by about
  3:1 (median 48.5 signed vs 16 spoken).
- Across the complete signing subset, 123 of 219 rows have
  `signed > spoken`; the average row-level share
  `signed / (signed + spoken)` is ~0.55.

But these aggregate numbers conceal a **semantic mismatch** in how
"signed" is coded by each contributing study. The shared data-prep
notebook (`dsegroup/research-data-analysis/projects/vocabulary/notebooks/n000-data-preparation.ipynb`)
makes this explicit:

| Source study                | What `signed` records                                                                                                                                                                                                                           | Construction                                                                                                                                                                                                                                     |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| UK 01 (EDG, 1990s–2000s)    | **Signed-only** — words signed but _not_ spoken. The original form codes each word with exactly one of `c` (comprehends), `v` (says), `s` (signs), so the categories are mutually exclusive at the per-word level.                              | `signed = sum(noun*s, …, verb*s)`, where the `*s` columns count words with the "signs" code only. `produced = spoken + signed` is therefore the total expressive count. The DuckDB view carries these signed counts; the flattened CSV does not. |
| UK 02 (EDG follow-on)       | **Total signed** — words signed, with or without also being spoken. The source has explicit `signed_only`, `signed_spoken`, `spoken_only` columns, but the current DuckDB view keeps only the aggregate `signed = signed_only + signed_spoken`. | Per-row totals from `signed_only` + `signed_spoken`; the decomposition is present in the source CSV but lost in `vocab_combined`.                                                                                                                |
| UK 04 (Mason-Apps; study 6) | Source column `signs`. Definition not documented in the prep code; provisionally treated as **total signed**. Provides 44 rows.                                                                                                                 | `signed = round(signs)` directly from source.                                                                                                                                                                                                    |
| UK 05 (Seager; study 7)     | Source column `signed`. Definition not documented in the prep code; provisionally treated as **total signed**. Provides 46 rows.                                                                                                                | `signed = signed` direct copy.                                                                                                                                                                                                                   |
| UK 06 (RLI; study 9)        | **Total signed** — the SPSS column is `CheckUnderAndSign`, "(words) understood and signed". This counts words signed regardless of whether they are also spoken. Provides 11 rows.                                                              | `signed = CheckUnderAndSign` after a rename.                                                                                                                                                                                                     |

The headline implication: a `signed = 50` value from UK 01 means
"50 words signed but not spoken"; a `signed = 50` value from UK 06
means "50 words signed, possibly also spoken". The two are **not on
the same scale**. The "signed-only" definition is a strict subset
of the "total signed" definition; the difference is the
"signed-and-spoken" overlap.

A second limitation compounds the first: the current model family
uses a single inventory size `N` per row to define the Beta-Binomial
success rate. The DuckDB view currently carries `survey_vocab_max`
for the signing rows (396/690 for UK 01, 800 for UK 02 and UK 06,
and 418 for UK 04/05), but those denominators need to be checked and
made explicit for a signing model. In particular, the UK 04/05
denominator is still marked `TODO` in `scripts/prepare_data.py`, and
the flattened CSV does not carry `form_max_spoken` /
`form_max_understood` for the 101-row signing subset.

### What the semantic mismatch implies for modelling

This is not a presentational nuisance; it determines which models
are even _coherent_ on the merged data.

1. **`signed + spoken` is not interpretable across studies.** For
   UK 01 EDG rows the sum equals total expressive vocabulary
   (signed and spoken are non-overlapping); for UK 06 RLI rows the
   sum double-counts every word that is both signed and spoken. Any
   model that uses `signed + spoken` as a derived quantity will
   produce uninterpretable population estimates _unless_ the
   convention is held constant across the rows being fitted.
2. **`q_G = p_G / p_U` (signed-probability ratio) means different
   things in different studies.** Under the signed-only convention,
   `q_G` answers _"of the words this child understands, what
   fraction do they sign but not speak?"_ Under the total-signed
   convention, `q_G` answers _"of the words this child understands,
   what fraction do they sign at all?"_ These are different
   developmental quantities — the first peaks and falls as signing
   gives way to speech, the second only ever rises (and converges
   to `q_U`) as the child acquires every modality of every word.
   Pooling rows across the two conventions and modelling a single
   `q_G(a)` would produce a curve with no clear interpretation.
3. **The cleanest joint model needs decomposed counts at the
   per-row level.** What we ideally want is, per row,
   `n_understood_only`, `n_spoken_only`, `n_signed_only`,
   `n_spoken_and_signed`, plus the inventory size and the observation
   convention. UK 02 has all of these in the original source but the
   DuckDB view drops the decomposition. UK 01 has a mutually
   exclusive instrument-specific decomposition (the per-word
   `*c/*v/*s` columns), with no observed "spoken and signed" category
   under that coding. UK 04, UK 05, UK 06 may or may not provide the
   full decomposition — we would need to audit the underlying source
   data.

### Implications for the candidate model structures

The data-prep heterogeneity sharpens the trade-offs between the
three options scoped earlier.

**Option S1 — Univariate "age → signed" baseline.** Mirror VG01
(spoken) and VG02 (understood) with a fitted GP + Beta-Binomial
trajectory for signed words. The semantic mismatch matters less
here than for the joint models, because S1 is descriptive: it just
asks "given each row's signed count, what is the smooth trajectory
of expected signed words by age?" It is still defensible if (a) we
fit it separately to the "signed-only" rows and the "total-signed"
rows, producing two distinct trajectories; or (b) we restrict to a
single convention. Cheap to fit; reveals whether the smooth curve
in either convention shows the expected rise-then-fall shape
(signed-only) or monotone climb (total-signed). Useful even on its
own for clinical reporting.

**Option S2 — Trivariate joint model "age → understood + spoken +
signed".** Extend VG05/VG07/VG09B's bivariate structure to three
coupled outcomes:

- `p_U(a)` — probability a word is understood;
- `p_S(a)` — probability a word is spoken, with `p_S ≤ p_U`
  enforced by `p_S = p_U · q_S(a)`;
- `p_G(a)` — probability a word is signed, again
  bounded above by `p_U` through `p_G = p_U · q_G(a)`.

The semantic problem is **decisive** for S2. If we fit S2 on rows
where `signed` is total-signed, `q_G` is the _signed-regardless_
ratio. If we fit on rows where `signed` is signed-only, `q_G` is
the _signed-but-not-spoken_ ratio. The two estimates cannot be
combined into a single coherent model without either (a) restricting
to one convention, (b) modelling the convention as a per-row
categorical variable with separate `q_G` functions per convention,
or (c) re-deriving the merged data so the decomposition
(`n_signed_only`, `n_signed_and_spoken`) is preserved per row, then
modelling each component separately.

**Option S3 — Modality-conditional production model.** Reframe the
question explicitly as _expressive modality choice given a word is
produced_: for each understood word the child can produce, what is
the probability they produce it (a) spoken only, (b) signed only,
(c) both, (d) neither? This is a categorical outcome per word — a
multinomial / Dirichlet-multinomial model — and would let us track
the _transition_ from sign-dominant to speech-dominant expression as
a first-class quantity rather than reading it off two parallel
ratios.

S3 is **the model the data actually wants**, _if_ we re-derive the
merged data to preserve each study's modality decomposition. UK 01
EDG provides a mutually exclusive coding (`*c/*v/*s` at the per-word
level); UK 02 records the four-way decomposition explicitly
(`signed_only`, `signed_spoken`, `spoken_only`, `understood_only`).
The current `vocab_combined` view throws away part of this
information for UK 02 and does not flag the UK 01 convention.
Recovering it would let S3 use the rows where modality is recorded
with honest semantics and an explicit observation model. Pursuing S3
is therefore primarily a data-preparation investment rather than a
likelihood-plumbing one.

### Practical questions to resolve before fitting

Four questions, in order:

1. **Re-derive the merged dataset so signing semantics are
   preserved.** UK 01 and UK 02 encode useful modality information at
   the source-data level, but the current flattened CSV drops their
   signing columns and the DuckDB view drops UK 02's decomposition.
   The first concrete piece of work is to extend the merged schema to
   carry `n_understood_only`, `n_spoken_only`, `n_signed_only`,
   `n_signed_and_spoken` where available, plus a flag for which
   definition `signed` follows for each study. Without this step, a
   pooled signing model is not on interpretable footing.
2. **Audit UK 04, UK 05 source data.** For Studies 6 and 7 the
   prep code does not document whether the source `signs` /
   `signed` column is signed-only or total-signed. The answer
   should be in the original survey instruments / data dictionaries
   for those studies. This is a 1-hour question for someone with
   access to the raw files; it materially changes which rows we can
   pool.
3. **Inventory denominators.** Verify and expose the CDI / MacArthur
   inventory size for every signing row. The DuckDB view already has
   `survey_vocab_max` for the current signing rows, but UK 04/05's
   value of 418 is still marked as needing confirmation in
   `scripts/prepare_data.py`, and a signing model should not depend
   on implicit or ambiguous denominators.
4. **Comparability across age bands.** Even after the semantic
   fix, age coverage is confounded with study and signing convention:
   UK 01 spans 15–115 months under a signed-only convention, UK 02
   spans 19–56 months with a decomposable total-signed convention,
   UK 04/05 contribute younger rows, and UK 06 contributes only 11
   older total-signed rows. Worth exploring whether the GP
   length-scale prior or the reporting age range needs special
   treatment for the signed outcome.

The recommended first experiment, once these questions are
resolved, is a **univariate "age → signed-only" Beta-Binomial
model on rows where signed-only counts are recoverable** — UK 01
directly, and UK 02 after reintroducing its source `signed_only`
column. If that fits cleanly, S3 on UK 01 + UK 02 is the natural
second step. Bringing UK 04/05/06 in requires either knowing which
convention they follow or accepting the cost of modelling them as
separate observation conventions with their own `q_G` functions.

## Conclusions

1. **Substantive scientific picture is stable across VG07 → VG08 →
   VG09 → VG09B.** DS vocabulary growth continues across the sampled
   age range; spoken production lags comprehension persistently; the
   DS–TD gap is structurally larger than a simple chronological
   delay; between-subject variation on the production ratio is the
   single largest source of heterogeneity in the family.
2. **VG09B is a defensible candidate to replace VG09 as the headline
   DS model.** Diagnostics are clean, the A+D parameterisation is a
   principled response to the GP–intercept ridge, and the variance
   partition is essentially identical to VG09. The mid-age
   production-ratio shift is compatible with the structural argument,
   but because VG09B changed both q-anchor priors and the GP anchor it
   should be reported as the result of the combined A+D variant rather
   than attributed to the GP anchor alone.
3. **The non-monotone tail of `q` beyond ~72 months is the main
   interpretive question that remains.** It is shared by VG09 and
   VG09B, while VG07 remains monotone, so it cannot be attributed to
   the A+D parameterisation fix. Plausible mechanisms include the
   wide subject-RE distribution interacting with sparse data at the
   upper end of the age range (Jensen on the inverse-logit), and
   measurement compression as children approach the finite checklist
   ceiling. This needs a deliberate investigation rather than a
   parameter tweak.
4. **The technical report needs updating for VG09B.** Commit
   `97023ac` already rewrote `docs/report/` into the `vgNN` chapter
   scheme and added chapters for VG07, VG08 and VG09. The remaining
   report gap is VG09B and the downstream model-comparison /
   discussion text if VG09B is promoted.

## Next steps

Ranked roughly by priority.

1. **Decide whether to promote VG09B to VG09 (the canonical name).**
   Evidence for promotion: clean diagnostics, preserved variance
   partition, structurally honest parameterisation. Cost: the
   technical report needs revising and the reported mid-age `q` values
   will move.
2. **If promoting, apply the GP anchor symmetrically to the rest of
   the family.** Specifically: `common.py` for the univariate
   VG01–VG04 models, `common_bivariate.py` for VG05–VG06, and the
   relevant VG07/VG08 definitions in `common_bivariate_re.py`. Those
   models all have the same GP–intercept ridge in principle; we just
   haven't seen it bite the diagnostics yet because they have fewer
   overlapping global components on each outcome. Consistency matters
   more than the marginal sampler win.
3. **Re-run K-fold LOSO with VG09B.** The structural argument
   predicts it will land essentially where VG09 did, but the
   empirical check is outstanding. 15 refits at `test` config, ≈ 40
   minutes wall time.
4. **Investigate the non-monotone q-tail.** This is independent of
   the A+D fix and is visible in VG09 and VG09B but not VG07. Options
   to consider:
   - tighten $\tau^{\text{subj}}_q$ (e.g. `HalfNormal(0.25)` instead of 0.5);
   - narrow the GP lengthscale prior so the GP cannot bend on
     decadal scales;
   - quantify how close observations and posterior trajectories are
     to the relevant CDI / MacArthur checklist ceilings;
   - restrict the query range to ages with substantive data and flag
     the upper tail explicitly as extrapolation and as within-checklist
     rather than total-vocabulary estimation.
5. **Update the technical report** (`docs/report/`):
   - add a VG09B chapter if the model is promoted, or document it as
     a named sensitivity / candidate model if not;
   - update `model-comparison.qmd`, `discussion.qmd` and any copied
     figures/tables so they do not mix VG09 and VG09B headline
     numbers;
   - re-render the report after the headline-model decision.
6. **Address the missing-understood-data root cause in future data
   collection.** 80 observations in the 40–60-month window have
   spoken data without matching understood data, predominantly from
   Studies 1 and 5. Future data collection or harmonisation should
   prioritise complete understood + spoken pairs in the 40–70-month
   range.
7. **Open GitHub issues for the items above.** The project currently
   has zero open issues; planning lives in notes and PR descriptions.
   For decisions of this weight (replace VG09 with VG09B, propagate
   the anchor to the rest of the family, VG09B report update)
   we want traceable tickets.
8. **Scope a signing extension — starting with data preparation.**
   The signing data in `vocab_combined` is _semantically
   heterogeneous_: in some contributing studies `signed` is
   signed-but-not-spoken, in others it is signed-regardless. The
   first concrete piece of work is to re-derive the merged schema so
   the available modality decomposition (`understood_only`,
   `spoken_only`, `signed_only`, `signed_and_spoken`) is preserved
   per row and the signing convention is flagged. UK 01 and UK 02
   carry the key source-level information, but the flattened CSV drops
   their signing columns and the DuckDB view drops UK 02's
   decomposition. In parallel, audit UK 04 / UK 05 / UK 06 source data
   to determine which convention each follows, and verify the
   form-level inventory denominators currently exposed as
   `survey_vocab_max`. _Only after these data-prep steps are settled_
   are the modelling options scoped in
   [Extending the model family](#extending-the-model-family-signed-vocabulary)
   worth coding. The recommended first model is then a univariate
   "age → signed-only" Beta-Binomial on rows where signed-only counts
   are recoverable — UK 01 directly, and UK 02 after reintroducing its
   source `signed_only` column.

## Pointers to source material

- VG09B findings note:
  `notes/202605141200-vg09b-findings.md`.
- Structural-options note that motivated VG09B:
  `notes/202605131500-vg09-structural-options.md`.
- Diagnostic investigation of VG09:
  `notes/202605131400-vg09-sampler-diagnostics.md`.
- Meeting-review document with the wider VG07–VG09 story:
  `notes/202605120945-meeting-project-review.md`.
- Per-model VG09B outputs (traces, figures, summary tables):
  `output/models/VG09B-age-understood-spoken-ds-re-subj-uq-anchored/`.
- Three-way q comparison data + figure:
  `output/comparisons/vg07_vg09_vg09b_q_by_age.{csv,png,svg}`.
- K-fold LOSO comparison (VG07 / VG08 / VG09):
  `output/comparisons/kfold_loso_compare.csv`.
- Source data-preparation notebook (defines per-study signing
  conventions referenced in the signing-extension section):
  `dsegroup/research-data-analysis/projects/vocabulary/notebooks/n000-data-preparation.ipynb`.
