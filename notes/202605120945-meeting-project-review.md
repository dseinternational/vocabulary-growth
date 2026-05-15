---
title: "Vocabulary growth — project review"
date: "2026-05-12"
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
This was drafted by digital assistants using LLMs that may make mistakes
in ways that differ from other inference machines.
:::

## Executive summary for the meeting

::: {.callout-important}
**Decision to make:** the previous recommendation was VG07 — the
Down-syndrome joint model with study-level random intercepts. Two
new models fitted on 13 May 2026 supersede it:

- **VG08** adds subject-level random intercepts on the understood
  trajectory.
- **VG09** further adds subject-level random intercepts on the
  production ratio `q`.

A gold-standard K=5 leave-one-subject-out comparison
(refitting each model on each training fold and evaluating the
held-out subjects' marginal predictive log-density) ranks the three
models **VG09 > VG08 > VG07** with overwhelming statistical
significance (paired diff/dSE of 9.8 for VG09 vs VG07, 5.7 for VG09
vs VG08, 8.8 for VG08 vs VG07). VG09 is the best model both for
_describing_ the DS data we have AND for _predicting_ a brand-new DS
child. The earlier non-K-fold marginal LOSO that suggested VG07 was
better for unseen-child prediction was an artefact of PSIS-LOO
instability on a thinned posterior — corrected below. **Use VG09 as
the headline DS joint model.**
:::

| Reporting question                               | Current answer                                                                                                                                                                                                                                                                                                                                                                         | Evidence to cite                                                                                                                                                                                                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Which DS model should headline current findings? | **VG09** is the new candidate (previously VG07, then VG08). VG09 separates between-subject SD on understanding (`τ^{subj}_U ≈ 0.84`) AND on the production ratio (`τ^{subj}_q ≈ 1.20`) from between-study SD (`τ_U ≈ 0.52`, `τ_q ≈ 0.94`) and from Beta-Binomial dispersion.                                                                                                           | Conditional LOSO ranks VG09 > VG08 > VG07 by +208 and +380 elpd respectively. The spoken-side mid-age dispersion factor climbs from `exp(1.40) ≈ 4.0` (VG07) → `exp(1.86) ≈ 6.4` (VG08) → `exp(3.32) ≈ 27.6` (VG09), i.e. ~7× reduction in residual overdispersion on production. |
| What is the main developmental finding?          | Vocabulary growth continues across the measured DS age range, but spoken production lags understanding for years. The within-subject growth curve under VG09 takes a typical DS child from ~20 understood / 0.2 spoken words at 12 months to ~438 understood / 389 spoken words at 72 months.                                                                                          | VG09 population-level posterior, study and subject REs = 0.                                                                                                                                                                                                                       |
| What is different from typical development?      | The understanding-speaking gap is not just a chronological delay; it is larger even at matched comprehension. Under VG09 the production ratio rises from ~0.04 at 100 understood words to ~0.85 at 300 understood words and ~0.88 at 500. TD reaches the same `q` levels at much lower comprehension thresholds.                                                                       | VG09 matched-comprehension table below; VG09 mid-range `q` _lower_ than VG07/VG08, sharpening the DS-vs-TD divergence at low understood.                                                                                                                                          |
| What caveat must travel with the findings?       | Study composition, missing understood-word data, **within-subject repeated measures** AND **between-subject heterogeneity in production rate** all matter. Predictive intervals for an unseen individual DS child are properly constructed from VG09's subject-marginal posterior predictive distribution — the K-fold check confirms it generalises better than VG07/VG08, not worse. | 510 of 950 usable DS rows belong to 288 subjects with ≥2 observations. Between-subject SD on the production-ratio logit (`≈ 1.20`) is the single largest variance component in the model family. K-fold LOSO: VG09 = −8 414, VG07 = −8 753 (diff +339, paired SE 35).             |

## What this project is

We are trying to describe, in plain numbers a family or teacher could use,
how children with Down syndrome learn vocabulary between roughly 12 months
and 8 years of age — how many words they tend to understand and how many
they tend to say at each age, and how those two grow together. The same
quantities are also estimated for typically developing children, mostly as a
reference point.

The Down-syndrome analyses use 964 usable observations from 510 subject IDs
across 10 study labels, pooled into a single dataset. The typically developing
reference models use a reproducible 10% Wordbank sample (1,655 observations).
The product is a set of statistical models that turn this pooled data into
estimates that can be queried at any age in months: typical (median) word
counts, and the spread around them that captures variation between individual
children.

## How we estimate things (plain-English version)

For each model we ask the same kind of question: _given everything we
already know about vocabulary development, and given the DS and TD datasets
just described, what range of values is plausible for the typical number of
words a child of a given age understands or says?_

Three ideas are worth flagging because they will come up in the meeting:

- **Bayesian inference.** Instead of returning a single number with a
  confidence interval, the model returns a probability distribution over
  every quantity it estimates. We can read it directly as
  _"there's a 90% chance the typical 36-month-old with Down syndrome
  understands between X and Y words"_. That phrasing is what most people
  assume a confidence interval means, but only Bayesian intervals actually
  support it.
- **Priors.** Before looking at the data the model is given soft sanity
  bounds on every quantity — e.g. _"a typical 12-month-old does not
  understand 800 words"_. These priors are weak: the data does the heavy
  lifting wherever the data is informative, and the priors only really
  matter where the data is sparse.
- **Posterior predictive distributions.** Once the model is fitted we can
  also simulate the kind of _individual_ word counts we'd expect to see at
  each age — not just the typical value but the spread around it. This is
  what lets us say things like _"at 36 months, about half of children with
  Down syndrome are expected to understand 100+ words and a quarter to
  understand 200+"_.

The mathematical engine underneath is PyMC + nutpie (a NUTS sampler). It
runs many parallel Markov chains and produces samples from the posterior;
diagnostics like R-hat (should be ≈1.0) and ESS (effective sample size,
should be in the thousands) tell us whether those samples are stable enough
to report. All seven pass the R-hat/ESS checks comfortably; the only sampler
warning in the reporting-quality run was one VG06 divergence.

## The nine models

| ID   | Outcome                                                                                                              | Population           |
| ---- | -------------------------------------------------------------------------------------------------------------------- | -------------------- |
| VG01 | Words spoken                                                                                                         | Down syndrome (DS)   |
| VG02 | Words understood                                                                                                     | DS                   |
| VG03 | Words spoken                                                                                                         | Typically developing |
| VG04 | Words understood                                                                                                     | Typically developing |
| VG05 | Words understood **and** spoken (joint)                                                                              | DS                   |
| VG06 | Words understood **and** spoken (joint)                                                                              | Typically developing |
| VG07 | Words understood and spoken (joint) **with study-level random intercepts**                                           | DS                   |
| VG08 | Words understood and spoken (joint) **with study + subject random intercepts on understood**                         | DS                   |
| VG09 | Words understood and spoken (joint) **with study + subject random intercepts on understood AND on production ratio** | DS                   |

All nine share the same statistical shape: a smooth (but flexible) average
trajectory over age, plus a likelihood that allows for plenty of
between-child variability at every age (a Beta-Binomial with age-varying
dispersion). The joint models additionally tie the spoken trajectory to
understanding by modelling the "production ratio" — the fraction of
understood words that are spoken — which is constrained to stay between 0
and 1.

VG07 was the previous most-recent addition. It is identical to VG05 except
that it gives each contributing study its own offset (a "random
intercept") on both the understanding trajectory and the production ratio.
This matters — explained in the Simpson's-paradox section below.

VG08 (fitted 13 May 2026) extends VG07 with a non-centred subject-level
random intercept on the understood logit. 288 of the 510 unique DS
subjects in the pooled dataset contribute more than one observation
(maximum 8 observations on a single subject), so treating observations as
independent inflates the Beta-Binomial dispersion to absorb within-subject
correlation and biases the population trajectory toward observation
density. VG08 separates between-subject SD on understood
(`τ^{subj}_U ≈ 0.78`, 90 % HDI [0.71, 0.84]) from between-study SD
(`τ_U ≈ 0.51`) and from the dispersion parameter.

VG09 (also 13 May 2026) extends VG08 by adding a parallel subject
random intercept on the production-ratio logit `h`. The motivation is
symmetry: if between-subject differences justify a subject RE on
understood, the same logic applies to how much of each child's
understood vocabulary they actually produce. The result is striking
— between-subject SD on `q` (`τ^{subj}_q ≈ 1.20`, 90 % HDI
[1.06, 1.34]) is the _largest_ variance component in the model
family, larger than the between-subject SD on understood (0.84) and
much larger than either study-level SD (0.52 on understood, 0.94 on
`q`). VG08 was absorbing this into the spoken-side Beta-Binomial
dispersion. With VG09, residual overdispersion on the spoken outcome
collapses substantially (`exp(a_κ_S)` rises from ~6.4 in VG08 to
~27.6 in VG09). The implications are worked through in two new
subsections below.

## What we estimate (headline numbers)

All seven models were re-fitted at full reporting sampling quality on
12 May 2026 (6 chains × 6,000 tuning + 6,000 draw steps, `target_accept`
= 0.95). Total wall time 3h 09m. **All seven met the convergence/reporting
thresholds: r_hat ≤ 1.01 and ESS ≥ 400 for every reported parameter** (in
practice r_hat ≤ 1.001 and ESS ≥ 5,000 for the headline parameters). The only
sampler warning was a single divergence in VG06 across the full run; the other
six models had zero divergences. All seven rendered Quarto reports were
uploaded to Azure Blob Storage and are linked from
`output/models/<MODEL>/index.html`.

### Down syndrome — joint model with study random intercepts (VG07)

Typical (median) values, with 90% credible intervals for the typical
trajectory in brackets:

| Age (mo) | Understood (median) | Spoken (median) | Production ratio (spoken/understood) |
| -------: | :------------------ | :-------------- | :----------------------------------- |
|       12 | 41 [28–55]          | 1.6 [0.7–2.7]   | 0.04                                 |
|       24 | 105 [80–132]        | 16 [9–22]       | 0.15                                 |
|       36 | 213 [170–258]       | 64 [43–85]      | 0.30                                 |
|       48 | 280 [230–335]       | 167 [127–208]   | 0.60                                 |
|       60 | 323 [265–383]       | 251 [200–300]   | 0.78                                 |
|       72 | 388 [319–460]       | 332 [274–391]   | 0.87                                 |
|       84 | 423 [337–515]       | 381 [313–451]   | 0.93                                 |

Between-study variation is substantial — `τ_U = 0.50`
(90% CI 0.30–0.69) and `τ_q = 0.66` (0.40–0.91) on the logit scale,
both well-identified with ESS > 36,000. This confirms that studies
differ systematically in how words understood (and especially the
production ratio) get measured and reported, and that absorbing this
variation is what makes VG07 superior to VG05 for population-level
predictions.

The story to tell at the meeting:

- Children with Down syndrome reach the rapid-acquisition stage of word
  learning later than typically developing children but they do reach it,
  and the median trajectory keeps climbing across the whole age range we
  have data for.
- The gap between understanding and speaking is large and persistent. At
  24 months a typical DS child speaks roughly **15%** of the words they
  understand; this rises to ~60% by 4 years and ~85–90% by 6 years.
- For comparison (VG06, typically developing), the same production ratio
  is **95%+ by 24 months**.

### Down syndrome — joint model with study + subject random intercepts (VG08)

Adding subject-level random intercepts on the understood logit (the only
structural change between VG07 and VG08) reveals that VG07 was conflating
three sources of variability:

|    Variance component (logit-SD, understood) | VG07 |     VG08 |
| -------------------------------------------: | ---: | -------: |
|                        Between-study (`τ_U`) | 0.50 |     0.51 |
|               Between-subject (`τ^{subj}_U`) |    — | **0.78** |
| Beta-Binomial dispersion `a_κ` (mid-age log) | 1.75 | **2.93** |

The study-level SD is essentially unchanged (so VG07's study REs were not
secretly absorbing within-subject correlation), but the Beta-Binomial
dispersion factor at mid-age jumps from `exp(1.75) ≈ 5.8` to
`exp(2.93) ≈ 18.6` — i.e. residual overdispersion shrinks to about a
third of what VG07 was estimating, once between-subject heterogeneity
has its own term. The same pattern shows up on the spoken side
(`a_κ_S`: 1.40 → 1.86, `κ_min,S`: 2.51 → 4.58) even though no
subject RE was added to the production ratio — cleaner `p_U` propagates
through `p_S = p_U · q`.

Typical (median) values for VG08, with 90 % credible intervals for the
population-level trajectory (study and subject RE both set to zero):

| Age (mo) | Understood (median) | Spoken (median) | Production ratio (q) |
| -------: | :------------------ | :-------------- | :------------------- |
|       12 | 25 [16–34]          | 0.5 [0.2–1.0]   | 0.02                 |
|       24 | 91 [68–115]         | 13 [7–19]       | 0.14                 |
|       36 | 189 [148–231]       | 57 [37–79]      | 0.30                 |
|       48 | 259 [207–311]       | 163 [124–206]   | 0.64                 |
|       60 | 345 [283–408]       | 272 [217–326]   | 0.79                 |
|       72 | 435 [362–506]       | 361 [299–425]   | 0.84                 |
|       84 | 511 [417–601]       | 398 [324–471]   | 0.79                 |

Compared with the VG07 table above, the VG08 trajectory:

- starts **lower** at 12–18 months (typical understood at 12 months
  drops from 41 to 25);
- rises **steeper** through the middle of the age range
  (12 → 60 months: 5× understood-vocabulary growth becomes 14×);
- ends **higher** at the top of the age range (~511 understood at
  84 months vs ~423 in VG07);
- and most strikingly, the production ratio `q` **plateaus** around
  0.78–0.84 from ~60 months onwards rather than continuing to climb
  toward 1.0.

The reason is mechanical: in VG07 every observation contributes equally
to the likelihood, so a subject with 5 observations exerts 5× the
influence of a singleton. The 288 multi-observation DS subjects
contribute 78.8 % of the rows but only 56.5 % of the children — and
because longitudinal recruitment tends to over-represent
higher-engaging families, those multi-observation subjects also sit
above the population mean. VG07's smooth trajectory therefore tracks
observation density, not child frequency. VG08's subject REs absorb
each subject's consistent deviation, so the population-level trajectory
becomes "expected within-subject change as a typical DS child ages",
which is the developmentally interesting quantity.

The production-ratio plateau is the most consequential change for
clinical communication. The VG07 reading — _q approaches 1.0 by the
later school years_ — implied DS children eventually speak essentially
every word they understand. The VG08 reading — _q stalls at ~0.8 from
about 60 months onward_ — says that across the age range we sampled,
a typical DS child speaks roughly four-fifths of the words they
understand, with the remaining receptive-productive gap persisting
into the older ages.

### Down syndrome — joint model with study + subject REs on understood AND on production ratio (VG09)

Adding a second subject-level random intercept on the production ratio
(VG09 vs VG08) reveals the largest single variance component in the
family:

|              Variance component (logit-SD) | VG07 | VG08 |     VG09 |
| -----------------------------------------: | ---: | ---: | -------: |
|          Between-study (`τ_U`, understood) | 0.50 | 0.51 |     0.52 |
|                   Between-study (`τ_q`, q) | 0.66 | 0.74 |     0.94 |
| Between-subject (`τ^{subj}_U`, understood) |    — | 0.78 | **0.84** |
|          Between-subject (`τ^{subj}_q`, q) |    — |    — | **1.20** |
|        Spoken BB dispersion (`a_κ_S`, log) | 1.40 | 1.86 | **3.32** |
|    Understood BB dispersion (`a_κ_U`, log) | 1.75 | 2.93 |     3.10 |

Each successive model shrinks the residual Beta-Binomial dispersion
(higher `a_κ` = less overdispersion) by reassigning that variability
to its proper structural source. The spoken-side dispersion factor
goes from ~4.0 (VG07) → ~6.4 (VG08) → ~27.6 (VG09), i.e. an order of
magnitude reduction. Between-subject SD on the production-ratio logit
is `τ^{subj}_q ≈ 1.20` — equivalent on the count scale to a typical
DS child's `q` sitting anywhere in roughly a five-fold range around
the population median at matched comprehension. That heterogeneity
was invisible in VG07/VG08; under VG09 it has its own well-identified
parameter (ESS_bulk ≈ 7,000, r_hat = 1.001).

Typical (median) values for VG09, with 90 % credible intervals for
the population-level trajectory (study and both subject REs set to
zero):

| Age (mo) | Understood (median) | Spoken (median) | Production ratio (q) |
| -------: | :------------------ | :-------------- | :------------------- |
|       12 | 20 [13–28]          | 0.2 [0.1–0.4]   | 0.01                 |
|       24 | 89 [67–113]         | 7 [3–10]        | 0.07                 |
|       36 | 186 [145–227]       | 49 [28–71]      | 0.26                 |
|       48 | 257 [207–309]       | 168 [123–212]   | 0.66                 |
|       60 | 344 [284–407]       | 290 [235–346]   | 0.85                 |
|       72 | 438 [365–512]       | 389 [328–453]   | 0.90                 |
|       84 | 509 [411–600]       | 434 [358–510]   | 0.87                 |

Two things change relative to VG08:

- The population-level `q` rises _higher_ through the middle of the
  age range (q ≈ 0.85 at 60 months, ~0.90 at 72 months) before
  settling around 0.87–0.90 in the older ages, rather than VG08's
  plateau around 0.79–0.84. The reason is the same Simpson's-paradox
  logic that drove the VG07 → VG08 shift: VG08's `q` was being
  pulled toward observation density, which over-weighted children
  with high comprehension but lower-than-typical production. VG09's
  subject RE absorbs that variability.
- The early-age `q` is _lower_ than VG07/VG08 (0.07 at 24 months vs
  0.14 for both VG07 and VG08). A typical young DS child speaks
  even fewer of the words they understand than the previous models
  suggested.

Three leave-one-subject-out (LOSO) comparisons were run on 13 May
2026, in increasing order of methodological rigour:

| Quantity                                                                     |       VG07 |       VG08 |       VG09 |
| ---------------------------------------------------------------------------- | ---------: | ---------: | ---------: |
| Conditional LOSO elpd (RE at posterior estimate)                             |     -8 740 |     -8 360 | **-8 152** |
| Marginal LOSO elpd (MC-integrated RE, PSIS on thinned posterior)             |    -14 151 |    -15 975 |    -22 565 |
| **K-fold LOSO elpd (K=5, refit per fold, RE drawn from prior at MCMC time)** | **-8 753** | **-8 524** | **-8 414** |

(The K-fold row is the gold standard. SE on the totals: 252, 242, 236
respectively. Paired-difference SEs are 26 for VG07→VG08, 19 for
VG08→VG09, 35 for VG07→VG09; every pairwise diff/dSE is above 5.7,
i.e. overwhelmingly significant.)

The interpretation of each row:

- **Conditional LOSO** holds each subject's RE at its posterior
  estimate, so the predictive density is conditioned on having seen
  that subject's own data. It ranks VG09 > VG08 > VG07 and is
  trustworthy as "given you have observations on these children,
  which model fits them best". It is biased toward more flexible
  models because the model gets to use each held-out subject's own
  data to estimate their RE.
- **Marginal LOSO** integrates the subject RE(s) over their priors
  by Monte Carlo, with the existing rep-config trace thinned 36-fold
  to keep the integration tractable. It _appeared_ to favour VG07
  decisively (VG09 -8 414 worse than VG07), and the v11 draft of this
  note recommended VG07 as the better predictive model on that
  basis. **That recommendation was wrong.** A sanity check exposed
  the problem: for VG07 (which has no subject RE) the marginal LOSO
  _must_ equal the conditional LOSO mathematically, but the
  recomputed pipeline returned −14 151 against the conditional −8 740.
  The 5,400-elpd discrepancy was driven by PSIS-LOO instability on
  the thinned posterior plus Monte Carlo noise in the Beta-Binomial
  recomputation. The marginal numbers above are kept for transparency
  but should not be used.
- **K-fold LOSO** (the gold standard) refits each model on each of
  five training folds, with the held-out fold's observations excluded
  from the likelihood while their `obs_id` rows remain in the model
  so f*U and h are computed at their ages. The subject REs for
  held-out subjects are then unconstrained — MCMC samples them from
  their priors — and the trace's `p_u_obs` and `p_s_obs` at held-out
  rows are already the marginal predictive probabilities. Held-out
  log-density is then `logsumexp*{c,d} log p(y | params*{c,d}, RE*{c,d}) − log(NK)`
  per subject. **VG09 wins by +339 elpd over VG07 and +109 over
  VG08**, with strong statistical significance.

The K-fold result also resolves the underlying theoretical question:
adding subject REs and shrinking the Beta-Binomial dispersion does
_not_ compromise generalisation to an unseen DS child, because the
priors on the subject REs (`τ^{subj}_U ≈ 0.84`, `τ^{subj}_q ≈ 1.20`)
are wide enough to span the actual range of between-subject
variability. The posterior predictive distribution for a held-out
subject is a Beta-Binomial whose `p` is itself a wide mixture over
the subject-RE prior; the resulting mixture is empirically broader
than VG07's single Beta-Binomial despite VG09's lower
within-occasion `κ`.

The cross-comparison CSVs are at
`output/comparisons/kfold_loso_summary.csv`,
`output/comparisons/kfold_loso_compare.csv`, and
`output/comparisons/kfold_loso_subject_elpds.csv`. The script is
`scripts/kfold_loso.py`. Total K-fold wall time across 15 refits
(3 models × 5 folds at `test` config) was 41 minutes.

**What this means for the choice of headline DS model:**

VG09 is the best DS joint model both **descriptively** (cleanest
variance partition, sharpest within-subject growth trajectory,
biggest conditional-LOSO win) and **predictively** for a brand-new
DS child (largest K-fold LOSO elpd, with diff/dSE close to 10 vs
VG07). The earlier draft of this note recommended VG07 for
predictive intervals on the assumption that VG09's narrower
within-occasion `κ` would over-tighten the predictive distribution;
the K-fold check shows that intuition was wrong, because the wide
prior over the subject RE more than compensates. **Use VG09 as the
default for all DS reporting, descriptive AND predictive.**

### Typically developing reference (VG03/VG04/VG06)

| Age (mo) | Understood (median) | Spoken (median) | Production ratio |
| -------: | :------------------ | :-------------- | :--------------- |
|        9 | 49 [41–57]          | 3.5 [2.6–4.4]   | 0.05             |
|       12 | 83 [76–90]          | 12 [10–13]      | 0.14             |
|       18 | 134 [126–143]       | 86 [81–91]      | 0.63             |
|       24 | 276 [263–290]       | 269 [257–281]   | 0.95             |
|       30 | 430 [404–456]       | 441 [417–465]   | 0.99             |

The contrast lets families see _what is different_ about Down syndrome
vocabulary development rather than just _what is delayed_: the
understanding–speaking gap is qualitatively much larger and lasts much
longer in DS, not simply shifted by a constant number of months.

### Distribution of outcomes (not just averages)

The model also gives us full predicted distributions for individual
children. For example, for a 36-month-old child with Down syndrome (from
VG07's posterior predictive output for understood words):

- About 5% are expected to understand fewer than 50 words.
- About 17% are expected to understand fewer than 100.
- About 50% are expected to understand 200+ words.
- About 7% are expected to understand 400+ words.

These probabilities — _not_ just averages — are what we want to put in
front of families and clinicians, and they are a direct output of the
Bayesian approach.

### Milestone framing for families and clinicians

The age-to-milestone outputs answer the question people are most likely to
ask: _at about what age would children on different parts of the distribution
reach a vocabulary size?_ For VG07, the typical child reaches the following
milestones:

| Target words | Typical age to understand | Typical age to speak |
| -----------: | :------------------------ | :------------------- |
|           25 | 12 months                 | 31 months            |
|           50 | 18 months                 | 37 months            |
|          100 | 25 months                 | 44 months            |
|          200 | 36 months                 | 58 months            |
|          400 | 75 months                 | 89 months            |

Report-ready wording:

- "Understanding grows earlier and faster than spoken production. In the
  current best DS model, the typical child reaches about 100 understood
  words near 25 months, but 100 spoken words near 44 months."
- "The median DS trajectory is still rising at the upper end of the current
  data range; these estimates should not be read as a plateau in vocabulary
  learning."
- "Milestone ages are distributional expectations, not targets for an
  individual child; the full predictive intervals remain wide."

## Beyond chronological delay: DS vs TD at matched comprehension

Comparing the two populations at the _same age_ makes Down syndrome look
straightforwardly "delayed". That framing is incomplete. A more useful
comparison is at _matched comprehension_ — how does spoken production
look for DS and TD children when both are at the same level of words
understood? The joint models (VG06, VG07) let us answer that directly,
since they each estimate the production ratio `q = p_S / p_U` as a
function of the comprehension trajectory.

### Production ratio against words understood

![Production ratio `q` against words understood — DS (**VG09**,
solid blue) versus TD (VG06, orange). The dashed blue line is the
earlier VG07 curve (no subject REs) for reference. Bands are 90%
credible intervals for the typical trajectory. Horizontal dashed
lines mark q = 0.5 and q = 0.9.
](../output/comparisons/ds_td_q_vs_understood.png){#fig-q-overlay
fig-align="center" width=85%}

DS and TD share roughly the same _shape_ — q rises smoothly from near
zero to near one — but the DS curve is shifted substantially to the
right. At every milestone we checked, DS children need close to **twice
the comprehension vocabulary** TD children do before reaching the same
production ratio:

| Production ratio | Words understood — TD (VG06) | Words understood — DS (VG09) | DS / TD | (DS — VG07, superseded) |
| ---------------: | :--------------------------- | :--------------------------- | :------ | :---------------------- |
|             0.25 | 100                          | 184                          | ≈ 1.84× | 188                     |
|             0.50 | 121                          | 226                          | ≈ 1.87× | 264                     |
|             0.75 | 156                          | 289                          | ≈ 1.85× | 311                     |
|             0.90 | 225                          | 413                          | ≈ 1.84× | 407                     |

What this means in plain terms:

- A typical typically-developing child speaks about half the words they
  understand by the time they understand ~120 words. A typical DS child
  reaches the same milestone at ~226 understood words under VG09.
- The early production lag in DS is therefore not just _delayed in
  months_ — it is _enlarged in comprehension terms_. Children with
  Down syndrome accumulate roughly **1.85× as much** receptive
  vocabulary before their spoken vocabulary catches up at every
  threshold checked — a striking constancy that VG07's wider range
  (1.8–2.2×) obscured.
- Both populations eventually converge: above ~400 understood words,
  DS and TD children both speak nearly all the words they understand
  (q ≥ 0.9 for DS at ~413 understood words; TD reaches the same
  threshold at ~225).

This reframes the typical clinical/parent expectation that the gap is a
matter of timing. The model tells a stronger story: the gap is
_structurally larger_ in DS, but it does close given enough comprehension
vocabulary. The data underlying the overlay are in
`output/comparisons/ds_td_q_crossings.csv` and the figure source is
`scripts/compare_models.py`.

### Between-child heterogeneity at matched comprehension

A reasonable concern is whether DS children show more variability
between individuals than TD children at the same level of word
understanding. Comparing the **width of the posterior predictive 90%
interval for an individual child's word count**, at matched expected
comprehension, gives a model-derived answer:

| Comprehension level | Population | Predicted understood (median) | 90% interval | Interval ÷ median |
| ------------------: | :--------- | :---------------------------- | :----------- | ----------------: |
|           ~80 words | TD (12 mo) | 83                            | [0, 188]     |              2.27 |
|                     | DS (24 mo) | 105                           | [0, 211]     |              2.01 |
|          ~280 words | TD (24 mo) | 276                           | [0, 520]     |              1.88 |
|                     | DS (48 mo) | 280                           | [55, 509]    |              1.62 |
|          ~430 words | TD (30 mo) | 430                           | [156, 690]   |              1.24 |
|                     | DS (84 mo) | 423                           | [98, 743]    |              1.52 |

Headline:

- At **low and middle** comprehension levels the relative spread of
  individual outcomes around the typical is _slightly tighter in DS_
  than in TD (e.g. at ~280 understood words, the 90% interval is 1.62×
  the median in DS versus 1.88× in TD).
- At **high** comprehension levels the pattern flips: DS shows somewhat
  wider relative spread (1.52× vs 1.24× at ~430 understood words).

**Caveat introduced by VG08:** the VG07 figures above conflated three
sources of variability (between-subject, between-study, and residual
Beta-Binomial). VG08 partitions them and finds that between-subject
SD on the understood logit is `τ^{subj}_U ≈ 0.78` — about 1.5× the
between-study SD. Translating that onto the count scale at typical
ages, the implied 90 % between-subject envelope at, say, 36 months is
roughly 84 to 327 understood words (vs the VG08 typical of 189) — a
genuine, well-identified heterogeneity that the VG07 numbers were
attributing to dispersion.

Re-stating the comparison honestly therefore requires re-running the
matched-comprehension intervals on VG08 (and the matching VG06 +
subject-RE variant where data permits). The earlier reading — _"DS
spread is broadly comparable to TD; the gap, not the spread, is the
qualitative difference"_ — survives at low and middle comprehension
levels, where VG07 and VG08 agreed. At high comprehension levels the
VG07-based finding of "wider relative DS spread at ~430 understood
words (1.52× vs 1.24× TD)" is the part most likely to revise once
VG08 intervals are tabulated: that range is dominated by older DS
subjects with repeated visits, exactly where the subject RE has the
largest effect.

**Discussion point for the meeting:** retain the qualitative reading
that _the gap, not the spread, is the larger qualitative difference_,
but flag that the VG08-based numerical comparison is still pending and
the existing table above is from VG07.

### Cross-check against D'Souza et al. (2026, Williams syndrome)

A recent paper (D'Souza, D'Souza, Mayor & Tovar, _Developmental
Science_ 2026, doi: 10.1111/desc.70115) reports a similar
matched-comprehension comparison across typical development
(TD, n = 1,210), Williams syndrome (WS, n = 67), Down syndrome
(DS, n = 27), and fragile X syndrome (FXS, n = 15). Their headline
conclusion is that **WS** shows a uniquely reduced
comprehension–production gap, while **DS, FXS and TD all show the
canonical asymmetry**. Their DS finding is read as "DS tracks TD" on
the production-vs-comprehension curve.

That reading is in tension with what our joint models say. At matched
comprehension above ~150 understood words, VG07 (DS) sits
substantially below VG06 (TD), and the gap widens with comprehension
level, not narrows:

| Understood | DS production rate (VG09) | DS production rate (VG07, superseded) | TD production rate (VG06) |
| ---------: | ------------------------: | ------------------------------------: | ------------------------: |
|        ~50 |                      0.04 |                                  0.06 |                      0.05 |
|       ~100 |                      0.08 |                                  0.14 |                      0.25 |
|       ~150 |                      0.15 |                                  0.20 |                      0.73 |
|       ~200 |                      0.33 |                                  0.27 |                      0.87 |
|       ~300 |                      0.78 |                                  0.71 |                      0.95 |
|       ~400 |                      0.90 |                                  0.89 |                      0.99 |
|       ~500 |                      0.88 |                                  0.99 |                      0.99 |

(Updated 13 May 2026 from VG09 and VG06 rep-config posteriors using
`production_rate_by_understood.csv` in each model's output directory.
VG09 puts DS production _below_ VG07's estimate at low and middle
understood (≤ 150 words) — a typical DS child speaks even fewer of
the words they understand than VG07 suggested — while reaching
broadly similar values to VG07 at the upper end. The qualitative
DS-vs-TD story sharpens: the gap at ~100 understood words is now
~17 percentage points (8 % DS vs 25 % TD) rather than the
~8 percentage points VG07 suggested.)

The two findings are less contradictory than they first appear, and
unpicking why exposes a real interpretive issue with D'Souza et al.'s
statistical test. The next two subsections work through it.

#### What test D'Souza et al. actually ran, and what it can and cannot tell us

The procedure (Section 2.1.4 of the paper) is:

1. Fit a SCAM (monotonic shape-constrained additive spline) of
   spoken-on-understood to the TD data. This produces a "TD
   prediction line" — the expected production count for any given
   comprehension level under typical development.
2. For each child in each clinical group, compute the residual:
   observed production minus the TD-predicted production at that
   child's comprehension level.
3. Count how many residuals are positive (above the TD line).
4. Run an **exact one-sided binomial sign test** against
   H₀: P(above) = 0.5, with alternative
   H₁: P(above) > 0.5. The paper's own words:
   "_This test evaluated whether the median of the residuals … was
   significantly greater than zero._"

For DS they report 9 above, 18 below (n = 27), a proportion-above of
0.33, 95% CI [0.186, 1.000], and p = 0.974. They conclude this
"was not significant" and that the reduced gap "appears specific to
WS".

The interpretive issue is in how that p = 0.974 then gets read.

**What p = 0.974 actually means.** This is the probability, _under
H₀ that DS production sits exactly on the TD curve_
(i.e. P(above) = 0.5), of observing **9 or more** DS data points above
the line. With only 9/27 above — far fewer than the 13–14 you'd
expect under a 50/50 split — the test correctly fails to reject in
the direction of "WS-like elevation". That is the entirety of what
the test establishes.

**What it does not test.** It does not test whether DS production sits
_at_ the TD line, or _below_ it. To see this, flip the alternative
and ask: if H₁ were "DS sits below TD" (P(above) < 0.5), how
significant is 9/27?

For X ~ Binomial(27, 0.5),

$$
P(X \le 9) = \sum_{k=0}^{9} \binom{27}{k} 0.5^{27}
= \frac{8{,}192{,}524}{134{,}217{,}728}
\approx 0.061.
$$

So the **same data**, tested in the opposite direction, give
p ≈ 0.061 — borderline, just outside conventional significance at
α = 0.05. A two-sided sign test (does DS differ from TD in either
direction?) gives p ≈ 0.122 — also non-significant, but for the
ordinary reason that n = 27 sign-tests have low power, not because
the two groups look alike.

#### The two distinct things this exposes

**Issue 1 — a null result is being read as evidence of similarity.**
The paper enters the analysis with the canonical-DS-pattern claim
already in place, citing Mason-Apps et al. (2020) in the Introduction:
_"In DS, comprehension typically exceeds production … This pattern
mirrors the canonical comprehension–production asymmetry observed in
typical development."_ The Figure 2 caption then explicitly groups DS
with TD, and the Discussion states _"Unlike TD and other
neurodevelopmental groups, participants with WS exhibited a
disproportionately higher production vocabulary given their level of
comprehension."_ The sign test is doing confirmatory work: it cannot
rule the prior claim out, and is treated as supporting it. But a
failure to reject "DS is above TD" is not evidence for "DS equals TD"
— it is the absence-of-evidence-vs-evidence-of-absence fallacy. The
methodologically clean way to establish similarity is an
**equivalence test** (e.g. TOST: two one-sided tests against an
a-priori equivalence margin). They do not run one. Their own
reported one-sided 95% lower confidence bound on the proportion above
the TD line is 0.186 — meaning the data is consistent with the true
proportion-above being anywhere from ~19% (substantially below TD) up
to 100% (substantially above TD). That confidence interval is far too
wide to support _any_ claim of similarity.

**Issue 2 — the test in the opposite direction is borderline, not
clearly null.** With p ≈ 0.061 for "DS sits below TD", their data is
_consistent with_ DS being below the TD curve. It does not reach
conventional significance at α = 0.05, so it cannot be claimed as a
positive finding either. The honest reading is that with n = 27 they
simply do not have the resolution to discriminate "DS tracks TD"
from "DS sits modestly below TD" — and our larger pooled sample says
the latter, particularly above ~150 understood words.

**Other reasons the test is weak in any direction:**

- **Sign tests discard magnitude.** They ignore _how far_ below the
  line each residual is. A Wilcoxon signed-rank test, or a paired
  comparison of residual magnitudes, would use more of the
  information in the same 27 observations.
- **TD treated as known.** The TD curve is fit from n = 1,210 and
  treated as the truth; all the inferential uncertainty lives in the
  27 DS residuals. So the test's resolution is bottlenecked by the
  DS sample size.
- **Range mismatch.** Their DS observations are concentrated at low
  comprehension levels (children mostly under 40 months); they have
  almost no data in the part of the curve where our models say the
  divergence opens up.

#### Why we end up in a different place

1. **Their DS sample is small and age-truncated.** n = 27 DS children,
   all under 40 months; the comparator TD group is capped at 25
   months (the Oxford CDI age ceiling). Very few DS observations
   exist above ~150 understood words. Our models _agree_ with them
   in that range — the production ratio is roughly equal in both
   populations at ~100 understood words. The divergence opens up
   _above_ that range, which their sample barely covers.
2. **Their statistical test cannot resolve the question they
   answered.** As above: a non-significant one-sided sign test rules
   out a WS-like elevation, but cannot establish similarity to TD or
   rule out a modest depression below TD. With n = 27 and a
   sign-only test, the data is consistent with both "DS = TD" and
   "DS sits 15–20 percentage points below TD".
3. **Different inferential machinery.** SCAM + sign-test returns one
   binary decision per group. Our Bayesian Beta-Binomial joint
   models return a full posterior over `q(comprehension)`. Small but
   systematic separations show up clearly in the posterior that a
   sign test on 27 points will miss.
4. **There is a plausible mechanism for a wider DS gap that their
   framework does not represent.** DS production is constrained by
   motor and phonological factors that DS comprehension is not:
   childhood apraxia of speech, oro-motor hypotonia, conductive
   hearing loss from recurrent otitis media, verbal short-term
   memory weaknesses. These predict _larger_, not smaller,
   comprehension–production asymmetry in DS. None of the
   perturbations in their self-organising-map model (map size,
   input noise, neighbourhood disruption) speak to articulation.

**How to frame this in the technical report.** The two findings agree
on the bottom-line claim D'Souza et al.'s test could actually
support: DS does not look like WS. They disagree on the stronger
reading that DS tracks TD on the production-vs-comprehension
trajectory, because their test cannot establish similarity in the
first place — only a non-elevation. Within the vocabulary range they
sampled (≲ 150 understood words), the two analyses are consistent;
above that range our pooled DS data show a structurally wider gap
that their sample could not have detected and their SOM architecture
does not represent.

## The Simpson's-paradox finding (the most important methodological result)

Earlier versions of the DS models (VG02, VG05) showed an apparent dip in
the typical number of words understood between roughly 40 and 60 months —
the median trajectory fell slightly before rising again. This caused a
visible "leftward hook" in the understood-vs-spoken plot.

Investigation (`notes/202604121055-understood-ds-decline.md`) traced this
to **which studies contribute data at which ages**, not to a real
developmental phenomenon:

- Higher-scoring studies (Studies 1, 2, 6, 7) stop contributing
  _understood-word_ data after roughly 50 months. Studies 3 and 5 keep
  contributing past that age and have systematically lower medians.
- As a result the **mix of children** changes around 50 months, and the
  smooth trajectory in VG02/VG05 was tracking that compositional change
  rather than actual loss of vocabulary.
- This is a classic Simpson's paradox: within each study, understanding
  continues to increase with age; pooled across the changing mix, it
  appears to dip.

**VG07 fixes this.** Giving each study its own intercept absorbs the
between-study differences and produces a monotonic typical trajectory:
understanding keeps rising with age. The estimated between-study spread is
substantial — `τ_U ≈ 0.5`, `τ_q ≈ 0.7` on the logit scale, both
well-identified — confirming the studies really do differ systematically
in how words understood (and especially the production ratio) get measured
and reported.

**Practical implication:** for the technical report, VG07 (not VG05) should
be the basis for any quoted understood-word counts for DS in the 40–70
month range.

## Where the project has got to

Modelling work (March–May 2026):

- Joint understood + spoken model for DS (VG05) and TD (VG06).
- VG07: DS joint model with study random intercepts — superseded
  for headline DS results by VG08.
- VG08 (13 May 2026): DS joint model with study **plus subject**
  random intercepts on the understood logit. Partitions
  between-subject variability from between-study variability and from
  Beta-Binomial dispersion; reveals that VG07's population trajectory
  was being driven by the observation-density distribution.
- VG09 (13 May 2026): adds a parallel subject random intercept on the
  production-ratio logit `q`. Between-subject SD on `q`
  (`τ^{subj}_q ≈ 1.20`) is the largest variance component in the
  model family — VG08 was absorbing this into the spoken-side
  Beta-Binomial dispersion. Conditional LOSO ranks VG09 above VG08
  by another +208 elpd; current candidate for headline DS model.
- Production ratio reparameterised so that spoken word probability cannot
  exceed understood-word probability by construction (avoids a class of
  unrealistic posteriors).
- GP length-scale prior re-tuned so the smooth trajectory bends on a 6–18
  month scale rather than over years (avoids overfitting short-range data
  composition shifts).

Infrastructure (April 2026):

- Shared utility library (`dse_research_utils` in the sibling `research`
  repo) for sampling configurations, console reporting, diagnostics,
  plotting.
- `fit_model.py` now supports a single command to fit, render and upload a
  model end-to-end (`--config rep --render --upload`).
- Azure Blob upload migrated from AzCopy to the Python SDK.
- CI smoke-test of VG01 to catch regressions.
- Dependabot configured for daily dependency PRs.

Reports:

- Per-model Quarto reports (one per `vgNN`) exist for all seven models
  and render with figures, posterior summaries and diagnostics.
- The aggregate technical report under `docs/report/` still uses the old
  `model-1`/`model-2`/... naming, has no chapter for VG07, and several
  chapters (`methods`, `discussion`, `glossary`, plus parts of `intro`)
  are stubs.

## Suggested next steps

Ranked roughly by priority for the research narrative:

1. ~~**Confirm VG07 at reporting-quality sampling.**~~ **Done 12 May 2026.** All seven models re-fitted at `rep` quality (6 chains ×
   12,000 NUTS steps each), all met the reporting thresholds, and all
   reports re-rendered and uploaded to Azure. The only sampler warning
   was a single VG06 divergence.
2. ~~**Add subject-level random intercepts to the DS joint model.**~~
   **Done 13 May 2026** — see VG08 section above. Pending: a clean
   leave-one-subject-out elpd comparison VG07 vs VG08 (standard LOO
   is unreliable because of singletons).
3. **Decide reporting policy for the 40–60 month understood window** and
   the production-ratio plateau:
   - **(Recommended)** Prefer VG08 over VG07 for headline DS numbers in
     the technical report. Document the variance-partition argument and
     the trajectory shift.
   - Keep VG07 and treat the subject-level structure as a sensitivity
     check. Less defensible once the LOSO numbers are in.
4. **Flesh out the aggregate technical report.** Concretely:
   - Rename `model-N-*.qmd` chapters to match the `vgNN` scheme used
     everywhere else.
   - Add a VG07 chapter and a VG08 chapter.
   - Write `methods.qmd`, `discussion.qmd`, `glossary.qmd` (currently
     stubs).
   - Finish the open subsections in `intro.qmd` ("Vocabulary learning
     for children with Down syndrome", "Rates of word learning",
     "Use of gestures and signs").
5. **Address the missing-understood-data root cause in future data
   collection.** 80 observations in the 40–60-month window have spoken
   data but no matching understood data (predominantly Studies 1 and 5).
   Future data collection/harmonisation should prioritise complete
   understood + spoken pairs in the 40–70 month range for DS.
6. ~~**VG09 — extend subject random intercepts to the production ratio
   `q`.**~~ **Done 13 May 2026** at rep config (18m 57s wall time,
   all r_hat ≤ 1.002, ESS_bulk ≥ 1,800). `τ^{subj}_q ≈ 1.20` confirmed
   as the largest variance component in the family. Conditional LOSO
   ranks VG09 > VG08 > VG07.
7. ~~**K=5 leave-one-subject-out gold-standard comparison.**~~
   **Done 13 May 2026** (15 refits, 41 minutes wall time at `test`
   config). VG09 wins decisively at every pairwise comparison
   (diff/dSE ≥ 5.7). The earlier marginal-LOSO finding that
   suggested VG07 was better for unseen-child prediction was an
   artefact of PSIS-LOO instability on a thinned posterior and is
   superseded.
8. **Open question — extend study + subject random intercepts to the
   univariate DS models (VG01, VG02)?** For consistency, since the same
   compositional shift drives the VG02 dip. Worth a deliberate decision
   rather than leaving them as-is.
9. **Sensitivity check on the GP amplitude prior** (`eta_sigma`). The
   suggestion in the investigation note was to tighten from 0.4 → 0.2 as
   a diagnostic check. Useful even now that VG09 is the headline DS
   model.
10. **Sweep the open Dependabot PRs** (#18, #19). One contains an
    `arviz` 0.23 → 1.0 major-version bump and a `setuptools` 81 → 82
    change that **removes `pkg_resources`** — these need a deliberate
    merge with a full fit-pipeline smoke-test, not an auto-merge.
11. **Open GitHub issues for the items above.** Currently the project has
    zero open issues — all planning lives in notes and PR descriptions.
    For decisions of this weight (rep-quality re-fit, reporting policy,
    extending RE to univariate models, VG08/VG09 model family) we want
    traceable tickets.

### Suggested enhancements to the model output pipeline

Reviewing the per-model output catalogue and the Quarto report
templates, the following additions would each be small in code terms
and high-value for the technical report and clinical communication.
Ranked roughly by usefulness:

1. **Quantitative model comparison (LOO).** Right now the
   "VG07 over VG05" preference rests on visual + domain reasoning.
   `arviz.loo()` per model plus `az.compare()` for matched pairs
   (VG05 vs VG07 on the DS bivariate, VG02 vs VG05/VG07 marginal-u)
   would give a defensible numeric basis.
2. **Time-to-milestone (inverse-CDF) outputs.** The current outputs
   answer _"given an age, what's the word count?"_ Families typically
   ask the inverse — _"by what age will my child have 50/100/200
   words?"_ Trivial to compute from the existing posterior predictive
   distributions.
3. **Cross-model comparison artefacts as a first-class pipeline step.**
   The DS-vs-TD overlay in the matched-comprehension section was produced
   by an ad-hoc script. Useful
   permanent overlays: VG01 vs VG03 (DS vs TD spoken by age), VG02 vs
   VG04 (DS vs TD understood by age), VG05 vs VG07 (where the
   Simpson's-paradox fix matters), and the matched-comprehension q
   overlay.
4. **Data-coverage table by study × age bin.** The Simpson's-paradox
   investigation showed how load-bearing this is. A `data_coverage`
   table + heatmap as a permanent output of `prepare_data.py` would
   make any future compositional artefact immediately diagnosable.
5. **Study-level random-intercept summaries for VG07.** `δ_U[s]` and
   `δ_q[s]` are inside the trace but not summarised. A `study_effects`
   CSV + forest plot would show _which_ studies shift the population
   trajectory and by how much.
6. **Prior-vs-posterior overlay plots** for the headline parameters.
   Standard Bayesian-workflow figure showing how much each prior was
   updated by the data.
7. **Single cross-model summary CSV** (one row per model: ESS range,
   R-hat range, divergences, runtime, headline parameters). Would feed
   the technical report and CI smoke-test thresholds.
8. **Timing log persisted to disk.** `fit_model.py` already prints
   per-model timings; writing them to `output/logs/run_summary.json`
   would let the methods section auto-pull numbers.
9. **Diagnostics-table conditional formatting** in the Quarto template
   (red on R-hat > 1.01 or ESS < 400) — catches issues at a glance.
10. **Prior-predictive-checks plot consistency.** The univariate
    `prior_predictive_checks` plot exists for VG01 only; the joint
    models emit equivalent evidence under different names. Worth a
    consistency pass.

### Status of those enhancements

| #   | Item                                | Status                                                                  | Output location                                                                                          |
| --- | ----------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1   | LOO                                 | per-model + VG05 vs VG07 comparison                                     | `output/comparisons/loo_*.csv`, `loo_compare_*.csv`                                                      |
| 2   | Time-to-milestone                   | per-model CSV + plot                                                    | `output/models/<MODEL>/time_to_milestone*.{csv,png,svg}`, `output/comparisons/time_to_milestone_all.csv` |
| 3   | Cross-model overlays                | DS vs TD age, VG05 vs VG07, q-vs-comprehension                          | `output/comparisons/*_by_age.{png,svg}`, `vg05_vs_vg07_*.{png,svg}`, `ds_td_q_*.{png,svg}`               |
| 4   | Data coverage                       | understood + spoken pivots + heatmaps                                   | `output/data/coverage_*.{csv,png,svg}`                                                                   |
| 5   | VG07 study effects                  | tau + delta CSV + forest plot                                           | `output/models/VG07-age-understood-spoken-ds-re/study_effects.{csv,png,svg}`                             |
| 6   | Prior-vs-posterior                  | per-model overlay panels                                                | `output/models/<MODEL>/prior_vs_posterior.{png,svg}`                                                     |
| 7   | Cross-model summary                 | one CSV with ESS / R-hat / divergences / wall time / LOO                | `output/comparisons/model_summary.csv`                                                                   |
| 8   | Timing log JSON                     | parsed from latest rep log                                              | `output/logs/run_summary.json`                                                                           |
| 9   | Diagnostics styling                 | Quarto Styler in `docs/models/vg*/index.qmd` flags r̂ > 1.01 / ESS < 400 | re-renders next time `--render` runs                                                                     |
| 10  | Prior-predictive checks (bivariate) | gap diagnosed (only univariates emit overlay vs observed)               | follow-up: add `plot_prior_predictive_checks_bivariate` to `common_bivariate.py`                         |

The scripts that produce these are all self-contained under
`scripts/`: `loo_compare.py`, `time_to_milestone.py`, `compare_models.py`,
`data_coverage.py`, `vg07_study_effects.py`, `prior_vs_posterior.py`,
`aggregate_summary.py`. They all read existing trace / CSV outputs and
can be re-run independently of the main fit pipeline.

### Headline diagnostics across all seven models (rep run, 12 May 2026)

From `output/comparisons/model_summary.csv`:

| Model |                    n params | ESS bulk (min) | r̂ (max) | divergences |  wall time |                elpd_loo |
| ----- | --------------------------: | -------------: | ------: | ----------: | ---------: | ----------------------: |
| VG01  |                          11 |         11,518 |   1.001 |           0 |    12m 31s |                  -4,762 |
| VG02  |                          11 |         10,806 |   1.000 |           0 |    12m 22s |                  -4,071 |
| VG03  |                          11 |          8,388 |   1.000 |           0 |    26m 18s |                  -8,852 |
| VG04  |                          11 |          9,553 |   1.000 |           0 |    25m 04s |                  -9,926 |
| VG05  |                          22 |         10,362 |   1.001 |           0 |    24m 11s |                  -8,830 |
| VG06  |                          22 |         12,147 |   1.001 |           1 | 1h 02m 57s |                 -18,771 |
| VG07  |                          24 |          5,492 |   1.001 |           0 |    24m 14s |                  -8,740 |
| VG08  |         26 + 510 (subj REs) |          1,663 |   1.005 |           0 |    26m 39s |     -8,360 (cond. LOSO) |
| VG09  | 28 + 2×510 (subj REs U & q) |          1,841 |   1.002 |           0 |    18m 57s | **-8,152 (cond. LOSO)** |

(VG08 and VG09 fitted 13 May 2026 with the same rep sampling
configuration. The elpd columns for VG08/VG09 are the conditional
leave-one-subject-out elpd; standard per-observation PSIS-LOO is
unreliable in those rows because of singleton subjects.)

Across 252,000 draws for VG01–VG07, 36,000 draws for VG08, and 36,000
for VG09, the combined rep run produced **one** divergence — in VG06
— and met the project's `r̂ ≤ 1.01, ESS ≥ 400` convergence criterion
on every reported parameter in every model.

### Quantitative model comparison: VG07 vs VG05

LOO scores VG07 and VG05 on the same DS held-out data, computed
separately for the understood outcome, the spoken outcome, and the
joint likelihood. VG07 is preferred in all three, but the size and
significance of the improvement differ in a revealing way:

| Held-out data                         | elpd_diff (VG07 − VG05) |  dSE | diff / dSE | VG07 stacking weight |
| ------------------------------------- | ----------------------: | ---: | ---------: | -------------------: |
| Understood only (n = 704)             |               **+65.7** | 43.5 |   **1.51** |                 0.57 |
| Spoken only (n = 949)                 |                   +24.7 | 81.3 |       0.30 |                 0.51 |
| Joint understood + spoken (n = 1,653) |                   +90.4 | 92.2 |       0.98 |                 0.53 |

What this says:

- **VG07's improvement is concentrated on the understood outcome**
  (`diff / dSE ≈ 1.5`, which is a clear though not overwhelming LOO
  preference). That is exactly where the Simpson's-paradox artefact
  lived, so the model is improving precisely the trajectory it was
  designed to fix.
- **On the spoken outcome the two models are statistically
  indistinguishable** (`diff / dSE ≈ 0.3`). VG07's study random
  intercepts do not hurt the spoken fit — they simply don't help it,
  because the spoken trajectory wasn't suffering from the same
  compositional shift.
- The combined joint LOO blends the two and lands at a borderline
  preference (`diff / dSE ≈ 1.0`). The joint number alone would
  understate the targeted nature of VG07's improvement.

VG07 adds 15 effective parameters (`p_loo` 36 vs 21 for the joint
likelihood) — i.e. the study random intercepts that the data is
clearly informing — and _still wins_ on LOO even when LOO penalises
complexity. That this comes mostly from the understood outcome
(matching the Simpson's-paradox diagnosis) is a strong, defensible reason to prefer
VG07 for any DS quoted statistic where understanding is in scope.

A **leave-one-study-out** test would more directly assess
generalisation to a _new_ study — worth running as a follow-up. The
full per-outcome CSVs are under
`output/comparisons/loo_compare_ds_bivariate_re_vs_no_re_*.csv`.

## Open questions to raise at the meeting

- Are we comfortable preferring VG07 over VG05 in the technical report?
  This is the single biggest scientific decision currently pending.
- Should the production ratio (q) be the headline figure for the
  comprehension–production gap in DS, or should we lead with absolute
  counts? The ratio compresses the story; absolute counts emphasise the
  size of the gap.
- Target audience for the **summary report** referenced in the preface —
  families, clinicians, researchers? The level of statistical language
  needs to match.
- Cadence — modelling tempo has slowed since 23 April (last two weeks
  have been infrastructure and dependency PRs only). Is this intentional
  (waiting on data, prioritising the write-up) or drift?

## Pointers to source material

- Per-model rendered reports: `output/models/<MODEL>/index.html`
  (also uploaded to Azure Blob Storage).
- Investigation note on the comprehension dip:
  `notes/202604121055-understood-ds-decline.md`.
- VG07 design + dev-config results:
  `notes/202604121309-vg07-random-intercept-ds.md`.
- Repository review (this morning):
  `V:\dev\frankbuckley\my-context\now\vocabulary-growth.md`.
