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

For each model we ask the same kind of question: *given everything we
already know about vocabulary development, and given the data from these
800-odd children, what range of values is plausible for the typical number
of words a child of a given age understands or says?*

Three ideas are worth flagging because they will come up in the meeting:

- **Bayesian inference.** Instead of returning a single number with a
  confidence interval, the model returns a probability distribution over
  every quantity it estimates. We can read it directly as
  *"there's a 90% chance the typical 36-month-old with Down syndrome
  understands between X and Y words"*. That phrasing is what most people
  assume a confidence interval means, but only Bayesian intervals actually
  support it.
- **Priors.** Before looking at the data the model is given soft sanity
  bounds on every quantity — e.g. *"a typical 12-month-old does not
  understand 800 words"*. These priors are weak: the data does the heavy
  lifting wherever the data is informative, and the priors only really
  matter where the data is sparse.
- **Posterior predictive distributions.** Once the model is fitted we can
  also simulate the kind of *individual* word counts we'd expect to see at
  each age — not just the typical value but the spread around it. This is
  what lets us say things like *"at 36 months, about half of children with
  Down syndrome are expected to understand 100+ words and a quarter to
  understand 200+"*.

The mathematical engine underneath is PyMC + nutpie (a NUTS sampler). It
runs many parallel Markov chains and produces samples from the posterior;
diagnostics like R-hat (should be ≈1.0) and ESS (effective sample size,
should be in the thousands) tell us those samples are trustworthy. All
seven models pass these checks comfortably.

## The seven models

| ID   | Outcome                              | Population            |
| ---- | ------------------------------------ | --------------------- |
| VG01 | Words spoken                         | Down syndrome (DS)    |
| VG02 | Words understood                     | DS                    |
| VG03 | Words spoken                         | Typically developing  |
| VG04 | Words understood                     | Typically developing  |
| VG05 | Words understood **and** spoken (joint) | DS                |
| VG06 | Words understood **and** spoken (joint) | Typically developing |
| VG07 | Words understood and spoken (joint) **with study-level random intercepts** | DS |

All seven share the same statistical shape: a smooth (but flexible) average
trajectory over age, plus a likelihood that allows for plenty of
between-child variability at every age (a Beta-Binomial with age-varying
dispersion). The joint models additionally tie the spoken trajectory to
understanding by modelling the "production ratio" — the fraction of
understood words that are spoken — which is constrained to stay between 0
and 1.

VG07 is the most recent addition. It is identical to VG05 except that it
gives each contributing study its own offset (a "random intercept") on both
the understanding trajectory and the production ratio. This matters —
explained in section 5 below.

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
| 12       | 41   [28–55]        | 1.6   [0.7–2.7] | 0.04 |
| 24       | 105  [80–132]       | 16    [9–22]    | 0.15 |
| 36       | 213  [170–258]      | 64   [43–85]    | 0.30 |
| 48       | 280  [230–335]      | 167  [127–208]  | 0.60 |
| 60       | 323  [265–383]      | 251  [200–300]  | 0.78 |
| 72       | 388  [319–460]      | 332  [274–391]  | 0.87 |
| 84       | 423  [337–515]      | 381  [313–451]  | 0.93 |

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

### Typically developing reference (VG03/VG04/VG06)

| Age (mo) | Understood (median) | Spoken (median) | Production ratio |
| -------: | :------------------ | :-------------- | :--------------- |
|  9       |  49   [41–57]       |  3.5  [2.6–4.4] | 0.05 |
| 12       |  83   [76–90]       | 12    [10–13]   | 0.14 |
| 18       | 134  [126–143]      | 86   [81–91]    | 0.63 |
| 24       | 276  [263–290]      | 269  [257–281]  | 0.95 |
| 30       | 430  [404–456]      | 441  [417–465]  | 0.99 |

The contrast lets families see *what is different* about Down syndrome
vocabulary development rather than just *what is delayed*: the
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

These probabilities — *not* just averages — are what we want to put in
front of families and clinicians, and they are a direct output of the
Bayesian approach.

## Beyond chronological delay: DS vs TD at matched comprehension

Comparing the two populations at the *same age* makes Down syndrome look
straightforwardly "delayed". That framing is incomplete. A more useful
comparison is at *matched comprehension* — how does spoken production
look for DS and TD children when both are at the same level of words
understood? The joint models (VG06, VG07) let us answer that directly,
since they each estimate the production ratio `q = p_S / p_U` as a
function of the comprehension trajectory.

### Production ratio against words understood

![Production ratio `q` against words understood — DS (VG07, blue)
versus TD (VG06, orange). Bands are 90% credible intervals for the
typical trajectory. Dashed lines mark q = 0.5 and q = 0.9.
](../output/comparisons/ds_td_q_vs_understood.png){#fig-q-overlay
fig-align="center" width=85%}

DS and TD share roughly the same *shape* — q rises smoothly from near
zero to near one — but the DS curve is shifted substantially to the
right. At every milestone we checked, DS children need close to **twice
the comprehension vocabulary** TD children do before reaching the same
production ratio:

| Production ratio | Words understood — TD | Words understood — DS | DS / TD |
| ---------------: | :-------------------- | :-------------------- | :------ |
| 0.25             | 100                   | 188                   | ≈ 1.9× |
| 0.50             | 121                   | 264                   | ≈ 2.2× |
| 0.75             | 156                   | 311                   | ≈ 2.0× |
| 0.90             | 225                   | 407                   | ≈ 1.8× |

What this means in plain terms:

- A typical typically-developing child speaks about half the words they
  understand by the time they understand ~120 words. A typical DS child
  reaches the same milestone at ~265 understood words.
- The early production lag in DS is therefore not just *delayed in
  months* — it is *enlarged in comprehension terms*. Children with
  Down syndrome accumulate roughly twice as much receptive vocabulary
  before their spoken vocabulary catches up.
- Both populations eventually converge: above ~400 understood words,
  DS and TD children both speak nearly all the words they understand
  (q ≥ 0.9).

This reframes the typical clinical/parent expectation that the gap is a
matter of timing. The model tells a stronger story: the gap is
*structurally larger* in DS, but it does close given enough comprehension
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
|  ~80 words          | TD (12 mo) |  83  | [0, 188]  | 2.27 |
|                     | DS (24 mo) | 105  | [0, 211]  | 2.01 |
| ~280 words          | TD (24 mo) | 276  | [0, 520]  | 1.88 |
|                     | DS (48 mo) | 280  | [55, 509] | 1.62 |
| ~430 words          | TD (30 mo) | 430  | [156, 690]| 1.24 |
|                     | DS (84 mo) | 423  | [98,  743]| 1.52 |

Headline:

- At **low and middle** comprehension levels the relative spread of
  individual outcomes around the typical is *slightly tighter in DS*
  than in TD (e.g. at ~280 understood words, the 90% interval is 1.62×
  the median in DS versus 1.88× in TD).
- At **high** comprehension levels the pattern flips: DS shows somewhat
  wider relative spread (1.52× vs 1.24× at ~430 understood words).

The first half of that pattern is mildly counter to the usual clinical
intuition that DS is "more variable across the board". The Beta-Binomial
dispersion parameters (κ) tell the same story: VG07's `κ_min` is
slightly *larger* than VG06's for both outcomes (1.87 vs 1.67 for
understood, 2.51 vs 2.01 for spoken), where larger κ means *less*
spread. The right-hand tail at high comprehension is interesting in its
own right — it reflects that the older DS sub-population the model
draws on at high ages is itself diverse, and warrants a closer look in
the technical report.

**Discussion point for the meeting:** how to frame this in the
write-up. The honest summary is "between-child spread is broadly
comparable to TD over most of the range; the early production lag is
the larger qualitative difference, not heterogeneity per se."

## The Simpson's-paradox finding (the most important methodological result)

Earlier versions of the DS models (VG02, VG05) showed an apparent dip in
the typical number of words understood between roughly 40 and 60 months —
the median trajectory fell slightly before rising again. This caused a
visible "leftward hook" in the understood-vs-spoken plot.

Investigation (`notes/202604121055-understood-ds-decline.md`) traced this
to **which studies contribute data at which ages**, not to a real
developmental phenomenon:

- Higher-scoring studies (Studies 1, 2, 6, 7) stop contributing
  *understood-word* data after roughly 50 months. Studies 3 and 5 keep
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

Modelling work (March–April 2026):

- Joint understood + spoken model for DS (VG05) and TD (VG06).
- VG07: DS joint model with study random intercepts — the current best
  bivariate model for Down syndrome.
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

1. ~~**Confirm VG07 at reporting-quality sampling.**~~ **Done 12 May
   2026.** All seven models re-fitted at `rep` quality (6 chains ×
   12,000 NUTS steps each), all met the reporting thresholds, and all
   reports re-rendered and uploaded to Azure. The only sampler warning
   was a single VG06 divergence. The April dev-config conclusions for
   VG07 are confirmed: the comprehension dip is gone, the typical
   trajectory is monotonic, and the production ratio resolves smoothly
   through the full 12–84 month range.
2. **Decide reporting policy for the 40–60 month understood window.** Two
   options:
   - **(Recommended)** Prefer VG07 over VG05 for DS understood-word
     counts in the 40–70 month range. Document why (Simpson's paradox
     finding, between-study spread).
   - Keep VG05 and treat the dip as a data limitation in the
     methods/discussion. Less defensible scientifically.
3. **Flesh out the aggregate technical report.** Concretely:
   - Rename `model-N-*.qmd` chapters to match the `vgNN` scheme used
     everywhere else.
   - Add a VG07 chapter.
   - Write `methods.qmd`, `discussion.qmd`, `glossary.qmd` (currently
     stubs).
   - Finish the open subsections in `intro.qmd` ("Vocabulary learning
     for children with Down syndrome", "Rates of word learning",
     "Use of gestures and signs").
4. **Address the missing-understood-data root cause in future data
   collection.** 80 observations in the 40–60-month window have spoken
   data but no matching understood data (predominantly Studies 1 and 5).
   Future data collection/harmonisation should prioritise complete
   understood + spoken pairs in the 40–70 month range for DS.
5. **Open question — extend study random intercepts to the univariate DS
   models (VG01, VG02)?** For consistency, since the same compositional
   shift drives the VG02 dip. Worth a deliberate decision rather than
   leaving them as-is.
6. **Sensitivity check on the GP amplitude prior** (`eta_sigma`). The
   suggestion in the investigation note was to tighten from 0.4 → 0.2 as
   a diagnostic check. Useful even if VG07 is the final answer.
7. **Sweep the open Dependabot PRs** (#18, #19). One contains an
   `arviz` 0.23 → 1.0 major-version bump and a `setuptools` 81 → 82
   change that **removes `pkg_resources`** — these need a deliberate
   merge with a full fit-pipeline smoke-test, not an auto-merge.
8. **Open GitHub issues for the items above.** Currently the project has
   zero open issues — all planning lives in notes and PR descriptions.
   For decisions of this weight (rep-quality re-fit, reporting policy,
   extending RE to univariate models) we want traceable tickets.

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
   answer *"given an age, what's the word count?"* Families typically
   ask the inverse — *"by what age will my child have 50/100/200
   words?"* Trivial to compute from the existing posterior predictive
   distributions.
3. **Cross-model comparison artefacts as a first-class pipeline step.**
   The DS-vs-TD overlay in §5 was produced by an ad-hoc script. Useful
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
   CSV + forest plot would show *which* studies shift the population
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

| # | Item | Status | Output location |
| --- | --- | --- | --- |
| 1 | LOO | per-model + VG05 vs VG07 comparison | `output/comparisons/loo_*.csv`, `loo_compare_*.csv` |
| 2 | Time-to-milestone | per-model CSV + plot | `output/models/<MODEL>/time_to_milestone*.{csv,png,svg}`, `output/comparisons/time_to_milestone_all.csv` |
| 3 | Cross-model overlays | DS vs TD age, VG05 vs VG07, q-vs-comprehension | `output/comparisons/*_by_age.{png,svg}`, `vg05_vs_vg07_*.{png,svg}`, `ds_td_q_*.{png,svg}` |
| 4 | Data coverage | understood + spoken pivots + heatmaps | `output/data/coverage_*.{csv,png,svg}` |
| 5 | VG07 study effects | tau + delta CSV + forest plot | `output/models/VG07-age-understood-spoken-ds-re/study_effects.{csv,png,svg}` |
| 6 | Prior-vs-posterior | per-model overlay panels | `output/models/<MODEL>/prior_vs_posterior.{png,svg}` |
| 7 | Cross-model summary | one CSV with ESS / R-hat / divergences / wall time / LOO | `output/comparisons/model_summary.csv` |
| 8 | Timing log JSON | parsed from latest rep log | `output/logs/run_summary.json` |
| 9 | Diagnostics styling | Quarto Styler in `docs/models/vg*/index.qmd` flags r̂ > 1.01 / ESS < 400 | re-renders next time `--render` runs |
| 10 | Prior-predictive checks (bivariate) | gap diagnosed (only univariates emit overlay vs observed) | follow-up: add `plot_prior_predictive_checks_bivariate` to `common_bivariate.py` |

The scripts that produce these are all self-contained under
`scripts/`: `loo_compare.py`, `time_to_milestone.py`, `compare_models.py`,
`data_coverage.py`, `vg07_study_effects.py`, `prior_vs_posterior.py`,
`aggregate_summary.py`. They all read existing trace / CSV outputs and
can be re-run independently of the main fit pipeline.

### Headline diagnostics across all seven models (rep run, 12 May 2026)

From `output/comparisons/model_summary.csv`:

| Model | n params | ESS bulk (min) | r̂ (max) | divergences | wall time | elpd_loo |
| ----- | --: | --: | --: | --: | --: | --: |
| VG01 | 11 | 11,518 | 1.001 | 0 | 12m 31s | -4,762 |
| VG02 | 11 | 10,806 | 1.000 | 0 | 12m 22s | -4,071 |
| VG03 | 11 |  8,388 | 1.000 | 0 | 26m 18s | -8,852 |
| VG04 | 11 |  9,553 | 1.000 | 0 | 25m 04s | -9,926 |
| VG05 | 22 | 10,362 | 1.001 | 0 | 24m 11s | -8,830 |
| VG06 | 22 | 12,147 | 1.001 | 1 | 1h 02m 57s | -18,771 |
| VG07 | 24 |  5,492 | 1.001 | 0 | 24m 14s | **-8,740** |

Across 252,000 draws (6 chains × 6,000 tuning + 6,000 sampling × 7
models), the rep run produced **one** divergence — in VG06 — and met
the project's `r̂ ≤ 1.01, ESS ≥ 400` convergence criterion on every
reported parameter in every model.

### Quantitative model comparison: VG07 vs VG05

LOO scores VG07 and VG05 on the same DS held-out data, computed
separately for the understood outcome, the spoken outcome, and the
joint likelihood. VG07 is preferred in all three, but the size and
significance of the improvement differ in a revealing way:

| Held-out data | elpd_diff (VG07 − VG05) | dSE | diff / dSE | VG07 stacking weight |
| --- | --: | --: | --: | --: |
| Understood only (n = 704)            | **+65.7** | 43.5 | **1.51** | 0.57 |
| Spoken only (n = 949)                | +24.7     | 81.3 | 0.30     | 0.51 |
| Joint understood + spoken (n = 1,653) | +90.4    | 92.2 | 0.98     | 0.53 |

What this says:

- **VG07's improvement is concentrated on the understood outcome**
  (`diff / dSE ≈ 1.5`, which is a clear though not overwhelming LOO
  preference). That is exactly where the Simpson's-paradox artefact
  lived in §6, so the model is improving precisely the trajectory it
  was designed to fix.
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
clearly informing — and *still wins* on LOO even when LOO penalises
complexity. That this comes mostly from the understood outcome
(matching the §6 diagnosis) is a strong, defensible reason to prefer
VG07 for any DS quoted statistic where understanding is in scope.

A **leave-one-study-out** test would more directly assess
generalisation to a *new* study — worth running as a follow-up. The
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
