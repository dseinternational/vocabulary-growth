# Model inventory

> [!NOTE]
> Drafted by LLM-based AI tools (Claude Code/Opus 4.8 and Codex/GPT-5; dispersion prior note by Claude Code/Opus 5).

> [!WARNING]
> This is work in progress. All models and their output are preliminary and likely
> to change as the models evolve and further data are received.
> The nested outcome likelihood, TD repeated-measures correction, signing-source harmonisation, and Edgin inclusion revision from issue #163 require new reporting-quality fits. Numerical prose derived from older traces is stale until those fits pass the convergence gate and the reports are regenerated.
>
> Two further changes on 2026-08-03 widen that staleness. **Every Down syndrome model** is stale: `us_01` now comes from the Edgin item-level contributor files rather than the age-truncated Wordbank by-child export. That admits 50 Words & Gestures administrations at 19–27 months — the study's only comprehension observations in that band — removes a ceiling-saturated preparation batch of 64 children on its provenance, drops two empty administrations scored as zeros, and links children across the two CDI forms that Wordbank had split into separate `child_id`s, changing the subject index every child random effect is keyed on. `us_01` goes from 195 to 230 rows and from 78 to 126 comprehension observations. See [`notes/202608031500-edgin-out-of-window-administrations.md`](../../notes/202608031500-edgin-out-of-window-administrations.md), whose §7 records a reversal in how the out-of-window administrations are handled. **VG11, VG12 and VG13** are stale for a second reason: their reference pool now spans English, Italian and Spanish (European) so the Down-syndrome-versus-typically-developing comparison covers several languages on both sides. See [`notes/202608031500-td-romance-extension.md`](../../notes/202608031500-td-romance-extension.md).

This document lists every model in the `vocab_growth` family, what each one
targets, and how the models relate to one another. It is a map, not a
specification: for the full statistical detail of any single model, read its
report under `docs/models/vgNN/` (e.g. [`vg01/`](vg01/)) or the consolidated
[technical report](../report/index.qmd).

Each model is a thin module in
[`src/vocab_growth/models/`](../../src/vocab_growth/models) (e.g.
[`model_vg01.py`](../../src/vocab_growth/models/model_vg01.py)) that
selects a definition from
[`definitions.py`](../../src/vocab_growth/models/definitions.py) and dispatches to
a shared fitting engine. All procedural code (model building, sampling,
diagnostics, plotting, reporting) lives in the `common_*.py` engines; the models
themselves differ only by **population**, **outcome(s)**, **structure**, and
**priors**.

## How the models are named

Models are numbered `VG01`–`VG16` in roughly the order they were developed. The
numbering is historical rather than hierarchical — a higher number means "added
later", not "supersedes". Models build on one another along two main lineages
(joint understood + spoken for Down syndrome, and the signing models), with
several typically-developing counterparts fitted in parallel for comparison.

The three axes that distinguish the models:

- **Population** — children with **Down syndrome (DS)** or **typically
  developing (TD)** children.
- **Outcome** — words **understood**, words **spoken**, words **signed**, or a
  **joint** model of more than one outcome.
- **Structure** — whether the model adds study-level and/or subject-level random
  intercepts, GP anchoring, or a sign–speech association term on top of the
  baseline trajectory model.

## Inventory

| Model                  | Population | Outcome(s)                   | What it adds / its purpose                                                                                                                                                                                                |
| ---------------------- | ---------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [VG01](vg01/index.qmd) | DS         | Spoken                       | Baseline age → words spoken trajectory.                                                                                                                                                                                   |
| [VG02](vg02/index.qmd) | DS         | Understood                   | Baseline age → words understood trajectory.                                                                                                                                                                               |
| [VG03](vg03/index.qmd) | TD         | Spoken                       | TD counterpart to VG01. English-only reference pool (no random effects to absorb between-language variation).                                                                                                             |
| [VG04](vg04/index.qmd) | TD         | Understood                   | TD counterpart to VG02. English-only reference pool (no random effects to absorb between-language variation).                                                                                                             |
| [VG05](vg05/index.qmd) | DS         | Understood + spoken (joint)  | Baseline joint model; spoken modelled as a fraction `q(a)` of understood.                                                                                                                                                 |
| VG06 _(retired)_       | TD         | Understood + spoken (joint)  | TD counterpart to VG05; retired after the WS-comprehension data issue (see below) — superseded by VG13.                                                                                                                   |
| [VG07](vg07/index.qmd) | DS         | Understood + spoken (joint)  | VG05 + study random intercepts.                                                                                                                                                                                           |
| [VG08](vg08/index.qmd) | DS         | Understood + spoken (joint)  | VG07 + subject random intercepts on understood.                                                                                                                                                                           |
| [VG09](vg09/index.qmd) | DS         | Understood + spoken (joint)  | VG08 + subject random intercepts on the production ratio `q`.                                                                                                                                                             |
| [VG10](vg10/index.qmd) | DS         | Understood + spoken (joint)  | VG09 + per-draw GP anchor at 54 months (stabilisation); `q` anchors match VG09.                                                                                                                                           |
| [VG11](vg11/index.qmd) | TD         | Spoken                       | VG03 + dataset and child random intercepts + GP anchor at 19 months; one-administration-per-child sensitivity available. Reference pool widened to English + Italian + Spanish (European).                                |
| [VG12](vg12/index.qmd) | TD         | Understood                   | VG04 + dataset and child random intercepts + GP anchor at 19 months; one-administration-per-child sensitivity available. Reference pool widened to English + Italian + Spanish (European).                                |
| [VG13](vg13/index.qmd) | TD         | Understood + spoken (joint)  | Young TD joint model, ages 8–18 months; dataset and child random intercepts on understood and `q` + GP anchor at 13 months. Reference pool widened to English + Italian + Spanish (European).                             |
| [VG14](vg14/index.qmd) | DS         | Understood + spoken + signed | Adds signing as a third ratio `r(a)`; total expressive vocabulary derived assuming sign/speech independence given age.                                                                                                    |
| [VG15](vg15/index.qmd) | DS         | Understood + spoken + signed | VG14 + within-understood sign–speech association `psi` + study & subject random intercepts + VG10 stabilisation.                                                                                                          |
| [VG16](vg16/index.qmd) | DS         | Understood + spoken          | VG09 + a within-child cross-lag (prior understood → current `q`): earlier receptive → later expressive. Population-relative headline (≈ null); the within-child (RI-CLPM) contrast is biased by short-T with 2-wave data. |

### Model roles

`methods-workflow.qmd` defines a four-way status taxonomy — **model of record** (the current source for a stated estimand), **development step** (retained to show how structure was added, but not preferred for headline estimates), **TD reference** (a typically-developing comparison model with a distinct population role), and **superseded** (replaced after a documented structural or data problem) — and names this inventory as the source of truth for them. Recording them here, because until now the taxonomy was defined but never applied.

| Model            | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VG10             | **Model of record** for the Down syndrome joint understood + spoken estimands. Confirmed by the study owner, 2026-08-05. Supersedes the 12 May review's recommendation of VG09.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| VG05, VG07, VG08 | **Development steps.** The study owner's position, 2026-08-06: not expected to supply any reported number, being superseded by later models in the lineage.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| VG14             | **Development step.** Wholly replaced by VG15 for reporting; confirmed by the study owner, 2026-08-06. VG15 supplies everything VG14 does — including `p_any_indep`, the independence-assumption total expressive that is VG14's headline output — on a better-specified model. VG14's independence assumption is rejected by the data (psi = 1.78, 89% [1.29, 2.45], P(psi>1) = 0.998), though the numerical cost of that assumption is small (at most ~9 words on the 810 scale); what actually separates the two models is VG15's study and subject random effects, worth up to 38 words at 36 months. |

This settles part of the prior-data conflict recorded in [`notes/202608051500-report-critical-review.md`](../../notes/202608051500-report-critical-review.md) §4a. VG05, VG07, VG08 **and VG14** all carry `b_kappa_mag_s` around four standard deviations beyond its prior, with the posterior wider than the prior — a parameter pinned at a boundary is not an estimate. For the three development steps that is disclosable rather than fixable, because a development step supplies no number. **VG14 is the one that matters**, and whether it needs migrating to the two-anchor dispersion form follows from its role decision rather than from the diagnostic.

### VG15's two open flags, settled 2026-08-06

Both resolve to **disclose, not change**; the reasoning is in [`notes/202608060900`](../../notes/202608060900-three-prior-conflicts.md) §5b.

- **`kappa_sign` stays on the legacy dispersion form.** It is well identified (contraction 0.429), sits comfortably inside its prior (CDF 0.276), and its sign constraint is not binding — none of which was true of the four models migrated for the same parameter. The resulting asymmetry with VG14 is **two separate correct calls, not an inconsistency**; do not "tidy" it.
- **`ell_unit_sign` stays sampled.** It is genuinely unidentified (contraction 0.033), but fixing it changes nothing measurable — a maximum median shift of 0.0023 on `r(a)`, and +0.1% band width — and removing the signed GP is worse: it fails the hard convergence tier and strips the model's only honest signal of ignorance above 60 months.

Because neither is being fixed, the signing chapter **must** disclose four things: that `r(a)`'s peak age is fixed by construction rather than estimated; that the signed GP's length-scale is not identified, so the trajectory is parametric in practice; that psi is estimated from uk_02 alone and applied pool-wide; and that signed evidence stops around 60 months while the reported range runs to 115.

**The remaining role assignments are outstanding and are owner decisions**, tracked in #190. They should not be inferred from this file's ordering. Two constraints are already fixed and any assignment must respect them: VG06 is retired and superseded by VG13; and `scripts/fit_recovery.py` treats `vg10`, `vg12` and `vg15` as the headline set, so those three at minimum carry reporting weight. A superseded model never supplies a number in the findings, so these assignments have to be settled before the findings chapters are written.

### Exploratory, unregistered prototypes

VG17 and VG18 are exploratory sign-group comparison modules. They are deliberately excluded from `MODEL_REGISTRY`, `fit_model.py all`, and the numbered model inventory because they have not yet passed the specification and reporting workflow required of registered models. VG17 still uses the same harmonised signing-source rules as the registered signing models so exploratory comparisons cannot silently reintroduce non-comparable fields.

## Shared architecture

Despite their differences, every model is built from the same components:

- **Outcome** — every model reports vocabulary on an **810-item common reference inventory**, so DS and TD estimates sit on a comparable scale regardless of which checklist (MB-CDI Words & Gestures, Words & Sentences, Oxford CDI, etc.) produced them. Univariate, understood and marginal-fallback likelihoods use that 810-item denominator; paired, logically nested spoken and signed likelihoods use the observed understood count as their denominator.
- **Likelihood** — a **Beta-Binomial** with **age-varying dispersion**, so the
  degree of between-child heterogeneity can change across development rather than
  being fixed. In joint models, paired spoken and signed counts are modelled conditionally on the observed understood count; rows without usable understood data retain a marginal Beta-Binomial fallback. Every model with an empirical dispersion calibration states that prior at two reference **ages** (a floor plus the age term at each anchor) rather than as an intercept and a slope, so the prior does not shift meaning when the pool changes and the sign of the age trend is a consequence of the data rather than a constraint. See [`PRIORS.md`](PRIORS.md).
- **Mean trajectory** — on the logit scale, the expected proportion is a
  **linear developmental trend plus a Hilbert-Space Gaussian Process (HSGP)**
  term that captures smooth nonlinear departures from the trend. The linear trend
  is parameterised indirectly through the expected proportion at two **reference
  (anchor) ages**, which makes the priors interpretable.
- **Inference** — fitted with **PyMC** using the **nutpie** NUTS sampler, in an
  iterative Bayesian workflow (prior predictive checks → sampling → diagnostics →
  posterior predictive checks → summaries and plots).

### Interval reporting convention

Every model reports the posterior **median** with an inner **50%** and an outer **89%** credible interval, alongside the **full posterior** (density and posterior-predictive plots). Intervals are **equal-tailed (ETI, percentile-based) by default**; a documented short-list of strongly skewed or boundary-censored estimands — the sign–speech association `psi`, the Beta-Binomial concentration/dispersion `kappa`, and milestone/peak ages — is reported with **highest-density intervals (HDI)** instead. The single source of truth is `vocab_growth.intervals` (`DEFAULT_CI_PROB = 0.89`, `INNER_CI_PROB = 0.50`, `HDI_ESTIMANDS`), carried through the shared `ReportingConfiguration(ci_prob=0.89, interval_kind="eti")`. 89% is a deliberately non-special width: it is not a decision threshold, and its 5.5th/94.5th-percentile limits are more MCMC-stable than the 2.5th/97.5th limits of a 95% interval (McElreath 2020; Kruschke 2021, _Bayesian Analysis Reporting Guidelines_). Column contract: the posterior-summary tables (`posterior_summary_*.csv`) carry the median with an inner `*_ci50_lo`/`*_ci50_hi` and an outer `*_ci_lo`/`*_ci_hi` interval; plot-sidecar CSVs carry the median and the outer `*_ci_lo`/`*_ci_hi`, and additionally `*_ci50_*` for the trajectory plots that draw a nested inner band.

### Reporting ages: 6-monthly tables, whole-month companions

Each model definition carries `ages_query`, the canonical reporting ages — 6-monthly (12–90 months) for the Down syndrome models, 3-monthly (9–30) for the typically-developing ones. Those ages drive `posterior_summary_*.csv` and the report, and they are part of the model graph: the GP is evaluated at them, so changing `ages_query` changes the fitted model.

`ages_query` is shared by every outcome a model reports, but the Down syndrome outcomes are not observed over the same range: understood has 905 rows with a 95th percentile of 64 months and 15 at or above 72, against 1346 spoken rows with a 95th percentile of 78 and 51 at or above 84. A second field, `report_max_age_understood`, therefore caps what is reported for **comprehension** — the understood and `q` tables and the production-ratio figure — at 72 months on VG02, VG05, VG07–VG10 and VG14–VG16, while spoken (and signed, and the signed ratio `r`) keep the full grid. VG01 is production-only and is left alone; the typically-developing grids stop at 30 months, well inside their data.

Unlike `ages_query` this is post-processing and cannot move the posterior — refitting VG10 across the change at a fixed seed reproduced its diagnostics bit-for-bit. It is not free, though: the summary tables are written during the fit pipeline and `--render-only` re-renders Quarto against the CSVs already on disk, so a new cap only takes effect on a refit, and the field is part of the recorded definition, so output produced under a different value is reported as stale. See [`PRIORS.md`](PRIORS.md), "Reported age range for comprehension".

Alongside them each fit writes a **whole-month** companion — `posterior_summary_monthly[_<outcome>].csv` and the figure `expected_counts_by_month[_<outcome>]` — one row per whole month of age, with the same columns and the same `P(Y<=k)` bucket thresholds. These are derived from the _plot_ grid (`n_plot = 500`) rather than from added query ages, so they are pure post-processing: no change to the model graph, the HSGP domain, or the `query_id` dimension the report and the comparisons consume. Three consequences worth knowing before quoting a monthly figure:

- **Coverage is the plot grid's span, the full observed age range**, which for the Down syndrome models is 8–115 months (108 rows) — _wider_ than the 12–90 canonical ages, not narrower. Every canonical age has a monthly counterpart. The extra months at both tails are mostly data-free: 33 of the 108 rows have no observation in that month at all, which is why the `n_obs` column exists. Read a row with `n_obs = 0` as the model interpolating or extrapolating, not as evidence. The monthly companions deliberately keep this full span even for comprehension, so an understood table capped at 72 months sits beside a monthly understood table running to 115: the `n_obs` column is the guard the capped table has no room for.
- **Each row's stated month takes the nearest plot-grid point**, recorded in `grid_age_months` / `grid_offset_months`. At the default `n_plot` the grid step is about 0.21 months, so the offset never exceeds about 0.11 months — three days.
- **That offset makes the monthly figure differ slightly from the canonical figure at the same nominal age.** The difference is largest where growth is steepest: for VG10-family understood counts it is about **1.3% at 12 months**, 1.2% at 18, and under 0.4% from 24 months on. A monthly row is therefore the trajectory a few days either side of the stated month, not a second estimate of the canonical value — when the two must agree exactly, quote the canonical table.

An `n_obs` column counts the administrations that observed **that outcome** in each whole month (not all administrations — the outcomes have different coverage). The joint sign/speech engine (VG15) draws no predictive counts on the plot grid, so its monthly tables carry the expected count (`p_*`, `Ey_*`) only — matching what its query-age tables report — and the predictive columns are absent rather than zero-filled.

The monthly tables use one column schema across all fifteen models, with the outcome in the filename (`p_median`, `Ey_median`, …). Note that VG15's _canonical_ tables are the exception in the codebase: they prefix by outcome (`p_u_median`, `Ey_u_median`), so joining VG15's monthly and canonical tables needs the rename.

Note the two estimands the tables keep separate, because conflating them misleads badly: `Ey_*` is the **expected** count, a credible interval on the mean trajectory carrying parameter uncertainty only, which narrows toward zero width as more children are observed; `Y_*` and `P(Y<=k)` are the posterior **predictive** count for an individual child, which also carry between-child and occasion-level dispersion and converge on the real population spread. For setting expectations about a child, `Y_*` is the one to quote.

### Structural decomposition (joint models)

The joint models do not model each outcome independently. Instead, production is
expressed as an age-varying fraction of comprehension:

- `p_U(a)` — proportion of the inventory **understood** at age `a`.
- `q(a) = P(speak | understood)` — the **production ratio**, so words spoken =
  `p_U(a) · q(a)`.
- `r(a) = P(sign | understood)` — the **signing ratio** (VG14/VG15 only), so
  words signed = `p_U(a) · r(a)`.

This decomposition keeps spoken and signed vocabulary bounded by comprehension by
construction, and lets the models report directly on quantities of practical
interest (how much of what a child understands they can also say or sign).

The primary signing analyses mask `uk_01` signing values because that source records sign-only rather than total signed words, and mask `uk_06` signing values pending source verification. Their understood and spoken outcomes remain in the models; explicit sensitivity variants can reintroduce either signing source.

### Random intercepts and GP anchoring

The DS data are pooled across multiple studies, and several contain repeated
observations of the same children. Later models add hierarchy to account for
this:

- **Study (dataset) random intercepts** absorb systematic level differences
  between source studies.
- **Subject random intercepts** capture stable between-child differences for
  children observed more than once.
- **GP anchoring** constrains the HSGP correction to be zero at a reference age
  for every posterior draw. This removes a redundancy ("ridge") between the
  linear trend, the GP, and the intercepts that otherwise degrades sampling once
  random intercepts are present.

## Model lineages

### Single-outcome baselines (VG01–VG04)

The simplest models: one population, one outcome, no hierarchy. VG01/VG02 cover
spoken and understood vocabulary for children with Down syndrome; VG03/VG04 are
the typically-developing counterparts. These establish the baseline trajectory
shape that the joint and hierarchical models build on.

### Joint understood + spoken, Down syndrome (VG05 → VG07 → VG08 → VG09 → VG10)

This is the main DS development lineage. Each step adds structure to the previous
model:

- **VG05** — baseline joint model of understood and spoken via the `q(a)`
  decomposition.
- **VG07** — adds **study random intercepts** to absorb between-study level
  differences.
- **VG08** — adds **subject random intercepts on understood**.
- **VG09** — extends subject random intercepts to the **production ratio `q`** as
  well.
- **VG10** — the same structure as VG09, plus a **per-draw GP anchor at 54
  months**, introduced to resolve sampling diagnostics on the `q`-trajectory
  hyperparameters by removing the trend/GP/intercept redundancy. (VG10 was
  originally also given tighter, VG07-posterior-informed `q` anchors; #155
  broadened those back to the shared weakly-informative DS-joint values to remove
  the prior-data double-dipping, so the `q` anchors now match VG09 and the GP
  anchor is the only structural difference.)

### Joint understood + spoken, typically developing (VG13; VG06 retired)

- **VG06** _(retired)_ — was the TD counterpart to VG05, used for
  DS-versus-TD comparison. Wordbank's CDI: Words & Sentences (WS) form records
  `comprehension` as a production proxy rather than an independent measurement,
  which had been telling VG06 that `U = S` for most TD rows above 18 months
  (see `notes/202605151630-vg06-ws-comprehension-issue.md`). VG06 was removed
  from the model family rather than re-validated; **VG13** is now the joint TD
  model used for DS-versus-TD comparison.
- **VG13** — a TD joint model restricted to **ages 8–18 months**, where the
  Wordbank Words & Gestures and Oxford CDI data are dense and the WS production
  proxy bias is avoided entirely (WS is excluded, not merely down-weighted).
  Uses dataset-level study random intercepts and a GP anchor at 13 months.

### Typically developing single-outcome, with hierarchy (VG11, VG12)

Re-fits of VG03/VG04 that use the **full** TD data pool (rather than
subsampling) together with **dataset-level study random intercepts** and a GP
anchor at 19 months. The study intercepts absorb between-lab variation, so all
qualifying observations can contribute.

### Signing models, Down syndrome (VG14 → VG15)

- **VG14** — extends the joint model with a third production ratio for
  **signing**, `r(a)`. Total expressive vocabulary (`p_any`) is derived by
  assuming sign and speech are **independent given age** — a deliberately simple
  starting point that gives an upper bound on combined production.
- **VG15** — replaces that independence assumption with a directly estimated
  **within-understood sign–speech association** (`psi`, a scalar Plackett odds
  ratio identified from the four-cell sign/speak cross-tabulations in the `uk_02`
  and `uk_07` datasets). It adds **study and subject random intercepts on all three
  trajectories** and carries VG10's stabilisation (per-draw GP anchor at 54
  months with the tightened `q`-GP amplitude `eta_q`), yielding a
  **data-identified** total expressive vocabulary rather than VG14's
  independence-based bound.
