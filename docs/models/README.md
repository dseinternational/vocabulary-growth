# Model inventory

> [!NOTE]
> Drafted by LLM-based AI tools (Claude Code/Opus 4.8 and Codex/GPT-5; dispersion prior note by Claude Code/Opus 5; reporting-age update by Claude Code/Fable 5).

> [!WARNING]
> This is work in progress. All models and their output are preliminary and likely
> to change as the models evolve and further data are received.
>
> **All fifteen models were refitted at reporting quality in the 13–16 August 2026 run** ([`notes/202608142000`](../../notes/202608142000-refit-run-record-and-disk-failure.md)), so the numerical prose is current as of that run. Fourteen clear the hard convergence gate; VG11 is published under a narrow recorded exception, and VG10, VG12 and VG13 carry soft-tier caveats — all four disclosures reach the report through `convergence_caveats.csv` and Appendix B. That run also settled the changes that had made the previous fits stale: the nested outcome likelihood, the TD repeated-measures correction, signing-source harmonisation and the Edgin inclusion revision from #163; `us_01` rebuilt from the Edgin item-level contributor files rather than the age-truncated Wordbank by-child export (195 → 230 rows, 78 → 126 comprehension observations — see [`notes/202608031500-edgin-out-of-window-administrations.md`](../../notes/202608031500-edgin-out-of-window-administrations.md), whose §7 records a reversal in how the out-of-window administrations are handled); the `uk_07` (PACT-DS) integration; and the widening of the VG11/VG12/VG13 reference pool to English, Italian and Spanish (European) so the Down-syndrome-versus-typically-developing comparison covers several languages on both sides ([`notes/202608031500-td-romance-extension.md`](../../notes/202608031500-td-romance-extension.md)).
>
> Two definition changes were taken **during** that run and are described below: the DS joint family adopted `CLAMP_Q_ONLY`, and comprehension reporting moved from 72 to 84 months under a project-wide age policy. (Comprehension reporting has since been lowered back to 72 months on 2026-08-22, and VG04/VG12 report comprehension only to 25 months — see the reporting-ages section below.)

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

Models are numbered `VG01`–`VG20` in roughly the order they were developed. The
numbering is historical rather than hierarchical — a higher number means "added
later", not "supersedes". `VG17`–`VG19` are reserved rather than skipped:
`VG17`/`VG18` for the exploratory sign-group modules and `VG19` for the child-slope
plan (`notes/202608141900-child-slope-implementation-plan.md`). Models build on one another along two main lineages
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

| Model                  | Population | Outcome(s)                   | What it adds / its purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------- | ---------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [VG01](vg01/index.qmd) | DS         | Spoken                       | Baseline age → words spoken trajectory.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [VG02](vg02/index.qmd) | DS         | Understood                   | Baseline age → words understood trajectory.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [VG03](vg03/index.qmd) | TD         | Spoken                       | TD counterpart to VG01. English-only reference pool (no random effects to absorb between-language variation).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [VG04](vg04/index.qmd) | TD         | Understood                   | TD counterpart to VG02. English-only reference pool (no random effects to absorb between-language variation).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [VG05](vg05/index.qmd) | DS         | Understood + spoken (joint)  | Baseline joint model; spoken modelled as a fraction `q(a)` of understood.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| VG06 _(retired)_       | TD         | Understood + spoken (joint)  | TD counterpart to VG05; retired after the WS-comprehension data issue (see below) — superseded by VG13.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [VG07](vg07/index.qmd) | DS         | Understood + spoken (joint)  | VG05 + study random intercepts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [VG08](vg08/index.qmd) | DS         | Understood + spoken (joint)  | VG07 + subject random intercepts on understood.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [VG09](vg09/index.qmd) | DS         | Understood + spoken (joint)  | VG08 + subject random intercepts on the production ratio `q`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [VG10](vg10/index.qmd) | DS         | Understood + spoken (joint)  | VG09 + per-draw GP anchor at 54 months (stabilisation); `q` anchors match VG09.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [VG11](vg11/index.qmd) | TD         | Spoken                       | VG03 + dataset and child random intercepts + GP anchor at 19 months; one-administration-per-child sensitivity available. Reference pool widened to English + Italian + Spanish (European).                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [VG12](vg12/index.qmd) | TD         | Understood                   | VG04 + dataset and child random intercepts + GP anchor at 19 months; one-administration-per-child sensitivity available. Reference pool widened to English + Italian + Spanish (European).                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [VG13](vg13/index.qmd) | TD         | Understood + spoken (joint)  | Young TD joint model, ages 8–18 months; dataset and child random intercepts on understood and `q` + GP anchor at 13 months. Reference pool widened to English + Italian + Spanish (European).                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [VG14](vg14/index.qmd) | DS         | Understood + spoken + signed | Adds signing as a third ratio `r(a)`; total expressive vocabulary derived assuming sign/speech independence given age.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [VG15](vg15/index.qmd) | DS         | Understood + spoken + signed | VG14 + within-understood sign–speech association `psi` + study & subject random intercepts + VG10 stabilisation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [VG16](vg16/index.qmd) | DS         | Understood + spoken          | VG09 + a cross-lag (prior understood → current `q`): children further ahead receptively convert more of it to speech. Population-relative headline `beta_lag` = **+0.203**, 89% ETI [0.093, 0.316]; best read as a between-child association. The within-child (RI-CLPM) contrast agrees in sign at about half the size (+0.103, spanning zero); its earlier negative value was `dev`-tier non-convergence.                                                                                                                                                                                                               |
| [VG19](vg19/index.qmd) | DS         | Understood + spoken          | VG10 + a child random **slope**: each child gets a rate as well as an offset, on both understood and `q`. VG10 is nested exactly at `tau1 = 0`. Structure chosen by Gate 1 on the fitted residuals before implementation — a random slope beats a constant intercept by `2 x delta logL` = 36.05 on spoken and survives restriction to the 334 children with repeated spoken measures (20.81), while an AR(1) transient collapses to zero persistence. **Gated against VG10, not VG20**: the two are parallel refinements and are not composable, so VG19 does not carry `rho_uq`. Registered 2026-08-21; not yet fitted. |
| [VG20](vg20/index.qmd) | DS         | Understood + spoken          | VG10 + one free correlation `rho_uq` between a child's understood and production-ratio random intercepts, replacing VG10's independent draws. VG10 is nested exactly at `rho_uq = 0`; the built graph differs by that parameter and nothing else. Estimates directly what VG16 could only leave behind as a residual correlation, on all 767 children rather than the 250 with a prior wave. Fitted at `rep` 2026-08-19: `rho_uq` = **+0.368**, 89% ETI [0.287, 0.447].                                                                                                                                                   |

### Model roles

`methods-workflow.qmd` defines a four-way status taxonomy — **model of record** (the current source for a stated estimand), **development step** (retained to show how structure was added, but not preferred for headline estimates), **TD reference** (a typically-developing comparison model with a distinct population role), and **superseded** (replaced after a documented structural or data problem) — and names this inventory as the source of truth for them. Recording them here, because until now the taxonomy was defined but never applied.

| Model            | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VG20             | **Model of record** for the Down syndrome joint understood + spoken estimands from 2026-08-19, on the study owner's instruction, after the gates of [#224](https://github.com/dseinternational/vocabulary-growth/issues/224) — three of four of which pass; gate 2 (recovery) does not, in that `rho_uq` recovers cleanly in 3 of 3 replicates but the surrounding coverage is under-nominal — measured against VG10 at the same tier, that shortfall is the engine's and VG20 is the better of the two on every recovery figure available. It is VG10 plus one parameter and nothing else, so it inherits VG10's role rather than reopening it, and the calibration defect the recovery run points at is VG10's equally. What changes is child-level, not population-level: the population trajectories and the production ratio are unmoved at every reported age, and the correlation widens the spoken subject-marginal intervals by 9–33%, most at the youngest ages. See [`notes/202608190900`](../../notes/202608190900-vg20-promotion.md). |
| VG10             | **Development step**, from 2026-08-19; **model of record 2026-08-05 to 2026-08-19**. Wholly replaced by VG20 for reporting. Its independence assumption on the two child effects is contradicted by its own fitted deviations, which correlate at +0.151 [0.106, 0.195] under a prior asserting zero. Numbers published from it are not withdrawn — the population trajectories it reported are reproduced by VG20 — but its child-level intervals are superseded, and so is every between-child DS-versus-TD spoken contrast derived from it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| VG05, VG07, VG08 | **Development steps.** The study owner's position, 2026-08-06: not expected to supply any reported number, being superseded by later models in the lineage.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| VG14             | **Development step.** Wholly replaced by VG15 for reporting; confirmed by the study owner, 2026-08-06. VG15 supplies everything VG14 does — including `p_any_indep`, the independence-assumption total expressive that is VG14's headline output — on a better-specified model. VG14's independence assumption is rejected by the data (psi = 2.34, 89% HDI [1.89, 2.81], P(psi>1) = 1.000 — re-pinned against the 13–16 August `rep` fits, was 1.78 [1.29, 2.45]), though the numerical cost of that assumption remains the smaller half: at most 14.3 words on the 810 scale, at 55 months. What actually separates the two models is VG15's study and subject random effects — VG14 puts total expressive vocabulary 45.9 words higher at 36 months and 68.6 higher at 46. **This role decision was not honoured by the comparison report until 2026-08-16**, which took its sign-inclusive contrast from VG14 for ten days after VG15 was named its replacement.                                                                               |

This settles part of the prior-data conflict recorded in [`notes/202608051500-report-critical-review.md`](../../notes/202608051500-report-critical-review.md) §4a. VG05, VG07, VG08 **and VG14** all carry `b_kappa_mag_s` around four standard deviations beyond its prior, with the posterior wider than the prior — a parameter pinned at a boundary is not an estimate. For the three development steps that is disclosable rather than fixable, because a development step supplies no number. **VG14 is the one that matters**, and whether it needs migrating to the two-anchor dispersion form follows from its role decision rather than from the diagnostic.

### VG15's two open flags, settled 2026-08-06

Both resolve to **disclose, not change**; the reasoning is in [`notes/202608060900`](../../notes/202608060900-three-prior-conflicts.md) §5b.

- **`kappa_sign` stays on the legacy dispersion form.** It is well identified (contraction 0.429), sits comfortably inside its prior (CDF 0.276), and its sign constraint is not binding — none of which was true of the four models migrated for the same parameter. The resulting asymmetry with VG14 is **two separate correct calls, not an inconsistency**; do not "tidy" it.
- **`ell_unit_sign` stays sampled.** It is genuinely unidentified (contraction 0.033), but fixing it changes nothing measurable — a maximum median shift of 0.0023 on `r(a)`, and +0.1% band width — and removing the signed GP is worse: it fails the hard convergence tier and strips the model's only honest signal of ignorance above 60 months.

Because neither is being fixed, the signing chapter **must** disclose four things: that `r(a)`'s peak age is fixed by construction rather than estimated; that the signed GP's length-scale is not identified, so the trajectory is parametric in practice; that psi is estimated from uk_02 alone and applied pool-wide; and that signed evidence stops around 60 months while the reported range runs to 115.

**The remaining role assignments are outstanding and are owner decisions**, tracked in #190. They should not be inferred from this file's ordering. Two constraints are already fixed and any assignment must respect them: VG06 is retired and superseded by VG13; and `scripts/fit_recovery.py` treats `vg20`, `vg12` and `vg15` as the headline set (it was `vg10` until the 2026-08-19 promotion), so those three at minimum carry reporting weight. A superseded model never supplies a number in the findings, so these assignments have to be settled before the findings chapters are written.

### Exploratory, unregistered prototypes

VG17 and VG18 are exploratory sign-group comparison modules. They are deliberately excluded from `MODEL_REGISTRY`, `fit_model.py all`, and the numbered model inventory because they have not yet passed the specification and reporting workflow required of registered models. VG17 still uses the same harmonised signing-source rules as the registered signing models so exploratory comparisons cannot silently reintroduce non-comparable fields.

### Registering a new model: everything that has to be updated

Adding an entry to `MODEL_REGISTRY` is the start, not the whole job. Registering VG20 tripped five separate obligations that no checklist recorded, four of them only discovered by running things. In rough order of when they bite:

1. **`MODEL_REGISTRY`** in `definitions.py`, and a `model_vgNN.py` module that selects the definition and dispatches to an engine.
2. **New definition fields go on a subclass, never on an existing definition class.** A fit is validated by comparing the serialised definition field for field, so a new field with a default still changes every sibling model's serialisation and invalidates every fitted output of that class. `UnivariateREModelDefinition` and `BivariateCorrelatedSubjectREModelDefinition` both exist for this reason.
3. **`docs/models/vgNN/index.qmd` must exist before the first fit**, not before the first render. The pipeline's report stage copies the source `.qmd` whether or not `--render` was passed, and raises `FileNotFoundError` at the very end of an otherwise complete fit if it is missing.
4. **`scripts/regenerate_plots.py`'s `ENGINE_BY_MODEL`.** Without an entry, a cosmetic plot fix silently requires a refit instead of a replot. A test enforces this, and accepts a recorded exemption with a reason as the alternative.
5. **The pinned test sets**, each of which exists to make a silent change visible: the κ-form lists in `test_model_definitions.py`, `CLAMP_Q_ONLY_MODELS` in `test_clamp_scope.py`, and the engine dispatch in `test_trend_gp_consolidation.py`. A model missing from the last of these fails with "Model has not been set in the context", which reads as a defect in the model rather than an omission in the test.
6. **This inventory**, and the model list in `CLAUDE.md` / `AGENTS.md` / `.github/copilot-instructions.md`, which are kept in sync with one another.

One hazard has no test and is worth stating on its own. **Registering a model makes its output subject to publication validation immediately**, so a provisional `dev` or `test` fit of it sitting in `output/models/` will fail `sync_report_figures` and abort the sync for _every_ model, with an error naming the new model rather than the act of registering it. Delete provisional output before syncing, or fit the new model against a different output root.

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

`ages_query` is shared by every outcome a model reports, but the Down syndrome outcomes are not observed over the same range. Since 2026-08-13 the caps are a project-wide **policy** rather than a per-figure decision, and the single source of truth is [`vocab_growth.reporting_ages`](../../src/vocab_growth/reporting_ages.py). The rule is per _quantity_, not per figure, because one figure can carry two outcomes with different support — the joint trajectory plots draw understood and spoken together and trim each independently:

| Quantity                                 | Cap | Why                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------- | --: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| understood                               |  72 | The 72–84 band is populated (25 rows / 20 children / 5 studies) but cannot distinguish the child structures, so comprehension-derived quantities are model-dependent there — see the paragraph below. VG04 and VG12 stop at 25: the typically-developing pool has no comprehension observations above 25 months (#228). |
| ratios of understood (`q`, `r`, `p_any`) |  72 | Conditioned on understood, so they inherit its cap. `p_any` has a second, independent reason — see below.                                                                                                                                                                                                               |
| signed                                   |  84 | Its own field, `report_max_age_signed`, since #212 — see below.                                                                                                                                                                                                                                                         |
| spoken                                   |  90 | The top of the query grid: 1428 rows, 95th percentile 81 months, 59 at or above 84.                                                                                                                                                                                                                                     |

The comprehension cap was **raised from 72 to 84 on 2026-08-13**, once `uk_07` and the reinstated `uk_06` signing rows rebuilt the older tail (understood then had 987 rows with a 95th percentile of 69 months, against 905 and 64 before), and **lowered back to 72 on 2026-08-22**: the 72–84 band is populated, but the Down syndrome child structures disagree there, so a comprehension number in that band depends on which model produced it. Raise it again when new older-child comprehension data let the band _distinguish_ the structures, not merely populate it — see [`notes/202608221200-reporting-source-by-quantity.md`](../../notes/202608221200-reporting-source-by-quantity.md). The 72-month cap is set on VG02, VG05, VG07–VG10 and VG14–VG16. VG01 is production-only and is left alone. Among the typically-developing models, VG04 and VG12 report comprehension only to **25 months** — their pool has no comprehension observations above 25, while their query grids run to 30 (#228); the remaining typically-developing grids stop at 30 months (18 for VG13), inside their data.

Signed is **not** covered by the comprehension cap. It has its own `report_max_age_signed`, set to 84 on VG14 and VG15. Before that field existed on the trivariate definition, VG14's sign-derived figures were trimmed by `report_max_age_understood`, so raising the comprehension cap moved the signed figures as a side effect that nobody decided — fixed in `4ff48e5`. Note the two caps rest on different arguments: comprehension stops at 72 because the models disagree beyond it, whereas signed is observed on 56 rows from 48 children at or above 84 and stops there because 84 is the trend anchor past which the mean is a levelled-off extrapolation.

`p_any` — total expressive vocabulary, the union of speaking and signing — stops at **the tighter of its two components' caps**, decided by the study owner on 2026-08-16. Past the signing cap one of the two components is no longer reported, and a union of a reported and an unreported quantity is not a quantity this project publishes. That currently agrees with the conditioning rule above, both giving 84, but the two arguments are independent and this one binds if the comprehension cap ever moves without the signing cap moving with it. It is recorded because it is a reporting decision rather than an arithmetic consequence: nothing objected when VG14's modality figure ran to 115 months above a `p_any` table trimmed to 84, because `modality_trajectories` carries no outcome suffix and so matched no rule in the policy test's stem map. Both the figure and the test are fixed in `204581f`.

Unlike `ages_query` this is post-processing and cannot move the posterior — refitting VG10 across the change at a fixed seed reproduced its diagnostics bit-for-bit. It is not free, though: the summary tables are written during the fit pipeline and `--render-only` re-renders Quarto against the CSVs already on disk, so a new cap only takes effect on a refit, and the field is part of the recorded definition, so output produced under a different value is reported as stale. `tests/test_reporting_age_policy.py` walks the fitted artefacts and fails on any that reports past its cap. See [`PRIORS.md`](PRIORS.md), "Reported age range for comprehension".

Alongside them each fit writes a **whole-month** companion — `posterior_summary_monthly[_<outcome>].csv` and the figure `expected_counts_by_month[_<outcome>]` — one row per whole month of age, with the same columns and the same `P(Y<=k)` bucket thresholds. These are derived from the _plot_ grid (`n_plot = 500`) rather than from added query ages, so they are pure post-processing: no change to the model graph, the HSGP domain, or the `query_id` dimension the report and the comparisons consume. Three consequences worth knowing before quoting a monthly figure:

- **Coverage is the plot grid's span, the full observed age range**, which for the Down syndrome models is 8–115 months (108 rows) — _wider_ than the 12–90 canonical ages, not narrower. Every canonical age has a monthly counterpart. The extra months at both tails are mostly data-free: 33 of the 108 rows have no observation in that month at all, which is why the `n_obs` column exists. Read a row with `n_obs = 0` as the model interpolating or extrapolating, not as evidence. The monthly companions deliberately keep this full span even for comprehension, so an understood table capped at 84 months sits beside a monthly understood table running to 115: the `n_obs` column is the guard the capped table has no room for.
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

The primary signing analyses mask `uk_01` signing values because that source records sign-only rather than total signed words; its understood and spoken outcomes remain in the models, and an explicit sensitivity variant can reintroduce its signing. `uk_06`'s signing was masked from 16 July 2026 pending source verification and was **reinstated on 12 August** — the source confirmed the standard DSE checklists, whose column 2 is "understands and signs", a total sign count (see `data/vocab_data_uk_06.md`).

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
  Wordbank Words & Gestures and Oxford CDI data are dense. Uses dataset-level
  study random intercepts and a GP anchor at 13 months. Note that the WS
  production-proxy bias is **not** what the age restriction avoids: `load_data`
  admits only `WORDBANK_BIVARIATE_FORMS` whenever `understood` is requested, so
  WS is excluded from every comprehension model at every age, and the cap adds
  nothing to that. The restriction's own justification is density — thin above
  18 months, and increasingly exposed to the Oxford CDI's 418-item ceiling. The
  `window-25` and `window-22` sensitivity variants measure what it costs; see
  `notes/202608171500-reporting-scope-audit.md` and issue #228.

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
