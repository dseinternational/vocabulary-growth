# Model report house style

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

This is the contract every `docs/models/vgNN/index.qmd` follows. It supersedes the recommendations in [`OUTPUT_TEMPLATE_REVIEW.md`](OUTPUT_TEMPLATE_REVIEW.md), which was written against an earlier state of the templates and is stale in several places (its VG04 anchor recommendation was followed into an error, and its VG13 items are now all implemented).

## The governing rule

**Numbers come from files at render time. Prose carries meaning, not measurements.**

Every model report is a template copied into its fitted output directory and rendered there, so a number typed into the template is a copy of a value that lives somewhere else — in `definitions.py`, or in a CSV the fit writes. Copies drift. A review of all fifteen reports found this same failure repeatedly:

| Report | Claim in prose                        | Truth                                    |
| ------ | ------------------------------------- | ---------------------------------------- |
| VG10   | `eta_q ~ HalfNormal(0.20)`, stated 3× | `0.8` since 2026-08-04                   |
| VG15   | the same, stated 4×                   | `0.8`                                    |
| VG15   | `q` high anchor `Beta(3,2)`, 2×       | `Beta(4, 1.2)`                           |
| VG02   | "346 rows"                            | 987, in a table on the same page         |
| VG14   | uk_02 union "~1.2 pp above"           | −6.7 pp, in a CSV five lines below       |
| VG14   | signed ratio "0.4–0.5"                | peaks at 0.371                           |
| VG13   | "the broad baseline `q` priors"       | VG13 is the one model that rejected them |

None of these was careless. Each was correct when written. That is precisely why the fix is structural rather than editorial: **if a number can go stale, it eventually will**, so the template must not hold one.

Consequences for authors:

- Never type a prior value. Call `render_priors_table()`.
- Never type a fitted quantity. Call `render_headline_quantities()` / `render_variation_table()`, or embed the CSV.
- Never type a frame size, study count or age range. Call `render_model_at_a_glance()`.
- Prose may state **structure** ("this model has no random effects", "signing is modelled as a rise and fall"), **direction** ("comprehension leads production"), and **caveats**. It may not state magnitudes.
- Where a figure needs a magnitude to be interpretable, put the magnitude in an adjacent table, not in the sentence.
- Where a historical magnitude must stay in prose — a gate record, an earlier fit's recovery z-scores — date it in the sentence ("on the 2026-08-19 fits"), so that staleness is visible rather than silent. The 2026-09-02 review found a dozen such numbers in Limits sections, one of which that run had already falsified.

## Shared blocks

All in `vocab_growth.report_cells`, all for a cell with `#| echo: false` and `#| output: asis`:

| Block                                  | Replaces                                        | Reads                                  |
| -------------------------------------- | ----------------------------------------------- | -------------------------------------- |
| `render_sampling_banner()`             | the hard-coded `{(chains, draws): label}` table | `fit_manifest.json`                    |
| `render_model_at_a_glance()`           | the hand-written glance callout                 | `fit_manifest.json`                    |
| `render_priors_table()`                | hand-written prior prose                        | manifest + `diagnostics.csv`           |
| `render_convergence_caveats()`         | a per-template reimplementation of the gate     | `diagnostics_summary.json`             |
| `render_headline_quantities()`         | nothing — this is new                           | the summary CSVs                       |
| `render_variation_table()`             | nothing — this is new                           | `diagnostics.csv`                      |
| `render_glossary([...])`               | nothing — this is new                           | static definitions                     |
| `render_calibration_section()`         | _(already in use)_                              | `posterior_predictive_calibration.csv` |
| `ppc_count_distribution_gallery()`     | _(already in use)_                              | the count-distribution figures         |
| `render_reading_routes(role, ...)`     | nothing — this is new                           | the role the template states           |
| `render_family_notes()`                | nothing — this is new                           | manifest + `diagnostics.csv`           |
| `render_expectations_table(...)`       | the raw `posterior_summary*` DataFrame display  | `posterior_summary*` + monthly tables  |
| `render_diagnostic_verdict()`          | the reader scanning the styled table            | `diagnostics_summary.json` + manifest  |
| `render_prior_posterior_contraction()` | "compare each posterior with its prior figure"  | `prior_posterior_contraction.csv`      |
| `render_frame_composition()`           | `describe()` plus normality tests               | manifest, then an exact frame rebuild  |
| `render_dispersion_scope()`            | two $\kappa$ figures with no stated scope       | manifest + contraction + frame rebuild |
| `render_loo_section()`                 | _(already in use)_                              | `loo_summary.csv`                      |

The six blocks added on 2026-09-02 came from a review of all twenty templates against a reporting-quality run (`notes/202609021200-report-template-review.md`). Two of them read artefacts a fit does not write and are fail-soft until those exist: `render_prior_posterior_contraction()` reads the per-fit CSV that `scripts/prior_vs_posterior.py --table --model <key>` writes from the trace, and VG22's implied correlation matrix reads `subject_factor_corr.csv` from `scripts/emit_factor_correlation.py`. Run both after a fit and before `--render-only`; each block prints how to produce its file when it is absent.

`render_dispersion_scope()` was added later the same day, after a reader compared the $\kappa$ figures across VG21 and VG22 and asked whether the differences were a model artefact. They largely are, in three ways the pages did not state: $\kappa_u$ is marginal on the item pool while $\kappa_s$ is conditional on the child's own understood count, so their **levels** are not comparable with each other; $\kappa$ is residual after whatever child structure a model carries, so it is never comparable **across** models; and a two-anchor $\kappa$ can have one end the data never informed while the figure still draws a confident median there (VG22's `kappa_excess_young_s` contracts to -0.23). The block renders immediately above the first $\kappa$ figure on every template that has one, and states all three from the fit's own record. See `notes/202609021620-dispersion-kappa-comparability.md`. **Never write a sentence comparing one $\kappa$ curve's level with another's** — the shared body carried one until that note.

**The bivariate random-effects family shares one prediction body.** VG10, VG19, VG20 and VG22 transclude `docs/models/_bivariate_re_body.qmd` (`{{< include _bivariate_re_body.qmd >}}`) rather than each carrying a copy: before it existed VG20's template referenced 23 of the 90 artefacts its fit wrote and sent the reader to VG10 — a development step — for the rest. `reporting.stage_report_sources` copies every `docs/models/_*.qmd` into the output directory beside `index.qmd` at fit time and on `--render-only`, because a Quarto include resolves relative to the rendered document. Nothing model-specific belongs in the include.

**Section anchors are fixed.** The reading-routes block links to `#sec-priors`, `#sec-prior-predictive`, `#sec-frame`, `#sec-diagnostics`, `#sec-findings`, `#sec-predictions`, `#sec-monthly`, `#sec-calibration`, `#sec-loo`, `#sec-robustness`, `#sec-limits` and, on joint pages, `#sec-spoken-given-understood`; a template renames a heading but keeps its id. An id on a callout is a Quarto cross-reference and must carry one of Quarto's own prefixes, so `render_family_notes()` puts its `#sec-one-child` anchor on a plain wrapping div.

Two of these fixed live defects rather than tidying: the banner told VG08, VG09, VG11, VG12 and VG13 they were "not fitted in reporting mode" when each was fitted at _more_ than the default reporting effort, and the caveats block told VG11 it had cleared a convergence gate it is published under a recorded exception to.

## Section skeleton

1. AI attribution callout
2. `render_sampling_banner()`
3. One sentence: what this model asks
4. `render_model_at_a_glance()`, then `render_reading_routes(role, ...)` with the role the page states — a development step or candidate names the model of record its non-research readers should use instead
5. Model diagram (`gp_model_graph.svg`)
6. **How to read this report** — `render_glossary([...])`, collapsed, listing only the terms this model uses; a "Terms specific to this model" callout where the page introduces any
7. **Statistical model** — structure only
8. **Priors** — `render_priors_table()`, then the prior figures, then qualitative rationale
9. **Prior predictive checks** — with an evaluative sentence, not just the figures
10. **Data** — `render_frame_composition()` first, then descriptives
11. **Diagnostics** — `render_convergence_caveats()`, `render_diagnostic_verdict()`, the styled table, figures, then `render_prior_posterior_contraction()` under "Prior to posterior"
12. **Findings** — `render_headline_quantities()`, `render_variation_table()` where applicable
13. **Posterior predictions** — preceded by `render_family_notes()` on any page a family or practitioner is routed to; `render_expectations_table(outcome)` above each raw summary table; every figure gets one sentence saying what to conclude
14. **Expected vocabulary by month** — `posterior_summary_monthly_*` / `expected_counts_by_month_*`
15. **Robustness** — the conditional robustness and recovery cells, on every model of record, reference and candidate
16. **Predictive calibration** — `render_calibration_section()`
17. **Out-of-sample prediction** — `render_loo_section()`
18. **Limits** — what this model must not be used for

## Accessibility

Target reader: an undergraduate science or maths student.

- Any term in `vocab_growth.glossary.GLOSSARY` used in the report goes in that report's `render_glossary` list. An unlisted term passed to it raises, so typos fail the render.
- Every figure needs a caption that says what it shows, not the filename. `![posterior_kappa](posterior_kappa.png)` is not a caption.
- State the interval convention wherever intervals appear. The default is an 89% outer and 50% inner equal-tailed interval.
- $\kappa$ must never appear without the reminder that **larger means less spread**. Reviewers flagged this on nine of fifteen reports.
- Distinguish **population-level** from **subject-marginal** every time both appear. The summary CSVs carry explicit `p_population_*` and `p_subject_marginal_*` columns, so name the column rather than describing the estimand loosely.

## Things that are not the template's fault

Some review findings need engine changes, tracked separately:

- `ppc_count_distribution_gallery` globs its directory instead of reading the capped table, so a stale figure from an earlier run survives an age-cap change. VG02 publishes a 90-month comprehension figure against an 84-month cap, and it has reached `docs/report/figures/`.
- `plot_modality_trajectories` takes no cap, so VG14's `p_any` figure runs to 115 months above a table that stops at 84.
- VG11's output has no `prior_*` figures at all, while its template references three.
- ~~`expected_counts_by_month_*.csv` and `posterior_summary_monthly_*.csv` are byte-identical~~ — fixed: the figure no longer writes a sidecar, because the caller has already written the same frame under the canonical `posterior_summary_monthly_*` name.
- The `*_smoothed.csv` sidecars are byte-identical to their unsmoothed originals, so a reader downloading the "smoothed" CSV gets the unsmoothed numbers. Fixed in the writers, but **the artefacts on disk still carry it**: the fix changes plot-stage output, and no model has been through the plot stage since. It clears on the next refit, or on a `regenerate_plots.py` pass for the four models whose traces are `full`.
- ~~LOO/ELPD is computed on every fit and printed only to the console, while the calibration section points readers to it.~~ — fixed: every fit now writes `loo_summary.csv`, `scripts/emit_loo_summaries.py` backfills it for fits already on disk, and `render_loo_section` prints it into the report's out-of-sample section.
- LOO's held-out unit is not the same for every model, and the reported prose must match the model it sits under. A univariate fit has one likelihood over administration rows, so its estimate is genuinely leave-one-administration-out. A multi-outcome fit gets one LOO per outcome likelihood, and a row there holds out one likelihood **term**: the spoken and signed likelihoods take the same administration's observed comprehension count as their trial count, so an expressive score is conditional on that observed count and an understood score leaves its own observed value in the expressive denominators. `render_loo_section` branches on the number of rows in the table and says so. What remains outstanding is the estimate itself — an administration-aggregated LOO for the multi-outcome engines, of the kind `scripts/loo_compare.py` already builds as `y_joint` — which needs regenerated outputs and is tracked separately.
- VG15 excludes both Dirichlet-Multinomial composition likelihoods from LOO, and those are the only terms that identify the association $\psi$, so $\psi$ is not scored by LOO at all. The report cell states this wherever the fit carries `psi` and `conc`.
