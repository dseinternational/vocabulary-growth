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

## Shared blocks

All in `vocab_growth.report_cells`, all for a cell with `#| echo: false` and `#| output: asis`:

| Block                              | Replaces                                        | Reads                                  |
| ---------------------------------- | ----------------------------------------------- | -------------------------------------- |
| `render_sampling_banner()`         | the hard-coded `{(chains, draws): label}` table | `fit_manifest.json`                    |
| `render_model_at_a_glance()`       | the hand-written glance callout                 | `fit_manifest.json`                    |
| `render_priors_table()`            | hand-written prior prose                        | manifest + `diagnostics.csv`           |
| `render_convergence_caveats()`     | a per-template reimplementation of the gate     | `diagnostics_summary.json`             |
| `render_headline_quantities()`     | nothing — this is new                           | the summary CSVs                       |
| `render_variation_table()`         | nothing — this is new                           | `diagnostics.csv`                      |
| `render_glossary([...])`           | nothing — this is new                           | static definitions                     |
| `render_calibration_section()`     | _(already in use)_                              | `posterior_predictive_calibration.csv` |
| `ppc_count_distribution_gallery()` | _(already in use)_                              | the count-distribution figures         |

Two of these fixed live defects rather than tidying: the banner told VG08, VG09, VG11, VG12 and VG13 they were "not fitted in reporting mode" when each was fitted at _more_ than the default reporting effort, and the caveats block told VG11 it had cleared a convergence gate it is published under a recorded exception to.

## Section skeleton

1. AI attribution callout
2. `render_sampling_banner()`
3. One sentence: what this model asks
4. `render_model_at_a_glance()`
5. Model diagram (`gp_model_graph.svg`)
6. **How to read this report** — `render_glossary([...])`, collapsed, listing only the terms this model uses
7. **Statistical model** — structure only
8. **Priors** — `render_priors_table()`, then the prior figures, then qualitative rationale
9. **Prior predictive checks** — with an evaluative sentence, not just the figures
10. **Data** — descriptives
11. **Diagnostics** — `render_convergence_caveats()`, the styled table, figures, and a verdict
12. **Findings** — `render_headline_quantities()`, `render_variation_table()` where applicable
13. **Posterior predictions** — every figure gets one sentence saying what to conclude
14. **Expected vocabulary by month** — `posterior_summary_monthly_*` / `expected_counts_by_month_*`
15. **Predictive calibration** — `render_calibration_section()`
16. **Limits** — what this model must not be used for

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
- LOO/ELPD is computed on every fit and printed only to the console, while the calibration section points readers to it.
