# Report template review: twenty templates, three readers

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

**Date:** 2026-09-02, during the reporting-quality refit of #281. **Scope:** every `docs/models/vg*/index.qmd`, reviewed one by one against `REPORT_STYLE.md` and against the artefacts each fit of that run wrote, for what each page shows about its priors, its predictive checks, its diagnostics and its findings — and what a family, a practitioner and a researcher would each still lack. The changes described here were made on a branch in a worktree while the fits ran, so no in-flight fit recorded a dirty checkout.

## What the review found

**The model of record was the thinnest page in its own lineage.** VG20's template referenced 23 of the 90 artefacts its fit writes. Thirty-four of the omissions were embedded by VG10's template and written byte-for-byte by VG20's fit: the summary tables, the monthly tables, PMF and CDF, the count galleries, derivatives, dispersion curves, the comprehension–production gap and `spoken_given_understood`. VG20's page said "each of these should be read as VG10's answer" — and VG10's own banner says it supplies no reported number. VG22 (20 of 90) and VG19 (16 of 89) had the same shape. VG20 also had no leave-one-out section at all, in the first run in which an administration-level score exists.

**Three templates had been made false by the run itself.** VG11 warned that its fit wrote no prior predictive figures (it wrote all three); VG22 said twice that every number on the page predated the refit under the designed prior (the run was that refit); VG23 opened "this report has not yet been fitted".

**Typed magnitudes survived in the sections most likely to be quoted.** VG20's Limits carried "+0.368 [0.287, 0.447]" against a fit that gives 0.390 [0.314, 0.463]; VG21 typed a prior value ("widens `eta_q` from 0.20 to 0.5"), the exact failure `REPORT_STYLE.md` was written against.

**No page routed its readers, and the usable numbers were shipped as raw DataFrames.** Fourteen templates mentioned parents or practitioners in passing; none said what a family should read. The summary CSVs (up to forty columns) were displayed as pandas output under headers such as `P(Y<=5)`.

**Checks were delegated to the reader.** All twenty pages asked the reader to compare each posterior with its prior figure by eye; where a page stated the answer, it stated it from memory. Prior predictive checks listed criteria and showed figures; three pages carried an evaluative sentence. The diagnostics section had no rendered verdict, so the worst R-hat and the smallest ESS had to be found by scanning a table.

## What changed

Six blocks in `vocab_growth.report_cells`, all reading the fit on disk and all fail-soft: `render_reading_routes` (three routes by stated role; development steps and candidates name the model of record), `render_family_notes` (what a count is, what the spread means for one child, that nobody is being ranked — population and hierarchy read from the fit), `render_expectations_table` (expected words, single-child ranges, threshold probabilities and nearby-observation counts at the reported ages, in plain headers, with extrapolated ages marked), `render_diagnostic_verdict` (the gate's extremes with the parameter that set them, or "an element the table does not list" when the gate's extreme belongs to a random-effect element the table omits; effort actually used; one sentence per tier), `render_prior_posterior_contraction` (from a per-fit CSV `scripts/prior_vs_posterior.py --table --model <key>` now writes) and `render_frame_composition` (administrations, children and per-study counts from the manifest; per-study children, age spans and the repeat share from an `analysis_frames` rebuild used only when its hash matches the fit's).

A shared include, `docs/models/_bivariate_re_body.qmd`, extracted from VG10's prediction body and transcluded by VG10, VG19, VG20 and VG22, with `reporting.stage_report_sources` copying every `docs/models/_*.qmd` beside `index.qmd` at both copy sites. VG20, VG19 and VG22 gained an out-of-sample section; VG20, VG15, VG12, VG22, VG14, VG13, VG21 and VG11 gained the conditional robustness cell and, where a recovery matrix exists, the recovery cell. VG22 gained a rendered 4×4 implied correlation matrix from a new `scripts/emit_factor_correlation.py`, which reads `subject_factor_corr` from the trace without loading the posterior; on this run's fit the level-to-rate cell no other model estimates is +0.39 [0.26, 0.52] and the rate-to-rate cell +0.62 [0.21, 0.89].

The three false templates were corrected; VG15's LOO prose was reconciled with the administration-level row that now scores $\psi$; VG19 gained an explicit glossary list and a terms-specific callout; every typed magnitude found was either rendered from a CSV or dated in the sentence.

## What was verified

All twenty templates edited by one script with an asserted anchor for every edit. Thirteen pages rendered from copies of this run's fits (trace symlinked), with zero unresolved cross-references, zero Quarto filter errors and zero stale phrases; the six development-step pages rendered on the first pass, and the pages carrying the family notes exposed one Quarto rule — an id on a callout is a cross-reference and must use Quarto's prefixes — fixed by wrapping the callout in a plain div. The contraction table and the factor matrix were rendered from real trace output, not only their fallback state. The fast test suite passes with the new blocks under test; `tests/test_environment.py::test_report_figs_dir_stays_repo_local` fails only in a worktree whose path begins with `/scratch/vg`, because it checks that literal as a substring.

## What remains, and needs the engines

Persist a `prior_predictive_summary.csv` at fit time so the prior predictive check can carry a computed verdict; write a per-study table from the multivariate engines (the render-time rebuild covers it meanwhile); shade the monthly figures where `n_obs` is small; emit VG22's two per-outcome offset–rate correlations as scalars so `render_variation_table` can tabulate the age-varying between-child spread; confirm whether this run's traces carry the persisted subject-marginal $q$ that #233 asked for.
