# Model report house style

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17.

Each `docs/models/vgNN/index.qmd` is copied into a fit's output directory and rendered there. The template explains the model; the fit supplies its numbers. This guide replaces the former template-review checklist, which is retained in Git history.

## Keep measurements in their source files

Read prior settings, fitted quantities, sample sizes and diagnostics at render time. Do not copy them into template prose. Use the shared report cells or display the exported table beside the explanation.

Prose should explain structure, interpretation and limits. A statement about a fitted direction, such as a positive correlation, also needs current evidence. Generate it from the fit or make it conditional. If a historical result is necessary, name the date and fit so the reader cannot mistake it for a current estimate.

## Shared report cells

The functions in `vocab_growth.report_cells` read the fit's manifest and exported tables. Use `echo: false` and `output: asis` for cells that print Markdown.

| Purpose                                   | Functions                                                                                                                     |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Sampling, model and reading route         | `render_sampling_banner`, `render_model_at_a_glance`, `render_reading_routes`                                                 |
| Prior settings and learning from the data | `render_priors_table`, `render_prior_posterior_contraction`                                                                   |
| Data included in the fit                  | `render_frame_composition`                                                                                                    |
| Diagnostics                               | `render_diagnostic_verdict`                                                                                                   |
| Estimates and variation                   | `render_headline_quantities`, `render_variation_table`, `render_expectations_table`, `render_sex_section`                     |
| Interpretation                            | `render_family_notes`, `render_dispersion_scope`, `render_conditional_production_check`, `render_reference_child_calibration` |
| Out-of-sample assessment                  | `render_loo_section`                                                                                                          |

Other shared helpers supply the glossary, convergence caveats and predictive calibration. Follow an existing model template for their imports. Check function signatures in the source before adding a call.

Some report cells read files generated after sampling. `scripts/prior_vs_posterior.py --table --model <key>` writes the prior-to-posterior comparison, and `scripts/emit_factor_correlation.py` writes VG22's implied correlation matrix. Generate these before `--render-only`. A missing-file message is a request for an artefact, not evidence that the check passed.

The bivariate models with child effects share `_bivariate_re_body.qmd`. Keep model-specific claims in each model's page. The reporting pipeline copies shared `_*.qmd` includes beside the rendered template.

## Report order

1. AI attribution, sampling banner, purpose, model summary and reading routes.
2. Model diagram and a short glossary of terms used on the page.
3. Statistical structure, prior settings, rationale and prior predictive checks.
4. Data composition and descriptive summaries.
5. Convergence verdict, diagnostics and prior-to-posterior comparison.
6. Findings, expected trajectories, predictions and monthly tables.
7. Sensitivity and recovery evidence, predictive calibration and out-of-sample assessment.
8. Limits on interpretation and use.

Keep these section IDs when changing headings, because reading routes link to them: `sec-priors`, `sec-prior-predictive`, `sec-frame`, `sec-diagnostics`, `sec-findings`, `sec-predictions`, `sec-monthly`, `sec-calibration`, `sec-loo`, `sec-robustness`, `sec-limits` and, on joint pages, `sec-spoken-given-understood`. Put a navigation anchor on a plain wrapping div when a callout cannot carry it.

## Explain the quantity being shown

Write for a reader who understands basic arithmetic but may not know statistical terminology. Define a term before relying on it. Use a descriptive caption and explain how each figure answers the model's question. State the interval convention; the default outer and inner intervals contain 89% and 50% of posterior draws, with equal probability in their two tails.

Distinguish these targets wherever they appear:

- A reference curve sets study and child effects to zero. It need not equal the average or median of the sampled children.
- A child-averaged estimate integrates over a stated distribution of child effects. Name the source column, such as `p_subject_marginal_*`, when needed to remove ambiguity.
- A new child's expected trajectory includes uncertainty about persistent child effects.
- A future observed count also includes variation between assessments. Use this distribution for statements about a child's possible observed score.

For models with sex as a covariate, the reference curve uses the midpoint on the logit scale. It is not generally the arithmetic average of girls' and boys' expected counts.

Explain concentration, $\kappa$, as residual variation at a given expected count. Larger values mean less count variation. Its level depends on the outcome, denominator and other variation already represented in the model. Comparing two concentration curves alone does not establish a difference in total variation between children.

A curve of production ratio against comprehension can trace the reference child's path through age. It does not, by itself, estimate the spoken share among all children who understand a given number of words. Retain age markers and observed comparisons, and name the target in the caption.

## Explain what predictive scores assess

Match the LOO description to the likelihood term held out. For a multi-outcome model, holding out a spoken likelihood term while retaining observed comprehension is a conditional assessment. It is not a forecast of every outcome for a new child. Aggregating terms by administration does not by itself remove outcome information used in denominators or lag predictors.

VG15's standard LOO excludes the composition likelihoods that identify $\psi$. That score therefore cannot validate the sign-speech association. Lagged models also require checks that prevent the held-out outcome from entering another row's predictor. Use the dedicated child-held-out or forward-scoring procedures for the questions they were designed to answer.

See the [model inventory](README.md), [prior guide](PRIORS.md) and [critical-review programme](../runbooks/critical-review.md) for model roles, assumptions and release checks.
