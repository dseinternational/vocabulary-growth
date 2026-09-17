# Critical review programme

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17.

Agreed on 2026-08-05. Apply these reviews at the milestones below. A check is complete only when its findings and their disposition are recorded.

## Why this exists

Six substantive errors were found in the fortnight to 2026-08-05, and none was caught by a scheduled check. Each surfaced by accident while someone was doing something else. The list is worth keeping in view, because the reviews below are designed around it rather than around a generic quality checklist:

| What was wrong                                                                        | How it was found                |
| ------------------------------------------------------------------------------------- | ------------------------------- |
| Numbers in prose stale against the fits behind them                                   | By chance, repeatedly           |
| A reported effect stated in the wrong direction                                       | Reading output, not the claim   |
| Dispersion described as between-child heterogeneity, which it is not                  | A challenge to the wording      |
| A cross-population comparator pointed at a model without subject effects              | Reading the comparator's source |
| An appendix the report cited as its disclosure mechanism was an empty stub            | Following a cross-reference     |
| Two incorrect analysis recommendations, one withdrawn and one falsified by experiment | Checking them before acting     |

The last row is the important one. Recommendations from any source, human or machine, need the same adversarial treatment as the numbers.

## The reviews

Each is defined by what it checks, what it must produce, and what would count as a failure. A review that produces no artefact has not been run.

### R1. Number provenance

**Checks:** every numeral in report prose is either generated at render time from a current artefact, a fixed constant (the 810-item scale, the gate thresholds), or a cited figure from the literature. Nothing fit-derived is hard-coded.

**Produces:** a list of hard-coded fit-derived numbers with file, line, claimed value and current value.

**Fails if:** any prose number derived from a fit differs from the current artefact, or cannot be traced to one.

**Cadence:** after every refit, and before any draft freeze. Automate the source-value comparison where possible.

> Use the shared functions in `vocab_growth.report_cells` and each report's data helper to derive values and direction words from the same artefact. Label illustrative numbers as examples.

### R2. Claim and evidence

**Checks:** each substantive claim uses the right direction, quantity and population, and follows from the cited output.

**Produces:** claim-by-claim disposition (supported / overstated / unsupported / miscited).

**Fails if:** any claim is stated in a direction the posterior does not support, or attributes to a parameter a meaning it does not carry.

**Cadence:** before every draft freeze. Requires a reader who can challenge the interpretation.

### R3. Prior-to-posterior checks

**Checks:** every prior across every registered model, for prior CDF at the posterior mean and contraction (`1 − posterior sd / prior sd`). Flag parameters in either prior tail and those with contraction at or below zero. Non-positive contraction means the posterior standard deviation is at least as large as the prior standard deviation. It does not establish that the posterior equals the prior or that the data supplied no information. Examine changes in location and shape as well.

**Produces:** a table per model, with flags.

**Requires review if:** a reported quantity has non-positive contraction or a posterior mean outside the prior's central 90% interval. Record the explanation and any sensitivity analysis. A tail shift may reflect learning from the data; these flags are not automatic grounds for rejecting an estimate.

**Cadence:** after every refit. Mechanical.

### R4. Cross-reference and mechanism integrity

**Checks:** every `@sec-`/`@fig-`/`@tbl-` reference resolves; unused labels and figures are reviewed for relevance; every mechanism the report claims to have (an appendix, a disclosure path, a sensitivity analysis) exists and does what is claimed.

**Produces:** dangling references, orphaned labels, unreferenced figures, and claimed-but-absent mechanisms.

**Fails if:** the report describes machinery that is not there. This is the check that would have caught Appendix B.

**Cadence:** every render. Mechanical.

### R5. Data-rule audit

**Checks:** the exclusions and masks documented in prose match what the code applies, and the source files still contain what the provenance manifest says.

**Produces:** rule-by-rule agreement, with row counts.

**Fails if:** a documented exclusion is not applied, or an applied one is undocumented.

**Cadence:** on any data change. Mechanical.

### R6. Reproducibility spot-check

**Checks:** a model of record can be refitted using its recorded definition, data, code and environment. Compare diagnostics and substantive estimates, allowing for Monte Carlo variation.

**Produces:** the diagnostic comparison.

**Cadence:** once before release, on the headline models.

### R7. Adversarial statistical review

**Checks:** an independent reader, briefed to _break_ the conclusions rather than confirm them, tests whether the target quantities are identified, how conclusions depend on priors, which population they describe, and whether a simpler explanation fits.

**Produces:** written challenges with responses recorded, including challenges that were accepted.

**Cadence:** once before release. Human and external. The 2026-07 review (#157) is the model.

### R8. Plain-language and overclaiming review

**Checks:** whether a parent, teacher or other non-specialist reader would take away something the models do not support, especially around individual prediction, and whether uncertainty survives the translation into plain language.

**Produces:** passages that mislead, with suggested rewording.

**Cadence:** before release, on the summary and findings chapters. Human, and not the author.

### R9. Representativeness and ethics

**Checks:** whether the sample's limits are stated where a reader will meet the numbers, not only in a methods chapter. The studies are not a random sample of children with Down syndrome. Check that descriptions of individual expectations do not imply population norms.

**Cadence:** before release. Human, ideally including someone outside the project.

## Milestone gates

| Milestone             | Must pass        |
| --------------------- | ---------------- |
| After any refit       | R1, R3, R4       |
| After any data change | R5, then a refit |
| Before a draft freeze | R1, R2, R4       |
| Before release        | all of R1–R9     |

A failed review blocks the unsupported claim. Disclosure can explain a limitation, but it does not make every result suitable for publication or override the fit-validation requirements. Revise, qualify or remove the claim according to the finding.

## Automation

Use checked-in tools where they cover the question. `scripts/prior_vs_posterior.py` writes the prior comparison, while the fit and comparison validators check recorded provenance. The test suite checks the model inventory, notes index and runbook model lists. These checks do not establish that every claim, link or interpretation is sound. Review rendered pages and record the remaining checks explicitly.
