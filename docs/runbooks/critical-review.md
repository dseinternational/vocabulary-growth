# Critical review programme

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Agreed 2026-08-05. Milestone-gated for now; the mechanical reviews are candidates for automation **once the report chapters exist**, because a stale-number check needs written prose to check.

## Why this exists

Six substantive errors were found in the fortnight to 2026-08-05, and none of them was caught by a scheduled check — each surfaced by accident while someone was doing something else. The list is worth keeping in view, because the reviews below are designed around it rather than around a generic quality checklist:

| What was wrong                                                                            | How it was found                |
| ----------------------------------------------------------------------------------------- | ------------------------------- |
| Numbers in prose stale against the fits behind them                                       | By chance, repeatedly           |
| A reported effect stated in the wrong direction                                           | Reading output, not the claim   |
| Dispersion described as between-child heterogeneity, which it is not                      | A challenge to the wording      |
| A cross-population comparator pointed at a model without subject effects                  | Reading the comparator's source |
| An appendix the report cited as its disclosure mechanism was an empty stub                | Following a cross-reference     |
| Two analysis recommendations that were wrong — one withdrawn, one falsified by experiment | Checking them before acting     |

The last row is the important one. Recommendations from any source, human or machine, need the same adversarial treatment as the numbers.

## The reviews

Each is defined by what it checks, what it must produce, and what would count as a failure. A review that produces no artefact has not been run.

### R1 — Number provenance

**Checks:** every numeral in report prose is either generated at render time from a current artefact, a fixed constant (the 810-item scale, the gate thresholds), or a cited figure from the literature. Nothing fit-derived is hard-coded.

**Produces:** a list of hard-coded fit-derived numbers with file, line, claimed value and current value.

**Fails if:** any prose number derived from a fit differs from the current artefact, or cannot be traced to one.

**Cadence:** after every refit, and before any draft freeze. Mechanical — automate first.

> Precedent: `results-words-understood-spoken.qmd` generates every number _and the direction word_ from `dq_facts()` at render time, so the paragraph cannot outlive its fits. That is the pattern; prose numbers should be the exception, and marked when illustrative.

### R2 — Claim and evidence

**Checks:** each substantive claim follows from the output it cites — right direction, right quantity, right population, and the cited artefact actually contains it.

**Produces:** claim-by-claim disposition (supported / overstated / unsupported / miscited).

**Fails if:** any claim is stated in a direction the posterior does not support, or attributes to a parameter a meaning it does not carry.

**Cadence:** before every draft freeze. Needs judgement — human or adversarial agent, not a script.

### R3 — Prior–posterior conflict sweep

**Checks:** every prior across every registered model, for prior CDF at the posterior mean and contraction (`1 − posterior sd / prior sd`). Flags parameters pressed into a prior tail, and parameters with contraction at or below zero, where the posterior is reporting the prior back.

**Produces:** a table per model, with flags.

**Fails if:** a reported quantity has contraction ≤ 0, or sits beyond the prior's 95th percentile, without that being stated where it is reported.

**Cadence:** after every refit. Mechanical.

### R4 — Cross-reference and mechanism integrity

**Checks:** every `@sec-`/`@fig-`/`@tbl-` reference resolves; every defined label is referenced; every figure the report displays is pointed at by prose; every mechanism the report claims to have (an appendix, a disclosure path, a sensitivity analysis) exists and does what is claimed.

**Produces:** dangling references, orphaned labels, unreferenced figures, and claimed-but-absent mechanisms.

**Fails if:** the report describes machinery that is not there. This is the check that would have caught Appendix B.

**Cadence:** every render. Mechanical.

### R5 — Data-rule audit

**Checks:** the exclusions and masks documented in prose match what the code applies, and the source files still contain what the provenance manifest says.

**Produces:** rule-by-rule agreement, with row counts.

**Fails if:** a documented exclusion is not applied, or an applied one is undocumented.

**Cadence:** on any data change. Mechanical.

### R6 — Reproducibility spot-check

**Checks:** a model of record can be refitted from its recorded manifest and reproduce its diagnostics.

**Produces:** the diagnostic comparison.

**Cadence:** once before release, on the headline models.

### R7 — Adversarial statistical review

**Checks:** an independent reader, briefed to _break_ the conclusions rather than confirm them, attacks the headline claims — the identification of each estimand, the sensitivity of each to its priors, the population each generalises to, and whether a simpler explanation fits.

**Produces:** written challenges with responses recorded, including challenges that were accepted.

**Cadence:** once before release. Human and external. The 2026-07 review (#157) is the model.

### R8 — Plain-language and overclaiming review

**Checks:** whether a non-specialist reader — a parent, a teacher — would take away something the models do not support, especially around individual prediction, and whether uncertainty survives the translation into plain language.

**Produces:** passages that mislead, with suggested rewording.

**Cadence:** before release, on the summary and findings chapters. Human, and not the author.

### R9 — Representativeness and ethics

**Checks:** whether the sample's limits are stated where a reader will meet the numbers, not only in a methods chapter. These are not a random sample of children with Down syndrome, and the report is intended to set expectations for individual children.

**Cadence:** before release. Human, ideally including someone outside the project.

## Milestone gates

| Milestone             | Must pass        |
| --------------------- | ---------------- |
| After any refit       | R1, R3, R4       |
| After any data change | R5, then a refit |
| Before a draft freeze | R1, R2, R4       |
| Before release        | all of R1–R9     |

A review that fails does not block the work; it blocks the _claim_. The disclose-and-publish path exists for exactly this reason — a known, stated limitation is publishable, an unstated one is not.

## Automation

Deferred until the report chapters exist. R1, R3, R4 and R5 are mechanical and should be automated first; R4's cross-reference scan and R3's prior sweep already exist as ad-hoc scripts and need only be promoted to checked-in tools with recorded output. R2 and R6–R9 need judgement and should stay human-triggered.
