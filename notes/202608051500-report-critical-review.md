# Critical review of the report draft, 2026-08-05

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> First run of the review programme in [`docs/runbooks/critical-review.md`](../docs/runbooks/critical-review.md), covering R1 (number provenance), R4 (cross-reference and mechanism integrity) and a partial R2 (claim and evidence). Run against the draft at commit `a4ea822`, while VG12 was refitting — so the artefacts several checks compare against are about to change. This is a **drafting worklist**, not a quality gate.

## 1. The headline finding: the report is a skeleton, not a draft

About **700 lines of actual prose**. One results chapter is written; the rest of the narrative is 3–7 line stubs.

| Chapter                                | Lines | State                        |
| -------------------------------------- | ----: | ---------------------------- |
| `methods-models.qmd`                   |   214 | written                      |
| `methods-workflow.qmd`                 |   121 | written                      |
| `intro.qmd`                            |   108 | written, 4 TODO subsections  |
| `glossary.qmd`                         |    95 | written                      |
| `appendix-specs.qmd`                   |    93 | convergence half written today; specifications still absent |
| `results-words-understood-spoken.qmd`  |    78 | **the only written results chapter** |
| `methods-data.qmd`                     |    60 | written                      |
| `_caveats-signing.qmd`                 |    11 | written                      |
| `appendix-ai.qmd`                      |    12 | written                      |
| `_caveats-ds.qmd`                      |     9 | **TODO**                     |
| `discussion.qmd`                       |     7 | **TODO**                     |
| `results-words-signed-total-expressive.qmd` | 6 | **TODO**                     |
| `appendix-reference-tables.qmd`        |     5 | **TODO**                     |
| `summary.qmd`                          |     3 | **TODO**                     |
| `acknowledgements.qmd`                 |     3 | **TODO**                     |

The methods are in good shape. The interpretation — the part that will be quoted — is unwritten. Any estimate of time to publication is dominated by this, not by the modelling.

The figure cache (`docs/report/figures/`) held **2 files** at the time of review, so the report cannot currently render with its figures. That is expected mid-refit and clears when `sync_report_figures.py` runs.

## 2. R1 — Number provenance: two stale claims, both in the introduction

The report is mostly clean here, because most numbers in prose are fixed constants (the 810-item scale, the gate thresholds `1.01` / `400` / `0.3`) or citations to the literature. Two exceptions, both in `intro.qmd`, both **hard-coded and both stale**:

| Location | Claim | Current VG10 |
| -------- | ----- | ------------ |
| `intro.qmd` §"Estimates as distributions" | "the typical 36-month-old child with Down syndrome learns about **5** new words per month … between **2 and 7**" | **6.25** words/month, 89% interval **[4.26, 8.15]** |
| `intro.qmd` §"Predictions for individual children" | "an 89% probability that a 36-month-old child with Down syndrome understands between **0 and 142** words" | **[56, 482]** population predictive; **[78, 458]** subject-marginal |

The second is the more serious. It appears as the showcase of what the method delivers for an individual child, and it is wrong in a way a reader will notice and act on: it implies a 36-month-old might understand **no words**, where the model puts the lower bound at 56 and the median at 218. The upper bound is out by more than a factor of three.

Both are presented as illustrations of what a Bayesian statement looks like, but neither is marked as illustrative, and both name the study population and a specific age. **Recommendation:** generate them at render time from VG10, as `results-words-understood-spoken.qmd` already does for its findings paragraph, or mark them explicitly as invented illustrations and change the population so they cannot be read as results.

### Checked and sound

- `_caveats-signing.qmd`'s "~60 uk_02 four-cell rows" — VG15 reports **56**. Consistent.
- The `810`, `1.01`, `400`, `0.3` constants throughout — all correct.
- Literature figures in `intro.qmd` (27 children, 19–38 months, 17 months) — citations, not our output.

### Needs verification, not yet an error

`_caveats-signing.qmd` and `methods-workflow.qmd` both say the sign–speech association ψ rests on "~34 children". This could not be confirmed from the fit output; an unmasked query gives 55 children over those 56 rows, which is an upper bound because VG15 applies further masking. **The number may be right.** It matters because `methods-workflow.qmd` uses it to justify leading with intervals over point estimates for ψ, so the figure carries argumentative weight.

## 3. R4 — Cross-reference and mechanism integrity

**Two genuinely dangling references**, both pointing at unwritten material:

- `@sec-findings-signing` — cited from **`intro.qmd` and `methods-models.qmd`**. The signing results chapter is a 6-line stub, so the target does not exist. `intro.qmd` uses it for the evidence-grading scheme applied to ψ.
- `@sec-lineage-ds` — cited from `intro.qmd`, for the iterative-workflow argument.

**Five figures displayed but never referenced in prose** — `fig-ds-spoken`, `fig-ds-understood`, `fig-ds-signed`, `fig-ds-us` (`methods-data.qmd`) and `fig-q-at-u` (`results-words-understood-spoken.qmd`). A reader meets a figure with nothing pointing at it and no instruction on what to take from it.

Eleven further labels are defined but unreferenced; most are section anchors that will be referenced once the narrative chapters exist, and are not defects yet.

**Mechanism integrity:** the disclosure path the report claims now exists — Appendix B renders the convergence table and the caveats table, and each per-model report carries its own caveat callout. This was the R4-class failure that prompted the programme, and it is closed.

## 4. R2 (partial) — Claim and evidence

Only the written chapters could be reviewed. Nothing miscited was found in them, and one pattern is worth preserving:

`results-words-understood-spoken.qmd` generates every number **and the direction word** in its findings paragraph from `dq_facts()`, over the same coverage-filtered table it tabulates. The paragraph cannot outlive the fits behind it, and cannot silently invert if the contrast changes sign. **This is the pattern the rest of the report should follow** — and had it been used in `intro.qmd`, §2's two errors could not have occurred.

Its limitations section is honest about the right things: the BFMI caveat on the typically-developing side, the coverage limit at ~200 understood words, and that matching on the *number* of words understood does not match *which* words.

One wording point for R2 when the chapter is revisited: the limitations say VG13 "carries an energy-BFMI caveat below 0.3". True, but §8 of [202608050900](202608050900-td-hierarchical-geometry.md) establishes something stronger and more useful — the contrast is **asymmetrically** affected, because the Down syndrome side samples markedly better, so the typically-developing interval is the less reliable of the two. Worth stating, since it bears directly on how the contrast should be read.

## 5. Worklist

**Blocking a draft freeze**

1. Fix or mark the two `intro.qmd` numbers (§2). Prefer generating them.
2. Resolve `@sec-findings-signing` and `@sec-lineage-ds`, or remove the references until their targets exist.
3. Reference the five orphaned figures in prose, or drop them.
4. Confirm or correct the "~34 children" figure for ψ (§2).

**Blocking release**

5. Write the six stub chapters. This is the critical path.
6. Complete the four `intro.qmd` TODO subsections — the literature review on expressive lag, rates of word learning, gestures and signs, and the study aims.
7. Write the model specifications half of Appendix B.
8. Produce the plain-language summary the introduction promises and links to.
9. Run R2 in full over the finished narrative, then R7–R9.

**Not blocking, but worth doing while the chapters are written**

10. Adopt the `dq_facts()` pattern wherever a number appears in prose, so R1 becomes a formality rather than an audit.
