# Critical review of the report draft, 2026-08-05

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> First run of the review programme in [`docs/runbooks/critical-review.md`](../docs/runbooks/critical-review.md), covering R1 (number provenance), R3 (prior-data conflict), R4 (cross-reference and mechanism integrity) and a partial R2 (claim and evidence). Run against the draft at commit `a4ea822`, while VG12 was refitting — so the artefacts several checks compare against are about to change. This is a **drafting worklist**, not a quality gate.

## 1. The headline finding: the report is a skeleton, not a draft

About **700 lines of actual prose**. One results chapter is written; the rest of the narrative is 3–7 line stubs.

| Chapter                                     | Lines | State                                                       |
| ------------------------------------------- | ----: | ----------------------------------------------------------- |
| `methods-models.qmd`                        |   214 | written                                                     |
| `methods-workflow.qmd`                      |   121 | written                                                     |
| `intro.qmd`                                 |   108 | written, 4 TODO subsections                                 |
| `glossary.qmd`                              |    95 | written                                                     |
| `appendix-specs.qmd`                        |    93 | convergence half written today; specifications still absent |
| `results-words-understood-spoken.qmd`       |    78 | **the only written results chapter**                        |
| `methods-data.qmd`                          |    60 | written                                                     |
| `_caveats-signing.qmd`                      |    11 | written                                                     |
| `appendix-ai.qmd`                           |    12 | written                                                     |
| `_caveats-ds.qmd`                           |     9 | **TODO**                                                    |
| `discussion.qmd`                            |     7 | **TODO**                                                    |
| `results-words-signed-total-expressive.qmd` |     6 | **TODO**                                                    |
| `appendix-reference-tables.qmd`             |     5 | **TODO**                                                    |
| `summary.qmd`                               |     3 | **TODO**                                                    |
| `acknowledgements.qmd`                      |     3 | **TODO**                                                    |

The methods are in good shape. The interpretation — the part that will be quoted — is unwritten. Any estimate of time to publication is dominated by this, not by the modelling.

The figure cache (`docs/report/figures/`) held **2 files** at the time of review, so the report cannot currently render with its figures. That is expected mid-refit and clears when `sync_report_figures.py` runs.

## 2. R1 — Number provenance: two stale claims, both in the introduction

The report is mostly clean here, because most numbers in prose are fixed constants (the 810-item scale, the gate thresholds `1.01` / `400` / `0.3`) or citations to the literature. Two exceptions, both in `intro.qmd`, both **hard-coded and both stale**:

| Location                                           | Claim                                                                                                            | Current VG10                                                        |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `intro.qmd` §"Estimates as distributions"          | "the typical 36-month-old child with Down syndrome learns about **5** new words per month … between **2 and 7**" | **6.25** words/month, 89% interval **[4.26, 8.15]**                 |
| `intro.qmd` §"Predictions for individual children" | "an 89% probability that a 36-month-old child with Down syndrome understands between **0 and 142** words"        | **[56, 482]** population predictive; **[78, 458]** subject-marginal |

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

Its limitations section is honest about the right things: the BFMI caveat on the typically-developing side, the coverage limit at ~200 understood words, and that matching on the _number_ of words understood does not match _which_ words.

One wording point for R2 when the chapter is revisited: the limitations say VG13 "carries an energy-BFMI caveat below 0.3". True, but §8 of [202608050900](202608050900-td-hierarchical-geometry.md) establishes something stronger and more useful — the contrast is **asymmetrically** affected, because the Down syndrome side samples markedly better, so the typically-developing interval is the less reliable of the two. Worth stating, since it bears directly on how the contrast should be read.

## 4a. R3 — Prior–data conflict sweep across the model family

Run with `scripts/prior_vs_posterior.py --table`, added for this purpose. 157 parameters across 13 models (VG14 and VG15 carry priors the script does not yet reconstruct; VG12 was skipped because its trace predates the `eta` revert). Two diagnostics per parameter: **prior CDF** at the posterior mean, and **contraction** = 1 − posterior sd / prior sd. Twelve flagged.

> The sweep's first run flagged VG12's `eta` at prior CDF 0.991 with contraction −0.670. That was an artefact — an `eta = 1.0` posterior being scored against the `eta = 0.5` prior it had just been reverted to. The script now checks each trace against the current registered definition and skips a stale one. A conflict diagnostic that silently compares mismatched pairs is worse than none.

### The dispersion slope constraint is being fought, hard (VG05, VG07, VG08)

| model | parameter       | posterior mean | prior CDF | contraction |
| ----- | --------------- | -------------: | --------: | ----------: |
| VG08  | `b_kappa_mag_s` |          1.229 | **1.000** |  **−0.164** |
| VG07  | `b_kappa_mag_s` |          1.242 | **1.000** |  **−0.077** |
| VG05  | `b_kappa_mag_s` |          1.213 | **1.000** |  **−0.095** |
| VG08  | `b_kappa_mag_u` |          0.984 |     0.999 |       0.346 |
| VG05  | `b_kappa_mag_u` |          0.773 |     0.990 |       0.441 |
| VG07  | `b_kappa_mag_u` |          0.738 |     0.986 |       0.448 |

The most severe conflict in the family. On the spoken side the posterior mean sits beyond essentially the _entire_ prior, and the posterior is **wider** than the prior — the likelihood is pushing against a boundary the prior will not let it cross.

These three still use the **legacy** dispersion parameterisation, in which `b_kappa_mag ≥ 0` forces dispersion to fall with age. That constraint is already documented as wrong elsewhere: `KappaAnchorPriorParams` records that the sign restriction "forces `kappa` to fall with age, which the typically-developing comprehension data reject", and the two-anchor form was introduced precisely to free it. VG05, VG07 and VG08 were never migrated.

They are development-step models rather than models of record, which limits the damage — but their dispersion estimates are quoted in the lineage comparison, and a parameter pinned at its prior boundary is not an estimate.

### `eta` presses in three further models, and widening is not the fix

| model | parameter | posterior mean | prior CDF | contraction |
| ----- | --------- | -------------: | --------: | ----------: |
| VG03  | `eta`     |          1.191 |     0.983 |       0.091 |
| VG01  | `eta`     |          1.028 |     0.960 |       0.277 |
| VG11  | `eta`     |          1.016 |     0.958 |       0.130 |

§5 identified this in VG12 and treated it as a TD oversight. It is not: it is family-wide, and reaches VG01 (Down syndrome spoken) and VG03 (typically-developing spoken) as well as VG11.

**The obvious remedy is now known to be wrong.** Widening VG12's `eta` from 0.5 to 1.0 took divergences from 2 to 29 while only moving the amplitude to prior CDF 0.810 with contraction 0.166 — it bought room without buying identification, and was reverted. Whatever fixes this has to _identify_ the GP amplitude, not merely free it. Recorded as an open problem rather than a pending change.

### VG13 cannot inform its GP hyperparameters at all

| model | parameter    | prior CDF | contraction |
| ----- | ------------ | --------: | ----------: |
| VG13  | `eta_q`      |     0.573 |  **−0.003** |
| VG13  | `ell_unit_u` |     0.517 |   **0.026** |
| VG13  | `ell_unit_q` |     0.493 |  **−0.017** |

All three sit mid-prior — so a prior-CDF check alone would pass them — but with contraction at or below zero. The 8–18 month window is too short to inform a length-scale or an amplitude, and VG13 is reporting its priors back for all three. **This is the case for the contraction diagnostic**: the pressing check would have missed it entirely.

VG13 supplies the typically-developing side of the matched-comprehension contrast in the only written results chapter, so this belongs in that chapter's limitations alongside the BFMI caveat.

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

**Raised by R3, needing decisions rather than edits**

11. Migrate VG05, VG07 and VG08 to the two-anchor dispersion form, or state in the lineage comparison that their `b_kappa_mag` estimates sit at a prior boundary and are not estimates (§4a).
12. `eta` presses in VG01, VG03, VG11 and VG12. Widening is known not to work. Needs a change that identifies the amplitude — open problem, no owner.
13. Record VG13's three uninformed GP hyperparameters in the limitations of the matched-comprehension chapter, which uses VG13 for its typically-developing side.

**Not blocking, but worth doing while the chapters are written**

10. Adopt the `dq_facts()` pattern wherever a number appears in prose, so R1 becomes a formality rather than an audit.
