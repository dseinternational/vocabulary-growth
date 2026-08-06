# Reporting-config refit, 4–6 August 2026 — run record

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Run record. Covers the full reporting-quality refit of every registered model and the work it turned up. **Every model of record in this record is current as of 2026-08-06** and published. Parameter recovery and the #147 prior-sensitivity matrix were still running when this was written; §7 says so where it matters.
>
> The analytical findings live in their own notes and are not repeated here: [202608050900](202608050900-td-hierarchical-geometry.md) (typically-developing geometry), [202608051200](202608051200-project-review-update.md) (project review), [202608051500](202608051500-report-critical-review.md) (report review), [202608060900](202608060900-three-prior-conflicts.md) (the three prior conflicts). This note records what was run, in what order, and what broke.

## 1. Outcome

**All 15 registered models hold valid, reporting-quality fits.** Twelve clear both convergence tiers; three carry disclosed soft-tier caveats.

| model                     | state | caveats                                        |
| ------------------------- | ----- | ---------------------------------------------- |
| VG01–VG10, VG14–VG16      | clean | —                                              |
| VG11                      | valid | 22 divergent transitions                       |
| VG12                      | valid | 2 divergent transitions; energy BFMI 0.207     |
| VG13                      | valid | energy BFMI 0.245                              |

All 15 reports were rendered and published as one consistent set on 2026-08-06 at 13:22–13:24, the three caveated ones carrying their caveats visibly on their own face. Both books render. 26 commits on `run/2026-08-reporting-refit`.

## 2. How the fits actually went

The Down syndrome pool was fitted five-wide, the typically-developing models one at a time. Two models needed a documented hightune retry (VG08, VG09) after failing the gate at plain `rep`; both then passed. That is the ladder behaving as designed — they are the two subject-RE models without the per-draw GP anchoring VG10 has.

**VG11 was lost to an OOM after 7h20m** at 247 GB anon-RSS on a 251 GB machine, and had to be refitted from scratch. The arithmetic was predictable rather than unlucky: memory goes as `n_obs × draws` for the observation-sized deterministics stored per draw, and VG11 has 18,522 rows against VG12's 7,052 while running 48,000 draws against 36,000. Four concurrent `test`-config arms were running alongside and brought it forward; the operator error was checking CPU load and not memory headroom. Refitted at plain `rep` alone it peaked at 157 GB and completed in 3h29m.

VG12 was refitted twice more — once to carry the geometry changes, once to revert the `eta` widening (§4).

## 3. What was changed, and why

| change                                        | driver                                                     |
| --------------------------------------------- | ---------------------------------------------------------- |
| Disclose-and-publish path (`--allow-caveats`) | Code refused what `methods-workflow.qmd` documented as policy |
| DS/TD dispersion comparator repointed to VG10 | Read VG07, which has no subject effects — contrast inverted |
| #190 §B corrections                           | `kappa` was being read as between-child heterogeneity        |
| Centred study block on VG11/VG12              | `tau` ESS 310 → 6,950 measured at `test`                     |
| Variance partition on VG11/VG12               | Divergences 59 → 14; **does not** fix BFMI (§4)              |
| VG14 dispersion migrated to two anchors       | `b_kappa_mag_s` pinned ~4σ beyond its prior                  |
| Trivariate engine accepts the anchored form   | It could not express the fix VG14 needed                     |
| `--render-only` refreshes the report template | A report fix could never reach an existing fit               |
| Free signed peak (implemented, not enabled)   | Peak age asserted at 36 months; identifiable at 29.4         |

## 4. Proposals that failed

Four analytical proposals were made during the run. **Three did not survive contact with evidence**, and recording that is the point of this section.

- **Port per-draw GP anchoring to the TD trio** — withdrawn before implementation, twice wrong. The trio already had it; the obvious redirect to VG08/VG09 would have collapsed VG09 into VG10 and destroyed the ladder contrast that produced the evidence for anchoring.
- **The variance partition fixes the energy BFMI** — falsified by direct experiment. BFMI moved 0.203 → 0.192, i.e. not at all. The rotation worked (`v_total` energy correlation −0.025) but `share` inherited the whole of it (−0.737): the ridge rotates rather than dissolving, which is what happens when the obstacle is missing information rather than bad coordinates. Retained for the divergence reduction only.
- **Widen the TD `eta` priors** — implemented, then reverted the same day. It cost VG12 27 divergences (2 → 29) while only moving the amplitude to prior CDF 0.810 with contraction 0.166. It bought room without buying identification. Three isolating fits established it was the sole cause.
- **The fixed signed peak biases the fitted curve** — withdrawn; the residual analysis was confounded by study composition. The properly adjusted answer moved the peak *earlier*, not later, so the confounded comparison pointed the wrong way.

The last two withdrawals were the **same error**: comparing a population-level model curve against a raw pooled empirical mean without adjusting for a grouping the model accounts for. In the second case the confound was already documented in a note from June that I had not read.

## 5. Defects found by running the pipeline, not by reading it

Five, none of which a code review would have caught:

1. **Appendix B was an empty stub** with a dangling cross-reference — the report cited a disclosure mechanism that did not exist.
2. **`--render-only` re-rendered the template frozen at fit time**, so VG13 — one of only three fits with anything to disclose — kept rendering without its caveat and exiting 0.
3. **A published figure was titled for the wrong model** ("Concentration (mean-independent, study-RE only)"), left stale by the comparator repointing.
4. **The runbook's render sequence omitted `prepare_report_figures.py`**, which the figure sync destroys. Following §3 as written fails.
5. **The R3 sweep excluded two model families wholesale**, hiding VG14's 4σ conflict on a parameter it could already reconstruct. A diagnostic that skips a subject reports nothing rather than a gap, and silence reads as a pass.

## 6. The standing limitation

VG12 and VG13 fail the soft tier on energy BFMI and **this is now understood rather than suspected**. The models cannot cleanly separate persistent between-child differences from within-child noise, because most typically-developing children in the pool are measured once. Nothing tried moves it, and the falsification test predicts nothing will: the only remedy is more repeat measurement.

VG11 clears the threshold despite the lowest repeat *rate*, which forced a correction to the mechanism — the operative quantity is the **absolute amount** of within-child replication, not the proportion of children with any.

Because the Down syndrome side samples markedly better, any DS/TD heterogeneity contrast is **asymmetrically affected**: the typically-developing interval is the less reliable of the two. That belongs in the text of any such contrast, not only in the convergence appendix.

## 7. Outstanding

**Running as this was written**: headline parameter recovery (VG10, VG12, VG15 at `test`, 3 replicates each).

**Not started**: the #147 Target 8 prior-sensitivity matrix.

**Report content, blocking a defensible draft**: two stale hard-coded numbers in `intro.qmd`, two dangling cross-references, five figures shown but never referenced, and the "~34 children" figure for ψ, which could not be verified and carries argumentative weight.

**Report content, blocking release**: six chapters are 3–7 line stubs — summary, discussion, signing results, shared DS caveats, reference tables, acknowledgements — plus four introduction subsections, the model-specifications half of Appendix B, and the plain-language summary the introduction promises. This dominates the timeline; the modelling does not.

**Owner decisions**: VG14's role against VG15; whether to adopt the free signed peak (a graph change on a headline model); and what happens to the superseded 2026-07-24 and 2026-08-04/05 publications, which remain live and publicly reachable alongside the current set.

**Recommended and not yet done**: trim signed reporting to where signed evidence stops. `r(a)` is reported to 115 months on 23 observations above 60 and none between 84 and 96, while comprehension was already capped at 72 months on exactly that argument.
