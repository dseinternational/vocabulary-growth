# Thinning VG08's replication to the typically developing profile reproduces the typically developing pathology

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

Date: 2026-09-07. Three fits of VG08 at `rep` on this project's own Windows workstation, plus a second seed of two of them. Runs [#229](https://github.com/dseinternational/vocabulary-growth/issues/229)'s proposed "cheap decisive test" and scopes the remedy for [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 4.6. Corrects one measurement in [202609061900](202609061900-td-bfmi-is-the-tau-kappa-ridge.md); the correction is appended there as a flagged block and argued here.

## The question

Every typically developing (TD) model carrying a child effect sits below the 0.3 energy-BFMI threshold, and in each the posterior correlation between the child scale `tau_subject` and the young-age concentration `kappa_young` is strongly positive (+0.57 to +0.76). [202609061900](202609061900-td-bfmi-is-the-tau-kappa-ridge.md) localised the energy to that τ–κ block and offered the Down syndrome (DS) family as a control: VG08 has the same two random effects on the same probability scale, half its children are seen more than once, and its ridge was reported as absent. #229's 2026-09-06 comment then proposed the test that would separate replication from everything else the two populations differ in: **thin the DS pool to the TD replication profile and see whether the ridge appears**, with a size-matched random-drop arm as the control.

## What was fitted

`scripts/experiments/vg08_replication_thinning.py` builds VG08's registered analysis frame (1,708 rows, 943 children, 46.6% seen twice or more, after the `us_03` ingestion) and derives two arms from it:

- **thinned** — the same 943 children, but repeaters are truncated to one randomly chosen visit until the share of children with two or more visits equals VG12's 17.2%. 162 repeaters are kept intact and 277 are reduced to a singleton; 502 rows leave. Only extra visits are removed, so the number of child effects is unchanged and replication is the only thing that moves. (The #229 comment sized this at 297 rows on the assumption that a truncated repeater loses one row; repeaters carry up to eight visits.)
- **control** — 502 rows dropped uniformly at random from the full frame. 174 children vanish with them; the repeater share falls only to 37.2%. Study and child codes are re-issued densely for the rows present, because the engine sizes the child block from the maximum code and a first run of this arm carried 943 child-effect slots for 769 children (that run's min BFMI was 0.393 against the 0.342 below; how much of the gap is the 174 prior-only dimensions and how much is run-to-run variation is not separable from one pair, and the dense run is the one reported).

Each arm and the unmodified **baseline** are fitted with the bivariate RE engine exactly as the fold-fit path fits them (`fold_fits.fit_holdout_fold` without the observation-level deterministics), at the `rep` configuration (6 chains × 6,000 draws after 6,000 tuning, target accept 0.95), and scored on the trace: min BFMI per chain (Betancourt 2016), the correlation of `tau_subj_u` with κ_u at the 24-month young anchor (read off the stored `kappa_u_query` grid — VG08's κ block emits no `kappa_young` deterministic), the correlation of each with the marginal energy, and the marginal energy SD against √(d/2). The traces are under `output/experiments/vg08-replication-thinning*/` on the workstation; the per-arm JSON and `summary.csv` beside them hold every number here.

## Result

Seed 47:

| arm      |  rows | children | singletons | repeaters |  min BFMI | corr(τ, κ young) | corr(τ, energy) | corr(κ young, energy) | energy SD ÷ √(d/2) | div | max R-hat |
| -------- | ----: | -------: | ---------: | --------: | --------: | ---------------: | --------------: | --------------------: | -----------------: | --: | --------: |
| baseline | 1,708 |      943 |        504 |     46.6% |     0.424 |       **+0.384** |          −0.627 |                −0.515 |               2.11 |   0 |    1.0082 |
| control  | 1,206 |      769 |        483 |     37.2% |     0.342 |           +0.508 |          −0.673 |                −0.635 |               2.32 |   0 |    1.0088 |
| thinned  | 1,206 |      943 |        781 |     17.2% | **0.267** |       **+0.611** |          −0.748 |                −0.708 |               2.67 |   0 |    1.0048 |

For comparison, the TD fits of record ([202609061900](202609061900-td-bfmi-is-the-tau-kappa-ridge.md)): VG12 min BFMI 0.208, ridge +0.757, energy ratio 3.08; VG13 0.248, +0.576, 2.81; VG23 0.243, +0.566, 2.84; VG21 0.263, +0.692, 2.68.

Seed 48 (thinned and control re-drawn; the baseline does not depend on the seed):

| arm     |  rows | children | repeaters |  min BFMI | corr(τ, κ young) | corr(τ, energy) | corr(κ young, energy) | energy SD ÷ √(d/2) | div | max R-hat |
| ------- | ----: | -------: | --------: | --------: | ---------------: | --------------: | --------------------: | -----------------: | --: | --------: |
| control | 1,220 |      760 |     38.6% |     0.364 |           +0.489 |          −0.672 |                −0.612 |               2.26 |   0 |    1.0038 |
| thinned | 1,220 |      943 |     17.2% | **0.278** |       **+0.630** |          −0.744 |                −0.693 |               2.53 |   0 |    1.0029 |

The second draw reproduces the first on every column: thinned 0.267 → 0.278 and +0.611 → +0.630, control 0.342 → 0.364 and +0.508 → +0.489. The ordering baseline > control > thinned, and the threshold crossing by the thinned arm alone, do not depend on which repeaters the thinning happened to keep.

**Reading it against the three outcomes #229 set out in advance.** The thinned arm shows the TD pathology in full: BFMI below the threshold, a ridge in the TD range, κ risen to the second energy correlate (`a_kappa_u` −0.447, from −0.158 in the baseline), and a heavy-tailed energy distribution at 2.67× the reference — on Down syndrome data, with the same 943 child effects the baseline has. The size-matched control moves in the same direction but stops short: it stays above the threshold and its ridge sits between the other two. So the answer is not the clean first row of #229's table ("thinned shows the ridge, control does not") but a graded one: **losing rows costs energy exploration, and losing replication costs more, and only the replication loss crosses the threshold.** The quantity the three arms are ordered by is not the row count and not the repeater share alone but the number of child effects resting on a single observation — 504, 483, 781 — with the control's fewer children partly offsetting its fewer rows.

## A correction to the 2026-09-06 note

That note's control table gives VG08's corr(τ_subj_u, κ_young_u) as **−0.001** and rests its headline on it: "the τ–κ ridge is not general … VG08's correlation between child scale and concentration is −0.001". The baseline arm here — the same model, the same frame, BFMI 0.424 against the VM fit's 0.421 and corr(τ, energy) −0.627 against −0.632 — puts that correlation at **+0.384**. The correlation of τ with `kappa_min_u`, the old-age floor of the κ curve, is +0.012.

The explanation is that VG08 is the only model in that table _with a child effect_ whose κ block is the `kappa_min + exp(a + b·z)` form (`KappaPriorParams`, shared with VG05 and VG07) and so emits no `kappa_young_u` deterministic; VG10, VG16 and every TD model use the two-anchor form and do. The note's script is not committed, but a −0.001 for VG08 is what a fallback to `kappa_min_u` gives, and it is not what κ at the young anchor gives. The "ridge" the TD models show is between the child scale and the **young-age** concentration, which is where the child effect and the dispersion compete; the floor is a different parameter.

What changes: the claim that the ridge is TD-specific is withdrawn. VG08 carries it at +0.38 — weaker than the TD models' +0.57 to +0.76, and with a BFMI that clears the gate, but present. What survives, and is now better supported than before: the energy cost of a child effect is general (VG07 0.784 → VG08 0.421 stands), and the ridge grows with the share of child effects that have no replication behind them. The thinned arm is that statement made on one dataset with everything else held fixed, which the two-population contrast could not do.

## What it means for task 4.6

Task 4.6 is written as "reparameterise the responsible blocks". This experiment says the pathology is not a parameterisation. The same graph, the same priors and the same population go from a passing BFMI to a failing one when 277 children lose their second visit. A child effect identified by one Beta-Binomial count and a dispersion identified by the same count are the textbook under-identified two-level decomposition (#229's reframe), and no coordinate change supplies the information the second visit carried. That is consistent with what the earlier arms found: the variance partition (`geom_arm.py`, 2026-08-05) rotated the coordinates and did not fix the BFMI, and VG12, which has it, is the worst of the five.

So the options for the TD refit narrow to the ones that supply information or stop asking for it:

1. **Calibrate the split from the children who can identify it** (#229 option 2, generalised). Fit each TD model restricted to its replicated children — 1,947 for VG11, 1,000 for VG12, 830 for VG13 — and use that posterior to set the prior on the child scale (or on `subject_variance_share` where the partition exists) for the full fit. The singletons then contribute to the trajectory and the dispersion under a prior on τ that the repeaters have already determined, rather than prising τ and κ apart by functional form. It is the only option that could repair the geometry without changing the estimand or the data; whether it does is an empirical question, and one that is answerable at `test` tier on VG12 before the refit — fit the repeaters-only subset, set the prior, refit, and read the BFMI and the ridge. Its cost is honesty about what it is: the reported τ_TD becomes a repeaters-informed quantity, which the report has to say.
2. **Move the published contrast to total scatter at matched age** (#229 option 4). The DS-versus-TD between-child contrast is the reason τ_TD matters, and `v_total` is what recovery says comes back cleanly. This is the reporting-side change and can be made whatever else is done.
3. **Admit more replicated TD data** (#229 option 6). The real fix and the slow one; it interacts with the language scope and the study threshold (#240 item 11) and is not a refit-window item.

**Recommendation for the owner's decision:** take 1 as the graph change for the TD refit, with 2 as the way the contrast is reported, and record 3 as the direction. Not recommended: a further coordinate change to the τ–κ block, for the reason above; or removing the TD child effects, which [202608050900](202608050900-td-hierarchical-geometry.md) §7 measured as pushing the variance into `kappa`, a reported estimand.

One thing this does **not** license is treating the TD fits' current BFMI as harmless because VG08 clears the gate with the same ridge at +0.38. The four failing TD models are at +0.57 to +0.76 with singleton shares of 82.8% (VG12) and 84.9% (VG13, VG23; VG21's was not measured); the thinned arm reproduces both numbers from 82.8% singletons. The caveat on the TD interval bounds stands until the refit.

## Caveats

- **One model, one population.** VG08 is a constant understood child effect on the DS joint frame. The TD models differ in instrument, age window, pool size and mean structure; this shows that replication alone is sufficient to produce their geometry, not that nothing else contributes.
- **Two seeds.** The thinning chooses which repeaters survive and which visit a truncated child keeps; the second seed is the check that the result is not a draw. It is not a distribution over draws.
- **The control confounds child count with row count.** No random-drop arm can hold both fixed; this one holds rows. Its 174 lost children are why its singleton count (483) is _below_ the baseline's, and why it should not be read as "sample size alone gets you most of the way": it lost rows and gained relative replication at the same time.
- **κ at the young anchor is read at 24 months** from the query grid, which is VG08's low slope anchor. The TD models' `kappa_young` is at their own young anchor (12 months for VG12). Both are "κ where the child effect and the dispersion compete", but they are not the same age.
- **BFMI is a statistic of one energy series per chain**, and the energy-SD ratio is another statistic of the same series; the two columns agree with each other by construction. The ridge column is the independent one.
- **Singleton share is sufficient within a dataset, not a law across models.** VG11 has the most singletons of any TD model (86.6%) and the weakest ridge (+0.29) and the best BFMI (0.363), as [202609061900](202609061900-td-bfmi-is-the-tau-kappa-ridge.md) already noted. It also has fourteen thousand children, an outcome whose young-age proportion sits at 0.012, and the variance partition. Holding everything else fixed, less replication produces the TD geometry; across models that differ in everything, other things move the ridge too.
- **These are not fits of record and nothing here is publishable.** The experiment root is outside `output/models/`; the harness is a dated record under `scripts/experiments/`, outside the fit-consumer ratchet, as that directory's README describes.

## Reproduction

```bash
uv run python scripts/prepare_data.py   # the local DuckDB must include us_03
uv run python scripts/experiments/vg08_replication_thinning.py all --output-dir output/experiments/vg08-replication-thinning
uv run python scripts/experiments/vg08_replication_thinning.py thinned control --seed 48 --output-dir output/experiments/vg08-replication-thinning-seed48
```

About 45 minutes for the three seed-47 arms on a 16-core, 95 GB Windows workstation with two arms running side by side (baseline 18.8 min, thinned 14.3, control 12.4).
