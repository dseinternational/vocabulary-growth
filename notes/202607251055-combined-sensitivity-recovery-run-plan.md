# Combined robustness run: Target-8 / P0-P1 sensitivities + parameter recovery (#147, #163) — plan

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> This is a **plan**, not a run record. Nothing below has been run. As the run proceeds, append a progress log and findings to this note, as `202607170935-full-refit-vm-run-147-163.md` did.

## Purpose

One VM run that closes the two remaining fitting-dependent acceptance items shared by [#147](https://github.com/dseinternational/vocabulary-growth/issues/147) and [#163](https://github.com/dseinternational/vocabulary-growth/issues/163):

- **#163:** "Run decision-relevant source, missingness, prior, dispersion, GP-anchor and measurement-scale sensitivities."
- **#163:** "Add quantitative posterior-predictive calibration plus simulation/parameter-recovery checks for the preferred models."
- **#147:** the whole issue — fit the registered variants, compare against baselines, document the robustness matrix in `docs/models/PRIORS.md`.

These are one workstream because they need the same fits. The 41 registered sensitivity variants include both the Target-8 anchor variants (#147) and the #163 P0/P1 variants (signing source, us_01 ceiling, TD single-administration, dispersion floor, child effects), and the parameter-recovery study needs a `test`-tier fit of each model of record — which is exactly what a sensitivity baseline is.

`compare_sensitivity.py` resolves the baseline from `output/models/<id>-<config_name>/`, **with no tier suffix** — the same path the reporting-quality model of record occupies. So the baselines must be fitted at the same tier as the variants, and doing that in the main output root would overwrite the published `rep` fits. Hence the central decision below.

## Central decision: one isolated output root, baselines serving double duty

Run everything under a single isolated output root (`/scratch/vg-robustness`), never the main one:

```bash
export DSE_VOCAB_GROWTH_OUTPUT_DIR=/scratch/vg-robustness
```

Two consequences, both good:

1. **The `rep` models of record are never touched.** No fit in this run writes to the main output root, so the published reports and their traces are safe regardless of what fails.
2. **The recovery study needs no `rep` trace.** `fit_recovery.py --truth posterior` reads `models/<id>-<config_name>/trace.nc` from the resolved output root — which, in the isolated root, is the `test`-tier baseline this run fits in Phase A. A single draw from a `test`-tier posterior of the same model on the same data is a perfectly legitimate truth: it is a plausible parameter setting in the regime the study reports, which is all a recovery truth has to be. The `rep` posterior is better *characterised*, but one draw from either is equally valid as a truth.

That second point removes what looked like a hard dependency. **Do not assume the post-#176 `rep` traces still exist** — they were written on `dse-vm-res-frank-buckley-07191545`, the upload excluded traces, and nothing archived them (see the memory note on rep-fit artefacts). This plan does not need them.

`compare_sensitivity.py` is the one script with no `--output-dir` flag, so it relies on the environment variable. Export it for the whole session rather than passing `--output-dir` per command, so no step can silently resolve to the repository-local `output/`.

## Prerequisites

1. **Commit and merge the parameter-recovery harness.** It is currently uncommitted. Every fit records its Git revision and dirty state, so fitting on the uncommitted harness would stamp `dirty=true` on all 50+ manifests and make the run unciteable.
2. **Porcelain-clean tree, including untracked build artefacts.** The #176 lesson: `git_metadata` computes dirty from `git status --porcelain --untracked-files=normal`, so leftover untracked figure copies set `dirty=true`. Check `git status --porcelain --untracked-files=normal` returns nothing before launching.
3. **Environment verified:** `dse-check-env environment.yml`, and confirm `pytest -q` is green on the VM (not just locally).
4. **Data prepared:** `python scripts/prepare_data.py`, and record the row counts and `source_data_hash`.
5. **Disk:** see the budget below. Confirm free space on `/scratch` before launching.

## Phases

### Phase A — baselines at `test` (5 fits)

Every variant and every recovery truth is defined relative to these, so they come first and nothing else can start until each model's baseline is in place.

```bash
for m in vg10 vg11 vg12 vg13 vg15; do
  python scripts/fit_model.py "$m" --config test
done
```

**Gate each baseline before proceeding with that model.** A baseline that fails the R-hat/ESS gate invalidates every comparison built on it. Note that the pipeline's hard convergence gate fires **only at reporting-quality configs**, so a `test` fit completes even when it under-converges — the banner and `diagnostics.csv` must be read, not assumed.

If a baseline fails at `test`, escalate **that whole model group** (baseline *and* its variants *and* its recovery replicates) to a heavier configuration. Mixing tiers between a baseline and its variants makes the comparison meaningless: an apparent prior sensitivity would be a sampling-effort difference. Record any escalation in this note.

Risk to watch: pre-#176, `vg11` under-converged even at `rep-lite` (min ESS 164). The #176 conditioning fixes removed those ridges and the TD trio now converges at plain `rep`, but `test` is lighter than `rep`, so the TD trio is where escalation is most likely.

### Phase B — sensitivity variants (41 fits)

| Model | Variants | Covers |
| --- | --- | --- |
| `vg10` | 11 | Target 1 (DS-joint q anchors), 4 (dispersion), 5 (RE scales, child effects, us_01 ceiling), **8 (`u-anchor-broad`, `eta-u-narrow`)** |
| `vg11` | 5 | Target 5 (RE scales, **single-admin**), **8 (`anchor-broad`, `eta-narrow`)** |
| `vg12` | 4 | Target 5 (**single-admin**), **8 (`lo-anchor-broad`, `hi-anchor-broad`, `eta-narrow`)** |
| `vg13` | 1 | Target 5 (**single-admin**) — deliberately limited; its full matrix is too heavy |
| `vg15` | 20 | Targets 2, 3 (**signing source: `sign-include-uk01`, `sign-include-uk06`, `sign-study-only`**), 4, 5, 6, 7 |

```bash
for m in vg10 vg11 vg12 vg13 vg15; do
  python scripts/fit_sensitivity.py "$m" all --config test
done
```

Then compare (needs the exported environment variable — no `--output-dir` flag):

```bash
for m in vg10 vg11 vg12 vg13 vg15; do
  python scripts/compare_sensitivity.py "$m"
done
```

Priority within the matrix, if the run has to be cut short:

1. **`vg12 hi-anchor-broad`** — #147 calls this the single most important check: the 26-month TD understood high anchor has no independent CDI comprehension norm (WS is production-only), so it is the one recalibrated anchor that is pure data-informed regularisation.
2. **`vg15 sign-include-uk01` / `sign-include-uk06` / `sign-study-only`** — #163's P0. This is the finding that can *manufacture* a headline result rather than merely widen an interval: uk_01 records signed-only words and owns 114 of the 136 signed observations above 60 months, so the signing peak-and-decline shape is exactly the artefact an unharmonised outcome would produce.
3. **`vg11`/`vg12`/`vg13 single-admin`** — #163's TD repeated-measures P1, and cheaper than their baselines (fewer rows).
4. **`vg10 u-anchor-broad` / `eta-u-narrow`** — the un-normed DS understood anchors.
5. Everything else.

**Prune variant traces as they complete.** The sensitivity comparison reads `posterior_summary_*.csv` and `diagnostics.csv` only — it never opens a trace. Deleting each variant's `trace.nc` after its fit completes cuts peak disk by roughly two thirds. The cost is that re-scoring a variant later needs a refit, which is acceptable for a variant (it is never published). Do **not** prune the baselines: Phase C reads their traces.

### Phase C — parameter recovery (9 fits)

VG10, VG12 and VG15 — the three models #163 names as preferred — three replicates each.

```bash
# Cheap: confirm every coherence check passes before committing sampling time.
python scripts/fit_recovery.py headline --config test --replicates 3 --simulate-only
python scripts/fit_recovery.py headline --config test --replicates 3 --fit-only
python scripts/fit_recovery.py headline --config test --replicates 3 --compare-only
```

Run the `--simulate-only` pass first and read its coherence checks (9 for VG15, 4 for VG10, 1 for VG12). A simulation whose nested denominators or cross-tab totals disagree with a model rebuilt from the synthetic frame aborts, and that must be resolved before any sampling.

Recovery fits land in `models/<id>-<config_name>-recovery-rNN/`, so they cannot collide with a baseline or a variant. Prune each recovery trace after its replicate is scored.

Escalate with the model group if that model's baseline was escalated (Phase A), so the truth and the refit come from the same tier.

`docs/runbooks/parameter-recovery.md` is the reference for reading the output — in particular, `coverage_ci89` is the fraction of target quantities whose interval contained the truth, **not** a coverage estimate, and three replicates cannot support a calibration claim. What three replicates *can* support: a quantity whose truth is far outside its posterior (`|z| ≥ 4`) is a real finding, and systematic one-directional error across a trajectory's query ages points at a constraint or anchor pulling the fit.

### Phase D — collate the calibration evidence (no fitting)

Quantitative posterior-predictive calibration is already written by every fit as `posterior_predictive_calibration.csv`. Two things are missing, and neither needs a fit:

1. **It is surfaced in no report.** `grep -rl calibration docs/**/*.qmd` returns nothing: the table is computed, synced into `docs/report/figures/`, and never read. Adding it to the model appendix and/or the workflow chapter is a reporting change that can be made *before or independently of* this run.
2. **The cached copies are stale.** The local `docs/report/figures/` cache holds calibration tables for only 9 models, dated 2026-07-16 to 07-18 — the **paused, pre-#176 run**. VG09–VG13 and VG16 have none, and the post-#176 refit (2026-07-23/24) wrote fresh ones that were never synced here.

For the record, the pattern in those stale tables (indicative only, superseded fits): empirical coverage of the 90% predictive interval ran 0.915–0.992 against a nominal 0.90, with mid-PIT variance below the uniform 1/12 in every case — predictive intervals slightly **wider** than the data warrant, which is the conservative direction. The nested/joint models were the most over-dispersed (VG15's three outcomes 0.972–0.992, PIT variance ≈ 0.04 against 0.083). Whether that survives the #176 refit is exactly what Phase D has to establish, so **do not quote these numbers**.

Collate from the post-#176 `rep` fits if they are still on a VM; otherwise this becomes the one part of the workstream that does need the models of record refitted at `rep`, and it should be scoped as its own run rather than bolted onto this one.

### Phase E — document and close

- **`docs/models/PRIORS.md`:** add a "Sensitivity results" subsection under "Sensitivity targets" with each model's robustness matrix, the tier used, and the date. State the verdict and `max_abs_delta` for each Target-8 variant. Update the Target-8 row and the closing "No final robustness conclusion should be made until…" line (line 605) to reflect what was established. Record any variant that failed to converge, so a skipped variant is never silently read as robust.
- **Recovery results:** the matrices plus a short interpretation, with the replicate count and its limits stated plainly. Natural home is a new subsection of the workflow chapter alongside the calibration evidence, once Phase D lands.
- **`docs/report/methods-workflow.qmd`:** add the recovery and calibration checks to "Validation checks". Deliberately not done yet — a findings chapter should not describe a check before the check has results.
- **Issues:** #147 closes on the PRIORS.md update. #163's two fitting-dependent boxes close on Phases B–D; its "refit the affected models at reporting quality" box is already satisfiable from the post-#176 run and can be ticked independently. The final box — declaring the methodology publication-ready — stays open: that is a human decision, and this run supplies evidence for it, not the decision itself.

## Scale, time and disk

**55 fits at `test`** (5 baselines + 41 variants + 9 recovery), plus simulation and comparison steps that cost minutes.

Time estimates are **extrapolated from `rep` timings in the previous run records, not measured at `test`** — treat them as planning figures and recalibrate after the first two baselines. `test` is 4 chains × 2000 draws (+2000 tune) against `rep`'s 6 × 6000, so roughly a fifth to a quarter of the sampling work per model.

| Group | Fits | Rough per-fit | Rough total |
| --- | --- | --- | --- |
| `vg10` (DS bivariate) | 12 + 3 recovery | 10–15 min | 3–4 h |
| `vg15` (DS joint) | 21 + 3 recovery | 12–20 min | 5–8 h |
| `vg11` (TD spoken, 16k rows) | 6 | 30–45 min | 3–5 h |
| `vg12` (TD understood, 6k rows) | 5 + 3 recovery | 25–40 min | 3–5 h |
| `vg13` (TD young joint) | 2 | 30–45 min | 1–1.5 h |

Sequential ≈ 15–24 h. With the previous run's 2-wide thread-pinned pool for the DS models (2 × 4 chains = 8 of 16 cores, `OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1`) and strictly sequential TD models, expect **10–16 h wall** — an overnight run. Use `tmux` and a detached driver; the previous run lost its monitor to an SSH drop while the `nohup`'d driver survived.

Disk, extrapolated from a measured local `dev` trace (VG07, 1000 draws, 1205 rows = 247 MB) scaled to 8000 draws and by observation count: roughly 2–3 GB per DS fit, 3–5 GB per VG12/VG13 fit, 8–10 GB per VG11 fit. Unpruned that is **200–260 GB**; with variant and recovery traces pruned as they complete it should stay under 60 GB. Measure the first VG10 trace and re-derive before trusting this.

## Risks and decision rules

| Risk | Decision rule |
| --- | --- |
| A baseline fails the gate at `test` | Escalate the whole model group (baseline + variants + recovery) to a heavier config. Never compare across tiers. |
| A variant fails the gate | Record it as non-converged. `compare_sensitivity.py` already refuses to call it robust; the danger is a skipped variant reading as a pass, so list it explicitly in PRIORS.md. |
| A recovery simulation aborts on a coherence check | Stop and fix. It means the synthetic data would be fitted under a different decomposition from the one that generated it, and any resulting "recovery failure" would be a harness artefact. |
| Disk fills mid-run | Prune completed variant traces first; they are never read again. |
| A recovery replicate shows `\|z\| ≥ 4` on a reported quantity | A real finding. Investigate whether it is non-identification or sampling before drawing any conclusion, and check whether the same quantity appears in that model's sensitivity matrix. |
| The run is interrupted | Both `fit_sensitivity.py` and `fit_recovery.py` write per-fit output independently, and recovery's stages are separable (`--simulate-only` / `--fit-only` / `--compare-only`). Re-running skips nothing automatically, so track completion in a status file as the previous run did. |

## What this run does not close

- **VG16 recovery.** Its cross-lag predictor is a function of earlier-wave comprehension, so single-pass simulation would fit synthetic-lag data against real-lag truth. Needs wave-sequential simulation; unsupported by design, and the harness says so.
- **The signing P0 itself.** The source-restricted fits (`sign-include-uk01`/`uk06`/`sign-study-only`) *test* the exposure; they do not harmonise the outcome. Item-level re-derivation of uk_01 signing from source material remains the actual fix, and it depends on source data, not code.
- **The Edgin cap.** Resolving it needs the source-form rationale, not a fit.
- **A calibration claim.** Three recovery replicates cannot establish interval coverage. The harness records `truth_quantile` per quantity per replicate precisely so a ~100-replicate simulation-based-calibration run can pool them later without new code — that is a separate, much larger run, and it should only be commissioned if the small study surfaces something.
- **Publication readiness.** Explicitly a human decision.
