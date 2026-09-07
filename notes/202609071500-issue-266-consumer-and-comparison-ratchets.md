# Closing issue #266's two coverage items

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

2026-09-07. Issue [#266](https://github.com/dseinternational/vocabulary-growth/issues/266) had four open completion criteria. Two of them needed fit time and are held for the VM refit; the other two were coverage ratchets that had stalled part-way, and a defect flagged on the issue on 2026-09-06 that was waiting for a run to drain. This note records what was done to all three, and — more usefully — why each of them stalled in a way nothing noticed.

## The pattern all three share

Each was a good mechanism with no invariant behind it.

- The prepared-frame hash worked, and five scripts of sixteen used it.
- The comparison manifest worked, and two comparison writers of thirteen wrote one.
- The sensitivity matrix recorded `baseline_fit_utc` faithfully, and the report cell that publishes the matrix did not display it.

In every case the _absence_ was indistinguishable from a deliberate decision. A reader of `emit_loo_summaries.py` could not tell whether it skipped validation because someone had thought about it or because nobody had; `sync_report_figures.py` reported unclaimed comparison files as warnings precisely so the ratchet could turn one script at a time, which also meant nothing failed when it stopped turning. So the substantive change here is not the eleven call sites and the eleven manifest entries — those are mechanical. It is that each ratchet now has a test that fails when the next script arrives in neither camp, and that every deliberate exemption is written down with its argument rather than left as a gap.

## 1. Every trace consumer checks the frame it was fitted on

`vocab_growth.fit_consumers` is one shared way to read a fit back. It asks the three questions that identify a posterior — the registered definition, the raw-data fingerprint, the exact prepared-frame hash — and deliberately not the publication apparatus. That scope is the same one `scripts/loso_compare.py` had already chosen for itself and the same reasoning `fit_validation_kwargs` records for `render`: a script printing a number off a model of record is not publishing it, and asking for the executable-code signature would mean an edit to any module in the package stopped every one of these scripts from running.

Sixteen top-level scripts open a stored trace. They now divide as:

| how it validates                   | scripts                                                                                                                                                                                                                                     |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit_consumers` (new)              | `aggregate_summary.py`, `emit_factor_correlation.py`, `emit_loo_summaries.py`, `loo_compare.py`, `predict_new_study.py`, `prior_vs_posterior.py`, `regenerate_vg09_marginal_predictive.py`, `time_to_milestone.py`, `vg07_study_effects.py` |
| directly, predating the helper     | `fit_model.py`, `loso_compare.py`, `regenerate_plots.py`, `resume_from_trace.py`                                                                                                                                                            |
| via `report_cells._verified_frame` | `compare_ds_td_re.py`                                                                                                                                                                                                                       |
| recorded exemption                 | `fit_recovery.py`, `compact_traces.py`                                                                                                                                                                                                      |

The two exemptions are argued in `EXEMPT_CONSUMERS` rather than asserted. `fit_recovery.py` reads a posterior only as a truth generator, and a draw from a fit whose frame has since moved is still a valid parameter vector to simulate from — the refit it is scored against is made by the current pipeline under the current rules, so staleness does not make the truth worse. `compact_traces.py` manipulates trace files as files and never reads a posterior value into a reported number; the risk actually present when rewriting a fit in place is liveness, and it has its own guard for that.

Refusal is the default and `--allow-stale-fit` is the one way past it, following `sync_report_figures.py --allow-provisional`. An override prints what it overrode, every time.

Two things came out of the wiring that were not on the issue:

- **`prior_vs_posterior.py`'s existing guard was the pre-#273 comparison.** It compared the stored definition payload to the current one as raw dictionaries. That cannot tell a graph change from a reporting one, and it has no `BACKFILL_DEFAULTS`, so a fit predating a field whose default reproduces exactly what that fit did was skipped as stale when it was not. It now uses `fit_errors`, which also brings the frame hash the raw comparison could not see at all. The guard was also only on the conflict-table path; `overlay_model`, which draws the prior-versus-posterior figures, had none.
- **`aggregate_summary.py` is the one consumer that records rather than refuses.** It is a summary _of a set of fits_, so a stale one belongs in the table with a `fit_current` column saying so. Refusing there would drop the only artefact where the staleness would have been visible for every model side by side.

All 21 registered models validate today, so none of this is disruptive in the current checkout — which is also the point at which it is cheapest to turn on.

## 2. Every comparison writer records what it came from

Thirteen scripts write into the comparisons root; two recorded a manifest entry. All thirteen do now, and the three that touch the directory without generating anything (`compare_sensitivity.py` writes only into the nested `sensitivity/` subdirectory, `publish_comparison.py` publishes what exists, `sync_report_figures.py` is the validator) carry their reason in `MANIFEST_EXEMPT_SCRIPTS`.

Three mechanisms were added rather than repeating `compare_models.py`'s shape eleven times:

- **`ComparisonOutputs` snapshots the directory** around the run, so `outputs` is derived from what was written rather than from a hand-maintained list. Hand lists go stale in the direction that matters: a script that gains a figure keeps claiming the old set and the new file surfaces as unclaimed. Detection is by `(size, mtime_ns)`; the failure mode is a file reported as unclaimed, never one wrongly vouched for.
- **`source_data_hash` on the entry**, for a comparison with no contributing fit. `pool_descriptives.py` describes the pool itself and `kfold_loso.py` fits its own folds; what their tables can outlive is a data change, not a refit. Recording nothing would have made them look exactly like a script nobody had wired up.
- **`fit_consumers.contributing_fits`** validates every fit a comparison reads _and_ returns the `{label: dir}` mapping the manifest records, so "validate what you read" and "record what you read" are one call and cannot drift apart. A comparison whose manifest lists a fit it did not check is the exact failure the manifest exists to prevent.

`compare_ds_td_re.py` records **per sub-comparison** rather than per script. Its outputs depend on the token run and three deprecated shims call `run_comprehension_matched` directly; one entry for the whole script would have unclaimed the other outcome's files on every partial run. The entry also carries `arguments`, so a record claiming three files where the directory holds nine reads as a correct account of a partial run rather than a defect.

Running the wired scripts against the current output root immediately produced a true finding: `compare_models.py`'s recorded fingerprints no longer match five contributing fits, because those models were refitted on 2026-09-06 after the comparison was generated. That is the mechanism doing its job.

## 3. The robustness matrix no longer mixes baselines silently

Flagged on the issue on 2026-09-06 and deferred while the task 1.2 arms were running. `compare_sensitivity.py --variant <name>` merges into the standing matrix rather than rewriting it, which is right — rewriting would drop every other verdict. What was missing is that a retained row was scored against _whatever baseline existed when it was computed_, and a refit moves the baseline. `robustness_matrix_vg10.csv` held one row against the pre-`us_03` VG10 and two against the refitted one, presented side by side; the only hint was a `nan` in `baseline_converged`.

Both halves of the suggested remedy are done. `merge_retained_rows` sets the status of a retained `compared` row whose `baseline_fit_utc` differs from the current baseline as `stale-baseline` and prefixes its verdict with the baseline it actually used, keeping its numbers — the row is a true record of a comparison that was made, and dropping it would lose the fact that the variant has been fitted at all. Only `compared` rows are marked: a `not-fitted` or `failed` row says something about the variant, and a new baseline does not change it. And all nine report cells that publish the matrix now display `baseline_fit_utc` beside `variant_fit_utc`, with a sentence saying what `stale-baseline` means.

The live VG10 matrix had already been fully rerun by the time this landed, so the artefact was clean; the defect was in the code and would have recurred on the next targeted rerun.

## What is left on #266

Only fits. VG22's recovery and whole-child predictive comparison are [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 3.1, and the reporting-quality refit plus resync is task 1.1. Nothing on the issue now needs a code change.

## Checks

`ruff` and `mypy` clean; 2,069 fast tests passed (53 skipped) and 315 slow tests passed, after rebasing onto the `dse-research-utils` 0.14.0 adoption in [#315](https://github.com/dseinternational/vocabulary-growth/pull/315). Twenty-eight of the fast tests are new here (thirty collected, one being a three-way parametrisation). `pool_descriptives.py`, `subject_effect_correlation.py`, `compare_ds_td_trajectories.py`, `aggregate_summary.py`, `prior_vs_posterior.py --table --model vg16` and `emit_factor_correlation.py` were each run end to end against the real output root, and the resulting `comparison_manifest.json` was validated.
