# Interrupted Mac refit, 16–17 July 2026

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). Condensed with assistance from OpenAI Codex/GPT-6 on 22 October 2026.

> [!IMPORTANT]
> This run was interrupted and replaced by the [17 July VM run](202607170935-full-refit-vm-run-147-163.md). The later run rejected the proposed `rep-lite` configuration for the hierarchical TD models. Use the [current runbook](../docs/runbooks/full-refit.md) for sampling, resumption and publication; a trace file alone no longer establishes a reusable fit.

## Run and interruption

The run began on 16 July at about 19:02 UTC on a 16-core Mac with 48 GB RAM and 307 GB free disk space, using the conda `dse-vocab-growth` environment and `dse-research-utils` 0.7.0. It planned `rep` fits for the thirteen smaller registered models and `rep-lite` for VG11 and VG12. Those were the fifteen registered models at that revision.

VG01–VG04 completed and rendered between 19:03 and 20:35 UTC, taking 283, 242, 4,087 and 892 seconds. VG05 then sampled and passed its recorded convergence gate (no divergences, maximum R-hat 1.0013, minimum ESS 8,602), but crashed during diagnostics. The run stopped before fitting the remaining joint models.

## Degenerate LOO failure and fix

The spoken likelihood conditioned on observed comprehension. Fourteen rows had comprehension zero, giving a spoken denominator of zero and a log likelihood constant across posterior draws. ArviZ's Pareto-tail fit raised `ValueError: All tail values are the same`. Diagnostics ran before trace persistence, so this failed attempt left no trace to resume.

The fix added `_loo_dropping_degenerate` in `common.py`, excluding likelihood terms with across-draw variance at most `1e-12` before computing LOO. A VG05 `dev` fit then reported fourteen excluded terms, computed both outcome scores and completed the pipeline. Ruff and the diagnostic tests passed. The fix was committed as PR #165 (`f08f7ed`) before the VM restart.

The Mac driver was resumed at about 06:46 UTC on 17 July. Its Bash 3.2 compatibility fix also repaired the uppercase conversion used by its old trace-existence check. No final convergence table or publication record was completed for this run. The empty result and upload sections and the obsolete restart recipe have been removed; the original working record remains in Git history.

## Recorded artefact locations

The output root was the checkout's ignored `output/`. The driver was `output/replication-driver.sh`; logs were under `output/replication-logs/latest/`, and fitted artefacts under `output/models/<MODEL>/`. These are historical locations, not a claim that the files remain available.
