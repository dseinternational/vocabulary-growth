# Full refit run of VG01–VG16 (rep + rep-lite) — run record and progress

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!WARNING]
> Live run record, started 2026-07-16 ~19:02 UTC while the run is in progress. Convergence, results tables, and timings below are filled in as the run drains. Numbers are not final until this banner is removed.

## Purpose

Full replication refit of every registered model (VG01–VG16) on the current 810-item reference scale, following `docs/runbooks/full-refit.md`. Every model is fitted at the `rep` (reporting) sampling config **except** the two full-data ("large") TD models — `vg11` and `vg12` — which are fitted at `rep-lite`, per the runbook's validated recommendation. All reports are then rendered and uploaded to the public research blob storage.

## Environment and method

- **Machine:** local Mac (darwin), 16 cores, 48 GB RAM, 307 GB free on the internal disk. This is materially smaller than the 32-core/251 GB VM used for the 2026-07-12 run, so wall time is expected to be longer (~1.5–2+ days) and the full-data TD fits (`vg11`/`vg12`) carry a real OOM risk at 48 GB — a reason the `rep-lite` (4-chain) config for those two is doubly helpful here.
- **Env:** conda `dse-vocab-growth`, `dse-research-utils 0.7.0` (≥ v0.6.0, so the convergence gate reports **unrounded** R-hat/ESS natively — no gate-rounding false-pass risk).
- **Output root:** repo-local `output/` (gitignored). Report figure cache stays in the checkout at `docs/report/figures/`.
- **Configs:**
  - `rep` — 6 chains / 6000 tune / 6000 draws / `target_accept` 0.95.
  - `rep-lite` — 4 chains / 4000 tune / 4000 draws / `target_accept` 0.95.
- **Data:** confirmed current (merged DuckDB newer than all 13 source CSVs); no re-prep needed, but `prepare_data.py` is re-run once at the start of the driver for safety.
- **Execution:** sequential per-model fitting via `scripts/run_replication.sh` (resumable — a model whose `trace.nc` exists is skipped). Sequential rather than the VM's DS-concurrent pool because 16 cores only fits ~2 six-chain `rep` models at once and the `rep` set here contains no huge full-data fits (those are the `rep-lite` pair), so sequential avoids the thread-oversubscription / OOM pitfalls for little wall-time cost.

## Model split

| Config     | Models                                                                 |
| ---------- | ---------------------------------------------------------------------- |
| `rep`      | vg01 vg02 vg03 vg04 vg05 vg07 vg08 vg09 vg10 vg13 vg14 vg15 vg16 (13)   |
| `rep-lite` | vg11 vg12 (2 — full-data TD)                                            |

## Progress log

- **2026-07-16 ~19:02 UTC** — prerequisites verified (env, gate v0.7.0, data current, disk, blob URL set). Run record created.
- **2026-07-16 ~19:03–20:35 UTC** — pass 1 fits: `vg01` PASS (283 s), `vg02` PASS (242 s), `vg03` PASS (4087 s), `vg04` PASS (892 s). All four univariate models clean, reports rendered.
- **2026-07-16 ~20:45 UTC** — `vg05` (first joint model) **crashed in the diagnostics stage** (see Incident 1). Run **stopped** to avoid wasting hours on the 8 further joint models that share the same defect.

## Incident 1 — PSIS-LOO crash on joint models (nested-likelihood regression)

**Symptom.** `vg05` sampled cleanly and **passed the convergence gate** (0 divergences, max R-hat 1.0013, min ESS 8602), then crashed in `diagnostics` → `az.loo(trace, var_name="y_s_obs")` with `ValueError: All tail values are the same` (`arviz_stats/base/diagnostics.py:1188`, `_ps_tail`).

**Root cause.** The DS joint likelihood models spoken conditionally on understood (`y_s ~ BetaBinomial(n = understood, …)`, the nested outcome likelihood from #163/#164). The merged DS data contains **14 rows with `understood == 0`** → spoken denominator `n = 0` → the spoken per-observation log-likelihood is **constant (≡ 0) across every posterior draw**. PSIS-LOO's Pareto tail fit requires the pointwise log-lik to vary; `arviz_stats 1.2.0`'s `az.loo` **raises** on such degenerate points (its moment-matching path lists this exact string in `fallback_errors`, but the plain `az.loo` path does not catch it).

**Why it's new.** This is the **first reporting-quality fit of the nested-likelihood models** (`docs/models/README.md` explicitly flags #163's nested outcome likelihood as requiring new rep fits). The 2026-07-12 run predates #163/#164, when spoken used a marginal fixed-810 denominator and never produced a constant per-obs log-lik — so the crash could not occur then.

**Blast radius.** Every joint/multivariate model: `vg05`, `vg07`, `vg08`, `vg09`, `vg10`, `vg13`, `vg14`, `vg15`, `vg16` (9 of 15). The 6 univariate models (`vg01`–`vg04` done; `vg11`/`vg12` pending) are unaffected. Crash occurs **after** sampling, in `diagnostics`, which runs **before** the posterior-predictive stage that writes `trace.nc` — so a failed joint fit **loses its trace** (no cheap recompute).

**Fix (implemented 2026-07-17).** `common.py` now computes each per-outcome LOO through a helper `_loo_dropping_degenerate(idata, var_name)` that drops observations whose pointwise log-likelihood is constant across draws before calling `az.loo`. The drop is deterministic (keyed off the structural `n = 0` degeneracy alone), so every joint model excludes the *same* observations and the per-outcome elpd stays comparable across models for `loo_compare`. Threshold: across-draw variance ≤ `1e-12` — the degenerate points sit at ~`1e-33` (numerically-zero log-lik plus fp noise) while genuinely informative observations have variance ≫ `1e-6`, so the two are cleanly separated (an initial `np.finfo(float).tiny` threshold was far too small and dropped nothing). Verified end-to-end: a `dev` fit of `vg05` reported "dropped 14 degenerate … observation(s)", computed LOO for both outcomes, and completed the full pipeline (exit 0). `ruff` clean; `tests/test_diagnostics_gate.py` passes.

**Working-tree change, flagged for review** (as with the 2026-07-12 DuckDB-lock fix): `src/vocab_growth/models/common.py`. Follow-up: a dedicated regression test for `_loo_dropping_degenerate` (deferred — needs a joint-trace fixture) and a proper PR.

- **2026-07-17 ~06:46 UTC** — fix landed and verified; run resumed (driver `bzpopzjn8`) for the 9 joint models + the `rep-lite` TD pair. The 4 univariate models `vg01`–`vg04` are already complete and are correctly skipped (also fixed `run_replication.sh`'s `has_trace`, which used a bash-4 `${1^^}` expansion that errored as "bad substitution" on macOS bash 3.2 and silently defeated skip-on-resume).

## Convergence (models of record — true, unrounded diagnostics)

Gate: max R-hat ≤ 1.01, min ESS ≥ 400, 0 divergences, BFMI ≥ 0.3.

_(filled in as fits complete)_

| Model | population / outcome | max R-hat | min ESS | div | verdict | config |
| ----- | -------------------- | --------- | ------- | --- | ------- | ------ |

### Known ridge to watch

Per the runbook, the DS joint/hierarchical models (`vg09`, `vg10`, `vg15`, `vg16`) historically leave the understood-trajectory GP block just over 1.01 at `rep`, and `vg13` needed `target_accept` 0.99 in the 2026-07-12 run. Remedy is a heavier-tuning refit (tune 12000 / draws 8000 / `target_accept` 0.97). The model code has changed since that run (#161–#164), so convergence is re-assessed empirically this run rather than assumed.

## Render + upload

_(filled in after fits pass the gate)_

## Reproduction / artefacts

- Driver: `output/replication-driver.sh` (two `run_replication.sh` fit-only passes: `rep` set, then `rep-lite` pair).
- Logs: `output/replication-logs/latest/run.log` and `status.tsv`.
- Traces / figures: `output/models/<MODEL>/`.
