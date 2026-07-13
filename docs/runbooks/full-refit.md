# Runbook: full reporting-config refit of all models

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

How to refit the whole `VG01`–`VG16` family at reporting quality (`rep`) on a
large VM, render every report, and produce comparisons — with the pitfalls that a
naive run hits. Distilled from the 2026-07-12 run
(`notes/202607121753-reporting-config-fit-run-and-findings.md`).

## TL;DR

- Canonical, resumable, sequential path: `scripts/run_replication.sh --config rep`.
- On a many-core VM, fit the **DS models concurrently** and the **TD models one at
  a time** (see [Parallel fitting](#parallel-fitting-on-a-large-vm)); this is the
  only reason to deviate from `run_replication.sh`.
- Three things bite every time: the **DuckDB lock** on concurrent fits, the
  **R-hat gate rounding** (need `dse-research-utils >= v0.6.0`), and the
  **understood-GP R-hat ridge** in the DS joint/hierarchical models.

## 0. Prerequisites

- Conda env `dse-vocab-growth` active; `dse-check-env environment.yml` clean.
- **`dse-research-utils >= v0.6.0`** — earlier versions' convergence gate rounds
  R-hat/ESS to 2 significant figures and can certify a fit that truly fails the
  ≤1.01 gate (research#65). A banner reading exactly `max R-hat = 1.0` is the
  tell-tale of the old rounding.
- Data current: `python scripts/prepare_data.py` (confirm the 810 reference scale;
  see `docs/report/methods-data.qmd`).
- Disk: `rep` traces are ~4–15 GB **each**. Budget ~20 GB × n_models and redirect
  output off the checkout: `--output-dir <scratch>` or `DSE_VOCAB_GROWTH_OUTPUT_DIR`.
  The report figure cache (`docs/report/figures/`) always stays in the checkout.
- `rep` config = 6 chains / 6 cores / 6000 tune / 6000 draws / `target_accept` 0.95.

## 1. Fit

### Default (sequential, resumable)

```bash
scripts/run_replication.sh --config rep --output-dir <scratch>
```

Idempotent: a model whose `trace.nc` exists is skipped (`--fresh` to force). It
fits+renders each model, runs comparisons, syncs figures, renders the report and
comparison book, and (optionally) uploads. Estimate ~15–25 h sequential.

### Parallel fitting on a large VM

The DS datasets are small; the full-data TD models (`vg11`, `vg12`) are
memory-heavy. So:

- **DS models** (`vg01 vg02 vg05 vg07 vg08 vg09 vg10 vg14 vg15 vg16`): run a pool,
  `concurrency × 6 ≤ physical cores` (e.g. 5 on 32 cores). **Pin each chain to one
  thread** first, or the pool oversubscribes (see the warning below):
  ```bash
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  printf '%s\n' vg01 vg02 vg05 vg07 vg08 vg09 vg10 vg14 vg15 vg16 \
    | xargs -P 5 -I {} python scripts/fit_model.py {} --config rep --output-dir <scratch>
  ```
- **TD models** (`vg03 vg04 vg13 vg11 vg12`): **strictly one at a time** — the
  full-data TD fits can OOM if stacked.

### Config choice for the full-data TD models (`rep-lite` is validated for these)

The full-data TD fits dominate wall time (`vg11`: 16,235 obs, ~9 h at `rep`; `vg12`:
~6,000 obs). At those sample sizes the posterior is **likelihood-dominated** and ESS
accumulates fast — `vg11`'s `rep` fit reached **min ESS ≈ 9,850, ~25× the 400 target**,
so raw draws are nowhere near the binding constraint. **Fit these large models at
`--config rep-lite`** (4 chains / 4000 tune / 4000 draws, same `target_accept = 0.95`):
it keeps reporting-grade rigour (ESS still clears 400 with wide margin), gives materially
identical estimates, and cuts ~⅓ off the wall time.

**Validated (2026-07-13, vg11).** Fitting vg11 both ways confirmed it: expected-word
trajectories agreed to **max 0.27 words (≤ 0.11%)** across the 9–30 mo grid, HDI widths
were essentially unchanged (ratio 0.99), `rep-lite` min ESS was 4,461 (~11× target), and
wall time fell from **9 h 17 m** (`rep`) to **5 h 59 m** (`rep-lite`), a ~35 % saving.
`rep-lite` even cleared the strict 0-divergence gate that `rep` missed by one (favourable
sampling luck, not a guarantee).

Caveats: `rep-lite` keeps `target_accept`, so it does **not** trade away divergence
control — but it has fewer tuning steps, so it won't _fix_ a divergence (and could nudge
the count up slightly). It is a wall-time optimisation, not a convergence fix. And the
small DS models are fast at `rep` anyway, so this only pays off on the big-data models.

> [!IMPORTANT]
> Concurrent fits require **read-only DuckDB connections**
> (`data_utils.load_combined_data`/`load_data` open with `read_only=True`). The
> default read-write connection takes an exclusive lock, so simultaneous fits die
> at data load with `IOException: Conflicting lock`.

> [!WARNING]
> Thread oversubscription is the biggest time sink. Without the thread-pinning env
> vars above, each of a fit's 6 chains spawns multiple BLAS/numba threads, so one
> fit uses ~10 cores, not 6, and a 5-wide pool drives the load past 2× the core
> count — in the 2026-07-13 run this made `vg03` take ~6 h instead of ~30 min.
> Mitigations: (1) pin threads to 1 (env vars above) so `concurrency × 6` is the
> real core count; (2) watch `uptime` — keep load near the core count; (3) don't
> stack the DS pool on top of a TD fit or the convergence refits. With threads
> pinned, 5 DS fits (30 cores) run cleanly on a 32-core box.

## 2. Verify convergence (do not trust the banner alone)

For every model confirm, on **unrounded** diagnostics: max R-hat ≤ 1.01, min ESS ≥
400, 0 divergences, BFMI ≥ 0.3. With `dse-research-utils >= v0.6.0` the gate is
correct natively; if any fit predates the fix, recompute from the trace:

```bash
python - <<'PY'
import arviz as az, xarray as xr
dt = xr.open_datatree("<scratch>/models/<MODEL>/trace.nc")
r = az.rhat(dt["posterior"].to_dataset())
print("max r_hat:", max(float(v.max()) for v in r.data_vars.values()))
PY
```

### Known ridge: the understood-GP block

The DS joint/hierarchical models (`vg09`, `vg10`, `vg15`, `vg16`) tend to leave the
**understood-trajectory GP block** (`g_u` / `g_unit_u` / `g_unit_u_hsgp_coeffs` /
`slope_u` / `p_slope_low_u`) just over 1.01 at `rep` — the trend/GP/intercept
redundancy the VG10 GP anchor addresses for the `q`-GP, here on the understood GP.
Remedy: refit with heavier tuning (**tune 12000 / draws 8000 / target_accept 0.97**,
6 chains), which cleared all four in the 2026-07-12 run (e.g. vg16 1.024 → 1.009).
Back up the non-converged output first; the refit becomes the model of record.

## 3. Render + comparisons

```bash
python scripts/sync_report_figures.py --output-dir <scratch>   # feeds docs/report/figures/
# comparisons (consume fitted traces/summaries):
for c in loo_compare loso_compare compare_models compare_ds_td \
         compare_ds_td_trajectories compare_ds_td_expressive \
         compare_ds_td_latency compare_ds_td_q_overlap compare_ds_td_re; do
  python scripts/$c.py
done
python scripts/sync_report_figures.py --output-dir <scratch>   # re-sync comparison artefacts
quarto render docs/report
# the comparison book reads its CSV/PNG artefacts by BARE filename from its own dir,
# and sync_report_figures only populates docs/report/figures/ — so stage them first:
cp <scratch>/comparisons/* docs/comparison/                    # (gitignored)
quarto render docs/comparison/index.qmd
```

Per-model reports render during `fit_model.py --render`; if you fit without
`--render`, render each `<scratch>/models/<MODEL>/index.qmd` directly. **Gotcha:**
rendering a model report whose output dir is **outside the git checkout** (e.g. a
scratch `--output-dir`) makes quarto exit non-zero on the `code-links: [repo]`
post-processor ("not a GitHub project") — the HTML is still produced and complete;
it just lacks the repo source-link button. It's clean when output lives under the
in-repo `output/`.

## 4. Completion checklist

- [ ] All registered models have a `trace.nc` and PASS the gate on **unrounded**
      diagnostics (R-hat ≤ 1.01, ESS ≥ 400, 0 divergences, BFMI ≥ 0.3).
- [ ] Understood-GP-ridge models refit with heavier tuning if needed.
- [ ] `sync_report_figures.py` run; all model reports + `docs/report` +
      `docs/comparison` render clean.
- [ ] Record the run in a dated `notes/` entry (config, incidents, convergence,
      timings).

## Standing caveats to re-check each run

- **Fixed-810 denominator** is a _validated_ approximation (dual-form crosswalk,
  `scripts/crosswalk_dse_oxford.py`; methods-data §Measures) — do **not** switch to
  per-form `n_trials`, which over-corrects. (Issue #149.)
- **Target 8 anchor prior-sensitivity** (#147) is a separate `test`-tier study, not
  part of a `rep` refit.
