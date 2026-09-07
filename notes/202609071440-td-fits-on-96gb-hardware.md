# Fitting the large typically-developing models on 96–128 GB machines

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

> [!IMPORTANT]
> Investigation note, 2026-09-07. Prompted by the study owner's question: can the large typically-developing (TD) fits — VG11, VG12, VG13, VG21, VG23 at `rep` — be made to run on 96 GB or 128 GB of RAM rather than the 251 GB VM? The answer turns on a fact nobody has measured since the 2026-08-23 change stopped sampling the observation-sized deterministics: **the runbook's "TD models must not share a box" rule and its 178–243 GB figures describe the old code.** §2 sizes what a current fit actually holds, §3 measures a current fit's memory profile stage by stage on a 96 GB Windows workstation, §4 extrapolates to `rep`, and §5 ranks the levers that would take the figure down further. Nothing here changes any model, sampler setting or definition.

## 1. What the record says, and what it no longer says

[202608050900](202608050900-td-hierarchical-geometry.md) §10 established that fit memory scaled as `n_obs × draws` through fifteen or so observation-sized `pm.Deterministic`s stored per draw, and the 2026-08 refit paid for it: VG13's `rep` fit plateaued at 178 GB and peaked at 243 GB on 2026-08-14 (`docs/runbooks/full-refit.md`, "Surviving an OOM"), and VG11 at the 48,000-draw configuration was killed at 247 GB. [202608231530](202608231530-observation-deterministics-not-sampled.md) removed that term at the source on 2026-08-23 — nutpie no longer evaluates or stores those variables — and measured VG10 at `test` going from 5.45 GB to 3.33 GB peak. Its §4 closed with: "the first `rep` fit on the new code should be measured rather than predicted, and the first TD one is the measurement that matters."

That measurement has not been made. The 2026-09-04 round ran the TD models on the new code and recorded their **trace sizes** (runbook, "Sizing", re-measured 2026-09-06: VG11 24.6 GB, VG21 15.2, VG13 and VG23 14.5, VG12 7.6) but no peak resident memory, and the runbook's parallel-fitting section still carries the serial-only rule with "treat them as VG13-class until [a measured peak] exists". The rule was right when written and is now unsupported either way; §3 and §4 are the first evidence about the current code at TD scale.

## 2. What a current fit holds per draw

Built without sampling, from each registered definition's own data preparation and model build (script in the session scratchpad; the counts are exact, the byte figures assume float64 and the sampler's `chain × draw` layout):

| model |   rows | children | free RV elements | sampled deterministics | log-likelihood elements | of which not sampled since 2026-08-23 |
| ----- | -----: | -------: | ---------------: | ---------------------: | ----------------------: | ------------------------------------: |
| VG12  |  7,049 |    5,819 |            5,849 |                  7,860 |                   7,049 |                                50,870 |
| VG11  | 18,500 |   14,553 |           14,587 |                 16,594 |                  18,500 |                               131,027 |
| VG13  |  6,356 |    5,496 |           11,052 |                 15,572 |                  12,712 |                                98,382 |
| VG21  |  6,783 |    5,707 |           11,474 |                 16,012 |                  13,566 |                               104,799 |
| VG23  |  6,356 |    5,496 |           11,053 |                 15,573 |                  12,712 |                                98,382 |

Two things to read off it. First, the per-draw footprint is now **child-sized, not observation-sized**: in every model the large arrays are the raw child effects (free) and their scaled copies (deterministic, `delta_subject` / `delta_subj_u` / `delta_subj_q`), which are the same size again — so the scaled duplicates are 41–47% of every posterior (VG13 lowest, VG11 highest). Second, what was removed was about four times what remains: VG11 stored 131,027 observation-sized elements per draw against 31,181 now, VG12 50,870 against 13,709.

The stored trace at each configuration follows directly (posterior + `log_likelihood` + `posterior_predictive`, the last as int64 counts):

| model | `rep` 6 × 6,000 | `rep-lite` 4 × 4,000 | `test` 4 × 2,000 | measured on disk, 2026-09-04 round (`rep`) |
| ----- | --------------: | -------------------: | ---------------: | -----------------------------------------: |
| VG12  |         7.5 GiB |              3.3 GiB |          1.7 GiB |                                     7.6 GB |
| VG11  |        18.3 GiB |              8.1 GiB |          4.1 GiB |                                    24.6 GB |
| VG13  |        14.0 GiB |              6.2 GiB |          3.1 GiB |                                    14.5 GB |
| VG21  |        14.6 GiB |              6.5 GiB |          3.3 GiB |                                    15.2 GB |
| VG23  |        14.0 GiB |              6.2 GiB |          3.1 GiB |                                    14.5 GB |

The arithmetic reproduces four of the five measured traces to within 5% — it runs slightly high on all four, which is what one expects if the recorded figures are binary gigabytes; the bivariate engines' administration-level log-likelihood (one more `n_obs × draws` array) pulls the other way. **VG11 is the exception: its fit of record is 24.6 GB against 19.6 GB from the counts**, a 5 GB gap that is about one more observation-sized array at `rep`. Nothing here explains it; the fit's manifest (`artefacts.trace`) and the trace's variable list will, and should be read before VG11's budget is set. The extrapolations below use the measured figure.

## 3. A current fit, measured stage by stage

VG12 at `test` (4 chains × 2,000 tune × 2,000 draws), fitted on 2026-09-07 on the study owner's Windows 11 workstation — Intel Core Ultra 9 285H, 16 cores, 95.4 GB RAM — from a clean `main` (`8dc427b`) into a scratch output root, with the fit process's working set sampled every 2 s and its Windows peak working set read at the end. Other software held 35–44 GB of the machine throughout; the figures below are the fit process alone.

| stage (wall clock)                        | process working set                                           |
| ----------------------------------------- | ------------------------------------------------------------- |
| build and prior predictive (1 m 0 s)      | 2.3–2.8 GB — the model, compiled functions, 1,000 prior draws |
| sampling (13 m 42 s)                      | grows linearly 2.8 → 3.81 GB: nutpie's Arrow buffers          |
| end of sampling: Arrow → NumPy conversion | brief spike, then **drops to 3.0 GB** once the buffers go     |
| diagnostics incl. PSIS-LOO (1 m 8 s)      | 3.0 → 3.8 GB                                                  |
| posterior predictive + `trace.nc` (25 s)  | 3.8 → 4.7 GB                                                  |
| summary, plots, report (8 s)              | 4.9 GB resident; **peak working set 5.48 GB**                 |

The written trace is 1.69 GiB (1.7 GB on disk), exactly the §2 estimate: posterior 0.82 GiB (of which `delta_subject_raw` 355 MiB and its scaled copy `delta_subject` 355 MiB), `log_likelihood` 0.42 GiB, `posterior_predictive` 0.45 GiB as int64. The gate returned REVIEW (71 divergences, R-hat 1.006, ESS 483) — irrelevant to memory, recorded for completeness.

So for the current code the fit's peak is **about 2.4 GB of fixed cost plus 1.8 × the stored trace**: 2.4 + 1.8 × 1.7 = 5.5 GB. The 1.8 is the sum of the copies the stages make at their worst moment — the posterior twice at the end of sampling, or the posterior once plus the log-likelihood three times inside `az.loo` (the array, the smoothed log-weights, and their sum), or the whole trace plus what `to_netcdf` and the extractors stage. Every one of those is linear in draws, which is what licenses the extrapolation; none of them is the `n_obs × draws` term that made the old plateaus move by 50% between runs of the same fit.

## 4. What that implies at `rep`

Applying peak ≈ fixed + 1.8 × trace, with the fixed cost allowed to grow with the model (VG11's prior predictive alone holds 1,000 draws of 131,027 elements, about 1 GB) and the measured trace sizes from §2:

| model at `rep`     |   trace | estimated peak | at a pessimistic 3 × trace |
| ------------------ | ------: | -------------: | -------------------------: |
| VG12               |  7.6 GB |         ~16 GB |                     ~25 GB |
| VG13, VG23         | 14.5 GB |         ~29 GB |                     ~46 GB |
| VG21               | 15.2 GB |         ~30 GB |                     ~48 GB |
| VG11               | 24.6 GB |         ~48 GB |                     ~77 GB |
| VG11 at `rep-lite` |   ~9 GB |         ~19 GB |                     ~30 GB |

**On this arithmetic every TD model of record fits on a 96 GB machine as the only significant tenant, and 128 GB is comfortable for all of them, including VG11 at a multiplier nearly twice the one measured.** What the arithmetic cannot do is stand in for the measurement: the multiplier was measured on a trace 4.5 times smaller than VG12's `rep` and 14 times smaller than VG11's, Windows working-set accounting is not Linux RSS, and the one unexplained number in the record (VG11's 5 GB) sits on the model that matters most. The runbook's rule — memory-heavy TD fit as the sole tenant, everything else in a scope you can stop in one command, per-process RSS sampled — stays exactly right as a _procedure_; only its numbers are stale.

Two practical qualifications for a 96 GB workstation specifically. The machine measured here was carrying 35–44 GB of other software during the fit; a VG11 `rep` run needs that closed down first, which is the difference between a 45 GB margin and none. And two TD fits at once remain out of the question at 96 GB (VG11 plus anything TD-sized), though a TD fit alongside the small DS models is now plausible where the runbook still forbids it — that is a measurement to make deliberately, not an inference to act on.

## 5. Levers, ranked by what they buy against what they cost

**1. Measure first, and make every fit measure itself.** Run VG11 at `rep` on the 96 GB machine as the sole tenant with `scripts/memwatch.ps1` (or the 2 s working-set sampler used here) alongside, and put the peak in the run record. Then close the gap permanently: record the process's peak resident memory in `fit_manifest.json`'s `runtime` block at the end of the pipeline (`psutil.Process().memory_info().peak_wset` on Windows, `resource.getrusage(RUSAGE_SELF).ru_maxrss` elsewhere), beside `nutpie_backend`, so the next budgeting question is answered by the manifests rather than by a note. It changes nothing about the posterior and belongs in `runtime` for the same reason the backend does.

**2. Stream the draws to disk while sampling (nutpie's Zarr store).** nutpie 0.16.0 (2025-10-09) added `zarr_store` to `nutpie.sample`, which writes each draw to a Zarr store as it is produced instead of accumulating Arrow buffers in RAM, and returns a lazily-loaded `DataTree`. That removes the linear growth during sampling and the conversion spike at its end — at VG11 `rep`, the 8–13 GB posterior would never be resident twice, and after sampling only what a stage actually touches is loaded (xarray caches each variable on first read, so the free RVs load for the log-likelihood and the gate; the scaled duplicates never do). Probed on 2026-09-07 with the locked versions (nutpie 0.16.11, PyMC 6.3.1, zarr 3.3.0, obstore 0.11.1, all already in the environment) on a 400-child hierarchical Binomial, on Windows:

- `nutpie.zarr_store.LocalStore(path, mkdir=True)` sampled, and `pm.compute_log_likelihood`, `az.loo`, `az.summary`, `pm.sample_posterior_predictive(extend_inferencedata=True)` and `to_netcdf` all ran on the result, which reloaded with matching means.
- **`pm.sample` cannot take it yet.** PyMC's nutpie wrapper calls `patch_nutpie_idata`, which reads `sample_stats.attrs["inference_library_settings"]`; the Zarr path writes the settings as a `sampler_settings` dict on the root instead, and the call raises `KeyError`. The `sample` stage would drive `nutpie.compile_pymc_model` and `nutpie.sample` directly — it already passes nutpie-specific arguments — and attach `observed_data`/`constant_data` itself as the wrapper does. Worth an upstream issue.
- `to_netcdf` refuses the result until its dict- and list-valued attrs (`sampler_settings`, `sample_dims`) are JSON-encoded; `save_trace` would do that.
- `warmup_posterior` is written in full even with `save_warmup=False` (disk, not RAM: one more posterior-sized group at 6,000 tuning draws), the unconstrained `*_log__` copies of transformed scalars appear in the posterior, and the store is chunked at 16 draws — thousands of small files per variable at `rep`, fine on local NVMe and unsuitable for a network filesystem, like the HDF5 trace already is.
- It changes no draws (same sampler, same seed path), but the 2026-08-23 note's bit-identity check on a `dev` fit is the standard to meet, and the setting must enter the manifest's `runtime` vocabulary like the backend did.

**3. Stop sampling the scaled child-effect duplicates.** `delta_subject` and `delta_subj_u`/`delta_subj_q` are 41–47% of every TD posterior (3.9 GiB at VG11 `rep`) and are `tau × raw`, recomputable from the free draws and the scale; `compact` already drops them at write time, which is why it saves disk and not RAM. Excluding them through the same `var_names` mechanism as the observation-sized deterministics makes it a RAM lever. [202608231530](202608231530-observation-deterministics-not-sampled.md) §6 deferred this as "a separate decision with readers to rewire", and the readers are few: nothing in the fit pipeline reads them; `comparison.py`'s DS/TD random-effect comparison, the recovery scorer and the `require_full_trace` consumers do, and each can go through `posterior_recompute.with_deterministics`, which already computes any absent name from a rebuilt graph. Old traces keep the variables, so readers must tolerate both, as they already do for the observation-sized ones.

**4. Store posterior-predictive counts as int16.** Every replicate is a word count at most 810, stored as int64: 0.45 GiB of a 1.69 GiB `test` trace, 5 GiB at VG11 `rep`. A cast before the group is attached takes it to a quarter; the calibration table and the extractors already cast to `int` on read.

**5. `rep-lite`.** Sanctioned as reporting quality (4 chains × 4,000, same `target_accept`), and it halves every figure above: VG11 at ~9 GB of trace and a ~19 GB peak. It is the fallback if the VG11 measurement in №1 disappoints, not the plan.

**Not levers.** Marginalising the singleton children was rejected on cost — a gradient 10–22 times dearer for a 1.6-fold cut in gradient evaluations ([202608231745](202608231745-singleton-marginalisation.md) §9); chains or draws below `rep-lite` are below the reporting tier minimums; thinning the stored `log_likelihood` or `posterior_predictive` changes what `loso_compare.py`, the recovery scorer and the calibration table read and would need its own decision. Reducing `n_obs` (subsampling the pool) was settled against in [202608050900](202608050900-td-hierarchical-geometry.md) §10.

## 6. Recommendation

Do №1 now — the VG11 `rep` measurement on the 96 GB machine, and peak-memory recording in the manifest — because §4 says the hardware question may already be answered and one fit settles it. Take №3 and №4 together as the next memory change: both are storage decisions with no statistical content, both have the observation-deterministics precedent for their tests and their manifest record, and together they remove about half of every TD trace (VG11 `rep`: 3.9 GiB of duplicates and 3.75 GiB of int64 padding out of 18–25 GB). Pursue №2 only if a measured `rep` peak on the current code is still uncomfortable at 96 GB after №3 and №4, since it carries an upstream integration gap and a change to how the `sample` stage is written; when it is pursued, the working probe is in this session's scratchpad and the four findings above are what it has to handle.

One housekeeping observation from the run: fitting from the repository root on this Windows machine left an untracked `packagespytensor/numba/` cache directory (19 MB of `.nbc`/`.nbi` files) in the checkout, created at the first compile of the fit. It was removed; if it recurs it wants a `.gitignore` entry or a `NUMBA_CACHE_DIR`.

## 7. JAX and a GPU: a wall-time question, not a memory one

Asked mid-investigation: should the JAX backend, with a GPU option, be part of the answer? Two of its parts are settled by what §2–§4 show about where the memory is, and the third was measured.

**A GPU does not touch the memory that matters.** Every large array in a fit — the child effects per draw, the pointwise log-likelihood, the predictive replicates — lives on the host, in the trace, after the sampler has produced it. The device would hold the model's density and gradient, which for VG11 is 18,500 Beta-Binomial terms and a 14,587-dimensional position: a few hundred kilobytes. nutpie's JAX path also hands each gradient back to the host as a NumPy array for every leapfrog step (`compile_pymc.py`, `make_logp_func`), so a GPU would add a device round trip to each of the tens of millions of gradient evaluations in a `rep` fit for arithmetic that is too small to fill it. That is the regime in which GPUs lose to CPUs, not the one nutpie's own documentation has in mind when it says the JAX backend "sometimes outperforms numba for larger models". Practically, too, the CUDA overlay (`jax[cuda]`, opt-in per `docs/runbooks/environment-locks.md`) needs an NVIDIA device on Linux or WSL2; the workstation measured here carries an Intel Arc Pro 140T, which no JAX backend in the locked stack can drive. A GPU is not a route to fitting these models on smaller hardware.

**The JAX backend on the CPU is a wall-time lever worth a clean benchmark.** It already exists as `--nutpie-backend jax` (#289 task 4.1) and is recorded in the manifest's `runtime` block rather than compared as a sampling parameter, so a fit made with it is publishable under current policy. VG12 at `test` was refitted with it on the same machine, same seed, straight after the numba run of §3:

| backend | sampling wall clock | divergences | max R-hat | min ESS | peak working set |  trace |
| ------- | ------------------: | ----------: | --------: | ------: | ---------------: | -----: |
| numba   |           13 m 42 s |          71 |    1.0061 |     483 |          5.48 GB | 1.7 GB |
| jax     |            9 m 39 s |          37 |    1.0073 |     605 |          5.14 GB | 1.7 GB |

Memory is the same to within the noise of the stages that follow sampling, as §4's arithmetic says it must be. The posterior is the same too — all eighteen summarised parameters agree, sixteen within one Monte Carlo standard error and the two derived intercept–slope quantities at 2.4–2.5 — but the **draws are not**: the two compilers round differently, the trajectories part after the first few leapfrog steps, and the divergence counts (71 against 37) are two realisations of the same sampler on the same geometry, not evidence that one backend has better geometry. The wall-clock gap is real but **not clean**: the numba run shared the machine with the sizing script and the Zarr probes for several minutes of its sampling, the JAX run ran alone. The plausible mechanism for a genuine gap is that XLA's CPU backend threads each gradient across cores while numba evaluates it on one, so four chains on sixteen cores leave numba twelve cores idle — which would make the advantage largest on a many-core VM with `rep`'s six chains, and smallest where cores are already saturated. A back-to-back pair of VG12 `test` fits on an otherwise idle machine, one per backend, settles it in half an hour; if it holds, VG11's `rep` sampling is where it pays.

One more reason to keep the JAX backend in view: the normalizing-flow adaptation deferred in [202608231410](202608231410-td-geometry-remaining-levers.md) §4 requires it. That is a geometry question, not a hardware one, and stays deferred on its own terms.

## 8. Procedure: the VG11 `rep` measurement on a 96 GB machine

Attempted on the workstation on 2026-09-07 at 15:01 and stopped by the study owner five minutes into sampling for want of a multi-hour window; the working set was 2.9 GB at the start of sampling and 3.9 GB at its peak in early warmup, consistent with §4 and uninformative about the post-sampling peak. Everything below is what that attempt established about running it, so the next one is a single command.

**Before launching.**

- **Commit first.** `fit_artifacts.git_metadata` counts untracked files as dirty, and a fit whose manifest records `dirty: true` fails `require_clean_fit` and can never be published. The 2026-09-07 attempt recorded a clean tree at `8dc427b` only because the uncommitted note and spellcheck words were stashed for the ninety seconds between launch and the manifest write (the manifest is written once, after the data-preparation stage, so nothing later re-reads the tree). Committing is the honest version of the same thing.
- **Sole tenant.** Close whatever holds tens of gigabytes — on the workstation SQL Server, its management studio and a browser held 35–44 GB — and check `free`/`FreePhysicalMemory` before starting. Windows' system-managed pagefile (96 GB on the workstation) is the backstop the runbook's swap advice describes; on Linux provision swap as the runbook says.
- **Disk.** About 60 GB free on the output root: a ~25 GB trace, a second copy in `.staging` during atomic promotion, and the previous fit of record kept as `.previous`.
- `data/vocabulary.duckdb` must exist (`uv run python scripts/prepare_data.py`), and `PYTHONUTF8=1` must be set.

**Launch** (PowerShell; the script also reads `/proc/meminfo` and `ps` on Linux):

```powershell
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # else the stage rules land in the log as mojibake
$mem = Start-Process pwsh -ArgumentList '-NoProfile','-File','scripts/memwatch.ps1','output/vg11-rep-memory.log','-IntervalSeconds','5' -PassThru -WindowStyle Hidden
try {
  uv run python scripts/fit_model.py vg11 --config rep --nutpie-backend jax 2>&1 |
    ForEach-Object { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $_" } |
    Tee-Object -FilePath output/vg11-rep-fit.log
} finally { Stop-Process -Id $mem.Id -Force }
```

`memwatch.ps1` samples every process whose command line names `fit_model.py` and prints whole gigabytes, which is the right resolution at this scale; 5 s rather than its 20 s default, because the runbook's own record is a +100 GB step in 90 s. Drop `--nutpie-backend jax` for a fit comparable with every fit of record, all of which used numba; keep it if the wall-clock question in §7 is part of the aim. Either is publishable. The fit inherits the lifetime of the terminal it is launched from; for an unattended run start the block above through `Start-Process pwsh -File <script> -WindowStyle Hidden` so closing the window does not kill it, and expect two to five hours.

**Record**, in this note or the run record: the maximum of the per-process field in the memory log (`grep -o '[0-9]*:vg11' output/vg11-rep-memory.log | sort -n | tail -1`); the timestamps of the `Posterior sampling`, `Diagnostics`, `Posterior predictions` and `Report` rules in the fit log; the size of `output/models/VG11-*/trace.nc`; and the gate line. Then compare with §4: the prediction is a ~48 GB peak. Under about 55 GB and the 96 GB claim holds for VG11 with margin, and the runbook's TD section can be rewritten from this note; materially above it and the multiplier has grown with scale, in which case read the memory log against the stage timestamps to see which stage did it before reaching for §5's levers.
