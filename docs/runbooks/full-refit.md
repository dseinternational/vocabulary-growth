# Runbook: full reporting-config refit of all models

> [!NOTE]
> Drafted by LLM-based AI tools (Claude Code/Opus 4.8 and OpenAI Codex/GPT-5; the Quarto kernel-resolution section, and the PowerShell driver and dirty-checkout material, by Claude Code/Opus 5).

How to refit the whole `VG01`–`VG16` family at reporting quality (`rep`) on a
large VM, render every report, and produce comparisons — with the pitfalls that a
naive run hits. Distilled from the 2026-07-12 run
(`notes/202607121753-reporting-config-fit-run-and-findings.md`).

## TL;DR

- Canonical, resumable path: `scripts/run_replication.ps1 -Config rep`. It needs
  PowerShell Core 7+ and runs unchanged on Linux, macOS and Windows.
- On a many-core VM, fit the **DS models concurrently** and the **TD models one at
  a time** (see [Parallel fitting](#parallel-fitting-on-a-large-vm)) — pass
  `-MaxParallel` rather than driving a pool by hand.
- Three things bite every time: the **DuckDB lock** on concurrent fits, the
  **R-hat gate rounding** (need `dse-research-utils >= v0.6.0`), and the
  **understood-GP R-hat ridge** in the DS joint/hierarchical models.
- **Before the TD phase**, do the two-minute setup in
  [Surviving an OOM](#surviving-an-oom-precautions-before-launching-the-memory-heavy-models):
  add swap (the box has none), and launch each phase in its own systemd scope. In
  the 2026-08-13 run a single overrun killed a seven-hour fit plus three unrelated
  ones, because they shared a scope and there was nothing to swap to.

## 0. Prerequisites

- Locked project environment installed and active: `uv sync --locked`, then `source .venv/bin/activate`. The commands below assume it is on `PATH`; without activation, prefix each with `uv run`. Activation matters for **rendering**, not only fitting — read [Rendering without an activated environment](#rendering-without-an-activated-environment) first if you drive the scripts by absolute interpreter path instead.
- **`dse-research-utils >= v0.6.0`** — earlier versions' convergence gate rounds
  R-hat/ESS to 2 significant figures and can certify a fit that truly fails the
  ≤1.01 gate (research#65). A banner reading exactly `max R-hat = 1.0` is the
  tell-tale of the old rounding.
- Data current: `python scripts/prepare_data.py` (confirm the 810 reference scale;
  see `docs/report/methods-data.qmd`).
- **On the DSE data-science fleet, most of this list is already on the image.**
  The stack's cloud-init installs `uv`, the newest stable CPython, Node.js active
  LTS, Quarto with a per-user TinyTeX (so `quarto render --to pdf` works without
  a separate LaTeX install), Pandoc, plotting fonts, `gh`, Graphviz, and — on the
  ARM64 CPU bootstrap — pinned **PowerShell 7.6 LTS exposed as `pwsh`**, which is
  what `run_replication.ps1` needs. Do not install competing versions by hand.
  The one thing worth checking rather than assuming is the report book's fonts:
  the image provides "plotting fonts", which is not the same claim as Source
  Sans 3 and Monaspace Neon, and those are needed only for the `pdf` format.
- **Graphviz `dot` on `PATH`.** Present on the DSE VM images since 2026-08-25. It
  is the one tool a _fit_ tolerates missing — `render_model_graph` catches the
  failure and prints a warning rather than aborting — but every model report
  references `gp_model_graph.svg`, so without it all twenty render with a broken
  figure and nothing fails loudly enough to notice.
- Disk: **choose the trace tier against the volume you actually have, and set it
  before you start.** Under about 1 TB use
  `DSE_VOCAB_GROWTH_TRACE_PERSISTENCE=compact`; at 1 TB or more leave the default
  `full`, and make sure that variable is _unset_ so it cannot silently override
  you. `compact` is byte-identical for reporting but blocks recovery scoring,
  `regenerate_plots.py` and `loso_compare.py` on those fits without a refit, so
  it is a saving worth making only when the space is genuinely tight. Either way,
  redirect output off the checkout with `--output-dir <scratch>` or
  `DSE_VOCAB_GROWTH_OUTPUT_DIR` — and point it at the **attached** disk rather
  than at a local temp disk, for the reasons in
  [Fit straight to the attached disk](#fit-straight-to-the-attached-disk-not-to-local-scratch). The report figure cache
  (`docs/report/figures/`) always stays in the checkout. Sizing and the
  exceptions are in [Surviving a full disk](#surviving-a-full-disk) — read it
  before the first fit, not after. The old advice here ("~20 GB × n_models") was
  wrong by more than a factor of two, because a run fits far more _variants_ than
  models.
- **Run `prepare_data.py` in any fresh checkout, worktree or clone before `pytest`.** `data/vocabulary.duckdb` and `data/vocab_data_merged.csv` are generated and gitignored, and the tests that read the real pool (the `dse_native_only` ones in `test_data_utils.py` and `test_joint_four_cell.py`) fail without them — with an error about the pool, not about a missing file, so it reads as a code regression. Hit while validating a merge in a `git worktree` on 2026-08-14.
- `rep` config = 6 chains / 6000 tune / 6000 draws / `target_accept` 0.95. The number of parallel cores is chosen for the host and does not affect fit compatibility.
- Publishing needs `DSERESEARCH_BLOB_CONTAINER_URL` **and** the right identity — see below.

### Uploading from an Azure VM: `DefaultAzureCredential` picks the wrong identity

`upload.py` authenticates with `DefaultAzureCredential`, which prefers the **VM's managed identity** over your `az login` session. On a DSE research VM that managed identity has no write role on the container, so the upload fails on the first model with:

```
ErrorCode:AuthorizationPermissionMismatch
This request is not authorized to perform this operation using this permission.
```

`az account show` reporting the right user is **not** evidence the upload will authenticate as that user. Check what the credential actually resolves to:

```bash
python -c "
from azure.identity import DefaultAzureCredential; import base64, json
t = DefaultAzureCredential().get_token('https://storage.azure.com/.default')
p = t.token.split('.')[1]; p += '='*(-len(p)%4)
c = json.loads(base64.urlsafe_b64decode(p)); print(c.get('upn') or c.get('appid'))"
```

An `appid` GUID rather than a `upn` means it chose the managed identity. Force the developer credential for the upload (azure-identity ≥ 1.23):

```bash
export AZURE_TOKEN_CREDENTIALS=dev
```

The failure is at least safe: validation runs for all models first, and the first blob write fails before anything is written, so a rejected upload cannot leave the published set half-replaced.

## 1. Fit

### Batch failure semantics

`python scripts/fit_model.py all --config rep --render --upload` treats convergence and rendering as per-model failures and publication as a batch-level decision. It continues fitting the remaining models after a `ConvergenceGateError`, atomically promotes every successful fit before rendering it, continues rendering after an individual Quarto failure, suppresses the entire upload phase so no partial batch is published, reports every failed model, and exits non-zero. A render failure leaves the completed fit available for `--render-only`; other fitting exceptions still abort immediately. The canonical `run_replication.ps1` path remains resumable and fits one model at a time unless `-MaxParallel` raises it.

### us_01 implausible-production sensitivity

The old `us01-ceiling-excluded` variants **no longer exist** and `fit_sensitivity.py` raises `KeyError` for them: they excluded records that the Edgin audit established as invalid and that are now masked by default, so they could not fail (`notes/202607261245-edgin-duplicated-outcome-records.md` §§9–10, 13).

They are replaced by the inverse. `us01-implausible-reinstated` puts the masked spoken observations back and refits, answering what changes if the default exclusion is itself mistaken. This is not optional in a published refit: the source author no longer holds the original data files, so the exclusion can never be confirmed at source, and this pair is the only evidence a reader has for whether the headline joint trajectories depend on our judgement.

```bash
python scripts/fit_sensitivity.py vg10 us01-implausible-reinstated --config rep
python scripts/fit_sensitivity.py vg15 us01-implausible-reinstated --config rep
python scripts/compare_sensitivity.py vg10 --variant us01-implausible-reinstated
python scripts/compare_sensitivity.py vg15 --variant us01-implausible-reinstated
```

Check the fit log's observation counts: each variant prints `us_01 implausible production reinstated` and it must read **11** against the current pool. A zero there means the variant has stopped biting and the comparison is worthless — treat it as a failure, not a pass.

> [!NOTE]
> **Re-pinned from 22 to 11 on 2026-08-14.** The old figure was correct when `us_01` came from the Wordbank by-child export; rebuilding it from the Edgin item-level contributor files changed which administrations trip the near-ceiling and longitudinal-collapse signatures. Verified three ways rather than assumed: `vg10` and `vg15` independently log 11; `vg10`'s frame goes from 1,428 spoken observations at baseline to 1,439 in the variant, exactly +11; and the loader confirms it directly — `include_implausible_production=True` takes `us_01` from 211 to 222 spoken observations, whole-pool 1,428 to 1,439. Row counts are unchanged at 230 either way, because the rule blanks the `spoken` value rather than dropping the row. The old note about "22 rather than 30, the other 8 under the duplicated-outcome rule" no longer describes the current pool and has been dropped; the duplicated-outcome rule is still independent and still has its own flag.

### The 810-item reference denominator

`dse-native-only` is the other pair that is not optional in a published refit, and it is the only check on an assumption the report otherwise states and moves past. Every model scores raw counts against `n_trials = 810`, so a 416-item Oxford CDI count enters on a denominator its form never used; that is sound only if the shorter forms hold the easier items. The sufficiency result (`notes/202607261540-item-difficulty-and-the-aggregate-likelihood.md`) is the proof that no aggregate analysis of these data can test that — a statistic sufficient for ability carries no information about item composition — so the assumption is probed by deleting the rows that need it rather than by modelling it.

```bash
python scripts/fit_sensitivity.py vg10 dse-native-only --config rep
python scripts/fit_sensitivity.py vg15 dse-native-only --config rep
python scripts/compare_sensitivity.py vg10 --variant dse-native-only
python scripts/compare_sensitivity.py vg15 --variant dse-native-only
```

Same discipline as above: each fit prints `Non-native-ceiling rows excluded`, which must read 1,243 against the current pool. It is the widest-scoped variant registered — 278 of 1,521 rows survive, from ie_01's 810 wave, ie_02, uk_02's DSE arm and uk_06 — so expect wide intervals and read it for whether the trajectory _shapes_ hold, not for agreement to three significant figures. On VG15 it also leaves uk_02 as the only cross-tab source, so `psi` falls back to its single-study branch: that fit answers the denominator question, not the association one.

### VG15's `psi` after the study-level term

Four further VG15 variants exist for the 2026-08-12 changes and cost a fit each. `tau-psi-narrow` and `tau-psi-wide` bracket the between-study scale, which was set from the measured spread and so is data-informed in exactly the way Target 8 was created for; with four informed studies it is weakly identified and governs how far the per-study values shrink toward the reported centre. `psi-drop-es01` and `psi-drop-uk07` remove one source's cross-tab while keeping its marginals, so U, q and r are untouched and only the association loses evidence — es_01 is the one to run if only one is affordable, being 185 of the 434 `psi`-informing rows and the only source sitting at independence.

### Default (sequential, resumable)

```powershell
./scripts/run_replication.ps1 -Config rep -OutputDir <scratch>
```

Idempotent: a model is skipped only when its state is `complete` and its model definition, requested sampling tier and minimum statistical effort, raw-data fingerprint, and Git commit match the current run (`--fresh` forces a refit). Host-dependent `cores` is ignored; a documented high-tuning refit is compatible when its draws, tuning iterations, chains and target acceptance meet or exceed the tier. A trace file by itself is never treated as complete. The script refuses to start from a dirty checkout, fits models, validates the set once, retries per-model rendering without resampling, runs comparisons, atomically synchronises figures, renders the report and comparison book, and optionally uploads. Development/test runs use provisional figure sync and do not upload. Any required-step failure stops all downstream comparison and publication phases and leaves a `FAILED` marker in the run log directory; an entirely successful run leaves `SUCCESS`. Estimate approximately 15–25 hours sequentially.

The dirty-checkout refusal is not fussiness, and `-AllowDirty` is a development-only escape. `write_fit_manifest` records `git_metadata` when the manifest is written — at the **end** of each fit, not its start — so editing the working tree mid-run stamps `dirty: true` on every fit still in flight, and `check_fit.py --purpose publish` then refuses them as "produced from a dirty or unverifiable checkout". Land any code or documentation change as a commit **before** starting a run, not during one; a run interrupted for a repository change has to be restarted, not resumed, because `--purpose resume` also compares the current commit against each fit's.

### Parallel fitting on a large VM

The DS datasets are small; the full-data TD models (`vg11`, `vg12`) are
memory-heavy. So:

> [!WARNING]
> **The two lists below must together cover every key in `MODEL_REGISTRY`.** They are an explicit `-Models` split, so the driver's registry-derived default does not apply and a model missing from both is never queued, never validated, and never reported as absent — the run ends `SUCCESS` having fitted a subset. `tests/test_runbook_model_lists.py` checks the split against the registry; if it fails, correct the lists here rather than the test.

- **DS models** (`vg01 vg02 vg05 vg07 vg08 vg09 vg10 vg14 vg15 vg16 vg19 vg20 vg22`): run
  a pool, `concurrency × 6 ≤ physical cores` (e.g. 5 on 32 cores):

  ```powershell
  ./scripts/run_replication.ps1 -Config rep -OutputDir <scratch> -MaxParallel 5 -NoCompare -NoRender -NoUpload -Models vg01,vg02,vg05,vg07,vg08,vg09,vg10,vg14,vg15,vg16,vg19,vg20,vg22
  ```

  `-MaxParallel` above 1 pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` to 1 for you, because each fit already
  runs its own chains in parallel and a pool otherwise oversubscribes the box
  through the BLAS/OpenMP thread pools (see the warning below). Setting any of
  them yourself beforehand keeps your value.

  `-MinFreeGB` (default 16) additionally holds a launch back while available
  memory is below the floor and a fit is already running, so the pool throttles
  itself rather than stacking peaks — the post-sampling assembly step is where
  these fits spike, not the sampling.

- **TD models** (`vg03 vg04 vg11 vg12 vg13 vg21 vg23`): **strictly one at a time** — the
  full-data TD fits can OOM if stacked. `vg03` and `vg04` are the exception and may
  share the box; `vg11`, `vg12`, `vg13`, `vg21` and `vg23` must not share it with anything,
  including a batch of small DS sensitivity fits (see below). Run them as a separate
  `-MaxParallel 1` pass; a single pool with a mixed model list cannot express this.

  ```powershell
  ./scripts/run_replication.ps1 -Config rep -OutputDir <scratch> -MaxParallel 1 -NoCompare -NoRender -NoUpload -Models vg03,vg04,vg11,vg12,vg13,vg21,vg23
  ```

  `vg21` and `vg23` join this pass because both are VG13 with one thing changed and
  neither is lighter than it: `vg23` is VG13's frame exactly, plus `rho_uq`, and
  `vg21` widens VG13's 8–18 month window to 8–22, so it sees strictly more of the
  TD pool than the model the serial rule was written for. Neither has a `rep` fit
  yet, so neither has a measured peak — treat them as VG13-class until one exists.

### Fit straight to the attached disk, not to local scratch

Asked when the 2026-08 run was provisioned with a 2 TB premium disk: fit to the
VM's local SSD and copy the results across afterwards, or write straight to the
attached disk? **Straight to the attached disk.** Three reasons, in order of
weight.

**The output root has to be one filesystem.** `create_staging_root` puts
`.staging` _inside_ the output root, and `promote_staged_fit` publishes with
`os.replace` — a rename. Across filesystems that raises `EXDEV` rather than
degrading to a copy, so the pipeline cannot stage on local and publish to the
attached disk; the rollback path (`.previous`, also under the output root) has
the same constraint. Fitting to scratch therefore means the _whole_ output root
lives on scratch, and the copy to the attached disk is a separate manual step
outside the atomicity machinery — a crash during a 320 GB copy leaves a partial
fit that nothing guards against. That trades a real protection for a saving the
next point shows is negligible.

**The saving is noise against the sampling.** Posterior sampling is about 92% of
a `rep` fit's wall clock (measured on VG12: 3h09m of 3h26m), and the trace is a
single burst at the end. Writing a 16 GB `trace.nc` costs on the order of a
minute to premium storage and a fraction of that to local NVMe — call it tens of
seconds per fit, perhaps 10–25 minutes across the whole run. Copying ~320 GB back
to the attached disk afterwards costs about the same again, so the round trip
saves nothing and may lose.

**Local disk is disposable by design, and the fits are long.** On the DSE data
science fleet the two mounts are explicit about this
([`dsegroup/infrastructure`](https://github.com/dsegroup/infrastructure), the
data-science stack's cloud-init):

| mount      | what it is                                 | survives teardown |
| ---------- | ------------------------------------------ | ----------------- |
| `/data`    | the persistent disk, mounted when attached | yes               |
| `/scratch` | every local NVMe the tier carries, striped | **no — wiped**    |

`$TMPDIR` points at `/scratch`. **Fit to `/data`.** The XL tier is
`Standard_E32pds_v6` — the `d` variants do carry local NVMe, so the choice
genuinely exists here and is not small: [infrastructure
#1804](https://github.com/dsegroup/infrastructure/pull/1804) stripes all of it
into one RAID0 volume, taking XL's `/scratch` from 440 GiB to about 1.3 TiB.
That PR was still open on 2026-08-25, so a box provisioned before it merges comes
up with the single-device 440 GiB `/scratch`; either way the size objection is
not what decides this.

What decides it is that `/scratch` is _meant_ to be lost. It is wiped on
teardown, and a deallocate/start wipes local NVMe outright even though #1804
gives the mount an `fstab` entry that survives a plain reboot. The stripe also
multiplies the device-failure surface across three disks — which that PR
correctly accepts, on the grounds that the failure "costs a workspace that
teardown was going to wipe anyway". A fifteen-hour VG12 fit is not that. This
project has already lost `rep` fits to a full disk (2026-08-14) and to a
concurrent OOM; host maintenance is not a third failure mode worth buying for
tens of seconds per fit.

**Where `/scratch` does help**: PyTensor's compile cache
(`PYTENSOR_FLAGS=base_compiledir=/scratch/...`). Many small latency-sensitive
files, disposable by design — exactly what the mount is for, and with the stripe
it has both the room and the throughput. Keep expectations low: CI measured the
compile cache as noise and dropped it in `8b7de41`.

### Surviving a full disk

On 2026-08-14 at 16:12 the output volume reached 100% and five in-flight refits died on `[Errno 28] No space left on device` — VG16 and VG14 part-way through writing `trace.nc`, and VG10, VG07 and VG05 before they had started. Nothing warned first: the fits simply failed at the moment they wrote.

**The cause was the trace tier, not the disk size.** Every fit that run wrote at the default `--trace-persistence full`. `compact` drops observation-sized deterministics that are recomputable from the free parameters, and the reporting output is byte-identical. Applied afterwards to 21 fits it took **229 GB to 77 GB** — the volume went from 100% to 63% with no loss of anything the report reads. (Since 2026-08-23 those observation-sized deterministics are not sampled at all, at any tier — see `--trace-persistence` in `CLAUDE.md` — so a new `full` trace is already close to what `compact` used to leave; the arithmetic in this section and the sizing below describe the fits that run produced, not what a fit writes now.)

**Set the tier before the run**, either on the driver or in the environment:

```powershell
./scripts/run_replication.ps1 -Config rep -TracePersistence compact
# or, for anything driven directly:
$env:DSE_VOCAB_GROWTH_TRACE_PERSISTENCE = 'compact'
```

**Three models should stay at `full`: VG10, VG12 and VG15.** They are `fit_recovery.py`'s headline set, and recovery scoring refuses a compacted trace up front — as do `regenerate_plots.py` and `loso_compare.py`. Pass `--trace-persistence full` for those three specifically. Everything else can be compacted, at the cost of needing a refit if its plots ever have to be regenerated.

**Sizing.** Budget by _fits_, not by models: a full round fits ~15 models of record plus ~20 registered sensitivity variants plus recovery replicates. At `full` that exceeds 400 GB; at `compact` it is roughly 130–150 GB. Add headroom for atomic promotion, which transiently holds a second copy of the largest trace in `.staging`. **500 GB is comfortable at `compact`; 1 TB at `full`.**

The 2026-08 refit is provisioned on a **2 TB attached disk**, so `full` is the tier to use and this section's `compact` advice does not apply to it. Measured against the current traces, its 28 planned fits come to about 320 GB at `full` — lower than the 400 GB above because that figure includes recovery replicates, which this run does not schedule. Either way it is comfortably inside the volume, and a fresh VM starts with an empty `output/`, so the peak equals the total rather than transiently holding both an old and a new trace.

**Recovering a volume that is already full**: `scripts/compact_traces.py` applies the tier to traces already written. It reuses the same policy code the fit pipeline uses, verifies each rewrite carries every free parameter the original had before atomically replacing it, and records the tier in `fit_manifest.json`. It processes smallest-first, which matters — each rewrite needs room for its output beside the original, so the small traces buy the space the large ones need.

```bash
python scripts/compact_traces.py --dry-run
python scripts/compact_traces.py --exclude VG10-... --exclude VG12-... --exclude VG15-...
```

Note that it will not touch a model whose fit is mid-promotion, and that it distinguishes a live staging directory from one an `ENOSPC` crash left behind by checking the PID embedded in the name — the aftermath of a full disk is full of stale staging directories, and a _live_ fit of some other model is usually exactly what the space is being reclaimed for.

**On provisioning.** Check `lsblk` before assuming the box is at its limit: this VM had two unmounted 440 GB NVMe devices alongside the one in use. Longer term the output root belongs on a **managed disk rather than the Azure temp disk** — not for speed (the local NVMe measured 472 MB/s sequential, and this workload writes each trace once, so storage is nowhere near the critical path) but because the temp disk is wiped on deallocation, which forces the VM to stay running through the idle stretches of a multi-day run, and because a managed disk can be grown online. Do not put `trace.nc` on blobfuse or Azure Files: netCDF here is HDF5, whose metadata I/O and POSIX assumptions make network filesystems a corruption risk. Blob remains the archive tier via `upload.py`.

### Surviving an OOM: precautions before launching the memory-heavy models

On 2026-08-13 a kernel OOM killed `vg13` after **7h05m of successful sampling** and
took three unrelated sensitivity fits and both drivers with it, costing about ten
hours. Every item below is a direct consequence. Do all of them before starting the
TD phase; they take about two minutes.

**1. Provision swap first. The box ships with none.** 251 GB of RAM and `Total swap = 0kB`
means a transient overshoot is an instant kill rather than a slowdown.

```bash
sudo fallocate -l 128G /scratch/swapfile && sudo chmod 600 /scratch/swapfile
sudo mkswap /scratch/swapfile && sudo swapon /scratch/swapfile
sudo sysctl -w vm.swappiness=10   # backstop only, not a working store
```

Put it on the scratch NVMe, not the 29 GB root. Deliberately **not** added to
`/etc/fstab`: it is a per-run measure, and a run does not survive a reboot anyway.

**2. Peaks are not predictable from plateaus, and plateaus are not stable between runs.** This is the trap, and the 2026-08-14 rerun sharpened it.

`vg13` was killed at **232 GB** on 2026-08-13 after seven hours. Re-run on 2026-08-14 with an identical configuration, it plateaued at **178 GB** rather than the previous ~120 GB and peaked at **243 GB** — higher than the figure that had killed it — before completing in 337 minutes. Same commit, same data, same tune/draws/chains.

So do **not** budget from a remembered plateau, and do not trust a "peak ≈ 2× plateau" rule; both numbers moved by ~50% between two runs of the same fit. What actually kept it alive was structural, and all three parts were needed:

- **Swap existed.** The 2026-08-13 kill happened with `Total swap = 0kB`.
- **Scopes were separate and capped**, so the co-tenants could be shed as one unit.
- **Someone was watching per-process RSS and shed load.** At 13:31 the box had 39 GB available with `vg13` at 178 GB and a capped DS batch holding 29 GB; stopping that batch immediately preceded `vg13` reaching 243 GB with room to spare.

The practical rule is therefore about _headroom and reversibility_, not about a target number: keep a memory-heavy TD fit as the only significant tenant, keep everything else in a scope you can stop in one command, and watch it. Anything co-scheduled with `vg11`/`vg12`/`vg13` should be work you are willing to throw away.

**3. Give every job its own systemd scope.** This is what turned one lost fit into
four. All jobs were launched inside one tmux window, so they shared a single systemd
scope, and when the kernel killed one process systemd applied `OOMPolicy` to the
whole scope: `tmux-spawn-….scope: Failed with result 'oom-kill'`. Launch each phase
in its own scope, and cap any batch that is not the one you are protecting:

```bash
# the protected long job
systemd-run --user --scope --collect --unit=vg-td -p OOMPolicy=continue \
  -- bash run_td.sh &
# small fits alongside it can never be what pushes the box over
systemd-run --user --scope --collect --unit=vg-sens -p MemoryMax=64G -p OOMPolicy=continue \
  -- bash run_sens.sh &
```

`MemoryMax` contains an overrun inside the offending cgroup, so a runaway 30-minute
sensitivity fit dies alone instead of killing a seven-hour one.

> [!IMPORTANT]
> **A systemd scope does not inherit conda activation.** The first relaunch after the
> crash failed all three TD jobs in 0m with `ModuleNotFoundError: No module named
'dse_research_utils'`. Source and activate inside the driver script, and assert it
> before any fit starts, so the failure is one line rather than a silent batch of
> zero-minute FAILs:
>
> ```bash
> source /opt/conda/etc/profile.d/conda.sh
> conda activate dse-vocab-growth
> python -c "import dse_research_utils" || { echo "env not activated" >&2; exit 1; }
> ```

**4. Sample per-process RSS, not machine-wide `used_GB`.** A machine-level sampler
records that the box hit 244 GB but not which process did it — after the 2026-08-13
kill, `vg13` could only be distinguished from its three co-tenants by reading the
kernel log. Sample per-process RSS filtered to the fit scripts, so the next model's
budget comes from measurement — `scripts/memwatch.ps1 <logfile>`, which reads
`/proc/meminfo` and `ps` on Linux and the equivalent CIM classes on Windows.

**5. Read the kernel log before re-running anything.** `sudo dmesg -T | grep -i oom`
and `journalctl --since …` distinguish the three cases that look identical from the
status file — a real convergence failure, a process killed by the OOM killer, and a
process killed as collateral scope teardown. The three sensitivity fits killed here
had left the **known non-fatal** PyTensor rewrite traceback
([pymc-devs/pytensor#2349](https://github.com/pymc-devs/pytensor/issues/2349)) at the
end of their logs, which reads convincingly as the cause and is not.

> [!WARNING]
> **The driver records nothing when the scope is torn down.** A `run_job` wrapper
> writes `OK`/`FAIL` only if it outlives the job, and it does not survive an
> `oom-kill` scope teardown. So an OOM leaves the status file showing `START` with no
> terminal line — indistinguishable from "still running" until you check `pgrep`.
> Never infer success or liveness from the status file alone.

**6. Detach the driver from whatever supervises it.** A fit inherits the lifetime of its parent process group, so anything that stops the supervisor kills the fit — no kernel log entry, no OOM, nothing to diagnose after the fact. On 2026-08-16 `vg11 anchor-broad` was killed three hours in, past its prior predictive checks and well into sampling, because the agent-harness background task running the driver was stopped; `sudo dmesg -T` for that day was empty, which is what distinguishes this case from item 5. Launch the driver so it outlives its launcher:

```bash
setsid nohup bash scripts/driver.sh >/dev/null 2>&1 < /dev/null &
ps -o ppid= -p "$(pgrep -f driver.sh)"   # must be 1
```

Check the reparenting rather than assuming it: `nohup … &` alone leaves the process in the launcher's group and does not survive a group kill.

**7. Order the queue cheapest-first unless something specific argues otherwise.** Putting the heaviest fit first to fail fast on disk or memory is only worth it while that resource is actually in doubt. Once headroom has been stable for hours, heaviest-first just maximises what a single interruption destroys — which is exactly what item 6 cost. Reordering after the loss got three of the four Target 8 variants moving while the multi-hour one waited.

### Reporting age caps, and what `regenerate_plots.py` can and cannot fix

Every figure and table stops where its own outcome's evidence stops. The policy
lives in `src/vocab_growth/reporting_ages.py` — understood **72**, spoken **90**,
signed **84**, and anything conditioned on understood (`q`, `r`, `p_any`,
comprehension gaps) **72**, because the conditioning rule takes the _lower_ of the
two components. Call sites name the _quantity_, not a cap attribute, because
choosing the wrong attribute is a defect that has already shipped twice.

> [!WARNING]
> Understood was **84** until `ae04e5e` (2026-08-22) returned it to 72, and the
> sign-ratio helper that makes `r` and `p_any` follow it —
> `reporting_ages.max_age_for_sign_ratio` — landed a day later in `565a769`.
> **VG14 and VG15 were fitted in the gap**, on 2026-08-22, so both write
> `posterior_summary_r` and `posterior_summary_p_any` out to 84. `check_fit.py`
> passes them, because their manifests record the current definition; only
> `tests/test_reporting_age_policy.py` sees it. Both need refits — see §2.

> [!IMPORTANT]
> **`regenerate_plots.py` re-runs the plot stage only.** Artefacts written by the
> _summary_ stage cannot be refreshed without a refit. On 2026-08-14 that was the
> difference between a policy change costing nothing and costing two fits: the
> univariate and bivariate engines emit their monthly summaries inside the plot
> stage, so twelve models were brought into line by regeneration alone, but VG14's
> `posterior_summary_p_any` / `posterior_summary_sign` and VG15's
> `posterior_summary_monthly_*` / `expected_counts_by_month_*` are summary-stage
> and stayed stale. `KNOWN_STALE` in `tests/test_reporting_age_policy.py` carried
> them until `fa9f836` emptied it; it is empty now, so any summary-stage artefact
> left past its cap fails the suite outright rather than being excused.

Check the policy against **output**, not call sites. `tests/test_reporting_age_caps.py`
walks the AST against a hand-written list of plot functions and so cannot see an
artefact nobody thought to cap; `tests/test_reporting_age_policy.py` reads a
fitted model's directory and checks every table. The second found sixteen
uncapped artefacts the first had passed for months — including
`posterior_predictive_pmf`/`_cdf`, which carry age in their **column names**
(`pmf_90m`) rather than in a column, and which a filename audit missed as well.

### Do not edit tracked files while a fit is launching

`write_fit_manifest` runs **immediately after stage 0** ("Prepare data"), within
seconds of launch, and records `git status --porcelain --untracked-files=normal` for
the **whole tree** — not just the code. Any tracked file modified in that window,
including a note or a runbook, sets `code.dirty = true` and makes the fit
unpublishable: `check_fit.py --purpose publish` rejects it with "The fit was produced
from a dirty or unverifiable checkout", and you discover it hours later.

The window is short, but the failure is silent and expensive. Commit or stash before
launching a batch, and confirm afterwards by reading the staged manifests rather than
waiting for the publish gate:

```bash
python - <<'PY'
import glob, json
for m in sorted(glob.glob("output/.staging/*/models/*/fit_manifest.json")):
    j = json.load(open(m))
    print(f"{m.split('/')[-2][:52]:54s} dirty={j['code']['dirty']} commit={j['code']['commit'][:7]}")
PY
```

### Config choice for the full-data TD models

> [!WARNING]
> **Superseded for the hierarchical TD models (2026-07-17, post-#164).** #164 added
> child (subject) random effects to `vg11`/`vg12`/`vg13` (the #163 P1 fix; their output
> dirs are now `…-td-re`), making them hierarchical. They now carry the same
> trend/GP/study-intercept ridge the DS models have, and **`rep-lite` no longer
> converges them**: `vg11` failed at `rep-lite` with max R-hat 1.023 / min ESS 164 on the
> trend/GP/study-RE block (not the per-child effects). **Fit `vg11`/`vg12`/`vg13` at
> `rep-hightune`** (tune 12000 / draws 8000 / 6 chains; `target_accept 0.99` for `vg13`,
> which also has divergences) — e.g. via `scripts/refit_hightune.py`. The `rep-lite`
> guidance below was validated only on the _pre-#164, non-hierarchical_ TD models and is
> retained for history. (See `notes/202607170935-full-refit-vm-run-147-163.md`.)

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

> [!NOTE]
> **The four env vars do not pin nutpie's own thread pool.** They control BLAS and
> numba, but nutpie's Rust sampler sizes its pool from the core count independently.
> Measured on 2026-08-14 with all four set to `1` and 6 chains requested: `vg13` held
> **33** threads and each concurrent `vg15` sensitivity **78** — 189 threads on 32
> cores. Observed load stayed near `concurrency × chains` (21.6 against 18 expected),
> so the surplus threads are mostly parked rather than competing, and this is a
> smaller effect than the BLAS oversubscription above. Still, budget by
> `concurrency × chains` and verify with `uptime` rather than trusting the env vars,
> and treat a thread count far above the chain count as expected, not as a fault.
> `RAYON_NUM_THREADS=1` is the likely lever but is **untested here** — measure before
> relying on it.

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

### Expect diagnostics to move on a refit — but only for models with child effects

Refitting the same commit against the same data does **not** reproduce diagnostics
bit-for-bit for every model, and a docstring claiming otherwise misled a day of this
run. The split is clean and worth knowing before you start comparing runs:

- **Models without child random effects reproduce exactly.** `vg05` (max R-hat
  1.00092, min ESS 7408), `vg07` and `vg14` were byte-identical across repeat fits —
  `vg14` across all three of its fits in the 2026-08-13 run.
- **Models with child random effects drift.** `vg08` moved 1.00437 → 1.00851 and
  `vg09` 1.00452 → 1.00623 with min ESS 1365 → 1102, on identical code and data.
  `vg10`, `vg15` and `vg16` drift likewise.

**The estimates are unaffected — only the diagnostics move.** `vg09`'s posterior
means agreed to **≤ 0.023 posterior SDs** across the refit. So treat a changed R-hat
on a random-effects model as sampling variation, not as evidence that something
changed; verify by comparing posterior means in SD units before investigating.

The corollary is a real scheduling risk: **`vg08` sits on the gate boundary**
(R-hat 1.00851 against 1.01, a margin of 0.0015) at the strongest tuning available
(16000/10000/0.99) and is the single longest DS fit at 202 m. Any change that forces
a DS refit is a coin-flip on `vg08` passing, and there is no stronger setting left to
escalate to. Weigh that before taking a change that re-fingerprints the DS family.

> [!WARNING]
> **Reporting-only fields live inside the fingerprinted model definition.** Changing
> a value that affects nothing but which ages get printed — `report_max_age_understood`,
> `report_max_age_signed` — changes the definition, which invalidates every affected
> model of record and forces a **full refit**. That cost three refits on 2026-08-13.
> `regenerate_plots.py` cannot rescue it either: it validates the definition first,
> and its own docstring names "a missing reporting-age cap" as the motivating case it
> does not cover. Batch all reporting-cap decisions **before** the fitting phase.

### Known ridge: the understood-GP block

The DS joint/hierarchical models (`vg09`, `vg10`, `vg15`, `vg16`) tend to leave the
**understood-trajectory GP block** (`g_u` / `g_unit_u` / `g_unit_u_hsgp_coeffs` /
`slope_u` / `p_slope_low_u`) just over 1.01 at `rep` — the trend/GP/intercept
redundancy the VG10 GP anchor addresses for the `q`-GP, here on the understood GP.
Remedy: refit with heavier tuning (**tune 12000 / draws 8000 / target_accept 0.97**,
6 chains), which cleared all four in the 2026-07-12 run (e.g. vg16 1.024 → 1.009).
Back up the non-converged output first; the refit becomes the model of record.

> [!NOTE]
> **Update (2026-07-17, post-#164/#161):** this DS understood-GP ridge did **not** recur —
> all ten DS models passed R-hat/ESS at plain `rep` (max R-hat 1.0066), so no DS
> `rep-hightune` refits were needed. **Re-assess the DS family empirically** rather than
> assuming hightune. The ridge now surfaces instead on the **hierarchical TD models**
> (`vg11`/`vg12`/`vg13`) — see the TD config warning above. (`vg13` additionally needs
> `target_accept 0.99` for divergences, as in July.)

## 3. Render + comparisons

Two report blocks read artefacts the fit itself does not write, and print a "run this" note until they exist. Produce them per fit **before** rendering (each opens the trace, so run them one model at a time rather than as a sweep while a heavy fit is on the box):

```bash
python scripts/prior_vs_posterior.py --table --model vg20 --model vg15   # writes prior_posterior_contraction.csv into each fit dir
python scripts/emit_factor_correlation.py <output>/models/VG22-*/         # writes subject_factor_corr.csv for the factor model
python scripts/regenerate_plots.py all --config rep --output-dir <scratch>   # re-runs the plot stage: since 2026-09-03 the joint RE pages reference study_fans.png and posterior_summary_monthly_weighted_{u,s}.csv, and the words/ratio figures carry the observed children, none of which a fit made before that date wrote
```

A template change is applied to an existing fit with `--render-only`, which re-stages `docs/models/<model>/index.qmd` **and** every `docs/models/_*.qmd` include beside it (the bivariate random-effects family transcludes one). Since 2026-09-02 that is a fresh render of every page, not only the changed ones, because the shared blocks changed.

```bash
python scripts/sync_report_figures.py --config rep --output-dir <scratch>   # validates fits, then feeds docs/report/figures/
# comparisons (consume fitted traces/summaries):
for c in loo_compare loso_compare compare_models compare_ds_td \
         compare_ds_td_trajectories compare_ds_td_expressive \
         compare_ds_td_latency compare_ds_td_q_overlap compare_ds_td_re; do
  python scripts/$c.py
done
python scripts/sync_report_figures.py --config rep --output-dir <scratch>   # re-sync comparison artefacts
# `sync_report_figures` ATOMICALLY REPLACES docs/report/figures/, which destroys
# the illustrative figures the introduction uses (bayes_update*.png). They are not
# model output, so nothing regenerates them. Regenerate AFTER every sync, or
# `quarto render docs/report` fails on a missing file:
python scripts/prepare_report_figures.py
# Pin Quarto's python to the project venv for BOTH book renders: a bare `quarto render`
# resolved a python without pyyaml on 2026-09-03 and both books died in their first
# cell with `ModuleNotFoundError: No module named 'yaml'`. fit_model.py pins this for
# the per-model pages; nothing pins it for the books.
export QUARTO_PYTHON="$(python -c 'import sys; print(sys.executable)')"
quarto render docs/report
# the comparison book reads its CSV/PNG artefacts by BARE filename from its own dir,
# and sync_report_figures only populates docs/report/figures/ — so stage them first.
# Files only: `cp <scratch>/comparisons/*` exits non-zero on the `sensitivity/`
# subdirectory (`cp: -r not specified`) after copying everything else, which reads
# as a failed step.
find <scratch>/comparisons -maxdepth 1 -type f -exec cp -a {} docs/comparison/ \;   # (gitignored)
cp <scratch>/comparisons/recovery/*.csv docs/comparison/       # only if a chapter cites recovery
quarto render docs/comparison/index.qmd
```

Two more things the 2026-09-03 tail found. The **first** `sync_report_figures` in the block above fails with "comparison_manifest.json is missing" whenever `output/comparisons/` already holds artefacts from a script run outside the tail (a smoke run of `compare_ds_td_re.py`, say) and no comparison script that writes a manifest has run since: the sync validates the comparison directory as a whole, and a directory with artefacts and no manifest is invalid. Either clear the directory first or accept that the first sync validates the fits and fails on the comparisons, and rely on the second. And `check_fit.py all --purpose publish` has no caveat allowance: a fit that cleared R-hat and ESS but carries a divergence or a BFMI below 0.3 is reported `[invalid]` there even though `sync_report_figures --allow-caveats` and `upload.py --allow-caveats` accept it. On 2026-09-03 six fits were in that state (VG11, VG12, VG13, VG21, VG22, VG23); the decision to publish them under `--allow-caveats` is recorded in `output/run-record.md`, and the checklist item below is therefore not met by them.

**A provisional fit of _any_ registered model blocks the whole sync.** `sync_report_figures` validates every directory under `output/models/` whose name resolves to a registered model, and one failure aborts the run for all of them — by design, so a half-valid figure cache never reaches the report. The trap is that **registering a new model makes its output subject to that check immediately**, long before anyone intends to publish it. A `dev` fit of VG20, made only to prove its pipeline ran, failed with `Sampling configuration mismatch: found 'dev', expected 'rep'` and took the other fifteen models' sync down with it.

So when a new model is added mid-run, either delete its provisional output before syncing, or point it at a different output root. `--allow-provisional` relaxes the check but is for local dev work, not for a publication sync. Note also that the publication sync needs `--allow-caveats` regardless: VG10, VG11, VG12 and VG13 all carry recorded soft-tier caveats, and without the flag the sync fails on those four with no mention of sampling configuration at all — which reads as a different problem than it is.

**Nothing enforces the comparison staging step, and it fails quietly in both directions.** `docs/comparison/` is gitignored, so a stale copy survives indefinitely and renders without complaint against artefacts from a previous run — on 2026-08-16 the comparison book was published against figures 20 hours older than the report beside it, and the only reason it was noticed is that a _newly cited_ file was absent, which fails loudly with a `KeyError` where an _outdated_ one does not. Treat a comparison-book render as invalid unless the copy immediately precedes it in the same shell.

Note also that `sync_report_figures._sync_dir` is flat: it copies files, not sub-directories. `comparisons/recovery/` and `comparisons/sensitivity/` are synced by an explicit loop, and anything else nested under `comparisons/` will silently not reach the report unless it is added there too.

Per-model reports render after successful fits during `fit_model.py --render`; if a render fails, retry it without resampling using `python scripts/fit_model.py <model> --config rep --render-only --output-dir <scratch>`. **Gotcha:**
rendering a model report whose output dir is **outside the git checkout** (e.g. a
scratch `--output-dir`) makes quarto exit non-zero on the `code-links: [repo]`
post-processor ("not a GitHub project") — the HTML is still produced and complete;
it just lacks the repo source-link button. It's clean when output lives under the
in-repo `output/`.

### Rendering without an activated environment

Quarto resolves the Jupyter kernel for a report's python cells from `PATH`, independently of the interpreter running the fit. Driving the scripts by absolute interpreter path (`.venv/bin/python scripts/fit_model.py …`) without also putting that `bin/` on `PATH` therefore renders against whichever `python` `PATH` finds — on macOS the system framework python, which has no `h5netcdf` and cannot open `trace.nc`. The tell-tale is a fit that samples, gates, and promotes normally, followed by `ModuleNotFoundError: No module named 'h5netcdf'` from the render (2026-08-03 `test`-config refit: fifteen clean fits, fifteen failed renders, all recovered with `--render-only`).

`fit_model.py` pins `QUARTO_PYTHON` to its own `sys.executable`, so per-model reports are immune. The two `quarto render` calls above are bare shell invocations and are not: either activate the env, or `export QUARTO_PYTHON="$PWD/.venv/bin/python"` before rendering the books. `run_replication.ps1` puts the project environment on `PATH` itself and needs neither.

On Linux the failure looks different and is worth recognising, because it does **not** stop the render: quarto reports `ModuleNotFoundError: No module named 'dse_research_utils'` for the affected chapters, prints `WARN: Error encountered when rendering files`, and still **exits 0** having produced a book with those chapters missing. Checking the exit status is not enough — confirm the chapter HTML you expected actually exists.

### `freeze: auto` does not track `{{< include >}}`

`execute.freeze` is `auto`, which re-executes a chapter when its own source changes. It does not notice a change to a file that chapter transcludes, and the frozen result stores the document with includes already resolved — so an edit to `_caveats-signing.qmd`, `_caveats-ds.qmd` or `_report_data.qmd` renders as its **pre-edit** self, with no warning and a successful exit.

Hit on 2026-08-16: a caveat rewritten to state a second bias rendered as the old single-bias paragraph, while a sibling chapter edited directly picked its change up normally — which is what makes this hard to spot, since some of the edits in a batch do appear.

After editing any `_*.qmd`, delete the frozen results for every chapter that includes it and re-render:

```bash
grep -l "_report_data" docs/report/*.qmd            # find the dependents
rm -rf docs/report/_freeze/<chapter>                # one per dependent
```

`_report_data.qmd` is included by nearly every chapter, so an edit there means clearing all of them.

## 4. Completion checklist

- [ ] `python scripts/check_fit.py all --config rep --purpose publish --output-dir <scratch>` passes; this includes complete lifecycle state, compatible provenance, reporting configuration, clean fit source state, rendered output, and `trace.nc` for every registered model.
- [ ] All registered models PASS the gate on **unrounded** diagnostics (R-hat ≤ 1.01, ESS ≥ 400, 0 divergences, BFMI ≥ 0.3).
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
