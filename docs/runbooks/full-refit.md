# Reporting-quality refits

> [!NOTE]
> Drafted with assistance from Claude Code and OpenAI Codex. Revised by OpenAI Codex/GPT-6.

Use `scripts/run_replication.ps1` for a resumable reporting run. Its default scope covers reporting models and unclassified candidates. Use `-Scope all` when the work requires every registered model.

The fitting workstation uses native Windows, PowerShell 7, 32 cores and about 137 GB of RAM. Its output root is `D:\output\vocabulary-growth`, set through `DSE_VOCAB_GROWTH_OUTPUT_DIR`. Run archives use `F:\projects\vocabulary-growth\<commit>\output`. The driver also supports Linux and macOS.

## 0. Prerequisites

- Install the [locked environment](environment-locks.md) with `uv sync --locked`. Use `uv run` for commands below. On Windows, set `PYTHONUTF8=1`.
- Prepare the data with `uv run python scripts/prepare_data.py`, including in a fresh worktree. The database and merged CSV are generated and gitignored.
- Install PowerShell 7, Quarto and Graphviz. `dot` is optional for fitting but required for the report's model diagram. Run `quarto check` to inspect the rendering environment. PDF output also needs TinyTeX and the report fonts.
- Finish and commit changes before launching fits. Keep the checkout unchanged through the run. The driver refuses dirty source unless `-AllowDirty` is set; that override is for development and does not make the resulting fits publishable.
- Confirm free space on the output volume and archive the outgoing run. Budget for models, sensitivity variants, recovery replicates and temporary copies during promotion.
- Keep traces at `full` if recovery scoring, plot regeneration or leave-one-subject-out comparison will be needed. Check `DSE_VOCAB_GROWTH_TRACE_PERSISTENCE` as well as command-line settings.
- For uploads, configure `DSERESEARCH_BLOB_CONTAINER_URL` and the Azure credentials required by the upload path. The driver sets `AZURE_TOKEN_CREDENTIALS=dev` unless already set, so it can use a valid `az login`.

The `rep` configuration uses 6 chains, 6,000 tuning iterations, 6,000 retained draws per chain and a target acceptance of 0.95. Host-dependent parallel cores do not affect fit compatibility.

## 1. Fit

### Default (sequential, resumable)

Run from the repository root in PowerShell:

```powershell
./scripts/run_replication.ps1 -Config rep -OutputDir <output-root> -NoUpload
```

Remove `-NoUpload` only when the run includes publication. The driver prepares outputs, fits or resumes models, validates them, renders model reports, runs comparisons, syncs report figures and renders the books. Check its help for phase-selection flags.

A model is skipped only when its lifecycle is complete and its definition, prepared data, sampling effort, implementation and source revision satisfy resume validation. A trace alone is insufficient. `cores` is ignored when comparing statistical effort; an adequately sampled high-tuning fit can satisfy a `rep` request.

### Batch failure semantics

A required-step failure stops downstream comparison and publication phases. Read the run log, `status.tsv` and the final `SUCCESS` or `FAILED` marker together. A launcher can return successfully while a detached child is still running, and a terminated driver may leave only `START` entries. Confirm process state before resuming.

The direct command `fit_model.py all` continues after per-model convergence or rendering failures, reports them and suppresses the batch upload. Other fitting exceptions abort it. A render failure leaves a completed fit available for `--render-only`.

### Which models a run covers

Explicit `-Models` takes precedence over `-Scope`.

| `-Scope`      | Coverage                                                                       |
| ------------- | ------------------------------------------------------------------------------ |
| `publication` | Default: models of record, TD references and unclassified candidates; 9 today. |
| `all`         | Every registered model, including development steps; 23 today.                 |

The nine publication-scope models comprise three models of record, four TD references and the unclassified VG25 and VG26. A candidate remains subject to full validation until the study owner assigns a role. See the [model inventory](../models/README.md#model-roles).

Use `-Scope all` for comparisons that require development models fitted against the same code and data. The log records models excluded by role. The parallel recipe below covers the entire registry explicitly.

### Parallel fitting

On the workstation, run the smaller DS models in a pool and the TD pass separately. Do not overlap these two commands.

- **DS models** (`vg01 vg02 vg05 vg07 vg08 vg09 vg10 vg14 vg15 vg16 vg19 vg20 vg22 vg24 vg25`): allow up to five concurrent fits on 32 cores.

  ```powershell
  ./scripts/run_replication.ps1 -Config rep -OutputDir <output-root> -MaxParallel 5 -NoCompare -NoRender -NoUpload -Models vg01,vg02,vg05,vg07,vg08,vg09,vg10,vg14,vg15,vg16,vg19,vg20,vg22,vg24,vg25
  ```

- **TD models** (`vg03 vg04 vg11 vg12 vg13 vg21 vg23 vg26`): run one fit at a time, without another fitting batch on the machine.

  ```powershell
  ./scripts/run_replication.ps1 -Config rep -OutputDir <output-root> -MaxParallel 1 -NoCompare -NoRender -NoUpload -Models vg03,vg04,vg11,vg12,vg13,vg21,vg23,vg26
  ```

The two lists must cover the registry without overlap. `tests/test_runbook_model_lists.py` checks them. These are fit-only passes; resume the required downstream phases after both complete.

At `-MaxParallel` above one, the driver sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMBA_NUM_THREADS` to one unless already set. This prevents each chain from starting its own large numerical thread pool. Check existing environment values. Budget by concurrent fits times chains and measure actual load. These variables do not control every thread nutpie creates.

`-MinFreeGB`, default 16, delays a new launch when available memory is below that floor and another fit is running. Fits can peak during post-sampling assembly, so sampling memory alone is not a safe budget. Data loading must remain read-only; a read-write DuckDB connection can block concurrent fits.

The 2026-09-08 Windows run measured these per-process peaks with `full` traces:

| Model | Peak memory | Fit time |
| ----- | ----------: | -------: |
| VG11  |       27 GB | 3 h 59 m |
| VG12  |       12 GB |     54 m |
| VG13  |       27 GB | 1 h 42 m |
| VG21  |       28 GB | 1 h 45 m |
| VG23  |       27 GB | 1 h 36 m |

These are dated measurements, not upper bounds or proof that concurrent TD fits are safe. The much larger pre-August memory figures describe engines that stored observation-sized deterministic arrays. See the [memory investigation](../../notes/202609071440-td-fits-on-96gb-hardware.md).

### The output root

The command-line `--output-dir` overrides `DSE_VOCAB_GROWTH_OUTPUT_DIR`; otherwise the checkout's `output/` is used. Read the disk preflight to confirm the resolved path.

Keep `.staging`, `.previous` and `models` on one local filesystem. Atomic promotion uses a rename and cannot cross volumes. A manual archive copy is separate from promotion; finish and verify it before treating it as complete.

### Archiving a run's output root

Copy the whole output root and record the destination and source commit. Retain:

| Directory           | Contents                                                                          |
| ------------------- | --------------------------------------------------------------------------------- |
| `models/`           | Traces, summaries, manifests and model reports.                                   |
| `recovery/`         | Truth draws, simulation metadata and synthetic frames.                            |
| `comparisons/`      | Recovery and sensitivity matrices, population contrasts and out-of-sample checks. |
| `failed/`           | Failed or quarantined fits and evidence for excluding them.                       |
| `replication-logs/` | Driver logs, status records and per-model errors.                                 |

Copying only `models/` loses the data needed to score recovery studies. A posterior-derived truth may be reconstructed from the exact source trace and recorded draw, but this does not replace an archive and does not cover prior-derived truth.

### Surviving a full disk

Budget per fit, using recent outputs from the same model and sampling configuration. In the 2026-09-04 round, 40 fits occupied about 218 GB at `full`. Means were 6.6 GB per reporting model, 3.8 GB per sensitivity fit and 5.0 GB per reporting recovery fit. VG11 was a 24.6 GB outlier. These averages exclude many registered variants and do not estimate a complete new study.

Allow temporary space for each concurrent fit's staged output and previous version. Check free space before launching, not just the volume's capacity. Archive an old cycle before reducing trace content needed by later work.

`compact` omits scaled random effects. `minimal` also omits log likelihood and posterior predictive draws. Observation-sized deterministic arrays are omitted at every tier. The tiers leave posterior sampling unchanged, but reduced storage blocks some later analyses. Preserve `full` for every model that needs recovery scoring, `regenerate_plots.py` or `loso_compare.py`, including the headline recovery set VG20, VG12 and VG15.

To inspect possible savings on existing traces:

```bash
uv run python scripts/compact_traces.py --dry-run
```

Review the selected files and required exclusions before running without `--dry-run`. Compaction needs room for a replacement beside the original and records the new tier in the manifest. It will not rewrite a fit during promotion.

### Surviving an OOM: precautions before launching the memory-heavy models

An out-of-memory failure can occur after hours of sampling. Keep the TD pass separate from other fitting batches and monitor per-process memory with `scripts/memwatch.ps1 <logfile>`. Machine-wide used memory includes file cache and can overstate a fit's footprint.

Leave the workstation's system-managed page file enabled. On Linux, confirm adequate swap and use separate process scopes for independent jobs. A scope-level memory failure can terminate otherwise healthy co-running fits. Choose swap and memory limits for the host; the old run's allocation is not a portable recipe.

Launch long runs in a terminal or scheduled task that outlives the controlling session. A background command owned by an agent session may die with that session. On Linux, check system logs for memory failures; on Windows, inspect allocation errors in the fit log. Distinguish resource failure from convergence failure before changing sampler settings.

### Reporting age caps, and what `regenerate_plots.py` can and cannot fix

The policy is in `reporting_ages.py` and the [inventory](../models/README.md#reporting-ages-6-monthly-tables-whole-month-companions). Check generated tables and figures, including age labels embedded in column names.

`regenerate_plots.py` reruns the plot stage only and requires a compatible `full` fit. It cannot repair stale summary-stage output or bypass a changed definition. `--render-only` restages the Quarto source and includes but reads the CSVs already on disk. Set reporting caps before fitting and use `tests/test_reporting_age_policy.py` to check outputs when available.

### Do not edit tracked files while a fit is launching

The manifest records source state after data preparation, before the expensive fitting stages. A modified tracked file, including documentation, or an untracked file can mark the fit dirty. Resume and publication also check source compatibility. Keep the checkout unchanged for the whole run rather than relying on the timing of one snapshot.

### Config choice for the full-data TD models

Start at `rep` and assess the actual diagnostics. The old `rep-lite` recommendation concerned pre-hierarchy models and is superseded. High tuning is an explicit sampling override, not a named `rep-hightune` configuration:

```bash
uv run python scripts/refit_hightune.py vg11 --tune 12000 --draws 8000 --target-accept 0.99 --chains 6
```

Use a setting justified by the failed diagnostics and record it. The script writes to the selected model's normal output directory, so archive evidence needed from the earlier attempt first. It accepts `--variant` for a registered sensitivity and `--output-dir` for another root.

### Required data-handling sensitivities

The refit plan must name its required variants. The standing checks include `us01-masked-production-reinstated` and `dse-native-only` on VG10 and VG15:

```bash
uv run python scripts/fit_sensitivity.py vg10 us01-masked-production-reinstated --config rep
uv run python scripts/fit_sensitivity.py vg15 us01-masked-production-reinstated --config rep
uv run python scripts/fit_sensitivity.py vg10 dse-native-only --config rep
uv run python scripts/fit_sensitivity.py vg15 dse-native-only --config rep
```

The first reinstates production masked by both the implausibility and same-day disagreement rules. The narrower `us01-implausible-reinstated` variant leaves the latter rule active. The [Edgin audit](../../notes/202607261245-edgin-duplicated-outcome-records.md) records why both judgements matter. Check that the variant changes its prepared frame as intended; a zero change requires investigation.

`dse-native-only` probes the 810-item reference assumption by retaining complete DSE forms. Since the 2026-09-15 short-form correction, `ie_02` is excluded. The dated frame check retained 153 rows and excluded 1,633 at that preparation stage. Those counts are not a subtraction from the final baseline frame because further filters follow. Recompute them after data-rule changes. VG15 then has only `uk_02` as a cross-tabulation source, so this arm cannot resolve between-study association differences.

VG15's `tau-psi-narrow` and `tau-psi-wide` variants vary association pooling. `psi-drop-es01` and `psi-drop-uk07` remove association cells while retaining marginal data. Their results still need checking; unchanged input marginals do not guarantee unchanged posterior trajectories.

### `vg15 fallback-dispersion` and nutpie's numba backend

This variant failed compilation on Linux aarch64 in the September run but completed under numba on Windows. If that compilation failure recurs, use:

```bash
uv run python scripts/fit_sensitivity.py vg15 fallback-dispersion --config rep --nutpie-backend jax
```

The backend changes compilation, not the model's posterior distribution, and is recorded in `runtime.nutpie_backend`. Use the default unless a measured compiler problem requires the alternative.

## 2. Verify convergence

Read unrounded diagnostics. The usual thresholds are maximum R-hat 1.01, minimum effective sample size 400, no divergent transitions and energy BFMI at least 0.3. R-hat and effective sample size assess sampling agreement and information; divergences and BFMI flag other sampling problems. The pipeline distinguishes hard failures from caveats and narrowly documented exceptions. Read the fit's actual verdict rather than inferring it from a rounded banner.

A repeated run may have different diagnostics and posterior summaries because sampling is numerical and stochastic. Compare estimates relative to their uncertainty as well as checking convergence. An earlier successful run does not excuse a new failure.

### Known ridge: the understood-GP block

Trend, Gaussian process and random effects can be difficult to separate. If the failed parameters belong to that block, inspect their chains and consider more tuning. A 12,000-tune, 8,000-draw run at target acceptance 0.97 resolved VG09's September 2026 failure. That is evidence for a possible remedy, not a guarantee for another model.

VG11 also has a recorded amplitude/length-scale exception and a subsequent amplitude-prior change. Read the [exception record](../../notes/202609160500-vg11-length-scale-exception.md) and [prior decision](../../notes/202609161440-vg11-eta-sigma-0.4.md) before interpreting its gate. Do not widen an exception solely to pass a fit.

## 3. Render and compare

Prefer the driver's downstream phases. If running them separately, generate and validate comparisons before syncing them into the report cache. Do not delete a comparison manifest to bypass a failed check; regenerate the affected comparison or use a separate provisional output root.

Some model-report sections use optional post-fit artefacts:

```bash
uv run python scripts/prior_vs_posterior.py --table --model vg20 --model vg15
uv run python scripts/emit_factor_correlation.py <output-root>/models/<VG22-directory>
```

Then render each required model with `fit_model.py <model> --config rep --render-only`. This refreshes its template and shared includes without resampling.

After the required comparisons have been generated:

```bash
uv run python scripts/sync_report_figures.py --config rep --output-dir <output-root>
uv run python scripts/prepare_report_figures.py
uv run quarto render docs/report
```

Use the documented `--allow-caveats` path only for fits whose limitations have been reviewed and recorded. `--allow-provisional` is for local development, not publication. Publication validation follows catalogue roles; a failing development model can be skipped, while a failing unclassified candidate blocks sync.

Use `scripts/publish_comparison.py` to stage, render, upload and verify the comparison book. It clears stale staged inputs and checks the published assets. It is a publishing command, not a local-preview command. `--run-id` updates an existing publication.

**Do not upload traces to the public container.** Leave `--include-traces` off; traces contain observation-level data and identifiers. Use designated internal or local storage for trace archives.

**Do not upload the technical report book.** The study owner's standing decision allows public model reports and the comparison book; the technical report is rendered for local review. A successful render does not prove that all findings are complete. Inspect placeholders and caveats before publication.

### Rendering without an activated environment

`fit_model.py` pins `QUARTO_PYTHON` to its interpreter. For a direct book render, use the project environment and explicitly pin the interpreter if Quarto resolves a different kernel. In PowerShell:

```powershell
$env:QUARTO_PYTHON = uv run python -c 'import sys; print(sys.executable)'
quarto render docs/report
```

On a POSIX shell:

```bash
export QUARTO_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"
quarto render docs/report
```

Read render warnings and confirm that every expected chapter and asset exists. Earlier Quarto runs returned success despite missing chapters.

### `freeze: auto` does not track `{{< include >}}`

The technical report uses `freeze: auto`. A chapter's cache may survive changes to an included file such as `_report_data.qmd` or `_caveats-signing.qmd`. Find the dependent chapters, remove only their generated frozen results and render again. Changes to fitted inputs also require a fresh execution; a source-only cache cannot establish that the numbers are current.

## 4. Completion checklist

- [ ] Every model required by the chosen scope has a complete, compatible fit and the required sampling effort.
- [ ] Diagnostics pass, or a permitted exception or caveat is recorded and disclosed through the appropriate validation purpose.
- [ ] Required sensitivities, recovery checks and comparisons use the intended data and fit revisions.
- [ ] Model reports, report cache and comparison book are current; chapters, figures and links have been inspected.
- [ ] Publication includes only authorised outputs and all referenced assets.
- [ ] The whole output root is archived, and a dated run record gives the commit, configuration, incidents, caveats and archive location.
