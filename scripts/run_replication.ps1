#!/usr/bin/env pwsh
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
<#
.SYNOPSIS
    One-command, recoverable full replication run. PowerShell Core port of
    run_replication.sh, which only ever ran on POSIX: it probes .venv/bin/python
    and re-execs under setsid, neither of which exists on Windows.

.DESCRIPTION
    Fits every registered model at a chosen sampling config, runs the read-only
    comparisons, syncs report figures, renders the reports, and uploads model
    output to blob storage. Built for long reporting-config runs, so it is
    resilient to a dropped session and to mid-run interruption.

    Key properties
      * Cross-platform: pwsh 7+ on Windows, Linux and macOS. The venv layout
        (Scripts vs bin) is resolved from $IsWindows rather than assumed.
      * Detached: -Detach re-launches a disowned child so a closed terminal or a
        dropped SSH session cannot kill the run.
      * Per-model isolation: each model fits in its own process, and -MaxParallel
        controls how many fit at once. Any failure still stops comparisons,
        rendering and publication for the batch.
      * Memory-aware: a new fit is not launched while available memory is below
        -MinFreeGB and something is already running. Reporting-config traces run
        to double-digit gigabytes and concurrent fits have OOM-killed each other
        on this project before.
      * Idempotent / resumable: only complete output made with the requested
        sampling tier, model definition, data and Git revision is skipped.
      * Upload decoupled from fitting: a broken blob credential can never sink a
        multi-hour fit. Uploads run last, using the az CLI credential.
      * Full logging: a run log, per-model stdout/stderr files, and a status TSV,
        with a "latest" pointer to the current run's log directory.

    Blob credential note: DefaultAzureCredential picks a VM managed identity
    (which lacks the blob data role) before the az CLI login, giving
    AuthorizationPermissionMismatch. AZURE_TOKEN_CREDENTIALS=dev makes it use the
    interactive az login instead. Requires a valid `az login`.

.EXAMPLE
    ./scripts/run_replication.ps1 -Config rep

.EXAMPLE
    # Fit the Down syndrome models three at a time, rendering each model report
    # as its own fit completes, and stop before the publication phases.
    ./scripts/run_replication.ps1 -Config rep -MaxParallel 3 -RenderOnFit -NoDescriptives -NoCompare -NoRender -NoUpload -Models vg01,vg02,vg05,vg20,vg10,vg07,vg08,vg09,vg15,vg14,vg16,vg19,vg22
#>
[CmdletBinding()]
param(
    # Sampling config: dev | test | rep | rep-lite.
    [string]   $Config = 'rep',
    # Model keys to run. Defaults to every registered model.
    [string[]] $Models,
    # Output root; overrides $env:DSE_VOCAB_GROWTH_OUTPUT_DIR.
    [string]   $OutputDir,
    # Where to write run logs. Default: <output>/replication-logs.
    [string]   $LogDir,
    # Repository checkout to run against. Defaults to this script's parent
    # directory, which is correct when it is run from scripts/ in the checkout.
    # Override to drive a checkout from a copy of this script held elsewhere.
    [string]   $RepoRoot,
    # How many models to fit concurrently.
    [int]      $MaxParallel = 1,
    # Hold a new fit back while available memory is below this and one is running.
    [double]   $MinFreeGB = 16,
    # Trace persistence tier passed to fit_model.py: full | compact | minimal.
    [ValidateSet('full', 'compact', 'minimal')]
    [string]   $TracePersistence,
    # Render each model's report as part of its own fit, rather than in the
    # render phase. A render failure then leaves the fit complete and retryable.
    [switch]   $RenderOnFit,
    # Skip the `uv sync --locked` preflight.
    [switch]   $NoSync,
    # Refit even where compatible complete output exists.
    [switch]   $Fresh,
    # Also run kfold_loso.py (expensive; refits per fold).
    [switch]   $IncludeKfold,
    [switch]   $NoDescriptives,
    [switch]   $NoFit,
    [switch]   $NoCompare,
    [switch]   $NoRender,
    [switch]   $NoUpload,
    # Proceed from a dirty checkout. Fits made this way record dirty provenance
    # and are refused by check_fit.py --purpose publish, so this is for local
    # development runs only.
    [switch]   $AllowDirty,
    # Re-launch disowned and return immediately.
    [switch]   $Detach
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
$ProvisionalSync = $false
switch -Regex ($Config) {
    '^(dev|development|test|testing)$'                          { $ProvisionalSync = $true }
    '^(rep|report|reporting|rep-lite|reporting-lite|rep_lite)$' { $ProvisionalSync = $false }
    default { Write-Error "Unknown sampling config: $Config"; exit 1 }
}
if ($ProvisionalSync -and -not $NoUpload) {
    Write-Warning 'Development/test replication does not publish; disabling upload.'
    $NoUpload = [switch]$true
}

# ---------------------------------------------------------------------------
# Paths + project environment (a disowned child inherits no activation)
# ---------------------------------------------------------------------------
$Repo = if ($RepoRoot) { (Resolve-Path $RepoRoot).Path } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
if (-not (Test-Path (Join-Path $Repo 'pyproject.toml'))) {
    Write-Error "No pyproject.toml at $Repo; pass -RepoRoot to point at the checkout."
    exit 1
}
Set-Location $Repo

# Exact replication must start from a clean checkout. Check before creating logs
# or fitted output so the preflight itself cannot dirty the repository.
$gitStatus = (& git status --porcelain --untracked-files=normal) -join [Environment]::NewLine
if ($LASTEXITCODE -ne 0) { Write-Error 'Could not inspect Git checkout state.'; exit 1 }
if ($gitStatus.Trim()) {
    if ($AllowDirty) {
        Write-Warning "Proceeding from a dirty checkout (-AllowDirty). Fits will record dirty provenance and cannot be published:"
        Write-Warning $gitStatus
    }
    else {
        Write-Error "Replication requires a clean Git checkout; commit, stash, or remove these changes:"
        Write-Error $gitStatus
        exit 1
    }
}

# Install exactly what uv.lock records. --locked fails on a stale lockfile rather
# than resolving something other than what was reviewed; an exact replication run
# must not silently move its own dependencies.
if (-not $NoSync) {
    & uv sync --locked
    if ($LASTEXITCODE -ne 0) { Write-Error 'uv sync --locked failed'; exit 1 }
}

# Activation by PATH rather than the activate script: the same PATH reaches the
# bare `quarto render` calls, which resolve their Jupyter kernel from PATH
# independently of the interpreter running the fits.
$Venv    = Join-Path $Repo '.venv'
$VenvBin = if ($IsWindows) { Join-Path $Venv 'Scripts' } else { Join-Path $Venv 'bin' }
$VenvPy  = if ($IsWindows) { Join-Path $VenvBin 'python.exe' } else { Join-Path $VenvBin 'python' }
if (-not (Test-Path $VenvPy)) {
    Write-Error "No project environment at $Venv; run 'uv sync --locked'."
    exit 1
}
$env:VIRTUAL_ENV = $Venv
$env:PATH        = $VenvBin + [IO.Path]::PathSeparator + $env:PATH

# The console progress output uses glyphs cp1252 cannot encode; UTF-8 mode
# renders them rather than degrading them to '?'.
$env:PYTHONUTF8 = '1'

# Pin each chain to one thread when fitting a pool. Every fit already runs its
# own chains in parallel, so a pool of them oversubscribes the box through the
# BLAS/OpenMP thread pools unless these are held at 1 -- the pitfall recorded
# under "Parallel fitting" in docs/runbooks/full-refit.md. A caller who has set
# them deliberately keeps their value.
if ($MaxParallel -gt 1) {
    foreach ($threadVar in 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS') {
        if (-not (Get-Item "env:$threadVar" -ErrorAction SilentlyContinue)) {
            Set-Item -Path "env:$threadVar" -Value '1'
        }
    }
}

# Output root: -OutputDir wins over env, which wins over repo-local output/.
if ($OutputDir) { $env:DSE_VOCAB_GROWTH_OUTPUT_DIR = $OutputDir }
$OutRoot = if ($env:DSE_VOCAB_GROWTH_OUTPUT_DIR) { $env:DSE_VOCAB_GROWTH_OUTPUT_DIR } else { Join-Path $Repo 'output' }
if (-not $LogDir) { $LogDir = Join-Path $OutRoot 'replication-logs' }

# Blob upload credential fix (see .DESCRIPTION); respect a caller override.
if (-not $env:AZURE_TOKEN_CREDENTIALS) { $env:AZURE_TOKEN_CREDENTIALS = 'dev' }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunTs = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')

# ---------------------------------------------------------------------------
# Detach: re-launch disowned, decoupled from this session. Done before the
# per-run dir is created so the parent leaves no orphan run dir.
# ---------------------------------------------------------------------------
if ($Detach -and -not $env:_REPL_DETACHED) {
    $childArgs = @('-NoProfile', '-File', $PSCommandPath, '-Config', $Config, '-LogDir', $LogDir,
                   '-MaxParallel', $MaxParallel, '-MinFreeGB', $MinFreeGB)
    if ($Models)           { $childArgs += @('-Models', ($Models -join ',')) }
    if ($OutputDir)        { $childArgs += @('-OutputDir', $OutputDir) }
    if ($TracePersistence) { $childArgs += @('-TracePersistence', $TracePersistence) }
    foreach ($s in 'RenderOnFit', 'NoSync', 'Fresh', 'IncludeKfold', 'NoDescriptives',
                   'NoFit', 'NoCompare', 'NoRender', 'NoUpload', 'AllowDirty') {
        if ((Get-Variable $s -ValueOnly)) { $childArgs += "-$s" }
    }
    $env:_REPL_DETACHED = '1'
    $outFile = Join-Path $LogDir "detached-$RunTs.out"
    $child = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList $childArgs -RedirectStandardOutput $outFile -RedirectStandardError "$outFile.err" -PassThru -WindowStyle Hidden
    Write-Host "Detached replication run started (pid $($child.Id))."
    Write-Host "  Follow: Get-Content -Wait '$LogDir/latest/run.log'"
    exit 0
}

# ---------------------------------------------------------------------------
# Per-run log dir + status file (only the actual worker gets here)
# ---------------------------------------------------------------------------
$RunDir = Join-Path $LogDir "run-$RunTs"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Latest = Join-Path $LogDir 'latest'
# A directory symlink needs Developer Mode or elevation on Windows; a junction
# does not, and a plain text pointer works where neither is available.
try {
    if (Test-Path $Latest) { Remove-Item $Latest -Force -Recurse -ErrorAction Stop }
    $linkType = if ($IsWindows) { 'Junction' } else { 'SymbolicLink' }
    New-Item -ItemType $linkType -Path $Latest -Target $RunDir -ErrorAction Stop | Out-Null
}
catch {
    Set-Content -Path (Join-Path $LogDir 'latest.txt') -Value $RunDir
}

$Log       = Join-Path $RunDir 'run.log'
$StatusTsv = Join-Path $RunDir 'status.tsv'
New-Item -ItemType File -Force -Path $StatusTsv | Out-Null
$script:RunFailed = $false

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
function Get-Ts { (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }

function Write-Log([string] $Message) {
    $line = "[$(Get-Ts)] $Message"
    Write-Host $line
    Add-Content -Path $script:Log -Value $line
}

function Add-Mark([string] $Name, [string] $State, [string] $Detail = '') {
    $tab = "`t"
    Add-Content -Path $script:StatusTsv -Value ($Name + $tab + $State + $tab + (Get-Ts) + $tab + $Detail)
}

# Run one synchronous step, appending its output to the run log.
function Invoke-Step([string] $Name, [string] $Exe, [string[]] $StepArgs) {
    $t0 = Get-Date
    Write-Log ">>> START $Name :: $Exe $($StepArgs -join ' ')"
    & $Exe @StepArgs *>> $script:Log
    $rc = $LASTEXITCODE
    $secs = [int]((Get-Date) - $t0).TotalSeconds
    if ($rc -eq 0) {
        Write-Log "<<< OK    $Name (${secs}s)"
        Add-Mark $Name 'OK' "${secs}s"
        return $true
    }
    $script:RunFailed = $true
    Write-Log "!!! FAIL  $Name rc=$rc (${secs}s)"
    Add-Mark $Name "FAIL rc=$rc" "${secs}s"
    return $false
}

function Write-Summary {
    Write-Log '===== RUN ENDED ====='
    Write-Log 'Status summary:'
    if (Test-Path $script:StatusTsv) {
        foreach ($row in Get-Content $script:StatusTsv) {
            $f = $row -split "`t"
            if ($f.Count -ge 4) { Write-Log ('  {0,-22} {1,-14} {2}' -f $f[0], $f[1], $f[3]) }
        }
    }
    $marker = if ($script:RunFailed) { 'FAILED' } else { 'SUCCESS' }
    New-Item -ItemType File -Force -Path (Join-Path $script:RunDir $marker) | Out-Null
}

function Stop-IfFailed {
    if ($script:RunFailed) {
        Write-Log 'Stopping before downstream phases because at least one required step failed.'
        Write-Summary
        exit 1
    }
}

# Available physical memory in GB, or $null where it cannot be determined, in
# which case the memory gate is simply not applied.
function Get-AvailableMemoryGB {
    try {
        if ($IsWindows) {
            return [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
        }
        if ($IsLinux -and (Test-Path '/proc/meminfo')) {
            $line = Get-Content /proc/meminfo | Where-Object { $_ -match '^MemAvailable:' } | Select-Object -First 1
            if ($line) { return [math]::Round((($line -split '\s+')[1] / 1MB), 1) }
        }
        if ($IsMacOS) {
            $stat  = & vm_stat
            $free  = (($stat | Where-Object { $_ -match '^Pages free:' })     -replace '\D', '')
            $inact = (($stat | Where-Object { $_ -match '^Pages inactive:' }) -replace '\D', '')
            if ($free) { return [math]::Round(((([double]$free + [double]$inact) * 4096) / 1GB), 1) }
        }
    }
    catch { }
    return $null
}

# Is model <key> complete and exactly compatible with this replication run?
function Test-CompatibleFit([string] $Model) {
    & uv run python scripts/check_fit.py $Model --config $script:Config --purpose resume --output-dir $script:OutRoot *>> $script:Log
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------------------
# Resolve model list from the registry (single source of truth) unless given.
# ---------------------------------------------------------------------------
if (-not $Models -or $Models.Count -eq 0) {
    $registry = & uv run python -c 'from vocab_growth.models.definitions import MODEL_REGISTRY; print(" ".join(MODEL_REGISTRY))'
    if ($LASTEXITCODE -ne 0 -or -not $registry) { Write-Error 'Could not enumerate MODEL_REGISTRY'; exit 1 }
    $Models = $registry.Trim() -split '\s+'
}
# Accept a single comma-joined string as well as a real array: the -Detach
# re-launch passes one argument, and so do most shells.
$Models = @($Models | ForEach-Object { $_ -split ',' } | Where-Object { $_ } | ForEach-Object { $_.Trim().ToLower() })

Write-Log '===== REPLICATION RUN START ====='
Write-Log "config=$Config pwsh=$($PSVersionTable.PSVersion) fresh=$($Fresh.IsPresent) output=$OutRoot"
Write-Log "models: $($Models -join ' ')"
Write-Log "parallel=$MaxParallel min-free=${MinFreeGB}GB render-on-fit=$($RenderOnFit.IsPresent)"
Write-Log ("phases: descriptives={0} fit={1} compare={2} render={3} upload={4} kfold={5}" -f (-not $NoDescriptives), (-not $NoFit), (-not $NoCompare), (-not $NoRender), (-not $NoUpload), $IncludeKfold.IsPresent)

# 1. Data prep + descriptives
if (-not $NoDescriptives) {
    Invoke-Step 'prepare_data'       'uv' @('run', 'python', 'scripts/prepare_data.py')                | Out-Null
    Invoke-Step 'descriptive_report' 'uv' @('run', 'python', 'scripts/prepare_report_figures.py', 'descriptives') | Out-Null
    Stop-IfFailed
}

# 2. Fit each model in its own process, up to -MaxParallel at once. Rendering is
#    a separate retryable phase unless -RenderOnFit folds it into the fit.
if (-not $NoFit) {
    $queue = [System.Collections.Generic.Queue[string]]::new()
    foreach ($m in $Models) { $queue.Enqueue($m) }
    $running = [System.Collections.ArrayList]::new()

    while ($queue.Count -gt 0 -or $running.Count -gt 0) {

        # Reap anything that has finished.
        foreach ($job in @($running | Where-Object { $_.Process.HasExited })) {
            $rc   = $job.Process.ExitCode
            $secs = [int]((Get-Date) - $job.Started).TotalSeconds
            if ($rc -eq 0) {
                Write-Log "<<< OK    fit_$($job.Model) (${secs}s)"
                Add-Mark "fit_$($job.Model)" 'OK' "${secs}s"
            }
            else {
                $script:RunFailed = $true
                Write-Log "!!! FAIL  fit_$($job.Model) rc=$rc (${secs}s) -- see $($job.OutFile)"
                Add-Mark "fit_$($job.Model)" "FAIL rc=$rc" "${secs}s"
            }
            $running.Remove($job)
        }

        # Launch while there is a free slot, work left, and memory to spare.
        while ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
            $free = Get-AvailableMemoryGB
            if ($running.Count -gt 0 -and $null -ne $free -and $free -lt $MinFreeGB) {
                Write-Log "Holding: ${free}GB free is below the ${MinFreeGB}GB floor with $($running.Count) fit(s) running."
                break
            }

            $model = $queue.Dequeue()
            if (-not $Fresh -and (Test-CompatibleFit $model)) {
                Write-Log "=== SKIP fit_$model (complete compatible fit; use -Fresh to refit) ==="
                Add-Mark "fit_$model" 'SKIP'
                continue
            }

            $fitArgs = @('run', 'python', 'scripts/fit_model.py', $model, '--config', $Config, '--output-dir', $OutRoot)
            if ($RenderOnFit)      { $fitArgs += '--render' }
            if ($TracePersistence) { $fitArgs += @('--trace-persistence', $TracePersistence) }

            $outFile = Join-Path $RunDir "fit_$model.out"
            $errFile = Join-Path $RunDir "fit_$model.err"
            Write-Log ">>> START fit_$model :: uv $($fitArgs -join ' ')"
            $proc = Start-Process -FilePath 'uv' -ArgumentList $fitArgs -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
            [void]$running.Add([pscustomobject]@{
                    Model   = $model
                    Process = $proc
                    Started = (Get-Date)
                    OutFile = $outFile
                })
        }

        if ($queue.Count -gt 0 -or $running.Count -gt 0) { Start-Sleep -Seconds 15 }
    }
    Stop-IfFailed
}

# Verify all inputs before read-only comparisons or publication. This also
# protects -NoFit runs from consuming stale or development-quality traces.
if (-not ($NoCompare -and $NoRender -and $NoUpload)) {
    Invoke-Step 'validate_models' 'uv' (@('run', 'python', 'scripts/check_fit.py') + $Models + @('--config', $Config, '--purpose', 'resume', '--output-dir', $OutRoot)) | Out-Null
    Stop-IfFailed
}

# 3. Read-only comparisons (consume fitted traces / summaries).
if (-not $NoCompare) {
    $cmp = @('loo_compare', 'loso_compare', 'compare_models', 'compare_ds_td',
             'compare_ds_td_trajectories', 'compare_ds_td_expressive',
             'compare_ds_td_latency', 'compare_ds_td_q_overlap', 'compare_ds_td_re')
    if ($IncludeKfold) { $cmp += 'kfold_loso' }
    foreach ($c in $cmp) {
        Invoke-Step "cmp_$c" 'uv' @('run', 'python', "scripts/$c.py") | Out-Null
    }
    Stop-IfFailed
}

# 4. Retry model-output rendering, sync figures, then render the combined reports.
if (-not $NoRender) {
    foreach ($m in $Models) {
        Invoke-Step "render_model_$m" 'uv' @('run', 'python', 'scripts/fit_model.py', $m, '--config', $Config, '--render-only', '--output-dir', $OutRoot) | Out-Null
    }
    Stop-IfFailed
    $syncArgs = @('run', 'python', 'scripts/sync_report_figures.py', '--config', $Config, '--output-dir', $OutRoot)
    if ($ProvisionalSync) { $syncArgs += '--allow-provisional' }
    Invoke-Step 'sync_figures'      'uv'     $syncArgs                                | Out-Null
    # Everything the report needs that is not model output (introduction
    # illustrations, the methods chapter's prior figures, placeholders for any
    # figure still absent): the sync neither validates nor regenerates these.
    Invoke-Step 'prepare_figures'   'uv'     @('run', 'python', 'scripts/prepare_report_figures.py', 'illustrations', 'priors', 'pending') | Out-Null
    Invoke-Step 'render_report'     'quarto' @('render', 'docs/report')               | Out-Null
    Invoke-Step 'render_comparison' 'quarto' @('render', 'docs/comparison/index.qmd') | Out-Null
    Stop-IfFailed
}

# 5. Upload model output (traces excluded), per-model so one failure is isolated.
if (-not $NoUpload) {
    foreach ($m in $Models) {
        Invoke-Step "upload_$m" 'uv' @('run', 'python', 'scripts/upload.py', $m, '--config', $Config) | Out-Null
    }
    Stop-IfFailed
}

Write-Log '===== REPLICATION RUN COMPLETE ====='
Write-Summary
exit $(if ($script:RunFailed) { 1 } else { 0 })
