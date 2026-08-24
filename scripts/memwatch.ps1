#!/usr/bin/env pwsh
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
<#
.SYNOPSIS
    Per-process RSS sampler for long fitting runs. PowerShell Core port of
    memwatch.sh, which reads `free` and `ps -eo rss=,args=` and so only ever ran
    on Linux.

.DESCRIPTION
    Why per-process. A machine-level sampler ("used_GB") records that the box hit
    its limit but not which fit did it. On 2026-08-13 vg13 was OOM-killed at
    232 GB while sharing the machine with three sensitivity fits, and the culprit
    could only be identified afterwards from the kernel log. Naming the process
    lets the next model's memory budget come from measurement rather than from a
    remembered figure.

    Peaks matter more than plateaus here: vg13 sampled for seven hours at a
    steady ~120 GB and then took +100 GB in 90 seconds during post-sampling
    assembly. Sample often enough to catch that -- the 20s default resolves it,
    60s would not.

    See docs/runbooks/full-refit.md, "Surviving an OOM".

.EXAMPLE
    # Alongside a fitting driver, stopped when the driver exits.
    $mem = Start-Process pwsh -ArgumentList '-NoProfile','-File','scripts/memwatch.ps1','memory.log' -PassThru -WindowStyle Hidden
    try { ./scripts/run_replication.ps1 -Config rep } finally { Stop-Process -Id $mem.Id -Force }
#>
[CmdletBinding()]
param(
    # File to append samples to.
    [Parameter(Mandatory)]
    [string] $LogFile,
    # Seconds between samples.
    [int]    $IntervalSeconds = 20,
    # Extra process-name patterns to sample beyond the fitting drivers.
    [string] $Pattern = '(fit_model|fit_sensitivity|refit_hightune|fit_recovery|kfold_loso)\.py'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

# Machine-level used / swap in whole GB, matching memwatch.sh's `free -g` fields.
function Get-MemorySummary {
    if ($IsWindows) {
        $os = Get-CimInstance Win32_OperatingSystem
        $usedGb = [int](($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB)
        # Windows has no swap partition; the page file is the equivalent.
        $swapGb = [int](((Get-CimInstance Win32_PageFileUsage | Measure-Object -Property CurrentUsage -Sum).Sum) / 1KB)
        return @{ Used = $usedGb; Swap = $swapGb }
    }
    if ($IsLinux -and (Test-Path '/proc/meminfo')) {
        $info = @{}
        foreach ($line in Get-Content /proc/meminfo) {
            if ($line -match '^(\w+):\s+(\d+)') { $info[$Matches[1]] = [double]$Matches[2] }
        }
        $total = $info['MemTotal']
        $avail = if ($info.ContainsKey('MemAvailable')) { $info['MemAvailable'] } else { $info['MemFree'] }
        $swapUsed = $info['SwapTotal'] - $info['SwapFree']
        return @{ Used = [int](($total - $avail) / 1MB); Swap = [int]($swapUsed / 1MB) }
    }
    # macOS and anything else: report what can be had cheaply.
    $used = (& ps -A -o rss= | Measure-Object -Sum).Sum
    return @{ Used = [int]($used / 1MB); Swap = 0 }
}

# Every matching fit process, largest first, as "<rss_gb>:<model>". The model id
# is recovered from the argv token matching vgNN, which covers fit_model.py,
# fit_sensitivity.py and refit_hightune.py alike.
function Get-FitProcesses {
    $rows = @()
    if ($IsWindows) {
        foreach ($p in Get-CimInstance Win32_Process) {
            if ($p.CommandLine -and $p.CommandLine -match $script:Pattern) {
                $tag = '?'
                foreach ($token in ($p.CommandLine -split '\s+')) {
                    if ($token -match '^vg[0-9]+$') { $tag = $token }
                }
                $rows += [pscustomobject]@{ Rss = $p.WorkingSetSize / 1GB; Tag = $tag }
            }
        }
    }
    else {
        foreach ($line in (& ps -eo rss=,args=)) {
            if ($line -match $script:Pattern) {
                $fields = $line.Trim() -split '\s+'
                $tag = '?'
                foreach ($token in $fields) { if ($token -match '^vg[0-9]+$') { $tag = $token } }
                $rows += [pscustomobject]@{ Rss = [double]$fields[0] / 1MB; Tag = $tag }
            }
        }
    }
    return $rows | Sort-Object -Property Rss -Descending
}

$dir = Split-Path -Parent $LogFile
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

while ($true) {
    $ts  = (Get-Date).ToUniversalTime().ToString('HH:mm:ss')
    $mem = Get-MemorySummary
    $procs = (Get-FitProcesses | ForEach-Object { '{0:N0}:{1}' -f $_.Rss, $_.Tag }) -join ' '
    Add-Content -Path $LogFile -Value "$ts used=$($mem.Used)G swap=$($mem.Swap)G | $procs"
    Start-Sleep -Seconds $IntervalSeconds
}
