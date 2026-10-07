#!/usr/bin/env pwsh
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
<#
.SYNOPSIS
    Sample fitting-process memory and machine memory use.

.DESCRIPTION
    Append timestamps, machine memory totals and each matching fit's resident
    memory. Short intervals help detect brief post-sampling peaks, but any
    interval can miss a peak between samples. See the full-refit runbook.

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
    # Regular expression selecting fitting-process command lines.
    [string] $Pattern = '(fit_model|fit_sensitivity|refit_hightune|fit_recovery|kfold_loso)\.py'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Continue'

# Machine memory and swap/page-file use, rounded to whole GiB.
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
    # Fallback: sum process RSS, which can double-count shared pages; swap is unknown.
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
