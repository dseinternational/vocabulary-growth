#!/usr/bin/env bash
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# memwatch.sh — per-process RSS sampler for long fitting runs.
#
# Why per-process. A machine-level sampler ("used_GB") records that the box hit its
# limit but not which fit did it. On 2026-08-13 vg13 was OOM-killed at 232 GB while
# sharing the machine with three sensitivity fits, and the culprit could only be
# identified afterwards from the kernel log. Naming the process lets the next model's
# memory budget come from measurement rather than from a remembered figure.
#
# Peaks matter more than plateaus here: vg13 sampled for seven hours at a steady
# ~120 GB and then took +100 GB in 90 seconds during post-sampling assembly. Sample
# often enough to catch that (the 20s default resolves it; 60s would not).
#
# Usage: bash scripts/memwatch.sh <logfile> [interval_seconds]
# Run it in the background alongside a fitting driver and kill it on exit:
#   bash scripts/memwatch.sh "$LOGS/memory.log" &
#   MEMPID=$!; trap 'kill $MEMPID 2>/dev/null' EXIT
#
# See docs/runbooks/full-refit.md, "Surviving an OOM".
set -u

out="${1:?usage: memwatch.sh <logfile> [interval_seconds]}"
interval="${2:-20}"

while true; do
  ts=$(date -u +%H:%M:%S)
  mem=$(free -g | awk '/^Mem:/ {print $3}')
  swp=$(free -g | awk '/^Swap:/ {print $3}')
  # Every fit process, largest first, as "<rss_gb>:<model>". The model id is
  # recovered from the argv token matching vgNN, which covers fit_model.py,
  # fit_sensitivity.py and refit_hightune.py alike.
  procs=$(ps -eo rss=,args= --sort=-rss \
            | awk '$0 ~ /(fit_model|fit_sensitivity|refit_hightune)\.py/ {
                     rss = $1 / 1048576
                     tag = "?"
                     for (i = 2; i <= NF; i++) if ($i ~ /^vg[0-9]+$/) tag = $i
                     printf "%.0f:%s ", rss, tag
                   }')
  echo "$ts used=${mem}G swap=${swp}G | $procs" >> "$out"
  sleep "$interval"
done
