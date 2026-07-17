#!/usr/bin/env bash
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# run_replication.sh — one-command, recoverable full replication run.
#
# Fits every registered model at a chosen sampling config, runs the read-only
# comparisons, syncs report figures, renders the reports, and uploads model
# output to blob storage. Built for long reporting-config runs (~15-25h) on a
# remote VM, so it is resilient to SSH disconnects and mid-run interruptions.
#
# Key properties
#   * Detached: --detach re-execs under setsid so an SSH drop can't kill it.
#   * Conda-safe: activates the env explicitly (setsid shells don't do this).
#   * Per-model isolation: each model fits in its own subprocess, so one
#     failure is logged and skipped rather than aborting the whole run.
#   * Idempotent / resumable: a model whose trace.nc already exists is skipped
#     (override with --fresh). Re-running after any interruption resumes.
#   * Upload decoupled from fitting: a broken blob credential can never sink a
#     multi-hour fit. Uploads run last, using the az CLI credential (see below).
#   * Full logging: timestamped, tee'd terminal log + a per-step status TSV,
#     with a "latest" symlink to the current run's log dir.
#
# Blob credential note: DefaultAzureCredential picks the VM managed identity
# (which lacks the blob data role) before the az CLI login, giving
# AuthorizationPermissionMismatch. We export AZURE_TOKEN_CREDENTIALS=dev so it
# uses the interactive az login instead. Requires a valid `az login`.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_replication.sh [options]

  --config <c>       Sampling config: dev | test | rep | rep-lite (default: rep)
  --models "<...>"   Space-separated model keys to run (default: all registered)
  --output-dir <d>   Output root (overrides $DSE_VOCAB_GROWTH_OUTPUT_DIR)
  --log-dir <d>      Where to write run logs (default: <output>/replication-logs)
  --env <name>       Conda env name (default: dse-vocab-growth)
  --fresh            Refit even if trace.nc exists (default: resume/skip)
  --include-kfold    Also run kfold_loso.py (expensive; refits per fold)
  --no-descriptives  Skip prepare_data + descriptive report
  --no-fit           Skip the fitting phase
  --no-compare       Skip the comparison phase
  --no-render        Skip sync + report rendering
  --no-upload        Skip the upload phase
  --detach           Re-exec under setsid (survives SSH disconnect) and return
  -h | --help        Show this help
EOF
}

# ---------------------------------------------------------------------------
# Defaults / arg parsing
# ---------------------------------------------------------------------------
CONFIG="rep"
MODELS=""
OUTPUT_DIR=""
LOG_DIR=""
CONDA_ENV="dse-vocab-growth"
FRESH=0
INCLUDE_KFOLD=0
DO_DESCRIPTIVES=1
DO_FIT=1
DO_COMPARE=1
DO_RENDER=1
DO_UPLOAD=1
DETACH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --config)          CONFIG="$2"; shift 2 ;;
    --models)          MODELS="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --log-dir)         LOG_DIR="$2"; shift 2 ;;
    --env)             CONDA_ENV="$2"; shift 2 ;;
    --fresh)           FRESH=1; shift ;;
    --include-kfold)   INCLUDE_KFOLD=1; shift ;;
    --no-descriptives) DO_DESCRIPTIVES=0; shift ;;
    --no-fit)          DO_FIT=0; shift ;;
    --no-compare)      DO_COMPARE=0; shift ;;
    --no-render)       DO_RENDER=0; shift ;;
    --no-upload)       DO_UPLOAD=0; shift ;;
    --detach)          DETACH=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Paths + conda (setsid/non-login shells do NOT activate conda automatically)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "$CONDA_ENV" ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh 2>/dev/null \
    || source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null \
    || { echo "Could not source conda.sh" >&2; exit 1; }
  conda activate "$CONDA_ENV" || { echo "Failed to activate conda env '$CONDA_ENV'" >&2; exit 1; }
fi

# Output root: --output-dir wins over env, which wins over repo-local output/.
[ -n "$OUTPUT_DIR" ] && export DSE_VOCAB_GROWTH_OUTPUT_DIR="$OUTPUT_DIR"
OUT_ROOT="${DSE_VOCAB_GROWTH_OUTPUT_DIR:-$REPO/output}"
[ -z "$LOG_DIR" ] && LOG_DIR="$OUT_ROOT/replication-logs"

# Blob upload credential fix (see header note); respect a caller override.
export AZURE_TOKEN_CREDENTIALS="${AZURE_TOKEN_CREDENTIALS:-dev}"

mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +'%Y%m%dT%H%M%SZ')"

# ---------------------------------------------------------------------------
# Detach: re-exec under setsid, fully decoupled from the SSH session. Done
# before the per-run dir is created so the parent leaves no orphan run dir.
# ---------------------------------------------------------------------------
if [ "$DETACH" = "1" ] && [ -z "${_REPL_DETACHED:-}" ]; then
  export _REPL_DETACHED=1
  set -- --config "$CONFIG" --env "$CONDA_ENV" --log-dir "$LOG_DIR"
  [ -n "$MODELS" ]        && set -- "$@" --models "$MODELS"
  [ -n "$OUTPUT_DIR" ]    && set -- "$@" --output-dir "$OUTPUT_DIR"
  [ "$FRESH" = 1 ]        && set -- "$@" --fresh
  [ "$INCLUDE_KFOLD" = 1 ] && set -- "$@" --include-kfold
  [ "$DO_DESCRIPTIVES" = 0 ] && set -- "$@" --no-descriptives
  [ "$DO_FIT" = 0 ]       && set -- "$@" --no-fit
  [ "$DO_COMPARE" = 0 ]   && set -- "$@" --no-compare
  [ "$DO_RENDER" = 0 ]    && set -- "$@" --no-render
  [ "$DO_UPLOAD" = 0 ]    && set -- "$@" --no-upload
  setsid bash "$SCRIPT_DIR/run_replication.sh" "$@" </dev/null >"$LOG_DIR/detached-$RUN_TS.out" 2>&1 &
  disown
  echo "Detached replication run started (pid $!)."
  echo "  Follow: tail -f $LOG_DIR/latest/run.log"
  exit 0
fi

# ---------------------------------------------------------------------------
# Per-run log dir + status file (only the actual worker gets here)
# ---------------------------------------------------------------------------
RUN_DIR="$LOG_DIR/run-$RUN_TS"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$LOG_DIR/latest"
LOG="$RUN_DIR/run.log"
STATUS="$RUN_DIR/status.tsv"
: > "$STATUS"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
mark() { printf '%s\t%s\t%s\n' "$1" "$2" "$(ts)" >> "$STATUS"; }

run_step() {  # name  command...
  local name="$1"; shift
  local t0=$SECONDS
  log ">>> START $name :: $*"
  if "$@" >>"$LOG" 2>&1; then
    log "<<< OK    $name ($((SECONDS - t0))s)"; mark "$name" "OK"; return 0
  else
    local rc=$?; log "!!! FAIL  $name rc=$rc ($((SECONDS - t0))s)"; mark "$name" "FAIL rc=$rc"; return $rc
  fi
}

# Does model <key> already have a completed trace.nc?
has_trace() {
  # -ipath is already case-insensitive, so no need for ${1^^} (a bash-4
  # uppercase expansion that errors as "bad substitution" on macOS bash 3.2 and
  # silently broke skip-on-resume there).
  [ -n "$(find "$OUT_ROOT/models" -maxdepth 2 -iname trace.nc -ipath "*/${1}-*" 2>/dev/null | head -1)" ]
}

on_exit() {
  log "===== RUN ENDED ====="
  log "Status summary:"; column -t -s $'\t' "$STATUS" 2>/dev/null | tee -a "$LOG"
  touch "$RUN_DIR/DONE"
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# Resolve model list from the registry (single source of truth) unless given.
# ---------------------------------------------------------------------------
if [ -z "$MODELS" ]; then
  MODELS="$(python -c 'from vocab_growth.models.definitions import MODEL_REGISTRY; print(" ".join(MODEL_REGISTRY))')" \
    || { echo "Could not enumerate MODEL_REGISTRY" >&2; exit 1; }
fi

log "===== REPLICATION RUN START ====="
log "config=$CONFIG env=$CONDA_ENV fresh=$FRESH output=$OUT_ROOT"
log "models: $MODELS"
log "phases: descriptives=$DO_DESCRIPTIVES fit=$DO_FIT compare=$DO_COMPARE render=$DO_RENDER upload=$DO_UPLOAD kfold=$INCLUDE_KFOLD"

# 1. Data prep + descriptives
if [ "$DO_DESCRIPTIVES" = 1 ]; then
  run_step "prepare_data"       python scripts/prepare_data.py
  run_step "descriptive_report" python scripts/generate_descriptive_report.py
fi

# 2. Fit each model independently, render each. Upload is deferred to phase 5.
if [ "$DO_FIT" = 1 ]; then
  for m in $MODELS; do
    if [ "$FRESH" = 0 ] && has_trace "$m"; then
      log "=== SKIP fit_$m (trace.nc present; use --fresh to refit) ==="; mark "fit_$m" "SKIP"; continue
    fi
    run_step "fit_$m" python scripts/fit_model.py "$m" --config "$CONFIG" --render
  done
fi

# 3. Read-only comparisons (consume fitted traces / summaries).
if [ "$DO_COMPARE" = 1 ]; then
  CMP="loo_compare loso_compare compare_models compare_ds_td compare_ds_td_trajectories \
       compare_ds_td_expressive compare_ds_td_latency compare_ds_td_q_overlap compare_ds_td_re"
  [ "$INCLUDE_KFOLD" = 1 ] && CMP="$CMP kfold_loso"
  for c in $CMP; do
    run_step "cmp_$c" python "scripts/$c.py"
  done
fi

# 4. Sync figures into the report cache, then render.
if [ "$DO_RENDER" = 1 ]; then
  run_step "sync_figures"      python scripts/sync_report_figures.py
  run_step "render_report"     quarto render docs/report
  run_step "render_comparison" quarto render docs/comparison/index.qmd
fi

# 5. Upload model output (traces excluded), per-model so one failure is isolated.
if [ "$DO_UPLOAD" = 1 ]; then
  for m in $MODELS; do
    run_step "upload_$m" python scripts/upload.py "$m"
  done
fi

log "===== REPLICATION RUN COMPLETE ====="
