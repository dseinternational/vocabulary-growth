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
#   * Environment-safe: puts the project's uv environment on PATH explicitly
#     (setsid shells inherit no activation, and Quarto resolves its Jupyter
#     kernel from PATH).
#   * Per-model isolation: each model fits in its own subprocess, while any
#     failure stops comparisons, rendering, and publication for the batch.
#   * Idempotent / resumable: only complete output made with the requested
#     sampling tier, model definition, data, and Git revision is skipped.
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
  --no-sync          Skip the `uv sync --locked` preflight
  --fresh            Refit even if compatible complete output exists
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
DO_SYNC=1
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
    --no-sync)         DO_SYNC=0; shift ;;
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

case "$CONFIG" in
  dev|development|test|testing) PROVISIONAL_SYNC=1 ;;
  rep|report|reporting|rep-lite|reporting-lite|rep_lite) PROVISIONAL_SYNC=0 ;;
  *) echo "Unknown sampling config: $CONFIG" >&2; usage; exit 1 ;;
esac
if [ "$PROVISIONAL_SYNC" = 1 ] && [ "$DO_UPLOAD" = 1 ]; then
  echo "Development/test replication does not publish; disabling upload." >&2
  DO_UPLOAD=0
fi

# ---------------------------------------------------------------------------
# Paths + project environment (setsid/non-login shells inherit no activation)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

# Exact replication must start from a clean checkout. Check before creating
# logs or fitted output so the preflight itself cannot dirty the repository.
GIT_STATUS="$(git status --porcelain --untracked-files=normal)" \
  || { echo "Could not inspect Git checkout state." >&2; exit 1; }
if [ -n "$GIT_STATUS" ]; then
  echo "Replication requires a clean Git checkout; commit, stash, or remove these changes:" >&2
  printf '%s\n' "$GIT_STATUS" >&2
  exit 1
fi

# Install exactly what uv.lock records. --locked fails on a stale lockfile
# rather than resolving something other than what was reviewed; an exact
# replication run must not silently move its own dependencies.
if [ "$DO_SYNC" = 1 ]; then
  uv sync --locked || { echo "uv sync --locked failed" >&2; exit 1; }
fi

# Activation by PATH rather than `source .venv/bin/activate`: the same export
# reaches the bare `quarto render` calls, which resolve their Jupyter kernel
# from PATH independently of the interpreter running the fits.
VENV="$REPO/.venv"
[ -x "$VENV/bin/python" ] || { echo "No project environment at $VENV; run 'uv sync --locked'." >&2; exit 1; }
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

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
  set -- --config "$CONFIG" --log-dir "$LOG_DIR"
  [ "$DO_SYNC" = 0 ]      && set -- "$@" --no-sync
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
RUN_FAILED=0

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
    local rc=$?; RUN_FAILED=1; log "!!! FAIL  $name rc=$rc ($((SECONDS - t0))s)"; mark "$name" "FAIL rc=$rc"; return $rc
  fi
}

# Is model <key> complete and exactly compatible with this replication run?
has_compatible_fit() {
  python scripts/check_fit.py "$1" --config "$CONFIG" --purpose resume \
    --output-dir "$OUT_ROOT" >>"$LOG" 2>&1
}

stop_if_failed() {
  if [ "$RUN_FAILED" = "1" ]; then
    log "Stopping before downstream phases because at least one required step failed."
    exit 1
  fi
}

on_exit() {
  local rc=$?
  log "===== RUN ENDED ====="
  log "Status summary:"; column -t -s $'\t' "$STATUS" 2>/dev/null | tee -a "$LOG"
  if [ "$rc" = "0" ] && [ "$RUN_FAILED" = "0" ]; then
    touch "$RUN_DIR/SUCCESS"
  else
    touch "$RUN_DIR/FAILED"
  fi
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
log "config=$CONFIG python=$(command -v python) fresh=$FRESH output=$OUT_ROOT"
log "models: $MODELS"
log "phases: descriptives=$DO_DESCRIPTIVES fit=$DO_FIT compare=$DO_COMPARE render=$DO_RENDER upload=$DO_UPLOAD kfold=$INCLUDE_KFOLD"

# 1. Data prep + descriptives
if [ "$DO_DESCRIPTIVES" = 1 ]; then
  run_step "prepare_data"       python scripts/prepare_data.py
  run_step "descriptive_report" python scripts/generate_descriptive_report.py
  stop_if_failed
fi

# 2. Fit each model independently. Rendering is a separate, retryable phase.
if [ "$DO_FIT" = 1 ]; then
  for m in $MODELS; do
    if [ "$FRESH" = 0 ] && has_compatible_fit "$m"; then
      log "=== SKIP fit_$m (complete compatible fit; use --fresh to refit) ==="; mark "fit_$m" "SKIP"; continue
    fi
    run_step "fit_$m" python scripts/fit_model.py "$m" --config "$CONFIG"
  done
  stop_if_failed
fi

# Verify all inputs before read-only comparisons or publication. This also
# protects --no-fit runs from consuming stale or development-quality traces.
if [ "$DO_COMPARE" = 1 ] || [ "$DO_RENDER" = 1 ] || [ "$DO_UPLOAD" = 1 ]; then
  # Validate the whole set in one process so Git/data provenance is computed once.
  # shellcheck disable=SC2086
  run_step "validate_models" python scripts/check_fit.py $MODELS --config "$CONFIG" \
    --purpose resume --output-dir "$OUT_ROOT"
  stop_if_failed
fi

# 3. Read-only comparisons (consume fitted traces / summaries).
if [ "$DO_COMPARE" = 1 ]; then
  CMP="loo_compare loso_compare compare_models compare_ds_td compare_ds_td_trajectories \
       compare_ds_td_expressive compare_ds_td_latency compare_ds_td_q_overlap compare_ds_td_re"
  [ "$INCLUDE_KFOLD" = 1 ] && CMP="$CMP kfold_loso"
  for c in $CMP; do
    run_step "cmp_$c" python "scripts/$c.py"
  done
  stop_if_failed
fi

# 4. Retry model-output rendering, sync figures, then render the combined reports.
if [ "$DO_RENDER" = 1 ]; then
  for m in $MODELS; do
    run_step "render_model_$m" python scripts/fit_model.py "$m" --config "$CONFIG" \
      --render-only --output-dir "$OUT_ROOT"
  done
  stop_if_failed
  SYNC_ARGS=(--config "$CONFIG" --output-dir "$OUT_ROOT")
  [ "$PROVISIONAL_SYNC" = 1 ] && SYNC_ARGS+=(--allow-provisional)
  run_step "sync_figures"      python scripts/sync_report_figures.py "${SYNC_ARGS[@]}"
  run_step "render_report"     quarto render docs/report
  run_step "render_comparison" quarto render docs/comparison/index.qmd
  stop_if_failed
fi

# 5. Upload model output (traces excluded), per-model so one failure is isolated.
if [ "$DO_UPLOAD" = 1 ]; then
  for m in $MODELS; do
    run_step "upload_$m" python scripts/upload.py "$m" --config "$CONFIG"
  done
  stop_if_failed
fi

log "===== REPLICATION RUN COMPLETE ====="
