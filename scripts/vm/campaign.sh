#!/usr/bin/env bash
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Run a refit campaign on a dedicated Linux VM, resumably, from a checkout pinned
# at a fits/YYYY-MM-DD tag. Read docs/runbooks/vm-refit.md first.
#
#   scripts/vm/campaign.sh plan [STAGE...]     list every step and whether it is done
#   scripts/vm/campaign.sh run STAGE...        run stages in order, in the foreground
#   scripts/vm/campaign.sh launch STAGE...     the same, detached into its own systemd scope
#   scripts/vm/campaign.sh status              the campaign log's tail and the failed steps
#
# Stages, in the order they assume one another:
#   A  refit the nine publication-scope models
#   B  refit the development steps
#   C  priority sensitivity arms, wave-forward scores and parameter recovery
#   D  the long tail: remaining arms, recovery for never-recovered models,
#      the Gompertz comparison (#330) and the TD held-out k-fold (#240)
#   P  comparisons, figure sync (--allow-caveats) and the report-book render.
#      Uploads and publishes nothing; publication is the study owner's call.
#
# Successful steps are skipped on rerun: success leaves <step>.ok under the state
# directory and a rerun skips it, so `run` after an interruption resumes. A
# failed step leaves <step>.failed and its log; delete nothing to retry, just
# rerun. Steps are keyed to the tag, so a new tag starts a new campaign.

set -uo pipefail

die() { echo "campaign: $*" >&2; exit 1; }

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_FILE=${VG_VM_ENV:-$HOME/.config/vocabulary-growth/vm.env}
[[ -f $ENV_FILE ]] || die "no $ENV_FILE; run scripts/vm/bootstrap.sh first"
# shellcheck source=/dev/null
source "$ENV_FILE"
: "${DSE_VOCAB_GROWTH_OUTPUT_DIR:?vm.env must set DSE_VOCAB_GROWTH_OUTPUT_DIR}"
export DSE_VOCAB_GROWTH_OUTPUT_DIR PYTHONUTF8=1 PYTHONUNBUFFERED=1
# A trace tier set in the environment would silently override `full`, and
# recovery scoring, plot regeneration and LOSO all refuse a reduced trace.
unset DSE_VOCAB_GROWTH_TRACE_PERSISTENCE

cd "$REPO" || die "cannot enter $REPO"
TAG=$(git describe --tags --exact-match --match 'fits/*' HEAD 2>/dev/null) \
    || die "HEAD is not at a fits/* tag; a campaign runs only from a pinned checkout"
OUT=$DSE_VOCAB_GROWTH_OUTPUT_DIR
STATE="$OUT/campaign/${TAG//\//-}"
POOL=${VG_POOL:-5}               # concurrent DS fits: chains (6) x POOL <= physical cores
MIN_FREE_GB=${VG_MIN_FREE_GB:-24}  # hold a pooled launch below this much available memory

PY=(uv run --locked python)
DRIVER=(pwsh -NoProfile -File scripts/run_replication.ps1 -Config rep -OutputDir "$OUT"
        -ReplaceModelOfRecord -RenderOnFit -NoCompare -NoRender -NoUpload)

log() { mkdir -p "$STATE"; printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$STATE/campaign.log"; }

require_clean() {
    [[ -z $(git status --porcelain --untracked-files=normal) ]] \
        || die "the checkout is dirty; fits would record dirty provenance and could not be published"
}

# step NAME COMMAND...: run once; success leaves NAME.ok and is skipped on rerun.
step() {
    local name=$1; shift
    if [[ -f $STATE/$name.ok ]]; then log "SKIP  $name"; return 0; fi
    if [[ ${PLAN_ONLY:-0} == 1 ]]; then echo "todo  $name :: $*"; return 0; fi
    require_clean
    log "START $name :: $*"
    local start=$SECONDS rc=0
    "$@" >"$STATE/logs/$name.log" 2>&1 || rc=$?
    if (( rc == 0 )); then
        touch "$STATE/$name.ok"; rm -f "$STATE/$name.failed"
        log "OK    $name ($(( SECONDS - start ))s)"
    else
        echo "$rc" >"$STATE/$name.failed"
        log "FAIL  $name rc=$rc ($(( SECONDS - start ))s) -- $STATE/logs/$name.log"
    fi
    return "$rc"
}

free_gb() { awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo; }

# pool WIDTH SPEC...: run steps WIDTH at a time. A SPEC is "name::command words";
# no argument may contain a space. Pooled fits pin the BLAS/OpenMP pools to one
# thread each, as run_replication.ps1 does, because every fit already runs its
# chains in parallel.
pool() {
    local width=$1 spec missing=0; shift
    for spec in "$@"; do
        if [[ ${PLAN_ONLY:-0} == 1 ]]; then step "${spec%%::*}" ${spec#*::}; continue; fi
        while (( $(jobs -rp | wc -l) >= width )) \
              || { (( $(jobs -rp | wc -l) > 0 )) && (( $(free_gb) < MIN_FREE_GB )); }; do
            sleep 30
        done
        (
            export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
            step "${spec%%::*}" ${spec#*::}
        ) &
        sleep 10  # stagger compilation and the read-only DuckDB opens
    done
    wait
    for spec in "$@"; do [[ -f $STATE/${spec%%::*}.ok ]] || missing=1; done
    return "$missing"
}

# serial SPEC...: run steps one at a time, continuing past a failure.
serial() {
    local spec missing=0
    for spec in "$@"; do step "${spec%%::*}" ${spec#*::} || missing=1; done
    return "$missing"
}

# arms TAG WIDTH "model variant"...: one sensitivity fit per arm at rep.
arms() {
    local tag=$1 width=$2 entry; shift 2
    local specs=()
    for entry in "$@"; do
        read -r model variant <<<"$entry"
        specs+=("$tag-arm-$model-$variant::${PY[*]} scripts/fit_sensitivity.py $model $variant --config rep")
    done
    if (( width > 1 )); then pool "$width" "${specs[@]}"; else serial "${specs[@]}"; fi
}

# ---------------------------------------------------------------------------
# The arm lists. tests/test_vm_campaign.py checks every "model variant" pair
# below against the sensitivity registry, so a renamed arm fails CI rather than
# a step two days into a campaign.
# ---------------------------------------------------------------------------

# Stage C, Down syndrome: the runbook's data-handling arms, every arm of the
# three models whose questions are open on #289 and #297, and VG14's fallback
# arms (#289 task 1.2, #236). VG16 3.7, VG20 3.6, VG25 #297 check 5.
C_DS_ARMS=(
    "vg10 us01-masked-production-reinstated" "vg10 dse-native-only"
    "vg10 ie02-comprehension-masked" "vg10 paired-only" "vg10 fallback-dispersion"
    "vg10 marginal-moments"
    "vg15 us01-masked-production-reinstated" "vg15 dse-native-only"
    "vg15 ie02-comprehension-masked" "vg15 paired-only" "vg15 fallback-dispersion"
    "vg15 marginal-moments"
    "vg14 paired-only" "vg14 fallback-dispersion" "vg14 marginal-moments"
    "vg16 conditional-only" "vg16 dse-native-only" "vg16 lag-gap-12" "vg16 no-us01"
    "vg16 lag-continuity" "vg16 lag-same-form" "vg16 beta-tight" "vg16 beta-wide"
    "vg16 corr"
    "vg20 ie02-comprehension-masked" "vg20 paired-only" "vg20 fallback-dispersion"
    "vg20 marginal-moments" "vg20 kappa-anchor-24-48" "vg20 kappa-floor-generic"
    "vg20 kappa-pre-promotion"
    "vg25 sign-lag-clip" "vg25 sign-lag-in-cells" "vg25 sign-lag-population"
    "vg25 sign-lag-gap-12" "vg25 sign-lag-same-form" "vg25 sign-lag-uk07-marginal"
    "vg25 no-ie02" "vg25 no-uk05" "vg25 beta-sign-tight" "vg25 beta-sign-wide"
)

# Stage C, typically developing: the arm #289 task 4.8 recovers.
C_TD_ARMS=("vg12 free-scales")

# Stage D, Down syndrome: every remaining registered arm.
D_DS_ARMS=(
    "vg10 q-broad" "vg10 q-wider" "vg10 kappa-broadfloor" "vg10 kappa-flat"
    "vg10 kappa-const" "vg10 tau-wide" "vg10 tau-narrow" "vg10 no-subject"
    "vg10 us01-implausible-reinstated" "vg10 u-anchor-broad" "vg10 eta-u-narrow"
    "vg10 clamp-both" "vg10 a1-tau-age-varying"
    "vg15 q-broad" "vg15 etasign-wide" "vg15 etasign-narrow" "vg15 ellsign-beta33"
    "vg15 ellsign-short" "vg15 sign-peak-lo" "vg15 sign-peak-hi"
    "vg15 sign-peak-age-uniform" "vg15 sign-peak-age-early" "vg15 sign-peak-age-late"
    "vg15 sign-old-hi" "vg15 sign-include-uk01" "vg15 kappa-broadfloor" "vg15 tau-wide"
    "vg15 sign-study-only" "vg15 us01-implausible-reinstated" "vg15 psi-neutral"
    "vg15 psi-broad" "vg15 psi-strong" "vg15 tau-psi-narrow" "vg15 tau-psi-wide"
    "vg15 psi-drop-es01" "vg15 psi-drop-uk07" "vg15 conc-broad" "vg15 conc-lo"
    "vg15 conc-hi"
    "vg19 max-age-84" "vg22 rank-1" "vg22 rank-2"
)

# Stage D, typically developing: every registered arm (#240), strictly serial.
D_TD_ARMS=(
    "vg11 tau-wide" "vg11 tau-narrow" "vg11 single-admin" "vg11 anchor-broad"
    "vg11 eta-wide" "vg11 no-study-threshold" "vg11 study-age-slopes"
    "vg11 a1-tau-age-varying"
    "vg12 single-admin" "vg12 lo-anchor-broad" "vg12 hi-anchor-broad" "vg12 eta-narrow"
    "vg12 no-study-threshold" "vg12 study-age-slopes" "vg12 a1-tau-age-varying"
    "vg13 single-admin" "vg13 window-25" "vg13 window-22" "vg13 window-22-vague-anchors"
    "vg21 vague-anchors" "vg21 no-study-threshold" "vg21 study-age-slopes"
    "vg21 a1-tau-age-varying" "vg21 eta-q-wide" "vg21 single-admin"
    "vg23 eta-flat" "vg23 no-study-threshold" "vg23 study-age-slopes" "vg23 eta-q-wide"
    "vg26 no-study-threshold" "vg26 study-age-slopes" "vg26 eta-q-wide"
    "vg26 single-admin" "vg26 vague-anchors"
)

REC=("${PY[*]} scripts/fit_recovery.py")

stage_A() {
    # The four DS publication models four-wide, then the TD five strictly alone.
    serial "A-fit-ds::${DRIVER[*]} -MaxParallel 4 -Models vg15,vg20,vg24,vg25" \
           "A-fit-td::${DRIVER[*]} -MaxParallel 1 -Models vg11,vg12,vg21,vg23,vg26"
}

stage_B() {
    serial "B-fit-ds::${DRIVER[*]} -MaxParallel $POOL -Models vg01,vg02,vg05,vg07,vg08,vg09,vg10,vg14,vg16,vg19,vg22" \
           "B-fit-td::${DRIVER[*]} -MaxParallel 1 -Models vg03,vg04,vg13"
}

stage_C() {
    local rc=0
    arms C "$POOL" "${C_DS_ARMS[@]}" || rc=1
    # Sequential validation for the two lag models (#289 task 3.8, #297).
    serial "C-wave-forward-vg16::${PY[*]} scripts/wave_forward_score.py --model vg16 --config rep" \
           "C-wave-forward-vg25::${PY[*]} scripts/wave_forward_score.py --model vg25 --config rep" || rc=1
    # Recovery, Down syndrome, three cells at a time. VG20 scores total spread
    # (#289 4.8/4.12). VG25's and VG16 corr's designed cells separate a lag
    # from a correlation (#297 check 4, #289 3.9): rho_uq = 2 * rho_uq_raw - 1.
    pool 3 \
        "C-rec-vg20::${REC[*]} vg20 --config rep --replicates 3" \
        "C-rec-vg25::${REC[*]} vg25 --config rep --replicates 3" \
        "C-rec-vg25-beta0::${REC[*]} vg25 --config rep --replicates 3 --set-truth beta_sign_lag=0" \
        "C-rec-vg25-rho0::${REC[*]} vg25 --config rep --replicates 3 --set-truth subject_re=independent" \
        "C-rec-vg16-corr::${REC[*]} vg16 --variant corr --config rep --replicates 3" \
        "C-rec-vg16-corr-beta0::${REC[*]} vg16 --variant corr --config rep --replicates 3 --set-truth beta_lag=0" \
        "C-rec-vg16-corr-rho0::${REC[*]} vg16 --variant corr --config rep --replicates 3 --set-truth rho_uq_raw=0.5" \
        || rc=1
    # Typically developing, alone on the machine: #289 task 4.8.
    arms C 1 "${C_TD_ARMS[@]}" || rc=1
    serial "C-rec-vg11::${REC[*]} vg11 --config rep --replicates 3" \
           "C-rec-vg12::${REC[*]} vg12 --config rep --replicates 3" \
           "C-rec-vg12-free-scales::${REC[*]} vg12 --variant free-scales --config rep --replicates 3" || rc=1
    return "$rc"
}

stage_D() {
    local rc=0
    arms D "$POOL" "${D_DS_ARMS[@]}" || rc=1
    # Recovery for the models that have never had it (#289 task 3.3).
    pool 3 \
        "D-rec-vg07::${REC[*]} vg07 --config rep --replicates 3" \
        "D-rec-vg08::${REC[*]} vg08 --config rep --replicates 3" \
        "D-rec-vg09::${REC[*]} vg09 --config rep --replicates 3" \
        "D-rec-vg10::${REC[*]} vg10 --config rep --replicates 3" \
        "D-rec-vg19::${REC[*]} vg19 --config rep --replicates 3" || rc=1
    arms D 1 "${D_TD_ARMS[@]}" || rc=1
    serial "D-rec-vg13::${REC[*]} vg13 --config rep --replicates 3" \
           "D-rec-vg21::${REC[*]} vg21 --config rep --replicates 3" \
           "D-rec-vg23::${REC[*]} vg23 --config rep --replicates 3" || rc=1
    # The Gompertz mean-function comparison (#330), at test tier as the issue specifies.
    local G="${PY[*]} scripts/experiments/gompertz_mean_arm.py"
    serial "D-gompertz-vg20-baseline::$G fit vg20 baseline --config test" \
           "D-gompertz-vg20-gompertz::$G fit vg20 gompertz --config test" \
           "D-gompertz-vg20-flexible::$G --ratio-mean flexible fit vg20 gompertz --config test" \
           "D-gompertz-vg12-baseline::$G fit vg12 baseline --config test" \
           "D-gompertz-vg12-gompertz::$G fit vg12 gompertz --config test" \
           "D-gompertz-vg11-baseline::$G fit vg11 baseline --config test" \
           "D-gompertz-vg11-gompertz::$G fit vg11 gompertz --config test" \
           "D-gompertz-compare::$G compare --config test" \
           "D-gompertz-compare-flexible::$G --ratio-mean flexible compare --curves vg20 --config test" \
           "D-gompertz-loso::$G loso --config test --folds 5" || rc=1
    # The typically developing held-out k-fold (#240), serially; about 100 hours.
    local K="${PY[*]} scripts/kfold_loso.py"
    serial "D-kfold-vg12-subject::$K --models VG12 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg12" \
           "D-kfold-vg12-study::$K --models VG12 --holdout-unit study --config rep-lite --suffix _vg12" \
           "D-kfold-vg11-subject::$K --models VG11 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg11" \
           "D-kfold-vg11-study::$K --models VG11 --holdout-unit study --config rep-lite --suffix _vg11" \
           "D-kfold-vg23-subject::$K --models VG23 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg23" \
           "D-kfold-vg23-study::$K --models VG23 --holdout-unit study --config rep-lite --suffix _vg23" \
           "D-kfold-vg21-vg26-subject::$K --models VG21,VG26 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg21_vg26" \
           "D-kfold-vg21-vg26-study::$K --models VG21,VG26 --holdout-unit study --config rep-lite --suffix _vg21_vg26" || rc=1
    return "$rc"
}

stage_P() {
    # The comparisons the 2026-09-17 cycle ran, the three it quarantined
    # (loo_compare, compare_models, loso_compare), and the robustness matrices.
    # kfold_loso's DS run refits per fold and belongs to a deliberate decision,
    # not to this stage. Nothing here uploads.
    local specs=() script model
    for script in loo_compare loso_compare compare_models compare_ds_td compare_ds_td_trajectories \
                  compare_ds_td_expressive compare_ds_td_latency compare_ds_td_q_overlap \
                  subject_effect_correlation pool_descriptives compare_matched_designs; do
        specs+=("P-cmp-$script::${PY[*]} scripts/$script.py")
    done
    specs+=("P-cmp-compare_ds_td_re::${PY[*]} scripts/compare_ds_td_re.py spoken understood comprehension"
            "P-cmp-prior_vs_posterior::${PY[*]} scripts/prior_vs_posterior.py --table --model vg20 --model vg15")
    for model in vg10 vg11 vg12 vg13 vg14 vg15 vg16 vg19 vg20 vg21 vg22 vg23 vg25 vg26; do
        specs+=("P-sens-$model::${PY[*]} scripts/compare_sensitivity.py $model")
    done
    serial "${specs[@]}" \
        "P-sync::${PY[*]} scripts/sync_report_figures.py --config rep --allow-caveats" \
        "P-prepare-figures::${PY[*]} scripts/prepare_report_figures.py" \
        "P-render-report::quarto render docs/report"
}

usage() { sed -n '5,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }

run_stages() {
    local stage rc=0
    for stage in "$@"; do
        [[ $stage =~ ^[ABCDP]$ ]] || die "unknown stage $stage"
        log "===== stage $stage ($TAG) ====="
        "stage_$stage" || rc=1
        log "===== stage $stage $([[ $rc == 0 ]] && echo complete || echo 'finished with failed steps') ====="
    done
    return "$rc"
}

command=${1:-}; shift || true
mkdir -p "$STATE/logs"
case $command in
    plan)
        PLAN_ONLY=1
        for stage in "${@:-A B C D P}"; do for s in $stage; do echo "== stage $s"; "stage_$s"; done; done
        ;;
    run)
        (( $# )) || usage
        run_stages "$@"
        ;;
    launch)
        (( $# )) || usage
        # Separate the campaign from the login session and sibling process groups.
        # This limits some OOM failure effects but cannot prevent the kernel
        # from selecting a process in this campaign when memory runs out.
        unit="vg-campaign-$(date -u +%Y%m%dT%H%M%S)"
        setsid nohup systemd-run --user --scope --collect --unit="$unit" -p OOMPolicy=continue \
            -- "${BASH_SOURCE[0]}" run "$@" >"$STATE/launch-$unit.out" 2>&1 </dev/null &
        echo "launched $unit; follow: tail -f $STATE/campaign.log"
        ;;
    status)
        tail -n 30 "$STATE/campaign.log" 2>/dev/null || echo "no campaign log yet"
        echo "-- done: $(find "$STATE" -maxdepth 1 -name '*.ok' | wc -l), failed:"
        find "$STATE" -maxdepth 1 -name '*.failed' -printf '   %f\n' | sed 's/\.failed$//'
        ;;
    *) usage ;;
esac
