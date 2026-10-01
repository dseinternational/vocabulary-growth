#!/usr/bin/env bash
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Prepare a dedicated Linux VM for a refit campaign. Read
# docs/runbooks/vm-refit.md first. Idempotent: rerun it after fixing whatever it
# reports.
#
#   bootstrap.sh TAG [--output-root DIR] [--checkout DIR] [--with-tests]
#
#   TAG            the cycle tag, e.g. fits/2026-10-02
#   --output-root  the attached persistent disk's output root (default /data/vocabulary-growth)
#   --checkout     where the pinned clone goes (default ~/vocabulary-growth-<tag date>)
#   --with-tests   also run the fast test suite in the pinned clone
#
# It checks the machine and its tools, clones the repository detached at TAG,
# installs the locked environment, prepares the data, and writes
# ~/.config/vocabulary-growth/vm.env, which scripts/vm/campaign.sh reads. The
# environment file lives outside the checkout on purpose: an untracked file in
# the clone would mark every fit dirty and unpublishable.

set -uo pipefail

REPO_URL=${VG_REPO_URL:-https://github.com/dseinternational/vocabulary-growth.git}
TAG=${1:-}
[[ $TAG == fits/* ]] || { echo "usage: bootstrap.sh fits/YYYY-MM-DD [--output-root DIR] [--checkout DIR] [--with-tests]" >&2; exit 2; }
shift
OUTPUT_ROOT=/data/vocabulary-growth
CHECKOUT="$HOME/vocabulary-growth-${TAG#fits/}"
WITH_TESTS=0
while (( $# )); do
    case $1 in
        --output-root) OUTPUT_ROOT=$2; shift 2 ;;
        --checkout) CHECKOUT=$2; shift 2 ;;
        --with-tests) WITH_TESTS=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

problems=0
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; problems=$(( problems + 1 )); }

echo "== machine"
arch=$(uname -m)
case $arch in
    x86_64) ok "x86_64" ;;
    aarch64) warn "aarch64: vg15 fallback-dispersion has failed numba compilation here; rerun that arm with --nutpie-backend jax if it does (runbook)" ;;
    *) fail "unsupported architecture $arch" ;;
esac
cores=$(nproc)
(( cores >= 32 )) && ok "$cores cores" || warn "$cores cores; the stage pools assume 32 (set VG_POOL lower)"
mem_gb=$(awk '/MemTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
(( mem_gb >= 128 )) && ok "${mem_gb} GB RAM" || fail "${mem_gb} GB RAM; the TD fits peak at 27-28 GB each and the DS pool needs headroom (128 GB minimum)"
swap_gb=$(awk '/SwapTotal/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
(( swap_gb >= 64 )) && ok "${swap_gb} GB swap" \
    || fail "${swap_gb} GB swap; add a backstop before any fit (runbook, 'Swap'): sudo fallocate -l 128G /scratch/swapfile && sudo chmod 600 /scratch/swapfile && sudo mkswap /scratch/swapfile && sudo swapon /scratch/swapfile"
if loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q 'Linger=yes'; then
    ok "user lingering enabled (systemd --user scopes outlive the login)"
else
    fail "user lingering is off; a launched campaign would die with the SSH session: sudo loginctl enable-linger $USER"
fi

echo "== output root"
mkdir -p "$OUTPUT_ROOT" 2>/dev/null || fail "cannot create $OUTPUT_ROOT"
if [[ -d $OUTPUT_ROOT ]]; then
    fs=$(df -PT "$OUTPUT_ROOT" | awk 'NR == 2 { print $2 }')
    free=$(df -P -BG "$OUTPUT_ROOT" | awk 'NR == 2 { gsub("G", "", $4); print $4 }')
    case $fs in
        nfs*|cifs|smb*|fuse*) fail "$OUTPUT_ROOT is on $fs; traces are HDF5 and must not be written to a network filesystem" ;;
        *) ok "$OUTPUT_ROOT on $fs" ;;
    esac
    (( free >= 1000 )) && ok "${free} GB free" || warn "${free} GB free; a full campaign at the full trace tier wants about 1 TB (runbook, 'Disk')"
    mountpoint -q /scratch 2>/dev/null && [[ $OUTPUT_ROOT == /scratch* ]] \
        && fail "$OUTPUT_ROOT is on /scratch, which is wiped on deallocation; use the attached disk"
fi

echo "== tools"
for tool in git uv pwsh quarto dot node npm az fc-list; do
    command -v "$tool" >/dev/null && ok "$tool" || fail "$tool is not on PATH"
done
for font in "Noto Sans" "Noto Sans Mono" "Noto Sans Math"; do
    fc-list : family 2>/dev/null | grep -qx "$font" && ok "font $font" \
        || fail "font $font is missing; matplotlib would silently substitute another (CLAUDE.md, Environment)"
done

echo "== pinned checkout"
if [[ ! -d $CHECKOUT/.git ]]; then
    git clone --quiet "$REPO_URL" "$CHECKOUT" || { fail "clone failed"; exit 1; }
fi
git -C "$CHECKOUT" fetch --quiet --tags origin
git -C "$CHECKOUT" checkout --quiet --detach "$TAG" || { fail "tag $TAG not found"; exit 1; }
[[ -z $(git -C "$CHECKOUT" status --porcelain --untracked-files=normal) ]] \
    && ok "$CHECKOUT clean at $TAG ($(git -C "$CHECKOUT" rev-parse --short HEAD))" \
    || fail "$CHECKOUT has local changes; fits from it would be unpublishable"

cd "$CHECKOUT" || exit 1
uv sync --locked --quiet && ok "locked environment" || fail "uv sync --locked failed"
uv run --locked python scripts/prepare_data.py >/dev/null && ok "data prepared" || fail "prepare_data.py failed"
npm ci --silent >/dev/null 2>&1 && ok "node tools" || warn "npm ci failed; only the documentation checks need it"
# matplotlib caches its font list; a cache built before the Noto fonts were
# installed keeps substituting until it is deleted.
cache=$(uv run --locked python -c 'import matplotlib; print(matplotlib.get_cachedir())' 2>/dev/null)
[[ -n $cache ]] && rm -f "$cache"/fontlist-*.json && ok "matplotlib font cache cleared"
uv run --locked python -c 'import dse_research_utils, vocab_growth' && ok "package imports" || fail "imports failed"

echo "== environment file"
mkdir -p "$HOME/.config/vocabulary-growth"
compile_dir="$HOME/.cache/pytensor"
mountpoint -q /scratch 2>/dev/null && compile_dir=/scratch/pytensor
cat >"$HOME/.config/vocabulary-growth/vm.env" <<EOF
# Written by scripts/vm/bootstrap.sh for $TAG on $(date -u +%F).
export DSE_VOCAB_GROWTH_OUTPUT_DIR=$OUTPUT_ROOT
export PYTHONUTF8=1
export QUARTO_PYTHON=$CHECKOUT/.venv/bin/python
export PYTENSOR_FLAGS=base_compiledir=$compile_dir
export AZURE_TOKEN_CREDENTIALS=dev
EOF
ok "$HOME/.config/vocabulary-growth/vm.env"
[[ -z $(git status --porcelain --untracked-files=normal) ]] && ok "checkout still clean after setup" \
    || fail "setup left the checkout dirty: $(git status --porcelain | head -3 | tr '\n' ' ')"

if (( WITH_TESTS )); then
    echo "== fast tests"
    uv run --locked pytest -n auto --dist loadfile -m "not slow" -q -o addopts="" && ok "fast suite" || fail "fast suite"
fi

echo
if (( problems )); then
    echo "$problems problem(s) to fix before launching; rerun this script afterwards."
    exit 1
fi
echo "Ready. Next: cd $CHECKOUT && scripts/vm/campaign.sh plan && scripts/vm/campaign.sh launch A"
