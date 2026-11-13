#!/usr/bin/env bash
# Overnight benchmark run — three sequential phases.
# Launch with:  bash scripts/overnight_run.sh 2>&1 | tee results/overnight_$(date +%Y%m%d)/driver.log
#
# Phase 1: nf zoo smoke test   — 2 cases × 3 origins (validates TFT/iTransformer/TimesNet)
# Phase 2: full core matrix    — all 11 cases × 24 origins (base + DL + Chronos-Bolt)
# Phase 3: full nf zoo matrix  — all 11 cases × 12 origins (TFT + iTransformer + TimesNet)
#
# Results land in results/overnight_YYYYMMDD/:
#   phase1_nf_smoke/     parquet + matrix PNGs
#   phase2_core_matrix/  parquet + matrix PNGs
#   phase3_nf_full/      parquet + matrix PNGs
#   driver.log           full console output

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_DIR="$REPO_ROOT/results/overnight_$(date +%Y%m%d)"

mkdir -p "$RUN_DIR"/{phase1_nf_smoke,phase2_core_matrix,phase3_nf_full}

cd "$REPO_ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── install extras ──────────────────────────────────────────────────────────
log "=== Installing extras: dl nf fm ==="
uv sync --extra dl --extra nf --extra fm
log "extras installed"

# ─── Phase 1: nf zoo smoke test ──────────────────────────────────────────────
log "=== PHASE 1: nf zoo smoke test (2 cases × 3 origins) ==="
PHASE1_OK=0
uv run python scripts/run_pilot.py \
    --with-nf --origins 3 \
    --cases surge_cuxhaven discharge_flashy_catskills \
    2>&1 | tee "$RUN_DIR/phase1_nf_smoke/run.log" && PHASE1_OK=1 || true

if [ "$PHASE1_OK" -eq 1 ] && [ -f "$REPO_ROOT/results/pilot_metrics.parquet" ]; then
    cp "$REPO_ROOT/results/pilot_metrics.parquet" \
       "$RUN_DIR/phase1_nf_smoke/metrics.parquet"
    for f in "$REPO_ROOT/results/pilot_matrix"*.png; do
        [ -f "$f" ] && cp "$f" "$RUN_DIR/phase1_nf_smoke/" || true
    done
    log "Phase 1 PASSED — nf zoo is functional"
else
    log "Phase 1 FAILED or produced no results — check $RUN_DIR/phase1_nf_smoke/run.log"
    log "Continuing to Phase 2 (without nf models — they will be skipped if broken)"
fi

# ─── Phase 2: full core matrix (base + DL + FM) ──────────────────────────────
log "=== PHASE 2: full core matrix — all cases × 24 origins (base + DL + Chronos-Bolt) ==="
PHASE2_OK=0
uv run python scripts/run_pilot.py \
    --with-dl --with-fm --origins 24 \
    2>&1 | tee "$RUN_DIR/phase2_core_matrix/run.log" && PHASE2_OK=1 || true

if [ "$PHASE2_OK" -eq 1 ] && [ -f "$REPO_ROOT/results/pilot_metrics.parquet" ]; then
    cp "$REPO_ROOT/results/pilot_metrics.parquet" \
       "$RUN_DIR/phase2_core_matrix/metrics.parquet"
    for f in "$REPO_ROOT/results/pilot_matrix"*.png; do
        [ -f "$f" ] && cp "$f" "$RUN_DIR/phase2_core_matrix/" || true
    done
    log "Phase 2 PASSED"
else
    log "Phase 2 FAILED — check $RUN_DIR/phase2_core_matrix/run.log"
fi

# ─── Phase 3: full nf zoo (only if Phase 1 passed) ───────────────────────────
if [ "$PHASE1_OK" -eq 1 ]; then
    log "=== PHASE 3: full nf zoo — all cases × 12 origins (TFT + iTransformer + TimesNet) ==="
    PHASE3_OK=0
    uv run python scripts/run_pilot.py \
        --with-nf --origins 12 \
        2>&1 | tee "$RUN_DIR/phase3_nf_full/run.log" && PHASE3_OK=1 || true

    if [ "$PHASE3_OK" -eq 1 ] && [ -f "$REPO_ROOT/results/pilot_metrics.parquet" ]; then
        cp "$REPO_ROOT/results/pilot_metrics.parquet" \
           "$RUN_DIR/phase3_nf_full/metrics.parquet"
        for f in "$REPO_ROOT/results/pilot_matrix"*.png; do
            [ -f "$f" ] && cp "$f" "$RUN_DIR/phase3_nf_full/" || true
        done
        log "Phase 3 PASSED"
    else
        log "Phase 3 FAILED — check $RUN_DIR/phase3_nf_full/run.log"
    fi
else
    log "Phase 1 failed — skipping Phase 3 (nf zoo not validated)"
fi

# ─── summary ─────────────────────────────────────────────────────────────────
log "=== OVERNIGHT RUN COMPLETE ==="
log "Results in: $RUN_DIR"
for phase_dir in "$RUN_DIR"/phase{1,2,3}*/; do
    if [ -f "$phase_dir/metrics.parquet" ]; then
        rows=$(uv run python -c "
import pandas as pd; df = pd.read_parquet('$phase_dir/metrics.parquet')
print(f'{len(df)} rows, {df[\"case_id\"].nunique()} cases, {df[\"method\"].nunique()} methods')
" 2>/dev/null || echo "unreadable")
        log "  $(basename "$phase_dir"): $rows"
    else
        log "  $(basename "$phase_dir"): NO RESULTS"
    fi
done
