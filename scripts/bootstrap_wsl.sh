#!/usr/bin/env bash
# Bootstrap the tsforecastbench dev environment on WSL2 / Linux.
#
# Why WSL: the broader DL/foundation-model ecosystem (neuralforecast -> ray, gluonts,
# uni2ts/Moirai, TimesFM) and clean NVIDIA CUDA are Linux-first. CI is already
# ubuntu-latest, so this aligns local dev with CI and the eventual open release.
#
# Usage (from a fresh clone in the WSL ext4 home, NOT /mnt/c — Windows-mount I/O is slow):
#     git clone <repo> ~/timeseries-forecasting-benchmark
#     cd ~/timeseries-forecasting-benchmark
#     bash scripts/bootstrap_wsl.sh           # CPU torch
#     bash scripts/bootstrap_wsl.sh --gpu     # + CUDA torch if an NVIDIA GPU is present
#
# Python 3.12 is used deliberately (3.13 has no ray wheels -> neuralforecast won't install).
set -euo pipefail

PYVER=3.12
WANT_GPU=0
[[ "${1:-}" == "--gpu" ]] && WANT_GPU=1

echo "==> 1/5  Ensure uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "==> 2/5  Pin Python $PYVER and sync core + portable extras (dev, data, dl, fm)"
uv python install "$PYVER"
uv sync --python "$PYVER" --extra dev --extra data --extra dl --extra fm

echo "==> 3/5  GPU torch (optional)"
if [[ "$WANT_GPU" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "    NVIDIA GPU detected — installing CUDA torch (cu124)"
    uv pip install --python "$PYVER" torch --index-url https://download.pytorch.org/whl/cu124
  else
    echo "    No nvidia-smi found in WSL — skipping. (Install the NVIDIA Windows driver;"
    echo "    CUDA is exposed to WSL2 automatically, no separate Linux driver needed.)"
  fi
else
  echo "    skipped (pass --gpu to install CUDA torch)"
fi

echo "==> 4/5  neuralforecast zoo (TFT / iTransformer / TimesNet) — Linux only"
# Marker-gated 'nf' extra; resolves on Linux + py<3.13.
if uv sync --python "$PYVER" --extra dev --extra dl --extra fm --extra nf 2>/dev/null; then
  echo "    neuralforecast installed (run_pilot --with-nf is now available)"
else
  echo "    'nf' extra did not resolve; the torch N-HiTS/PatchTST (--with-dl) still work."
fi

echo "==> 5/5  Verify"
uv run python scripts/precheck.py || uv run pytest -q

cat <<'NOTE'

Done. Next:
  * Copy your large raw downloads into  data/raw/<case_id>/  (GESLA zip, Mendeley WWTP).
    Everything else (USGS, NDBC, ERA5) re-fetches automatically and credential-free.
  * Smoke the benchmark:   uv run python scripts/run_pilot.py --fast
  * Deep-learning columns:  uv run python scripts/run_pilot.py --with-dl --origins 5 \
                              --cases surge_cuxhaven discharge_flashy_catskills
  * neuralforecast zoo:     uv run python scripts/run_pilot.py --with-nf --origins 5 ...
NOTE
