# Session handoff → WSL

Written 2026-06-14 to hand the project from the Windows Claude Code session to a fresh
session running in WSL. Delete this file once the transition is done.

## TL;DR

- **`origin/main` is the single source of truth and is fully current** (PRs #3–17 merged):
  11 cases, 16 methods, the review-case skill, RESULTS.md, the WSL bootstrap. A fresh WSL
  clone of `main` has everything — nothing important is stranded on the Windows box.
- The WSL clone is already at `~/SourceWSL/timeseries-forecasting-benchmark`. First do
  `git pull` (to get this file), then run the bootstrap below.
- **Claude memory does NOT transfer** (it's keyed to the Windows project path). The new WSL
  session starts with empty memory — this file is the bridge. Re-create memories there as you go.

## First steps in WSL

```bash
cd ~/SourceWSL/timeseries-forecasting-benchmark
git pull
bash scripts/bootstrap_wsl.sh --gpu      # uv + Python 3.12 + extras + CUDA torch + nf zoo
uv run pytest -q                          # expect ~74 passing
uv run python scripts/run_pilot.py --fast # smoke the pipeline
```

Notes:
- Use **Python 3.12** (the bootstrap pins it). 3.13 is exactly why `ray`/`neuralforecast`
  wouldn't install on Windows.
- GPU is an **NVIDIA RTX 500 Ada, 4 GB** — real but small. Keep model/batch sizes modest;
  it won't run large foundation models.
- The Windows host has **AC sleep disabled** (`powercfg`), so overnight WSL runs on mains
  won't be killed by the laptop sleeping. (If you ever revert it: a 120-min idle sleep is
  what cut the first overnight run short.)

## What is NOT in git (Windows-only — safe to discard, regenerate in WSL)

- A **partial set of per-case reports** (bellinge, camels, gesla) under `docs/reports/` and
  `docs/figures/reports/`, left untracked when the generator was interrupted. Just regenerate
  all 11 cleanly in WSL: `uv run python scripts/case_report.py --with-fm`.
- Everything under `results/` (matrix parquet/PNGs, overnight logs) is **gitignored** by
  design — regenerate with `scripts/run_pilot.py`.
- Only `docs/reports/usgs_groundwater.md` (+ its figures) is committed so far.

## Suggested first work in WSL (the GPU-deferred backlog)

1. **Validate the neuralforecast zoo** — `src/tsforecastbench/bench/methods/nf_models.py`
   (TFT / iTransformer / TimesNet) is written but **never run** (no neuralforecast on
   Windows). Expect possible small API fixes. Test:
   `uv run python scripts/run_pilot.py --with-nf --origins 5 --cases gesla_storm_surge usgs_river_discharge`
2. **Full DL on GPU** — N-HiTS/PatchTST were only a 4-case CPU probe (PatchTST ~400 s/cell).
   On GPU run them across all cases: `run_pilot --with-dl --with-fm --origins 12`, then
   refresh `docs/RESULTS.md` + the matrix figures with the complete DL columns.
3. **Generate all 11 per-case reports** and commit them.
4. **Roadmap** (research_plan.md): covariate-capable FM (Chronos-2 / Moirai), fine-tune the
   top zero-shot performer, more cases, the Streamlit browser.

## Project state / orientation

- **Two namespaces** (will eventually become two repos): `tsforecastbench.data` (case
  building — schema, registry, `cases/`, the `_usgs`/`_ndbc`/`_era5` fetchers) and
  `tsforecastbench.bench` (methods, eval, viz). `bench` depends on `data` only via `Case`.
- **Entry points**: `scripts/run_pilot.py` (matrix; `--with-fm`/`--with-dl`/`--with-nf`,
  `--fast`, `--origins`), `scripts/run_leadtime.py`, `scripts/case_report.py`,
  `scripts/review_case.py` (and the `/review-case` skill — run it on every new case).
- **11 cases**: 3 hydrology (groundwater `camels_dk`, flashy `usgs_river`, snowmelt
  `usgs_snowmelt`), `usgs_groundwater`, `usgs_stream_temp`, `bellinge_sewer_level`,
  `wd_battledim`, `wwtp_influent`, `gesla_storm_surge`, `ndbc_waves`, `ndbc_waves_atlantic`.
- **16 methods**: naive ×4, AutoARIMA/AutoETS/Theta, LinearLags/LightGBM/MLP (+`+exo`
  variants), DLinear, N-HiTS, PatchTST (torch), Chronos-Bolt (zero-shot FM), + the nf zoo.
- **Data**: USGS/NDBC/ERA5 fetch credential-free (Open-Meteo for ERA5). GESLA + Mendeley
  WWTP need a manual archive in `data/raw/<case>/` — **copy those raw archives over** if you
  want to rebuild those two from source (processed Parquets are committed, so not required).

## Key findings so far (the scientific context — see docs/RESULTS.md)

From one consistent 11-case × 14-method × 12-origin rolling pass:
- **Covariates win 8/11 where physics predicts** — stream-temp +0.35→+0.84, surge
  +0.00→+0.52, snowmelt −0.08→+0.47; groundwater is the null (integrates recharge).
- **Chronos zero-shot wins the flashy river (+0.80)** and is strong on regular signals
  (demand, WWTP flow) — a strong but **regime-dependent free prior** (the KER5 verdict).
- Honest nuances: covariates can **hurt** (bellinge 15-min rain +0.31→−0.45); univariate
  **LightGBM is brittle on flashy daily** (−3.11) where DLinear/ARIMA/Chronos win; **MLP**
  is the least robust (bellinge −54); seasonal-naive is a trap on trending hydrology.

## Git workflow (important)

- **PR-only.** Direct push to `main` is blocked by the harness. Branch → `gh pr create` →
  wait for CI (`gh pr checks`) → `gh pr merge --merge`.
- **Leave PR #2 (`rpal_review`) open** — it's Rocco's comment-only review, for reference.
- PR #1 (dependabot aiohttp) is open — review/merge or close at leisure.
- End commit messages with the `Co-Authored-By: Claude …` trailer; PR bodies with the
  Generated-with line.
