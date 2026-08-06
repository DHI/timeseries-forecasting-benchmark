# Contributing to tsforecastbench

Thanks for helping extend the benchmark. The two things you'll most often add are a
**case** (a dataset/forecasting problem) or a **method** (a forecasting model). Both
plug into a common interface, so additions are local.

## Setup

```bash
uv sync                 # core deps
uv sync --extra data    # + dataset download deps (xarray, utide, cdsapi, ...)
uv sync --extra dev     # + pytest, ruff
uv run pytest           # tests run entirely on synthetic data (no network, no GPU)
uv run ruff check .
```

## Before pushing

CI runs both `ruff check` and `pytest`. **Run the same locally first** so a failing CI
isn't a surprise:

```bash
git add -A                              # stage everything; don't try to be selective
uv run python scripts/precheck.py       # ruff + pytest, plus prints uncommitted files
git commit -m "..."
git push
```

`scripts/precheck.py` also prints `git status` so any unstaged changes are visible
before you push — easier than chasing a failing CI on a file that didn't get added.

The package is split into two concerns: **`tsforecastbench.data`** (case-building:
schema, registry, downloads) and **`tsforecastbench.bench`** (model comparison: methods,
eval, viz). `bench` depends on `data` only through the public `Case` interface.

## Adding a case (data side)

Create `src/tsforecastbench/data/cases/<case_id>/` with:

- `case.toml` — metadata validated into a `Case` (see `data/schema.py`): `id`, `domain`,
  `target`, `freq`, `horizon_primary`, `exogenous`, `splits` (or `split_mode="fraction"`),
  and `source`. If `exogenous` (or `exogenous_known_future`) is non-empty, list which of
  those columns are required for a meaningful forecast in `necessary_exogenous` (e.g. rain
  for a flood case) — a domain-expert judgment. Everything else is implicitly optional
  (`Case.optional_exogenous`, derived, not a TOML field). See `research_log/details.md` (2026-07-02).
  `seasonalities` is **required**: seasonal periods in steps of `freq`, dominant first,
  `[]` = non-seasonal. It is the single source of truth for seasonality in the
  train/predict cycle (methods and the MASE season resolve from it; there is no
  frequency-based fallback). Fix it once at curation time: open
  `notebooks/seasonality.py`, review the train-split evidence (periodogram prominence +
  ACF over a fixed physical candidate menu), and commit the proposed value — or deviate
  from it with a justification in `notes`. See `research_log/details.md` (2026-07-03).
- `synth.py` — `generate(case) -> pd.DataFrame` returning the canonical long format
  (`unique_id`, `ds` UTC, `y`, plus exogenous columns). This lets the pipeline and CI
  run before any real data exists.
- `download.py` — `main(argv) -> int` that fetches the real data and writes
  `data/processed/<case_id>.parquet` via `tsforecastbench.data.cases._io.write_canonical`.
  Shared fetchers are available: `_usgs.py` (USGS NWIS), `_ndbc.py` (NDBC buoys),
  `_era5.py` (credential-free ERA5 forcing via Open-Meteo).

The registry auto-discovers any `data/cases/*/case.toml`. **Run `/review-case <id>`** (or
`scripts/review_case.py`) on a new case before committing — it checks coverage, exogenous
predictive skill, and forecastability.

## Adding a method (bench side)

Subclass `tsforecastbench.bench.methods.base.Forecaster`:

```python
class MyForecaster(Forecaster):
    name = "MyMethod"            # column header in the matrix
    requires_gpu = False
    uses_exogenous = False       # set True to receive future covariates
    def fit(self, case, train): ...
    def predict(self, case, history, horizon, future_exog=None): ...  # -> DataFrame[ds, y_hat]
```

Export it in `bench/methods/__init__.py` and add it to the method list in
`scripts/run_pilot.py` (optionally behind a flag like `--with-dl` for heavy methods).

## Conventions

- Canonical storage is long-format Parquet; in-memory is pandas.
- Timestamps are UTC and tz-aware.
- Gaps are explicit NaN rows; training imputation and NaN-masked scoring are handled in
  `eval/loop.py`, so methods can assume a complete `y` at fit time.
- Keep methods CPU-runnable where possible; mark heavy ones `requires_gpu = True`.
