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

## Adding a case

Create `src/tsforecastbench/cases/<case_id>/` with:

- `case.toml` — metadata validated into a `Case` (see `schema.py`): `id`, `domain`,
  `target`, `freq`, `horizon_primary`, `exogenous`, `splits` (or `split_mode="fraction"`),
  and `source`.
- `synth.py` — `generate(case) -> pd.DataFrame` returning the canonical long format
  (`unique_id`, `ds` UTC, `y`, plus exogenous columns). This lets the pipeline and CI
  run before any real data exists.
- `download.py` — `main(argv) -> int` that fetches the real data and writes
  `data/processed/<case_id>.parquet` via `tsforecastbench.cases._io.write_canonical`.

The registry auto-discovers any `cases/*/case.toml`.

## Adding a method

Subclass `tsforecastbench.methods.base.Forecaster`:

```python
class MyForecaster(Forecaster):
    name = "MyMethod"           # column header in the matrix
    requires_gpu = False
    def fit(self, case, train): ...
    def predict(self, case, history, horizon): ...   # -> DataFrame[ds, y_hat]
```

Export it in `methods/__init__.py` and add it to the method list in `scripts/run_pilot.py`.

## Conventions

- Canonical storage is long-format Parquet; in-memory is pandas.
- Timestamps are UTC and tz-aware.
- Gaps are explicit NaN rows; training imputation and NaN-masked scoring are handled in
  `eval/loop.py`, so methods can assume a complete `y` at fit time.
- Keep methods CPU-runnable where possible; mark heavy ones `requires_gpu = True`.
