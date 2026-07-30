# Plan: exogenous inputs for MLP, DLinear, N-HiTS, PatchTST

## Status quo

`_WindowedMLBase` (in `ml.py`) has end-to-end exogenous support: column extraction
(`_exo_cols`), last-value fallback, a `_future_X` builder, and `X_df` at predict time.
LightGBM, LinearLags, and CatBoost each have a trivial `+exo` subclass that flips
`use_exogenous = True`. MLP inherits the same base but has no `+exo` variant — an
oversight, since the plumbing already works.

DLinear, N-HiTS, and PatchTST are standalone implementations (not `_WindowedMLBase`)
and are univariate by design.

---

## Tier 1 — MLP+exo (trivial, ≤ 10 lines)

The base already does everything. Add a subclass and register it.

### Changes

**`ml.py`** — append:

```python
class MLPExoForecaster(MLPForecaster):
    name = "MLP+exo"
    use_exogenous = True
```

**`__init__.py`** — add `MLPExoForecaster` to imports and `__all__`.

**`test_pipeline_synth.py`** — add `MLPExoForecaster` to the `test_exogenous_methods_run`
parametrised list (and import it).

### Why it works without further changes

`test_exo_variants_inherit_their_base_declarations` in `test_method_scorecards.py`
already asserts that every `*ExoForecaster` shares its base's capabilities and scorecard.
`_WindowedMLBase` declares `capabilities.exogenous = True` on the base, and
`MLPForecaster` inherits it, so the invariant holds automatically.

---

## Tier 2 — DLinear+exo (moderate, ~80 lines)

DLinear fits a Ridge from `[trend | remainder]` (shape `(N, 2L)`) to `Y` (`(N, H)`).
Exogenous support extends the feature vector without changing the solver.

### Design

1. **Windowing** — alongside each target window `(L,)` and horizon `(H,)`, extract
   covariate arrays of the same spans: `exo_hist (L, n_exo)` for the lookback,
   `exo_fut (H, n_exo)` for the horizon. Use `segment_arrays` generalised to
   extra columns, or build a parallel windowing loop over the exogenous columns
   of the training frame.

2. **Feature vector** — flatten and concatenate per sample:
   `[trend | remainder | exo_hist_flat | exo_fut_flat]`, giving shape
   `(N, 2L + (L+H) × n_exo)`. When `n_exo = 0` the vector is identical to today's.

3. **Predict** — fill the lookback exogenous block from `history` and the future
   block from `future_exog`, using the same last-value fallback pattern as
   `_WindowedMLBase._last_exo`.

4. **Subclass** — `DLinearExoForecaster(DLinearForecaster)` with
   `use_exogenous = True`, `capabilities` updated to `exogenous=True`, `limitations`
   docstring adjusted.

### Changes

**`dl.py`**:

- Import `_exo_cols` from `ml.py` (or duplicate the three-line helper locally to
  avoid a cross-module dependency — either is fine, the helper is trivial).
- Add `use_exogenous: bool = False` and `uses_exogenous` to `__init__`.
- In `fit`: extract exogenous column arrays per segment, window them alongside the
  target, concatenate to the feature matrix.
- In `predict`: build the exogenous lookback+future block the same way, concatenate
  to the decomposed window before `_model.predict`.
- Add `DLinearExoForecaster` subclass.

**`__init__.py`** — register.

**`test_pipeline_synth.py`** — add to the exogenous-methods test.

### Risk

Low. The Ridge solver is unchanged; the only new code is array slicing and
concatenation. The main thing to verify is that the exogenous window indices stay
aligned with the target windows (off-by-one in the future block would shift covariates
relative to the target).

---

## Tier 3 — N-HiTS+exo / PatchTST+exo (significant, ~120 lines each)

Both networks take `(B, L) → (B, H)`. Supporting covariates means changing the input
tensor shape and the first layer of each architecture.

### Design — shared

- `_build_windows` gets a multi-channel variant that returns `(N, L, C)` input windows
  and `(N, H)` target windows, where `C = 1 + n_exo`. It reads covariate columns from
  the training frame and stacks them as extra channels. When `n_exo = 0`, `C = 1` and
  the tensor is equivalent to today's `(N, L)` unsqueezed.
- `_TorchDLBase.fit` and `.predict` gain the same exogenous plumbing as
  `_WindowedMLBase`: column extraction, last-value fallback, future-exo builder.
- RevIN instance normalisation applies per-channel: mean/std computed over the time
  axis per channel (shape `(B, 1, C)`).

### Design — N-HiTS

- Each `_NHiTSBlock`: the `AvgPool1d` already operates on a channel dimension
  (input `(B, 1, L)` → `(B, 1, ceil(L/k))`). With `C` channels the pooled tensor
  is `(B, C, ceil(L/k))`; flatten to `(B, C × ceil(L/k))` before the MLP.
  The MLP input size changes from `ceil(L/k)` to `C × ceil(L/k)`.
- Backcast and forecast basis dimensions stay the same — they map to the target
  channel only.
- Future exogenous are not needed inside the block (N-HiTS produces a forecast
  purely from the lookback representation; the original paper has no future-covariate
  pathway). Past covariates in the lookback window are the natural extension.

### Design — PatchTST

- Channel-independent patching (the paper's default): reshape each patch from
  `(patch_len, C)` to `(patch_len × C,)` before embedding.
  `nn.Linear(patch_len, d_model)` becomes `nn.Linear(patch_len × C, d_model)`.
- The rest of the transformer encoder is unchanged (operates on patch-position
  tokens).
- Prediction head unchanged (`n_patches × d_model → H`).

### Changes

**`torchdl.py`**:

- `_build_windows_mc(segs_y, segs_exo, L, H)` → `(N, L, C)`, `(N, H)`.
- `_make_modules()`: accept `n_exo` arg, adjust `NHiTSNet` and `PatchTSTNet`
  constructors to take a `channels` parameter. When `channels = 1` the architecture
  is numerically identical to today's.
- `_TorchDLBase`: add `use_exogenous`, exo-column extraction, last-value fallback.
- Subclasses: `NHITSExoForecaster(NHITSForecaster)`,
  `PatchTSTExoForecaster(PatchTSTForecaster)`.

**`__init__.py`** — register both.

**`test_pipeline_synth.py`** — add to the exogenous-methods test (guarded by
`pytest.importorskip("torch")`).

### Risk

Medium. The architectural changes are local (first layer + pooling reshape), but
any shape mismatch between the channel-aware input and the original single-channel
path will surface as a runtime error. The `channels = 1` equivalence should be
verified by running the existing `test_torch_dl_runs_if_installed` after the
refactor to confirm identical-shape outputs.

---

## Implementation order

1. **Tier 1** — ship first, no risk, immediate value.
2. **Tier 2** — straightforward linear algebra extension; test on a case with
   exogenous columns.
3. **Tier 3** — implement and test separately; most architectural surface area.

## Registration checklist (per new forecaster)

- [ ] Class in its source module
- [ ] Import + `__all__` in `__init__.py`
- [ ] `uses_exogenous` / `capabilities.exogenous` set correctly
- [ ] `limitations` docstring updated
- [ ] Pipeline test (`test_pipeline_synth.py`) covers it
- [ ] Scorecard test (`test_method_scorecards.py`) passes (automatic if inheritance is
      correct)
- [ ] Existing tests still pass (no regression in the univariate path)
