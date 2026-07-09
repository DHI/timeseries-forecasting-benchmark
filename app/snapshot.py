"""Dummy forecast-snapshot panel for the results-matrix app.

Renders, for one case x method cell, a row of `N_SNAPSHOTS` small charts each showing the
actual series around one forecast-origin timestamp with the (placeholder) predicted values
overlaid — the interactive analogue of `app/snapshot_example.py`'s static matplotlib figure.

The eval pipeline doesn't persist per-origin forecasts yet (see RESEARCH_LOG.md), so the
actual/predicted values here are deterministic placeholders, seeded by case/method/origin so
repeated clicks are stable. `freq` and `horizon_primary` are the case's real values (from the
case registry), so the time axis and horizon length are accurate even though the series
values aren't.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsforecastbench.data.registry import get_case

N_SNAPSHOTS = 3
N_ACTUAL_POINTS = 300


def _seeded_rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()
    return np.random.default_rng(int(digest[:8], 16))


def _dummy_actual_series(case_id: str, freq: str, n_points: int = N_ACTUAL_POINTS) -> pd.Series:
    """Placeholder actual series for a case, shared across methods (real actuals don't
    depend on which method is being inspected)."""
    rng = _seeded_rng(case_id)
    values = np.sin(np.linspace(0, 4 * np.pi, n_points)) + rng.normal(0, 0.3, n_points)
    index = pd.date_range(start="2023-01-01", periods=n_points, freq=freq)
    return pd.Series(values, index=index)


def _dummy_predicted(
    case_id: str, method: str, start: pd.Timestamp, horizon_steps: int, freq: str
) -> pd.Series:
    rng = _seeded_rng(case_id, method, str(start))
    index = pd.date_range(start=start, periods=horizon_steps, freq=freq)
    values = np.sin(np.linspace(0, 2 * np.pi, horizon_steps)) + rng.normal(0, 0.5, horizon_steps)
    return pd.Series(values, index=index)


def _snapshot_starts(
    actual_index: pd.DatetimeIndex,
    horizon_steps: int,
    n_history: int,
    n_snapshots: int = N_SNAPSHOTS,
) -> list[pd.Timestamp]:
    """`n_snapshots` evenly spaced forecast-origin timestamps, each with room for
    `n_history` steps of history before and `horizon_steps` of predictions after."""
    valid_positions = list(range(n_history, len(actual_index) - horizon_steps))
    if not valid_positions:
        valid_positions = [min(n_history, len(actual_index) - 1)]
    picks = np.linspace(0, len(valid_positions) - 1, n_snapshots).round().astype(int)
    return [actual_index[valid_positions[p]] for p in picks]


def render_snapshots(case_id: str, method: str) -> go.Figure:
    """Plotly panel of `N_SNAPSHOTS` forecast-origin snapshots for one case x method cell."""
    case = get_case(case_id)
    horizon_steps = pd.Timedelta(case.horizon_primary) // pd.Timedelta(case.freq)
    n_history = max(horizon_steps // 4, 1)

    actual = _dummy_actual_series(case_id, case.freq)
    starts = _snapshot_starts(actual.index, horizon_steps, n_history)

    fig = make_subplots(rows=1, cols=len(starts), shared_yaxes=True)
    for i, start in enumerate(starts):
        predicted = _dummy_predicted(case_id, method, start, horizon_steps, case.freq)
        window_start = start - n_history * pd.Timedelta(case.freq)
        window = actual.loc[window_start : predicted.index[-1]]
        show_legend = i == 0

        fig.add_trace(
            go.Scatter(
                x=window.index,
                y=window.values,
                name="Actual",
                mode="lines",
                line={"color": "black"},
                legendgroup="actual",
                showlegend=show_legend,
            ),
            row=1,
            col=i + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=predicted.index,
                y=predicted.values,
                name="Predicted",
                mode="lines",
                line={"color": "orangered"},
                legendgroup="predicted",
                showlegend=show_legend,
            ),
            row=1,
            col=i + 1,
        )
        fig.add_vline(
            x=start,
            line={"color": "gray", "dash": "dash"},
            annotation_text="Prediction start",
            annotation_font={"color": "gray", "size": 10},
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        height=320,
        margin={"l": 40, "t": 20, "r": 20, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig
