import { format as d3format } from 'd3-format';
import { useMemo } from 'react';
import type { Annotations, Data, Layout, Margin, PlotData, Shape } from 'plotly.js';
import { CELL, DOMAIN_KEY_DICT, MATRIX_MARGIN } from '../../data/constants';
import { colorscaleFor } from '../../data/colorscale';
import { metricConfig, TRAINING_TIME_CONFIG, type MetricConfig } from '../../data/metricConfig';
import {
  formatSnakeCaseLabel,
  groupBoundaries,
  groupLabels,
  orderedCases,
  orderedMethods,
  pivotValues,
  rowBestMethods,
} from '../../data/selectors';
import { exogenousOverlayShapes, globalOverlayShapes, selectionShapes } from './overlayShapes';
import { getPlotTheme } from '../../theme/plotTheme';
import type { SelectionState } from '../../state/selectionReducer';
import type { ScoreRow } from '../../types/data';

export type MatrixFigure = {
  data: Data[];
  layout: Partial<Layout>;
  rowOrder: string[];
  colOrder: string[];
};

/** Round-number seconds ticks (0, 1, 10, 100, ...) up to the first one at or past
 * `maxSeconds`, positioned in the same log10(seconds + 1) space the training-time
 * overlay colors cells in — so the colorbar reads in real seconds despite the
 * underlying color mapping being log-scaled to cope with training time's heavy tail.
 * Unitless (no "s" suffix): the colorbar's "Train<br>time<br>[seconds]" title already carries
 * the unit. */
function trainingTimeColorTicks(maxSeconds: number): { tickvals: number[]; ticktext: string[] } {
  const candidates = [0, 1, 10, 100, 1000, 10000, 100000];
  const ticks: number[] = [];
  for (const t of candidates) {
    ticks.push(t);
    if (t > maxSeconds) break;
  }
  return { tickvals: ticks.map((t) => Math.log10(t + 1)), ticktext: ticks.map((t) => `${t}`) };
}

/** 1-2-5 log-spaced ticks across [min, max] (raw, unscaled values), displayed at `scale`
 * (matching how cell text is scaled — see metricConfig.ts's `scale`). Unlike cell text's
 * fixed decimal count, tick labels drop decimals entirely once the displayed value passes
 * 1, so a heavy-tailed range doesn't force needless ticks like "1000.00"/"2000.00" — e.g.
 * for MAE (scale 1) this yields ...0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, ..., 500, 1000. Used for
 * metrics whose color (not cell text) pivots on log10(value) — see `colorScale: 'log'` in
 * metricConfig.ts. */
function niceLogTicks(
  min: number,
  max: number,
  scale: number,
): { tickvals: number[]; ticktext: string[] } {
  if (!(min > 0) || !(max > 0) || min >= max) return { tickvals: [], ticktext: [] };
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const scaleExp = Math.round(Math.log10(scale));
  const tickvals: number[] = [];
  const ticktext: string[] = [];
  // A tolerance guards the boundary check: `m * 10 ** e` and `min`/`max` are computed via
  // different floating-point paths (exponentiation vs. a data min/max), so a candidate that
  // is mathematically exactly `min` can come out a hair below it and get skipped.
  const tolerance = 1e-9;
  for (let e = lo; e <= hi; e++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** e;
      if (v < min * (1 - tolerance) || v > max * (1 + tolerance)) continue;
      const displayExp = e + scaleExp;
      const decimals = Math.max(0, -displayExp);
      tickvals.push(Math.log10(v));
      ticktext.push((m * 10 ** displayExp).toFixed(decimals));
    }
  }
  return { tickvals, ticktext };
}

/** Colorbar title: "Skill<br>vs<br>X" for Skill_vs_X metrics (narrower than the full metric
 * name), otherwise "MetricName<br>[unit]". Plotly text only breaks on `<br>`, not `\n`. */
function colorbarTitle(metric: string, cfg: MetricConfig): string {
  if (metric.startsWith('Skill_vs_')) {
    return `Skill<br>vs<br>${metric.slice('Skill_vs_'.length)}`;
  }
  return `${metric}<br>[${cfg.unit}]`;
}

/** Builds the case x method heatmap figure for one metric. `rows` must be filtered only by
 * metric/case-eligibility/family (matching app/app.py's `df_metric`) — NOT by
 * `used_extended_history`, since the extended-history rows a case may have are exactly
 * what `showGlobal` swaps in. Port of app/app.py's `render_matrix`. */
export function useMatrixFigure(
  rows: ScoreRow[],
  metric: string,
  selection: SelectionState,
  showGlobal: boolean,
  showExo: boolean,
  showTrainingTime: boolean,
  isDark: boolean = false,
): MatrixFigure | null {
  return useMemo(() => {
    if (!metric || rows.length === 0) return null;
    const pt = getPlotTheme(isDark);

    // When the training-time overlay is held, every cell pivots on `seconds` instead of
    // `value` — including the showGlobal/showExo swap-in cells below — so the whole
    // matrix temporarily shows timing instead of the selected metric's score.
    const valueOf = showTrainingTime ? (r: ScoreRow) => r.seconds : (r: ScoreRow) => r.value;

    const dfLocal = rows.filter((r) => !r.used_extended_history);
    const rowOrder = orderedCases(dfLocal);
    const colOrder = orderedMethods(dfLocal);
    if (rowOrder.length === 0 || colOrder.length === 0) return null;

    const values = pivotValues(dfLocal, rowOrder, colOrder, valueOf);
    const cfg = showTrainingTime ? TRAINING_TIME_CONFIG : metricConfig(metric);
    const fmt = d3format(cfg.format);

    const domainOf = new Map<string, string>();
    const familyOf = new Map<string, string>();
    for (const row of dfLocal) {
      domainOf.set(row.case_id, row.domain);
      familyOf.set(row.method, row.family);
    }

    // Every method has a `used_extended_history=false` ("local") row on the shared
    // curated span, plotted by default. `is_global` methods additionally get a
    // `used_extended_history=true` row wherever the case has real history beyond the
    // curated span. When `showGlobal` is on, those cells are swapped in for the local
    // score and marked with a corner triangle — they're fit on more data than the rest
    // of the row and aren't directly comparable. Port of app/app.py's `show_global`
    // branch (render_matrix lines 262-273).
    const globalOverlayCells: Array<[string, string]> = [];
    if (showGlobal) {
      for (const row of rows) {
        if (!row.used_extended_history || !row.is_global) continue;
        const r = rowOrder.indexOf(row.case_id);
        const c = colOrder.indexOf(row.method);
        const v = valueOf(row);
        if (r === -1 || c === -1 || v === null) continue;
        values[r]![c] = v;
        globalOverlayCells.push([row.case_id, row.method]);
      }
    }

    // A method with an exogenous-aware sibling (e.g. "LightGBM" / "LightGBM+exo") is
    // shown as a single column. When `showExo` is on, cases with real exogenous data
    // (case_exo_tier != "none") get that sibling's score swapped in and marked with a
    // teal corner. Port of app/app.py's `show_exo` branch (render_matrix lines 275-293).
    const exoOverlayCells: Array<[string, string]> = [];
    if (showExo) {
      const localMethods = new Set(dfLocal.map((r) => r.method));
      const exoPairs = new Map(
        colOrder
          .filter((m) => localMethods.has(`${m}+exo`))
          .map((m): [string, string] => [m, `${m}+exo`]),
      );
      if (exoPairs.size > 0) {
        const tierOf = new Map<string, string>();
        for (const row of rows) tierOf.set(row.case_id, row.case_exo_tier);
        const exoMethodNames = new Set(exoPairs.values());
        for (const row of dfLocal) {
          if (!exoMethodNames.has(row.method)) continue;
          const base = row.method.slice(0, -'+exo'.length);
          const r = rowOrder.indexOf(row.case_id);
          const c = colOrder.indexOf(base);
          const v = valueOf(row);
          if (r === -1 || c === -1 || v === null) continue;
          if ((tierOf.get(row.case_id) ?? 'none') === 'none') continue;
          values[r]![c] = v;
          exoOverlayCells.push([row.case_id, base]);
        }
      }
    }

    // Training time is heavy-tailed (most methods are fast, a few much slower), so cell
    // color pivots on log10(seconds + 1) rather than the raw seconds `values` used for
    // cell text/ranking below — a linear scale would leave almost every cell looking
    // identical. The colorbar ticks further down relabel this log space back to real
    // seconds. Several metrics (MAE/RMSE/MAPE/sMAPE) are heavy-tailed the same way —
    // `colorScale: 'log'` in metricConfig.ts opts them into an analogous log10(value) color
    // mapping (floored at the smallest positive value present, so an exact 0 doesn't map to
    // -Infinity).
    const isLogScale = !showTrainingTime && cfg.colorScale === 'log';
    const positiveValues = values.flat().filter((v) => Number.isFinite(v) && v > 0);
    const logFloor = positiveValues.length ? Math.min(...positiveValues) : Number.EPSILON;
    const logMax = positiveValues.length ? Math.max(...positiveValues) : 1;
    const colorZ = showTrainingTime
      ? values.map((row) => row.map((v) => (Number.isNaN(v) ? NaN : Math.log10(v + 1))))
      : isLogScale
        ? values.map((row) =>
            row.map((v) => (Number.isNaN(v) ? NaN : Math.log10(Math.max(v, logFloor)))),
          )
        : values;

    // Greys runs from white (t=0) to black (t=1), same as Plotly's own zmin/zmax mapping
    // (cfg.center is null for every Greys-cmap config today, so Plotly auto-ranges to this
    // same colorZ min/max) — once a cell's background crosses mid-grey, default dark
    // annotation text stops being readable and needs to switch to a light color instead.
    const isGreyscale = cfg.cmap === 'Greys';
    const finiteColorZ = colorZ.flat().filter((v) => Number.isFinite(v));
    const colorZMin = finiteColorZ.length ? Math.min(...finiteColorZ) : 0;
    const colorZMax = finiteColorZ.length ? Math.max(...finiteColorZ) : 1;

    // No "s" suffix on training-time cells: the colorbar title already labels the unit.
    // `scale` lets a metric's cell text be a plain number (e.g. MAPE's `12`) while the unit
    // itself (e.g. "%") is stated once, on the colorbar, instead of repeated in every cell.
    const scale = cfg.scale ?? 1;
    const text = values.map((rowValues) =>
      rowValues.map((v) => (Number.isNaN(v) ? '—' : fmt(v * scale))),
    );
    // The overlay is a quick glance at timing, not a ranking — skip the fastest-method
    // bolding that normally highlights the best-scoring method per row.
    const bestOfRow = showTrainingTime
      ? new Map<string, string>()
      : rowBestMethods(rowOrder, colOrder, values, cfg.lowerIsBetter);

    const annotations: Partial<Annotations>[] = [];
    rowOrder.forEach((caseId, r) => {
      const bestMethod = bestOfRow.get(caseId);
      colOrder.forEach((method, c) => {
        const isBest = method === bestMethod;
        const t =
          isGreyscale && colorZMax > colorZMin
            ? (colorZ[r]![c]! - colorZMin) / (colorZMax - colorZMin)
            : NaN;
        const isDarkCell = Number.isFinite(t) && t > 0.5;
        annotations.push({
          x: method,
          y: caseId,
          xref: 'x',
          yref: 'y',
          text: isBest ? `<b>${text[r]![c]}</b>` : text[r]![c],
          showarrow: false,
          font: { size: isBest ? 13 : 10, color: isDarkCell ? '#f5f5f5' : undefined },
        });
      });
    });

    const nRows = rowOrder.length;
    const nCols = colOrder.length;
    const boundaryLine = { color: pt.boundaryLineColor, width: 4 };
    const shapes: Partial<Shape>[] = [
      ...groupBoundaries(rowOrder, domainOf).map((y): Partial<Shape> => ({
        type: 'line',
        x0: -0.5,
        x1: nCols - 0.5,
        y0: y,
        y1: y,
        line: boundaryLine,
      })),
      ...groupBoundaries(colOrder, familyOf).map((x): Partial<Shape> => ({
        type: 'line',
        x0: x,
        x1: x,
        y0: -0.5,
        y1: nRows - 0.5,
        line: boundaryLine,
      })),
      ...globalOverlayShapes(globalOverlayCells, rowOrder, colOrder),
      ...exogenousOverlayShapes(exoOverlayCells, rowOrder, colOrder),
      ...selectionShapes(selection, rowOrder, colOrder),
    ];

    for (const [pos, label] of groupLabels(colOrder, familyOf)) {
      annotations.push({
        x: pos,
        y: -0.5,
        xref: 'x',
        yref: 'y',
        yanchor: 'bottom',
        yshift: 6,
        text: `<b>${formatSnakeCaseLabel(label, '<br>')}</b>`,
        showarrow: false,
        font: { size: 13, color: pt.labelText },
      });
    }
    for (const [pos, label] of groupLabels(rowOrder, domainOf)) {
      annotations.push({
        x: -0.5,
        y: pos,
        xref: 'x',
        yref: 'y',
        xanchor: 'right',
        xshift: -6,
        text: `<b>${DOMAIN_KEY_DICT[label] ?? label}</b>`,
        showarrow: false,
        font: { size: 13, color: pt.labelText },
        textangle: '-90',
      });
    }

    // @types/plotly.js is missing `zmid` (a real, documented plotly.js heatmap option —
    // just absent from this package's typings), hence the inline extension below.
    const heatmap: Partial<PlotData> & { zmid?: number } = {
      type: 'heatmap',
      z: colorZ,
      x: colOrder,
      y: rowOrder,
      colorscale: colorscaleFor(cfg),
      // Native per-cell hover is off; MatrixGraph draws its own row/column tooltip
      // overlay instead (see HoverTooltips), matching app/app.py's clientside callback.
      hoverinfo: 'none',
      colorbar: {
        // "Train<br>time<br>[seconds]" is no wider than colorbarTitle()'s longest line
        // ("[target unit]"), so it still fits within margin.r below (see the
        // margin.autoexpand note further down for why that matters).
        title: {
          text: showTrainingTime ? 'Train<br>time<br>[seconds]' : colorbarTitle(metric, cfg),
        },
        len: 0.6,
        thickness: 16,
        x: 1.0,
        xanchor: 'left',
        ...(showTrainingTime
          ? {
              tickmode: 'array' as const,
              ...trainingTimeColorTicks(
                values.flat().reduce((m, v) => (Number.isFinite(v) ? Math.max(m, v) : m), 0),
              ),
            }
          : isLogScale
            ? {
                tickmode: 'array' as const,
                ...niceLogTicks(logFloor, logMax, scale),
              }
            : {}),
      },
    };
    if (cfg.center !== null) {
      const finite = values.flat().filter((v) => Number.isFinite(v));
      const lo = cfg.vlo ?? (finite.length ? Math.min(...finite) : 0);
      const hi = cfg.vhi ?? (finite.length ? Math.max(...finite) : 1);
      heatmap.zmin = Math.min(lo, cfg.center);
      heatmap.zmax = Math.max(hi, cfg.center);
      heatmap.zmid = cfg.center;
    }

    // @types/plotly.js is missing `margin.autoexpand` (a real, documented plotly.js
    // option — just absent from this package's typings, like `zmid` above). Without it,
    // Plotly lets a wide colorbar title grow past this fixed margin, silently shrinking
    // the plot area within the figure's fixed width/height — which is what made the
    // training-time overlay's grid look narrower than the normal metric view. Disabling
    // it guarantees the heatmap cells occupy exactly the same pixels either way. `r` is
    // sized to fit the colorbar title's widest possible second line ("[target unit]"),
    // left-anchored at the colorbar's own x — a plain 80px clipped it mid-word.
    const margin: Partial<Margin> & { autoexpand?: boolean } = {
      l: 90,
      t: 40,
      r: 120,
      b: 80,
      autoexpand: false,
    };

    const layout: Partial<Layout> = {
      annotations,
      shapes,
      height: CELL * nRows + MATRIX_MARGIN,
      // Derived from margin.l/margin.r (not a separate constant) so the two can never drift
      // out of sync — the heatmap's pixel area is exactly width - (margin.l + margin.r).
      width: CELL * nCols + (margin.l ?? 0) + (margin.r ?? 0),
      margin,
      paper_bgcolor: pt.paperBg,
      plot_bgcolor: pt.plotBg,
      font: { color: pt.labelText },
      dragmode: false,
      xaxis: {
        side: 'bottom',
        tickangle: -35,
        constrain: 'domain',
        fixedrange: true,
        showticklabels: false,
        showgrid: false,
        ticks: '',
      },
      yaxis: {
        autorange: 'reversed',
        scaleanchor: 'x',
        scaleratio: 1,
        fixedrange: true,
        showticklabels: false,
        showgrid: false,
        ticks: '',
      },
    };

    return { data: [heatmap as Data], layout, rowOrder, colOrder };
  }, [rows, metric, selection, showGlobal, showExo, showTrainingTime, isDark]);
}
