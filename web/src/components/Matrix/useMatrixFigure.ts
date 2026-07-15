import { format as d3format } from 'd3-format';
import { useMemo } from 'react';
import type { Annotations, Data, Layout, PlotData, Shape } from 'plotly.js';
import { CELL, DOMAIN_KEY_DICT, MATRIX_MARGIN } from '../../data/constants';
import { colorscaleFor } from '../../data/colorscale';
import { metricConfig } from '../../data/metricConfig';
import {
  groupBoundaries,
  groupLabels,
  orderedCases,
  orderedMethods,
  pivotValues,
  rowBestMethods,
} from '../../data/selectors';
import { exogenousOverlayShapes, globalOverlayShapes, selectionShapes } from './overlayShapes';
import type { SelectionState } from '../../state/selectionReducer';
import type { ScoreRow } from '../../types/data';

export type MatrixFigure = {
  data: Data[];
  layout: Partial<Layout>;
  rowOrder: string[];
  colOrder: string[];
};

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
): MatrixFigure | null {
  return useMemo(() => {
    if (!metric || rows.length === 0) return null;

    const dfLocal = rows.filter((r) => !r.used_extended_history);
    const rowOrder = orderedCases(dfLocal);
    const colOrder = orderedMethods(dfLocal);
    if (rowOrder.length === 0 || colOrder.length === 0) return null;

    const values = pivotValues(dfLocal, rowOrder, colOrder);
    const cfg = metricConfig(metric);
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
        if (r === -1 || c === -1 || row.value === null) continue;
        values[r]![c] = row.value;
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
          if (r === -1 || c === -1 || row.value === null) continue;
          if ((tierOf.get(row.case_id) ?? 'none') === 'none') continue;
          values[r]![c] = row.value;
          exoOverlayCells.push([row.case_id, base]);
        }
      }
    }

    const text = values.map((rowValues) => rowValues.map((v) => (Number.isNaN(v) ? '—' : fmt(v))));
    const bestOfRow = rowBestMethods(rowOrder, colOrder, values, cfg.lowerIsBetter);

    const annotations: Partial<Annotations>[] = [];
    rowOrder.forEach((caseId, r) => {
      const bestMethod = bestOfRow.get(caseId);
      colOrder.forEach((method, c) => {
        const isBest = method === bestMethod;
        annotations.push({
          x: method,
          y: caseId,
          xref: 'x',
          yref: 'y',
          text: isBest ? `<b>${text[r]![c]}</b>` : text[r]![c],
          showarrow: false,
          font: { size: isBest ? 13 : 10 },
        });
      });
    });

    const nRows = rowOrder.length;
    const nCols = colOrder.length;
    const boundaryLine = { color: 'white', width: 4 };
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
        text: `<b>${label.replace(/_/g, '<br>').replace(/^./, (c) => c.toUpperCase())}</b>`,
        showarrow: false,
        font: { size: 13 },
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
        font: { size: 13 },
        textangle: '-90',
      });
    }

    // @types/plotly.js is missing `zmid` (a real, documented plotly.js heatmap option —
    // just absent from this package's typings), hence the inline extension below.
    const heatmap: Partial<PlotData> & { zmid?: number } = {
      type: 'heatmap',
      z: values,
      x: colOrder,
      y: rowOrder,
      colorscale: colorscaleFor(cfg),
      // Native per-cell hover is off; MatrixGraph draws its own row/column tooltip
      // overlay instead (see HoverTooltips), matching app/app.py's clientside callback.
      hoverinfo: 'none',
      colorbar: {
        title: { text: metric.replace(/_/g, '\n') },
        len: 0.6,
        thickness: 16,
        x: 1.0,
        xanchor: 'left',
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

    const layout: Partial<Layout> = {
      annotations,
      shapes,
      height: CELL * nRows + MATRIX_MARGIN,
      width: CELL * nCols + 170,
      margin: { l: 90, t: 40, r: 80, b: 80 },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      dragmode: false,
      xaxis: {
        side: 'bottom',
        tickangle: -35,
        constrain: 'domain',
        fixedrange: true,
        showticklabels: false,
      },
      yaxis: {
        autorange: 'reversed',
        scaleanchor: 'x',
        scaleratio: 1,
        fixedrange: true,
        showticklabels: false,
      },
    };

    return { data: [heatmap as Data], layout, rowOrder, colOrder };
  }, [rows, metric, selection, showGlobal, showExo]);
}
