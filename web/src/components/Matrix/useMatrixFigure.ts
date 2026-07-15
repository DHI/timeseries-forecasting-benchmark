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
import type { ScoreRow } from '../../types/data';

export type MatrixFigure = {
  data: Data[];
  layout: Partial<Layout>;
  rowOrder: string[];
  colOrder: string[];
};

/** Builds the case x method heatmap figure for one metric, restricted to `rows` (already
 * filtered by domain/family/metric-eligibility). Port of app/app.py's `render_matrix`,
 * minus the extended-history/exogenous value swaps, corner overlays, and click-selection
 * outlines — those depend on interactive state added in a later milestone. */
export function useMatrixFigure(rows: ScoreRow[], metric: string): MatrixFigure | null {
  return useMemo(() => {
    if (!metric || rows.length === 0) return null;

    const rowOrder = orderedCases(rows);
    const colOrder = orderedMethods(rows);
    if (rowOrder.length === 0 || colOrder.length === 0) return null;

    const values = pivotValues(rows, rowOrder, colOrder);
    const cfg = metricConfig(metric);
    const fmt = d3format(cfg.format);
    const text = values.map((rowValues) => rowValues.map((v) => (Number.isNaN(v) ? '—' : fmt(v))));

    const domainOf = new Map<string, string>();
    const familyOf = new Map<string, string>();
    for (const row of rows) {
      domainOf.set(row.case_id, row.domain);
      familyOf.set(row.method, row.family);
    }

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
      hoverinfo: 'x+y+z',
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
  }, [rows, metric]);
}
