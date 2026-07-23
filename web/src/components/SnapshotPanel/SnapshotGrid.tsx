import { format as d3format } from 'd3-format';
import { useMemo } from 'react';
import type { Annotations, AxisName, Data, Layout, LayoutAxis, Shape } from 'plotly.js';
import Plot from '../../plotly';
import { METHOD_COLORS } from '../../data/constants';
import { freqToMs } from '../../data/freq';
import { useTheme } from '../../theme/ThemeContext';
import { getPlotTheme } from '../../theme/plotTheme';
import { snapshotPanelLayout } from './snapshotLayout';
import type { CaseMeta, PredictionVariant, SnapshotBundle } from '../../types/data';

const AXIS_STYLE = {
  showgrid: false,
  zeroline: false,
  showline: true,
  ticks: 'outside' as const,
};

const X_AXIS_STYLE = {
  ...AXIS_STYLE,
  tickangle: 0 as const,
  tickformat: '%b %d, %H:%M',
  nticks: 4,
};

function effectiveVariant(
  bundle: SnapshotBundle,
  method: string,
  showGlobal: boolean,
): PredictionVariant {
  return showGlobal && bundle.predictions[method]?.extended ? 'extended' : 'local';
}

type Props = {
  caseId: string;
  caseMeta: CaseMeta;
  bundle: SnapshotBundle;
  effectiveMethods: string[];
  metric: string;
  showGlobal: boolean;
};

/** Grid of curated forecast-origin snapshots for one case, overlaying each of
 * `effectiveMethods` (already resolved to their "+exo" sibling where the exogenous
 * toggle applies). Stacked in a single column (`maxCols = 1`), matching the narrow
 * detail-column layout `app/app.py`'s `render_panel` actually uses. Port of
 * app/snapshot.py's `render_snapshots`, consuming the export step's precomputed
 * curation/predictions instead of re-deriving them. */
export function SnapshotGrid({
  caseId,
  caseMeta,
  bundle,
  effectiveMethods,
  metric,
  showGlobal,
}: Props) {
  const { theme } = useTheme();
  const figure = useMemo(() => {
    if (effectiveMethods.length === 0) return null;
    const anchor = effectiveMethods[0]!;
    const anchorVariant = effectiveVariant(bundle, anchor, showGlobal);
    const curated = bundle.curation[metric]?.[anchor]?.[anchorVariant] ?? [];
    if (curated.length === 0) {
      return { empty: `No observed actuals to score any stored origin for ${caseId}.` };
    }

    const pt = getPlotTheme(theme === 'dark');
    const step = freqToMs(caseMeta.freq);
    const originById = new Map(bundle.origins.map((o) => [o.origin_id, o]));
    const panels = snapshotPanelLayout(curated.length, 1);
    const colorOf = new Map(
      effectiveMethods.map((m, i) => [m, METHOD_COLORS[i % METHOD_COLORS.length]]),
    );
    const fmt = d3format('.3~g');

    const data: Data[] = [];
    const shapes: Partial<Shape>[] = [];
    const annotations: Partial<Annotations>[] = [];
    const axisLayout: Record<string, Partial<LayoutAxis>> = {};

    curated.forEach((entry, i) => {
      const origin = originById.get(entry.origin_id);
      const axes = panels[i]!;
      if (!origin) return;
      const originMs = Date.parse(origin.origin_ds);
      const nHistory = bundle.n_history;
      const timestamps = Array.from({ length: nHistory + bundle.horizon_steps }, (_, k) =>
        new Date(originMs + (k - nHistory) * step).toISOString(),
      );
      const actualValues = bundle.actual[String(entry.origin_id)] ?? [];

      data.push({
        type: 'scatter',
        mode: 'lines',
        x: timestamps,
        y: actualValues,
        name: 'Actual',
        line: { color: pt.actualLineColor, width: 1 },
        xaxis: axes.xaxisRef,
        yaxis: axes.yaxisRef,
      } as Data);

      for (const method of effectiveMethods) {
        const variant = effectiveVariant(bundle, method, showGlobal);
        const predValues = bundle.predictions[method]?.[variant]?.[String(entry.origin_id)] ?? [];
        const predTimestamps = timestamps.slice(nHistory, nHistory + predValues.length);
        const label = variant === 'extended' ? `${method} (global fit)` : method;
        data.push({
          type: 'scatter',
          mode: 'lines',
          x: predTimestamps,
          y: predValues,
          name: label,
          line: { color: colorOf.get(method), width: 2 },
          xaxis: axes.xaxisRef,
          yaxis: axes.yaxisRef,
        } as Data);
      }

      const originIso = new Date(originMs).toISOString();
      shapes.push({
        type: 'line',
        xref: axes.xaxisRef as Shape['xref'],
        yref: `${axes.yaxisRef} domain` as Shape['yref'],
        x0: originIso,
        x1: originIso,
        y0: 0,
        y1: 1,
        line: { color: pt.mutedLineColor, dash: 'dash', width: 1 },
      });

      annotations.push({
        xref: 'paper',
        yref: 'paper',
        x: axes.titleX,
        y: axes.titleY,
        yanchor: 'bottom',
        yshift: 4,
        text: `${entry.label} · (${metric} ${fmt(entry.value)})`,
        showarrow: false,
        font: { size: 12, color: pt.labelText },
      });

      axisLayout[axes.xaxisKey] = {
        domain: axes.xDomain,
        anchor: axes.yaxisRef as AxisName,
        ...X_AXIS_STYLE,
        linecolor: pt.axisLineColor,
      };
      axisLayout[axes.yaxisKey] = {
        domain: axes.yDomain,
        anchor: axes.xaxisRef as AxisName,
        ...AXIS_STYLE,
        linecolor: pt.axisLineColor,
      };
    });

    const layout: Partial<Layout> = {
      ...(axisLayout as Partial<Layout>),
      autosize: true,
      height: 220 * curated.length + 30,
      margin: { l: 50, t: 30, r: 20, b: 40 },
      paper_bgcolor: pt.paperBg,
      plot_bgcolor: pt.plotBg,
      font: { family: 'system-ui, sans-serif' },
      annotations,
      shapes,
      showlegend: false,
    };

    return { data, layout };
  }, [bundle, effectiveMethods, metric, showGlobal, caseId, caseMeta.freq, theme]);

  if (!figure) return null;
  if ('empty' in figure) return <p style={{ color: 'var(--text-muted)' }}>{figure.empty}</p>;
  return (
    <Plot
      data={figure.data}
      layout={figure.layout}
      config={{ displayModeBar: false }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}
