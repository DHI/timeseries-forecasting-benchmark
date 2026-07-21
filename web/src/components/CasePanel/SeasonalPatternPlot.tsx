import { useMemo } from 'react';
import type { Annotations, Layout, PlotData } from 'plotly.js';
import Plot from '../../plotly';
import type { SeasonalPattern } from '../../types/data';

const ROW_HEIGHT = 190;
const N_COLS = 2;
const H_GAP = 0.08;
const V_GAP = 0.22;
const TITLE_BAND = 0.1;

const AXIS_STYLE = {
  showgrid: false,
  zeroline: false,
  showline: true,
  linecolor: 'black',
  ticks: 'outside' as const,
  automargin: true,
};

/** One subplot's domain (as fractions of the whole figure) for a grid position, leaving a
 * thin band above the plot area for its "Period = N steps" title annotation. */
function subplotDomain(index: number, nCols: number, nRows: number) {
  const row = Math.floor(index / nCols);
  const col = index % nCols;
  const colWidth = (1 - (nCols - 1) * H_GAP) / nCols;
  const rowHeight = (1 - (nRows - 1) * V_GAP) / nRows;
  const x0 = col * (colWidth + H_GAP);
  const rowTop = 1 - row * (rowHeight + V_GAP);
  const rowBottom = rowTop - rowHeight;
  return {
    x: [x0, x0 + colWidth] as [number, number],
    y: [rowBottom, rowTop - TITLE_BAND] as [number, number],
    titleY: rowTop,
    col,
  };
}

/** Per-seasonality scatter of the training tail against its position in the cycle, with a
 * LOESS curve overlaid to show the seasonal shape — one subplot per declared period, 2
 * columns wrapping to more rows as needed, sharing a single y axis so amplitude is
 * comparable across periods. Data (scatter + curve) is precomputed server-side; see
 * `export_web_data.py`'s `_seasonal_patterns`. */
export function SeasonalPatternPlot({ patterns }: { patterns: SeasonalPattern[] }) {
  const figure = useMemo(() => {
    if (patterns.length === 0) return null;
    const nCols = Math.min(N_COLS, patterns.length);
    const nRows = Math.ceil(patterns.length / nCols);

    const data: Partial<PlotData>[] = [];
    const layout: Partial<Layout> = {
      autosize: true,
      height: ROW_HEIGHT * nRows,
      margin: { l: 0, t: 10, r: 10, b: 30 },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      font: { family: 'system-ui, sans-serif' },
      showlegend: false,
      annotations: [],
    };
    const annotations: Partial<Annotations>[] = [];

    patterns.forEach((pattern, i) => {
      const suffix = i === 0 ? '' : String(i + 1);
      const { x: xDomain, y: yDomain, titleY, col } = subplotDomain(i, nCols, nRows);

      data.push({
        type: 'scatter',
        x: pattern.x,
        y: pattern.y,
        mode: 'markers',
        marker: { color: 'rgba(0,0,0,0.15)', size: 4 },
        xaxis: `x${suffix}`,
        yaxis: `y${suffix}`,
        hoverinfo: 'skip',
      });
      data.push({
        type: 'scatter',
        x: pattern.curve_x,
        y: pattern.curve_y,
        mode: 'lines',
        line: { color: 'black', width: 2 },
        xaxis: `x${suffix}`,
        yaxis: `y${suffix}`,
        hoverinfo: 'skip',
      });

      layout[`xaxis${suffix}` as 'xaxis'] = { ...AXIS_STYLE, domain: xDomain };
      layout[`yaxis${suffix}` as 'yaxis'] = {
        ...AXIS_STYLE,
        domain: yDomain,
        showticklabels: col === 0,
        ...(i > 0 ? { matches: 'y' } : {}),
      };

      annotations.push({
        text: `Period = ${pattern.period} steps`,
        xref: 'paper',
        yref: 'paper',
        x: (xDomain[0] + xDomain[1]) / 2,
        y: titleY,
        xanchor: 'center',
        yanchor: 'top',
        showarrow: false,
        font: { size: 11, color: '#444444' },
      });
    });

    layout.annotations = annotations;
    return { data, layout };
  }, [patterns]);

  if (!figure) return null;
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
