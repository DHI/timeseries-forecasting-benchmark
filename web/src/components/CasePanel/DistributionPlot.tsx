import { useMemo } from 'react';
import type { Layout, PlotData } from 'plotly.js';
import Plot from '../../plotly';
import type { Distribution } from '../../types/data';

const CASE_PLOT_HEIGHT = 150;

const AXIS_STYLE = {
  showgrid: false,
  zeroline: false,
  showline: true,
  linecolor: 'black',
  ticks: 'outside' as const,
};

/** Train vs test value distributions for one case, as KDE lines (precomputed by the
 * export script) rather than overlaid histogram bars — two curves on the same axes are
 * easier to compare than two sets of overlapping bars. Port of app/snapshot.py's
 * `render_split_distribution`. */
export function DistributionPlot({
  distribution,
  caseId,
}: {
  distribution: Distribution;
  caseId: string;
}) {
  const figure = useMemo(() => {
    if (distribution.x_grid.length === 0) return { empty: `No data available for ${caseId}.` };
    if (!distribution.train_density && !distribution.test_density) {
      return { empty: `Not enough variance to estimate a distribution for ${caseId}.` };
    }
    const data: Partial<PlotData>[] = [];
    if (distribution.train_density) {
      data.push({
        type: 'scatter',
        x: distribution.x_grid,
        y: distribution.train_density,
        name: 'Train',
        mode: 'lines',
        line: { color: '#000000', width: 2 },
      });
    }
    if (distribution.test_density) {
      data.push({
        type: 'scatter',
        x: distribution.x_grid,
        y: distribution.test_density,
        name: 'Test',
        mode: 'lines',
        line: { color: '#888888', width: 2, dash: 'dash' },
      });
    }
    const layout: Partial<Layout> = {
      autosize: true,
      height: CASE_PLOT_HEIGHT,
      margin: { l: 0, t: 10, r: 10, b: 30 },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      font: { family: 'system-ui, sans-serif' },
      xaxis: { ...AXIS_STYLE },
      yaxis: { ...AXIS_STYLE, showticklabels: false },
      legend: {
        orientation: 'h',
        yanchor: 'bottom',
        y: 1.02,
        xanchor: 'left',
        x: 0,
        bgcolor: 'rgba(0,0,0,0)',
        borderwidth: 0,
      },
    };
    return { data, layout };
  }, [distribution, caseId]);

  if ('empty' in figure) return <p style={{ color: '#666' }}>{figure.empty}</p>;
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
