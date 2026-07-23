import { useMemo } from 'react';
import type { Layout, PlotData } from 'plotly.js';
import Plot from '../../plotly';
import { useTheme } from '../../theme/ThemeContext';
import { getPlotTheme } from '../../theme/plotTheme';
import type { Series } from '../../types/data';

// Shared height for the case-panel's train/test snapshots and distribution plot — kept
// short so all four case-panel plots fit on screen together without much scrolling. Port
// of app/snapshot.py's CASE_PLOT_HEIGHT.
const CASE_PLOT_HEIGHT = 150;

const AXIS_STYLE = {
  showgrid: false,
  zeroline: false,
  showline: true,
  ticks: 'outside' as const,
};

/** Quick-look line plot for a single split's (ds, y) series — used for both the training
 * and test snapshots, which differ only in what data they're handed. Port of
 * app/snapshot.py's `_line_snapshot`. */
export function LineSnapshot({ series, emptyMessage }: { series: Series; emptyMessage: string }) {
  const { theme } = useTheme();
  const figure = useMemo(() => {
    if (series.ds.length === 0) return null;
    const pt = getPlotTheme(theme === 'dark');
    const data: Partial<PlotData>[] = [
      {
        type: 'scatter',
        x: series.ds,
        y: series.y,
        mode: 'lines',
        line: { color: pt.actualLineColor, width: 1.5 },
      },
    ];
    const layout: Partial<Layout> = {
      autosize: true,
      height: CASE_PLOT_HEIGHT,
      margin: { l: 0, t: 10, r: 10, b: 30 },
      paper_bgcolor: pt.paperBg,
      plot_bgcolor: pt.plotBg,
      font: { family: 'system-ui, sans-serif', color: pt.labelText },
      xaxis: { ...AXIS_STYLE, linecolor: pt.axisLineColor },
      yaxis: { ...AXIS_STYLE, linecolor: pt.axisLineColor, automargin: true },
    };
    return { data, layout };
  }, [series, theme]);

  if (!figure) return <p style={{ color: 'var(--text-muted)' }}>{emptyMessage}</p>;
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
