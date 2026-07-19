import { useMemo } from 'react';
import type { Layout, PlotData } from 'plotly.js';
import Plot from '../../plotly';
import type { SplitCounts } from '../../types/data';

// Greyscale, chronological-order encoding — darkest first, since this bar isn't
// comparing method identity like the matrix's colour-blind-safe palette. Port of
// app/snapshot.py's SPLIT_COLORS.
const SPLIT_COLORS: Record<keyof SplitCounts, string> = {
  train: '#000000',
  val: '#888888',
  test: '#cccccc',
};

/** Horizontal stacked bar of the train/val/test proportion of observed points. Port of
 * app/snapshot.py's `render_split_proportions`. */
export function ProportionsBar({ counts, caseId }: { counts: SplitCounts; caseId: string }) {
  const figure = useMemo(() => {
    const total = counts.train + counts.val + counts.test;
    if (total === 0) return null;
    const data: Partial<PlotData>[] = (Object.keys(SPLIT_COLORS) as Array<keyof SplitCounts>).map(
      (split) => {
        const pct = counts[split] / total;
        const label = split.charAt(0).toUpperCase() + split.slice(1);
        return {
          type: 'bar',
          y: [''],
          x: [pct],
          name: label,
          orientation: 'h',
          marker: { color: SPLIT_COLORS[split] },
          text: `${label} ${(pct * 100).toFixed(0)}%`,
          textposition: 'inside',
          insidetextanchor: 'middle',
          textfont: { color: split === 'test' ? 'black' : 'white' },
          hovertemplate: `${split}: %{x:.0%}<extra></extra>`,
        };
      },
    );
    const layout: Partial<Layout> = {
      barmode: 'stack',
      height: 90,
      margin: { l: 0, t: 10, r: 10, b: 10 },
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      showlegend: false,
      font: { family: 'system-ui, sans-serif' },
      xaxis: { visible: false, range: [0, 1] },
      yaxis: { visible: false },
    };
    return { data, layout };
  }, [counts]);

  if (!figure) return <p style={{ color: '#666' }}>No data available for {caseId}.</p>;
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
