import type { Config } from 'plotly.js';
import Plot from '../../plotly';
import { useMatrixFigure } from './useMatrixFigure';
import type { ScoreRow } from '../../types/data';

// Locks the heatmap down to a read-only view: keeps hover, but strips every pan/zoom/
// select tool from the mode bar so the grid can't be dragged, scaled, or rearranged.
// Port of app/app.py's _MATRIX_CONFIG.
const MATRIX_CONFIG: Partial<Config> = {
  displayModeBar: true,
  modeBarButtonsToRemove: [
    'zoom2d',
    'pan2d',
    'select2d',
    'lasso2d',
    'zoomIn2d',
    'zoomOut2d',
    'autoScale2d',
    'resetScale2d',
  ],
  displaylogo: false,
  scrollZoom: false,
  doubleClick: false,
};

type Props = {
  rows: ScoreRow[];
  metric: string;
};

export function MatrixGraph({ rows, metric }: Props) {
  const figure = useMatrixFigure(rows, metric);

  if (!figure) {
    return <p>Select a metric, at least one domain, and one method family.</p>;
  }

  return (
    <Plot
      data={figure.data}
      layout={figure.layout}
      config={MATRIX_CONFIG}
      style={{ width: `${figure.layout.width}px`, height: `${figure.layout.height}px` }}
    />
  );
}
