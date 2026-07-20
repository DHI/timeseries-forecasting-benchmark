import { useCallback, useRef, useState } from 'react';
import type { Config } from 'plotly.js';
import Plot from '../../plotly';
import { computeHoverOverlay } from './hoverGeometry';
import type { HoverOverlay, PlotlyGraphDiv } from './hoverGeometry';
import { HoverTooltips } from './HoverTooltips';
import { useMatrixFigure } from './useMatrixFigure';
import type { SelectionState } from '../../state/selectionReducer';
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

type HoverPoint = { x?: unknown; y?: unknown };
type PlotMouseEvent = { points?: HoverPoint[] };

type Props = {
  rows: ScoreRow[];
  metric: string;
  selection: SelectionState;
  showGlobal: boolean;
  showExogenous: boolean;
  showTrainingTime: boolean;
  onCellClick: (caseId: string, method: string) => void;
};

export function MatrixGraph({
  rows,
  metric,
  selection,
  showGlobal,
  showExogenous,
  showTrainingTime,
  onCellClick,
}: Props) {
  const figure = useMatrixFigure(
    rows,
    metric,
    selection,
    showGlobal,
    showExogenous,
    showTrainingTime,
  );
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<HTMLDivElement | null>(null);
  const [overlay, setOverlay] = useState<HoverOverlay | null>(null);

  const handleHover = useCallback(
    (event: PlotMouseEvent) => {
      const pt = event.points?.[0];
      const wrapper = wrapperRef.current;
      const gd = graphRef.current as PlotlyGraphDiv | null;
      if (!pt || !wrapper || !gd || !figure) return;
      setOverlay(
        computeHoverOverlay(
          String(pt.x),
          String(pt.y),
          figure.colOrder.length,
          figure.rowOrder.length,
          wrapper,
          gd,
        ),
      );
    },
    [figure],
  );

  const handleUnhover = useCallback(() => setOverlay(null), []);

  const handleClick = useCallback(
    (event: PlotMouseEvent) => {
      const pt = event.points?.[0];
      if (!pt) return;
      // Reads the clicked cell's case/method off x/y rather than customdata: Dash's
      // clickData only reconstructs customdata for traces indexed by a flat
      // pointNumber, which a Heatmap never sets — the same is true of plotly.js's own
      // click event, so this reads the category labels directly either way.
      onCellClick(String(pt.y), String(pt.x));
    },
    [onCellClick],
  );

  if (!figure) {
    return <p>Select a metric, at least one domain, and one method family.</p>;
  }

  return (
    <div ref={wrapperRef} style={{ position: 'relative' }}>
      <Plot
        ref={graphRef}
        data={figure.data}
        layout={figure.layout}
        config={MATRIX_CONFIG}
        style={{ width: `${figure.layout.width}px`, height: `${figure.layout.height}px` }}
        onHover={handleHover}
        onUnhover={handleUnhover}
        onClick={handleClick}
      />
      <HoverTooltips overlay={overlay} />
    </div>
  );
}
