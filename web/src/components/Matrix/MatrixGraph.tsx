import { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Config } from 'plotly.js';
import Plot from '../../plotly';
import { computeHoverOverlay } from './hoverGeometry';
import type { HoverOverlay, PlotlyGraphDiv } from './hoverGeometry';
import { HoverTooltips, tooltipBoxStyle } from './HoverTooltips';
import { useMatrixFigure } from './useMatrixFigure';
import type { TrainingTimeBreakdown } from './useMatrixFigure';
import { useTheme } from '../../theme/ThemeContext';
import type { SelectionState } from '../../state/selectionReducer';
import type { ScoreRow } from '../../types/data';

function formatBreakdownSeconds(v: number | null): string {
  return v === null || !Number.isFinite(v) ? '—' : `${v.toFixed(2)}s`;
}

/** Shown only while the training-time overlay is held (see `showTrainingTime`): the
 * combined fit+predict number a cell displays doesn't say which of the two dominates, so
 * hovering breaks it back out. Positioned at the hovered cell's own center, reusing the
 * pixel geometry `HoverTooltips` already computes for the row/column callouts. */
function TrainingTimeBreakdownTooltip({
  breakdown,
  left,
  top,
}: {
  breakdown: TrainingTimeBreakdown;
  left: number;
  top: number;
}) {
  const style: CSSProperties = {
    position: 'absolute',
    pointerEvents: 'none',
    zIndex: 10,
    left,
    top,
    transform: 'translate(-50%, -140%)',
    padding: '12px 16px',
    lineHeight: 1.6,
    ...tooltipBoxStyle,
  };
  return (
    <div style={style}>
      <div>Fit: {formatBreakdownSeconds(breakdown.fitSeconds)}</div>
      <div>Predict: {formatBreakdownSeconds(breakdown.predictSeconds)}</div>
    </div>
  );
}

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
  /** Reports the figure's rendered pixel width (heatmap + colorbar + axis-title margins)
   * whenever it changes, so a caller can size surrounding content to match. */
  onWidthChange?: (width: number) => void;
};

function MatrixGraphComponent({
  rows,
  metric,
  selection,
  showGlobal,
  showExogenous,
  showTrainingTime,
  onCellClick,
  onWidthChange,
}: Props) {
  const { theme } = useTheme();
  const figure = useMatrixFigure(
    rows,
    metric,
    selection,
    showGlobal,
    showExogenous,
    showTrainingTime,
    theme === 'dark',
  );
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const graphRef = useRef<HTMLDivElement | null>(null);
  const [overlay, setOverlay] = useState<HoverOverlay | null>(null);
  const [breakdown, setBreakdown] = useState<TrainingTimeBreakdown | null>(null);

  useEffect(() => {
    if (figure?.layout.width) onWidthChange?.(figure.layout.width);
  }, [figure, onWidthChange]);

  const handleHover = useCallback(
    (event: PlotMouseEvent) => {
      const pt = event.points?.[0];
      const wrapper = wrapperRef.current;
      const gd = graphRef.current as PlotlyGraphDiv | null;
      if (!pt || !wrapper || !gd || !figure) return;
      const x = String(pt.x);
      const y = String(pt.y);
      setOverlay(
        computeHoverOverlay(x, y, figure.colOrder.length, figure.rowOrder.length, wrapper, gd),
      );
      // The fit/predict breakdown only ever makes sense while the combined number is on
      // screen — otherwise a stale value could hang around into a normal metric hover.
      setBreakdown(showTrainingTime ? (figure.trainingBreakdownOf.get(`${y}|${x}`) ?? null) : null);
    },
    [figure, showTrainingTime],
  );

  const handleUnhover = useCallback(() => {
    setOverlay(null);
    setBreakdown(null);
  }, []);

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
      {showTrainingTime && breakdown && overlay && (
        <TrainingTimeBreakdownTooltip
          breakdown={breakdown}
          left={overlay.col.left}
          top={overlay.row.top}
        />
      )}
    </div>
  );
}

export const MatrixGraph = memo(MatrixGraphComponent);
