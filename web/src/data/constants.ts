// Pixel size of one heatmap cell — the figure's own width/height derive from this plus
// its margin object (see useMatrixFigure.ts), so cells render square. Port of app/app.py's
// CELL.
export const CELL = 42;

/** Full name for each method family — FAMILY_ORDER's own keys don't spell out what
 * "ml"/"dl"/"fm" stand for (see formatSnakeCaseLabel), so this supplies the readable form
 * used both as the matrix's column-group headers and the sidebar's FamilyMultiSelect. */
export const FAMILY_FULL_NAMES: Record<string, string> = {
  baseline: 'Baseline',
  ensemble: 'Ensemble',
  classical: 'Classical',
  ml: 'Machine learning',
  dl: 'Deep learning',
  fm: 'Foundation models',
};

/** Colour-blind-safe palette (Wong 2011, dropping its low-contrast yellow), used to
 * colour-match a selected method's chip/outline/forecast line. Port of
 * app/snapshot.py's METHOD_COLORS. */
export const METHOD_COLORS = [
  '#0072B2',
  '#D55E00',
  '#009E73',
  '#CC79A7',
  '#56B4E9',
  '#E69F00',
  '#8B008B',
];

/** Default number of points in the training-tail quick-look plot. Port of
 * app/snapshot.py's TRAINING_TAIL_N. */
export const TRAINING_TAIL_N = 1000;

/** A collapsed method-info card is exactly its title bar. Shared with App.tsx (which reserves
 * this much space below the matrix column so a collapsed card never overlaps the last line of
 * text above it) so the two can't drift apart — kept in this constants module rather than
 * MethodInfoCard's own file so that a synchronous import of just this number doesn't drag
 * MethodInfoCard's (lazy-loaded) react-markdown dependency into the main bundle. */
export const TITLE_BAR_HEIGHT = 34;

/** Matrices with fewer rows than this can't show the training-time overlay: Plotly.js
 * fails to repaint the heatmap for a matrix this short when switching to the overlay's
 * color scale (confirmed down to 3 rows; 4 rows renders correctly) — a Plotly bug, not
 * something in this app's own data or logic (see RESEARCH_LOG.md, 2026-07-21). The "Show
 * training time" control is disabled below this row count rather than left to fail silently. */
export const TRAINING_TIME_MIN_ROWS = 4;
