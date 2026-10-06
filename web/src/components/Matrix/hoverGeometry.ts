// Port of app/app.py's `hover-tooltips` clientside callback: reads the live pixel
// geometry off the Plotly graph div (`_fullLayout` axis offsets/lengths and the
// data->pixel converters) to position two floating tooltip boxes (row name pinned to the
// left edge, column name pinned to the top edge) plus four translucent rectangles that
// dim the grid outside the hovered row/column crosshair. Framework-agnostic — the only
// React-specific change from the original is *when* it runs (inside an onHover handler
// instead of a Dash clientside callback) and *where* the result goes (component state
// instead of 8 Dash Outputs).

/** The subset of Plotly's internal (undocumented, but stable) per-axis layout fields this
 * needs — not covered by @types/plotly.js, which only types the public API. */
type PlotlyAxisInternal = {
  _offset: number;
  _length: number;
  // Category axes (case_id/method strings, this heatmap's x/y) key these converters by
  // the category label itself, not a numeric index — hence `string | number` here rather
  // than `number`, matching the Python clientside callback's `d2c(pt.x)` on a string.
  d2p?: (v: string | number) => number;
  d2c?: (v: string | number) => number;
  c2p: (v: number) => number;
};

export type PlotlyGraphDiv = HTMLElement & {
  _fullLayout?: { xaxis: PlotlyAxisInternal; yaxis: PlotlyAxisInternal };
};

export type Rect = { left: number; top: number; width: number; height: number };

export type HoverOverlay = {
  rowLabel: string;
  colLabel: string;
  row: { left: number; top: number };
  col: { left: number; top: number };
  dimTL: Rect;
  dimTR: Rect;
  dimBL: Rect;
  dimBR: Rect;
};

function rect(l: number, t: number, w: number, h: number): Rect {
  return { left: l, top: t, width: Math.max(0, w), height: Math.max(0, h) };
}

/** `x`/`y` are the hovered cell's category values (column/row names) off the Plotly hover
 * event's `points[0]`. `nCols`/`nRows` are the heatmap's category counts (`colOrder.length`/
 * `rowOrder.length`), needed to find the true edge of the plotted cells (see below).
 * Returns null when the graph div's layout isn't ready yet (it always is by the time a real
 * hover event fires — Plotly only emits one after rendering — this is just a defensive
 * type-narrowing guard, not a real runtime path). */
export function computeHoverOverlay(
  x: string,
  y: string,
  nCols: number,
  nRows: number,
  wrapperEl: HTMLElement,
  gd: PlotlyGraphDiv,
): HoverOverlay | null {
  if (!gd._fullLayout) return null;
  const xa = gd._fullLayout.xaxis;
  const ya = gd._fullLayout.yaxis;
  const d2pix = (ax: PlotlyAxisInternal, v: string) =>
    ax.d2p ? ax.d2p(v) : ax.c2p(ax.d2c ? ax.d2c(v) : Number(v));

  const wr = wrapperEl.getBoundingClientRect();
  const gr = gd.getBoundingClientRect();
  const dx = gr.left - wr.left;
  const dy = gr.top - wr.top;
  const gridLeft = dx + xa._offset;
  const gridTop = dy + ya._offset;

  const colCenter = gridLeft + d2pix(xa, x);
  const rowCenter = gridTop + d2pix(ya, y);
  const halfW = Math.abs(xa.c2p(1) - xa.c2p(0)) / 2;
  const halfH = Math.abs(ya.c2p(1) - ya.c2p(0)) / 2;
  const colLeft = colCenter - halfW;
  const colRight = colCenter + halfW;
  const rowTop = rowCenter - halfH;
  const rowBottom = rowCenter + halfH;

  // `_offset`/`_length` describe the axis's full *padded* autorange box, not where the
  // cells themselves start and end: `scaleanchor`/`constrain: 'domain'` (locking the heatmap
  // to a 1:1 aspect ratio) pads the range asymmetrically whenever the container's aspect
  // ratio doesn't exactly match nCols:nRows, shifting the true cell edges inward from
  // `_offset`/`_offset + _length`. The true edges sit at category index ±0.5 from the first/
  // last row or column, which `c2p` maps correctly (unlike `_offset` alone) — using
  // `_offset`/`_length` here (as the crosshair math above does) would let the dim rects
  // bleed past the grid into the row/column-group label margins.
  const dimLeft = gridLeft + xa.c2p(-0.5);
  const dimRight = gridLeft + xa.c2p(nCols - 0.5);
  const yEdgeA = gridTop + ya.c2p(-0.5);
  const yEdgeB = gridTop + ya.c2p(nRows - 0.5);
  const dimTop = Math.min(yEdgeA, yEdgeB);
  const dimBottom = Math.max(yEdgeA, yEdgeB);

  return {
    rowLabel: y,
    colLabel: x,
    row: { left: gridLeft, top: rowCenter },
    col: { left: colCenter, top: gridTop },
    dimTL: rect(dimLeft, dimTop, colLeft - dimLeft, rowTop - dimTop),
    dimTR: rect(colRight, dimTop, dimRight - colRight, rowTop - dimTop),
    dimBL: rect(dimLeft, rowBottom, colLeft - dimLeft, dimBottom - rowBottom),
    dimBR: rect(colRight, rowBottom, dimRight - colRight, dimBottom - rowBottom),
  };
}
