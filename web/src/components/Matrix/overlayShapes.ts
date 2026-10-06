import type { Shape } from 'plotly.js';
import { METHOD_COLORS } from '../../data/constants';
import type { SelectionState } from '../../state/selectionReducer';

/** SVG path for a small filled triangle in one corner of a cell's box, sized as `frac` of
 * the cell's edge length. Port of app/app.py's `_corner_triangle_path`. */
export function cornerTrianglePath(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  frac = 0.35,
  corner: 'top-right' | 'bottom-left' = 'top-right',
): string {
  const s = (x1 - x0) * frac;
  if (corner === 'bottom-left') {
    return `M ${x0},${y1} L ${x0 + s},${y1} L ${x0},${y1 - s} Z`;
  }
  return `M ${x1},${y0} L ${x1 - s},${y0} L ${x1},${y0 + s} Z`;
}

function overlayTriangles(
  cells: Array<[string, string]>,
  rowOrder: string[],
  colOrder: string[],
  corner: 'top-right' | 'bottom-left',
  color: string,
): Partial<Shape>[] {
  const shapes: Partial<Shape>[] = [];
  for (const [caseId, method] of cells) {
    const y = rowOrder.indexOf(caseId);
    const x = colOrder.indexOf(method);
    if (y === -1 || x === -1) continue;
    shapes.push({
      type: 'path',
      path: cornerTrianglePath(x - 0.5, y - 0.5, x + 0.5, y + 0.5, 0.35, corner),
      line: { color, width: 0 },
      fillcolor: color,
      opacity: 0.8,
      layer: 'above',
    });
  }
  return shapes;
}

/** Small hotpink corner triangle over cells overridden with the extended-history (full
 * data) fit, so they read as "not directly comparable to the rest of the row" without
 * covering the cell's number. Port of app/app.py's `_global_overlay_shapes`. */
export function globalOverlayShapes(
  cells: Array<[string, string]>,
  rowOrder: string[],
  colOrder: string[],
): Partial<Shape>[] {
  return overlayTriangles(cells, rowOrder, colOrder, 'top-right', '#FF1493');
}

/** Small teal corner triangle (opposite corner from the global-history marker, so the two
 * never collide) over cells whose displayed value was swapped in from the method's
 * exogenous-aware ("+exo") sibling. Port of app/app.py's `_exogenous_overlay_shapes`. */
export function exogenousOverlayShapes(
  cells: Array<[string, string]>,
  rowOrder: string[],
  colOrder: string[],
): Partial<Shape>[] {
  return overlayTriangles(cells, rowOrder, colOrder, 'bottom-left', '#0B7285');
}

/** Outline shapes for the selected case's clicked cells, one per selected method,
 * colour-matched to that method's chip/forecast line. Port of app/app.py's
 * `_selection_shapes`. */
export function selectionShapes(
  selection: SelectionState,
  rowOrder: string[],
  colOrder: string[],
): Partial<Shape>[] {
  const y = rowOrder.indexOf(selection.case_id ?? '');
  if (y === -1) return [];
  const shapes: Partial<Shape>[] = [];
  selection.methods.forEach((method, i) => {
    const x = colOrder.indexOf(method);
    if (x === -1) return;
    const color = METHOD_COLORS[i % METHOD_COLORS.length];
    shapes.push({
      type: 'rect',
      x0: x - 0.5,
      x1: x + 0.5,
      y0: y - 0.5,
      y1: y + 0.5,
      line: { color, width: 3 },
      fillcolor: 'rgba(0,0,0,0)',
      layer: 'above',
    });
  });
  return shapes;
}
