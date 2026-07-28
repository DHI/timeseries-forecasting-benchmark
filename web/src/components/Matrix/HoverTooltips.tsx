import type { CSSProperties } from 'react';
import type { HoverOverlay, Rect } from './hoverGeometry';

// Port of app/app.py's _TOOLTIP_BG / _TOOLTIP_BOX_STYLE / _ROW_TRIANGLE_STYLE /
// _COL_TRIANGLE_STYLE / _DIM_RECT_STYLE.
const TOOLTIP_BG = '#2a3f5f';

export const tooltipBoxStyle: CSSProperties = {
  background: TOOLTIP_BG,
  color: 'white',
  padding: '3px 8px',
  borderRadius: 3,
  fontSize: 12,
  whiteSpace: 'nowrap',
  boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
};

const rowTriangleStyle: CSSProperties = {
  width: 0,
  height: 0,
  borderTop: '6px solid transparent',
  borderBottom: '6px solid transparent',
  borderLeft: `7px solid ${TOOLTIP_BG}`,
};

const colTriangleStyle: CSSProperties = {
  width: 0,
  height: 0,
  borderLeft: '6px solid transparent',
  borderRight: '6px solid transparent',
  borderTop: `7px solid ${TOOLTIP_BG}`,
};

const dimRectBaseStyle: CSSProperties = {
  position: 'absolute',
  background: 'var(--hover-dim-overlay)',
  pointerEvents: 'none',
  zIndex: 5,
};

function DimRect({ rect }: { rect: Rect }) {
  return (
    <div
      style={{
        ...dimRectBaseStyle,
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
      }}
    />
  );
}

/** The two floating name tooltips (row pinned to the left edge, column pinned to the top
 * edge) plus four translucent rectangles dimming the grid outside the hovered
 * row/column crosshair. Renders nothing when `overlay` is null (cursor outside the
 * grid). Port of the DOM structure app/app.py's layout builds for `row-tooltip` /
 * `col-tooltip` / `dim-tl` / `dim-tr` / `dim-bl` / `dim-br`. */
export function HoverTooltips({ overlay }: { overlay: HoverOverlay | null }) {
  if (!overlay) return null;
  return (
    <>
      <DimRect rect={overlay.dimTL} />
      <DimRect rect={overlay.dimTR} />
      <DimRect rect={overlay.dimBL} />
      <DimRect rect={overlay.dimBR} />
      <div
        style={{
          position: 'absolute',
          display: 'flex',
          alignItems: 'center',
          pointerEvents: 'none',
          zIndex: 10,
          left: overlay.row.left,
          top: overlay.row.top,
          transform: 'translate(-100%, -50%)',
        }}
      >
        <div style={tooltipBoxStyle}>{overlay.rowLabel}</div>
        <div style={rowTriangleStyle} />
      </div>
      <div
        style={{
          position: 'absolute',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          pointerEvents: 'none',
          zIndex: 10,
          left: overlay.col.left,
          top: overlay.col.top,
          transform: 'translate(-50%, -100%)',
        }}
      >
        <div style={tooltipBoxStyle}>{overlay.colLabel}</div>
        <div style={colTriangleStyle} />
      </div>
    </>
  );
}
