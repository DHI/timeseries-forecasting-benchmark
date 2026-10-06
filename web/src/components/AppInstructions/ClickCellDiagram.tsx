// Small illustration of clicking a matrix cell: a 3x3 stand-in grid with one cell picked out and
// an arrow pointing at it, in the same curved-arrow style as Matrix/SelectionHint.tsx.
const GRID_ORIGIN = { x: 26, y: 10 };
const CELL_SIZE = 20;
const CELL_GAP = 4;
const HIGHLIGHT = { row: 1, col: 1 };

export function ClickCellDiagram() {
  const cells = [];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const x = GRID_ORIGIN.x + col * (CELL_SIZE + CELL_GAP);
      const y = GRID_ORIGIN.y + row * (CELL_SIZE + CELL_GAP);
      const isHighlighted = row === HIGHLIGHT.row && col === HIGHLIGHT.col;
      cells.push(
        <rect
          key={`${row}-${col}`}
          x={x}
          y={y}
          width={CELL_SIZE}
          height={CELL_SIZE}
          fill="var(--panel-bg)"
          stroke="var(--border)"
          strokeWidth={isHighlighted ? 2.5 : 1.5}
        />,
      );
    }
  }

  return (
    <div style={{ textAlign: 'center', margin: '8px 0' }}>
      <svg width={120} height={90} viewBox="0 0 120 90" aria-hidden="true">
        {cells}
        <defs>
          <marker
            id="click-cell-hint-arrowhead"
            markerWidth="8"
            markerHeight="8"
            refX="4"
            refY="4"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--text-muted)" />
          </marker>
        </defs>
        <path
          d="M100,78 Q86,58 74,49"
          fill="none"
          stroke="var(--text-muted)"
          strokeWidth={2}
          strokeLinecap="round"
          markerEnd="url(#click-cell-hint-arrowhead)"
        />
      </svg>
      <p style={{ color: 'var(--text-muted)', margin: 0 }}>
        Click a cell to select a method for that case.
      </p>
    </div>
  );
}
