// Small illustration of dragging a method card's title bar to resize it: a card outline with its
// title strip picked out and a double-headed vertical arrow, in the same arrow style as
// Matrix/SelectionHint.tsx.
export function ResizeHandleDiagram() {
  return (
    <div style={{ textAlign: 'center', margin: '8px 0' }}>
      <svg width={120} height={90} viewBox="0 0 120 90" aria-hidden="true">
        <rect
          x={20}
          y={10}
          width={80}
          height={60}
          fill="var(--panel-bg)"
          stroke="var(--border)"
          strokeWidth={1.5}
        />
        <rect x={20} y={10} width={80} height={14} fill="var(--text-muted)" opacity={0.25} />
        <defs>
          <marker
            id="resize-hint-arrowhead"
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
          d="M108,17 L108,4"
          fill="none"
          stroke="var(--text-muted)"
          strokeWidth={2}
          strokeLinecap="round"
          markerEnd="url(#resize-hint-arrowhead)"
        />
        <path
          d="M108,17 L108,30"
          fill="none"
          stroke="var(--text-muted)"
          strokeWidth={2}
          strokeLinecap="round"
          markerEnd="url(#resize-hint-arrowhead)"
        />
      </svg>
      <p style={{ color: 'var(--text-muted)', margin: 0 }}>
        Drag a card&apos;s title bar to resize it.
      </p>
    </div>
  );
}
