// Floating prompt shown in the whitespace beside the heatmap until the user selects a
// cell. Positioned absolutely against a `position: relative` ancestor sized to the
// heatmap's own width, so `left: 100%` lands it just to the heatmap's right.
export function SelectionHint() {
  return (
    <div
      style={{
        position: 'absolute',
        left: '100%',
        top: '50%',
        transform: 'translateY(-50%)',
        width: 180,
        marginLeft: 16,
        textAlign: 'center',
        pointerEvents: 'none',
      }}
    >
      <svg width="100" height="60" viewBox="0 0 100 60" aria-hidden="true">
        <defs>
          <marker
            id="selection-hint-arrowhead"
            markerWidth="8"
            markerHeight="8"
            refX="4"
            refY="4"
            orient="auto"
          >
            <path d="M0,0 L8,4 L0,8 Z" fill="#666" />
          </marker>
        </defs>
        <path
          d="M90,50 Q40,50 10,12"
          fill="none"
          stroke="#666"
          strokeWidth={2}
          strokeLinecap="round"
          markerEnd="url(#selection-hint-arrowhead)"
        />
      </svg>
      <p style={{ color: '#666', margin: 0 }}>Click a cell to inspect a case</p>
    </div>
  );
}
