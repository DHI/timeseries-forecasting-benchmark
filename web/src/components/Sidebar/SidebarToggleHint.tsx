// Floating prompt shown near the top-left corner until the user opens the controls
// panel. Positioned absolutely against the sidebar's `position: sticky` wrapper, which
// also hosts the toggle button, so it stays anchored to the page's top-left corner.
export function SidebarToggleHint() {
  return (
    <div
      style={{
        position: 'absolute',
        top: 50,
        left: 8,
        width: 180,
        pointerEvents: 'none',
      }}
    >
      <svg width="100" height="60" viewBox="0 0 100 60" aria-hidden="true">
        <defs>
          <marker
            id="sidebar-toggle-hint-arrowhead"
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
          d="M75,50 Q30,45 12,10"
          fill="none"
          stroke="#666"
          strokeWidth={2}
          strokeLinecap="round"
          markerEnd="url(#sidebar-toggle-hint-arrowhead)"
        />
      </svg>
      <p style={{ color: '#666', margin: 0 }}>Reveal controls panel</p>
    </div>
  );
}
