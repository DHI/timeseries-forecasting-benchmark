// Minimalist wireframe of the page layout: sidebar, matrix, and case panel side by side, with
// the method-card strip docked below the matrix. Box outlines are SVG; labels are plain HTML
// overlaid on top, matching the existing hint components' "SVG for shape, HTML for text" split.
const LABELS = [
  { text: 'Sidebar', left: 27, top: 45 },
  { text: 'Matrix', left: 130, top: 45 },
  { text: 'Case panel', left: 241, top: 45 },
  { text: 'Method cards', left: 130, top: 112 },
];

export function PageOverviewDiagram() {
  return (
    <div style={{ position: 'relative', width: 276, height: 124, margin: '8px auto' }}>
      <svg width={276} height={124} viewBox="0 0 276 124" aria-hidden="true">
        <rect
          x={0}
          y={0}
          width={54}
          height={90}
          fill="var(--panel-bg)"
          stroke="var(--border)"
          strokeWidth={1.5}
        />
        <rect
          x={60}
          y={0}
          width={140}
          height={90}
          fill="var(--panel-bg)"
          stroke="var(--border)"
          strokeWidth={1.5}
        />
        <rect
          x={206}
          y={0}
          width={70}
          height={90}
          fill="var(--panel-bg)"
          stroke="var(--border)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
        <rect
          x={60}
          y={100}
          width={140}
          height={24}
          fill="var(--panel-bg)"
          stroke="var(--border)"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      </svg>
      {LABELS.map(({ text, left, top }) => (
        <span
          key={text}
          style={{
            position: 'absolute',
            left,
            top,
            transform: 'translate(-50%, -50%)',
            fontSize: '0.75em',
            color: 'var(--text-muted)',
            textAlign: 'center',
            pointerEvents: 'none',
          }}
        >
          {text}
        </span>
      ))}
    </div>
  );
}
