import { cornerTrianglePath } from '../Matrix/overlayShapes';

const SIZE = 20;

export function MarkerSwatch({
  corner,
  color,
}: {
  corner: 'top-right' | 'bottom-left';
  color: string;
}) {
  return (
    <svg
      width={SIZE}
      height={SIZE}
      style={{ flexShrink: 0, verticalAlign: 'middle' }}
      aria-hidden="true"
    >
      <rect
        x={0.5}
        y={0.5}
        width={SIZE - 1}
        height={SIZE - 1}
        fill="var(--panel-bg)"
        stroke="var(--border)"
      />
      <path d={cornerTrianglePath(0, 0, SIZE, SIZE, 0.35, corner)} fill={color} opacity={0.8} />
    </svg>
  );
}
