import { useCallback, useEffect, useRef } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';

type Axis = 'x' | 'y';

const CURSOR: Record<Axis, string> = { x: 'col-resize', y: 'row-resize' };

function clientPos(event: { clientX: number; clientY: number }, axis: Axis): number {
  return axis === 'x' ? event.clientX : event.clientY;
}

/**
 * Drag handler for a panel resized by dragging one of its edges, where the opposite edge
 * stays anchored in place. Moving the mouse toward the anchored edge grows the panel — e.g.
 * a panel anchored to the right of the screen, resized by its left border, grows as the
 * mouse moves left.
 */
export function useEdgeResize(
  axis: Axis,
  size: number,
  onResize: (size: number) => void,
): { onMouseDown: (event: ReactMouseEvent) => void } {
  const dragStart = useRef<{ pos: number; size: number } | null>(null);

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!dragStart.current) return;
      onResize(dragStart.current.size - (clientPos(event, axis) - dragStart.current.pos));
    },
    [axis, onResize],
  );

  const stopDragging = useCallback(() => {
    dragStart.current = null;
    document.body.style.removeProperty('cursor');
    document.body.style.removeProperty('user-select');
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', stopDragging);
  }, [handleMouseMove]);

  const onMouseDown = useCallback(
    (event: ReactMouseEvent) => {
      event.preventDefault();
      dragStart.current = { pos: clientPos(event, axis), size };
      document.body.style.cursor = CURSOR[axis];
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', stopDragging);
    },
    [axis, size, handleMouseMove, stopDragging],
  );

  useEffect(() => stopDragging, [stopDragging]);

  return { onMouseDown };
}
