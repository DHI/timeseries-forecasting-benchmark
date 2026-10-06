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
 * mouse moves left. Pass `invert: true` for the mirror-image case — a panel anchored to the
 * left of the screen, resized by its right border, grows as the mouse moves right.
 */
export function useEdgeResize(
  axis: Axis,
  size: number,
  onResize: (size: number) => void,
  { invert = false }: { invert?: boolean } = {},
): { onMouseDown: (event: ReactMouseEvent) => void; moved: { current: boolean } } {
  const dragStart = useRef<{ pos: number; size: number } | null>(null);
  // Set on the first mousemove of a drag, so callers (e.g. a title bar that both toggles on
  // click and resizes on drag) can tell a drag apart from a plain click in their onClick handler
  // — mousedown/mouseup/click all fire synchronously for one gesture, so this ref is settled by
  // the time onClick runs.
  const moved = useRef(false);
  // Read via a ref rather than closing over `onResize` directly: if the caller passes an
  // inline callback (a new function identity every render), closing over it would change
  // handleMouseMove's identity mid-drag, which tears down the drag's own event listeners
  // through the stopDragging/useEffect cleanup chain after the first move.
  const onResizeRef = useRef(onResize);
  onResizeRef.current = onResize;

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!dragStart.current) return;
      moved.current = true;
      const delta = clientPos(event, axis) - dragStart.current.pos;
      onResizeRef.current(dragStart.current.size + (invert ? delta : -delta));
    },
    [axis, invert],
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
      moved.current = false;
      dragStart.current = { pos: clientPos(event, axis), size };
      document.body.style.cursor = CURSOR[axis];
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', stopDragging);
    },
    [axis, size, handleMouseMove, stopDragging],
  );

  useEffect(() => stopDragging, [stopDragging]);

  return { onMouseDown, moved };
}
