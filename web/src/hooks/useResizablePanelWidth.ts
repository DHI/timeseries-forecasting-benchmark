import { useCallback, useEffect, useRef } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';

/**
 * Drag handler for a panel anchored to the right edge of the screen, resized by
 * dragging its left border. Moving the mouse left widens the panel.
 */
export function useResizablePanelWidth(
  width: number,
  onResize: (width: number) => void,
): { onMouseDown: (event: ReactMouseEvent) => void } {
  const dragStart = useRef<{ x: number; width: number } | null>(null);

  const handleMouseMove = useCallback(
    (event: MouseEvent) => {
      if (!dragStart.current) return;
      onResize(dragStart.current.width - (event.clientX - dragStart.current.x));
    },
    [onResize],
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
      dragStart.current = { x: event.clientX, width };
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', stopDragging);
    },
    [width, handleMouseMove, stopDragging],
  );

  useEffect(() => stopDragging, [stopDragging]);

  return { onMouseDown };
}
