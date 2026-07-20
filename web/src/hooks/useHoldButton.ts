import { useCallback, useEffect, useRef } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';

/**
 * Momentary press handler: `onChange(true)` fires on mousedown over the element,
 * `onChange(false)` fires the instant the mouse button is released anywhere on the
 * page — via a window-level `mouseup` listener, so release is caught even if the
 * cursor was dragged off the element first, matching useEdgeResize's drag pattern.
 */
export function useHoldButton(onChange: (active: boolean) => void): {
  onMouseDown: (event: ReactMouseEvent) => void;
} {
  // Read via a ref rather than closing over `onChange` directly: if the caller passes an
  // inline callback (a new function identity every render), closing over it would change
  // stopHolding's identity mid-press, tearing down and re-adding the window listener.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const stopHolding = useCallback(() => {
    onChangeRef.current(false);
    window.removeEventListener('mouseup', stopHolding);
  }, []);

  const onMouseDown = useCallback(
    (event: ReactMouseEvent) => {
      event.preventDefault();
      onChangeRef.current(true);
      window.addEventListener('mouseup', stopHolding);
    },
    [stopHolding],
  );

  useEffect(() => stopHolding, [stopHolding]);

  return { onMouseDown };
}
