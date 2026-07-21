import { useCallback, useEffect, useRef } from 'react';
import type { MouseEvent as ReactMouseEvent } from 'react';

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.tagName === 'INPUT' ||
    target.tagName === 'TEXTAREA' ||
    target.tagName === 'SELECT' ||
    target.isContentEditable
  );
}

/**
 * Momentary press handler: `onChange(true)` fires on mousedown over the element or on
 * holding the space bar anywhere on the page, `onChange(false)` fires once every active
 * source (mouse, keyboard) has released. A ref-counted source count — rather than a plain
 * boolean — means holding the button with the mouse and tapping space at the same time
 * doesn't let the keyup end the hold early while the mouse button is still down.
 */
export function useHoldButton(onChange: (active: boolean) => void): {
  onMouseDown: (event: ReactMouseEvent) => void;
} {
  // Read via a ref rather than closing over `onChange` directly: if the caller passes an
  // inline callback (a new function identity every render), closing over it would change
  // the callbacks' identity mid-press, tearing down and re-adding the window listeners.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const activeSourcesRef = useRef(0);

  const activate = useCallback(() => {
    activeSourcesRef.current += 1;
    if (activeSourcesRef.current === 1) onChangeRef.current(true);
  }, []);

  const deactivate = useCallback(() => {
    if (activeSourcesRef.current === 0) return;
    activeSourcesRef.current -= 1;
    if (activeSourcesRef.current === 0) onChangeRef.current(false);
  }, []);

  const stopHolding = useCallback(() => {
    deactivate();
    window.removeEventListener('mouseup', stopHolding);
  }, [deactivate]);

  const onMouseDown = useCallback(
    (event: ReactMouseEvent) => {
      event.preventDefault();
      activate();
      window.addEventListener('mouseup', stopHolding);
    },
    [activate, stopHolding],
  );

  useEffect(() => stopHolding, [stopHolding]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || event.repeat || isTypingTarget(event.target)) return;
      event.preventDefault();
      activate();
    };

    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code !== 'Space') return;
      // Browsers fire a focused button's click as the *keyup* default action for space,
      // separately from keydown — preventing only keydown (above) stops page scroll but
      // doesn't stop a focused button (e.g. the sidebar's collapse toggle) from firing on release.
      event.preventDefault();
      deactivate();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [activate, deactivate]);

  return { onMouseDown };
}
