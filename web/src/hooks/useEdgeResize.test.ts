import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useEdgeResize } from './useEdgeResize';

describe('useEdgeResize', () => {
  it('x axis: widens when the mouse moves left, and stops on mouseup', () => {
    const onResize = vi.fn();
    const { result } = renderHook(() => useEdgeResize('x', 460, onResize));

    act(() => {
      result.current.onMouseDown({
        preventDefault: () => {},
        clientX: 500,
        clientY: 0,
      } as React.MouseEvent);
    });
    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 440 }));
    });
    expect(onResize).toHaveBeenCalledWith(520);

    act(() => {
      window.dispatchEvent(new MouseEvent('mouseup'));
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 300 }));
    });
    expect(onResize).toHaveBeenCalledTimes(1);
  });

  it('x axis, inverted: widens when the mouse moves right', () => {
    const onResize = vi.fn();
    const { result } = renderHook(() => useEdgeResize('x', 380, onResize, { invert: true }));

    act(() => {
      result.current.onMouseDown({
        preventDefault: () => {},
        clientX: 500,
        clientY: 0,
      } as React.MouseEvent);
    });
    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 560 }));
    });
    expect(onResize).toHaveBeenCalledWith(440);
  });

  it('y axis: grows when the mouse moves up', () => {
    const onResize = vi.fn();
    const { result } = renderHook(() => useEdgeResize('y', 220, onResize));

    act(() => {
      result.current.onMouseDown({
        preventDefault: () => {},
        clientX: 0,
        clientY: 400,
      } as React.MouseEvent);
    });
    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientY: 350 }));
    });
    expect(onResize).toHaveBeenCalledWith(270);
  });

  it('keeps dragging across renders that pass a brand-new onResize identity each time', () => {
    // Regression test: a caller that passes an inline callback (a new function every render,
    // as React re-renders on each onResize-triggered state update) must not have its drag cut
    // short after the first mousemove.
    let height = 220;
    const seen: number[] = [];
    const makeCallback = () => (h: number) => {
      seen.push(h);
      height = h;
    };

    const { result, rerender } = renderHook(
      ({ size, cb }: { size: number; cb: (h: number) => void }) => useEdgeResize('y', size, cb),
      { initialProps: { size: height, cb: makeCallback() } },
    );

    act(() => {
      result.current.onMouseDown({
        preventDefault: () => {},
        clientX: 0,
        clientY: 400,
      } as React.MouseEvent);
    });
    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientY: 380 }));
    });
    rerender({ size: height, cb: makeCallback() });
    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientY: 340 }));
    });

    expect(seen).toEqual([240, 280]);
  });
});
