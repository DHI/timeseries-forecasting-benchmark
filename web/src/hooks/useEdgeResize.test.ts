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
});
