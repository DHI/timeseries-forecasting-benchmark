import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useResizablePanelWidth } from './useResizablePanelWidth';

describe('useResizablePanelWidth', () => {
  it('widens the panel when the mouse moves left, and stops on mouseup', () => {
    const onResize = vi.fn();
    const { result } = renderHook(() => useResizablePanelWidth(460, onResize));

    act(() => {
      result.current.onMouseDown({
        preventDefault: () => {},
        clientX: 500,
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
});
