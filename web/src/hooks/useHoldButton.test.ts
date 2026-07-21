import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useHoldButton } from './useHoldButton';

describe('useHoldButton', () => {
  it('activates on mousedown and deactivates on window mouseup', () => {
    const onChange = vi.fn();
    const { result } = renderHook(() => useHoldButton(onChange));

    act(() => {
      result.current.onMouseDown({ preventDefault: () => {} } as React.MouseEvent);
    });
    expect(onChange).toHaveBeenCalledWith(true);

    act(() => {
      window.dispatchEvent(new MouseEvent('mouseup'));
    });
    expect(onChange).toHaveBeenLastCalledWith(false);
  });

  it('deactivates on mouseup even when the cursor left the element first', () => {
    const onChange = vi.fn();
    const { result } = renderHook(() => useHoldButton(onChange));

    act(() => {
      result.current.onMouseDown({ preventDefault: () => {} } as React.MouseEvent);
    });
    act(() => {
      window.dispatchEvent(new MouseEvent('mousemove', { clientX: 999 }));
      window.dispatchEvent(new MouseEvent('mouseup'));
    });
    expect(onChange).toHaveBeenLastCalledWith(false);
  });

  it('keeps working across renders that pass a brand-new onChange identity each time', () => {
    const seen: boolean[] = [];
    const makeCallback = () => (active: boolean) => seen.push(active);

    const { result, rerender } = renderHook(
      ({ cb }: { cb: (active: boolean) => void }) => useHoldButton(cb),
      { initialProps: { cb: makeCallback() } },
    );

    act(() => {
      result.current.onMouseDown({ preventDefault: () => {} } as React.MouseEvent);
    });
    rerender({ cb: makeCallback() });
    act(() => {
      window.dispatchEvent(new MouseEvent('mouseup'));
    });

    expect(seen).toEqual([true, false]);
  });

  it('activates on holding space and deactivates on release', () => {
    const onChange = vi.fn();
    renderHook(() => useHoldButton(onChange));

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space' }));
    });
    expect(onChange).toHaveBeenCalledWith(true);

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keyup', { code: 'Space' }));
    });
    expect(onChange).toHaveBeenLastCalledWith(false);
  });

  it('ignores repeated keydown events from the OS key-repeat', () => {
    const onChange = vi.fn();
    renderHook(() => useHoldButton(onChange));

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space' }));
      window.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', repeat: true }));
    });
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it('ignores space presses while typing in a text input', () => {
    const onChange = vi.fn();
    renderHook(() => useHoldButton(onChange));

    const input = document.createElement('input');
    document.body.appendChild(input);

    act(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', bubbles: true }));
    });
    expect(onChange).not.toHaveBeenCalled();

    document.body.removeChild(input);
  });

  it('keeps holding via mouse when space is released first', () => {
    const onChange = vi.fn();
    const { result } = renderHook(() => useHoldButton(onChange));

    act(() => {
      result.current.onMouseDown({ preventDefault: () => {} } as React.MouseEvent);
      window.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space' }));
    });
    expect(onChange).toHaveBeenCalledTimes(1);

    act(() => {
      window.dispatchEvent(new KeyboardEvent('keyup', { code: 'Space' }));
    });
    expect(onChange).toHaveBeenCalledTimes(1);

    act(() => {
      window.dispatchEvent(new MouseEvent('mouseup'));
    });
    expect(onChange).toHaveBeenLastCalledWith(false);
  });
});
