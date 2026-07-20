import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { EMPTY_SELECTION } from '../../state/selectionReducer';
import { useMatrixFigure } from './useMatrixFigure';
import type { ScoreRow } from '../../types/data';

function row(overrides: Partial<ScoreRow>): ScoreRow {
  return {
    case_id: 'case_a',
    domain: 'water_distribution',
    case_exo_tier: 'none',
    method: 'LightGBM',
    family: 'ml',
    uses_exogenous: false,
    is_global: false,
    used_extended_history: false,
    horizon: '24h',
    metric: 'MAE',
    value: 1.0,
    seconds: 0.1,
    frac_finite_test: 1.0,
    status: 'ok',
    n_origins: 40,
    ...overrides,
  };
}

// This is a regression test for a real bug caught during M1->M2 development: passing
// both the local and extended-history row for the same (case, method) straight into the
// pivot silently picked whichever came last in array order, rather than always defaulting
// to the local row until `showGlobal` is on.
describe('useMatrixFigure — extended-history swap', () => {
  const rows = [
    row({ value: 1.0, used_extended_history: false, is_global: true }),
    row({ value: 9.0, used_extended_history: true, is_global: true }),
  ];

  it('uses the local value by default, regardless of row order', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[1.0]]);
  });

  it('swaps in the extended value and marks it when showGlobal is on', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, true, false, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[9.0]]);
    // One boundary-less single-cell grid: any shape present is the global overlay triangle.
    expect(result.current!.layout.shapes?.some((s) => s.fillcolor === '#FF1493')).toBe(true);
  });

  it('never swaps in a non-global method’s extended row', () => {
    const localOnly = [
      row({ value: 1.0, used_extended_history: false, is_global: false }),
      row({ value: 9.0, used_extended_history: true, is_global: false }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(localOnly, 'MAE', EMPTY_SELECTION, true, false, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[1.0]]);
  });
});

describe('useMatrixFigure — exogenous swap', () => {
  const rows = [
    row({ method: 'LightGBM', value: 1.0, case_exo_tier: 'necessary' }),
    row({ method: 'LightGBM+exo', value: 0.5, case_exo_tier: 'necessary' }),
  ];

  it('uses the base method’s value by default', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[1.0]]);
  });

  it('swaps in the +exo sibling’s value and marks it when showExo is on', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, true, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[0.5]]);
    expect(result.current!.layout.shapes?.some((s) => s.fillcolor === '#0B7285')).toBe(true);
  });

  it('does not swap for a case with no exogenous data (case_exo_tier "none")', () => {
    const noExo = [
      row({ method: 'LightGBM', value: 1.0, case_exo_tier: 'none' }),
      row({ method: 'LightGBM+exo', value: 0.5, case_exo_tier: 'none' }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(noExo, 'MAE', EMPTY_SELECTION, false, true, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[1.0]]);
  });

  it('never shows the +exo variant as its own column', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    expect(result.current!.colOrder).toEqual(['LightGBM']);
  });
});

describe('useMatrixFigure — training-time overlay', () => {
  const rows = [
    row({ case_id: 'case_a', method: 'LightGBM', value: 1.0, seconds: 4.32 }),
    row({ case_id: 'case_a', method: 'CatBoost', value: 2.0, seconds: 0.5 }),
  ];

  it('pivots on seconds instead of value when the overlay is active', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    // colOrder sorts alphabetically within a family: CatBoost before LightGBM.
    expect(heatmap.z).toEqual([[0.5, 4.32]]);
  });

  it('formats cell text with a trailing "s" and skips fastest-method bolding', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const texts = result.current!.layout.annotations!.map((a) => String(a.text));
    expect(texts).toContain('4.32s');
    expect(texts).toContain('0.50s');
    const cellTexts = texts.filter((t) => t.endsWith('s'));
    expect(cellTexts.every((t) => !t.includes('<b>'))).toBe(true);
  });

  it('shows a training-time colorbar title', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const heatmap = result.current!.data[0] as {
      colorbar: { title: { text: string } };
    };
    expect(heatmap.colorbar.title.text).toBe('Training\ntime (s)');
  });

  it('pivots the showGlobal swap-in cell on seconds too', () => {
    const globalRows = [
      row({
        case_id: 'case_a',
        method: 'LightGBM',
        used_extended_history: false,
        is_global: true,
        value: 1.0,
        seconds: 4.32,
      }),
      row({
        case_id: 'case_a',
        method: 'LightGBM',
        used_extended_history: true,
        is_global: true,
        value: 9.0,
        seconds: 8.64,
      }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(globalRows, 'MAE', EMPTY_SELECTION, true, false, true),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[8.64]]);
  });
});
