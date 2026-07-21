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
    // MAE colors on log10(value) (see metricConfig.ts's `colorScale: 'log'`).
    expect(heatmap.z).toEqual([[Math.log10(1.0)]]);
  });

  it('swaps in the extended value and marks it when showGlobal is on', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, true, false, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[Math.log10(9.0)]]);
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
    expect(heatmap.z).toEqual([[Math.log10(1.0)]]);
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
    expect(heatmap.z).toEqual([[Math.log10(1.0)]]);
  });

  it('swaps in the +exo sibling’s value and marks it when showExo is on', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, true, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[Math.log10(0.5)]]);
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
    expect(heatmap.z).toEqual([[Math.log10(1.0)]]);
  });

  it('never shows the +exo variant as its own column', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    expect(result.current!.colOrder).toEqual(['LightGBM']);
  });
});

describe('useMatrixFigure — colorbar title', () => {
  function heatmapOf(result: { data: unknown[] }) {
    return result.data[0] as { colorbar: { title: { text: string } } };
  }

  it('gives Skill_vs_* metrics a short three-line "Skill<br>vs<br>X" title', () => {
    const rows = [row({ metric: 'Skill_vs_Persistence', value: 0.1 })];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'Skill_vs_Persistence', EMPTY_SELECTION, false, false, false),
    );
    expect(heatmapOf(result.current!).colorbar.title.text).toBe('Skill<br>vs<br>Persistence');
  });

  it('gives other metrics a "MetricName<br>[unit]" title', () => {
    const rows = [row({ metric: 'MAE', value: 1.0 })];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    expect(heatmapOf(result.current!).colorbar.title.text).toBe('MAE<br>[target unit]');
  });

  it('states the % unit on the colorbar title instead of in MAPE/sMAPE cell text', () => {
    const rows = [row({ metric: 'MAPE', value: 0.123 })];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAPE', EMPTY_SELECTION, false, false, false),
    );
    const heatmap = result.current!.data[0] as {
      colorbar: { title: { text: string } };
    };
    expect(heatmap.colorbar.title.text).toBe('MAPE<br>[%]');
    const texts = result.current!.layout.annotations!.map((a) => String(a.text));
    expect(texts).toContain('<b>12</b>');
    expect(texts.some((t) => t.includes('%'))).toBe(false);
  });
});

describe('useMatrixFigure — log color scale for heavy-tailed metrics', () => {
  it('colors MAE cells by log10(value), floored at the smallest positive value', () => {
    const rows = [
      row({ case_id: 'case_a', method: 'CatBoost', value: 0.5 }),
      row({ case_id: 'case_a', method: 'LightGBM', value: 500 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[Math.log10(0.5), Math.log10(500)]]);
  });

  it('labels the colorbar with 1-2-5 log-spaced ticks, dropping decimals above 1', () => {
    const rows = [
      row({ case_id: 'case_a', method: 'CatBoost', value: 0.02 }),
      row({ case_id: 'case_a', method: 'LightGBM', value: 1000 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    const heatmap = result.current!.data[0] as {
      colorbar: { tickmode: string; tickvals: number[]; ticktext: string[] };
    };
    expect(heatmap.colorbar.tickmode).toBe('array');
    expect(heatmap.colorbar.ticktext).toEqual([
      '0.02',
      '0.05',
      '0.1',
      '0.2',
      '0.5',
      '1',
      '2',
      '5',
      '10',
      '20',
      '50',
      '100',
      '200',
      '500',
      '1000',
    ]);
  });

  it('shifts tick decimals by the metric scale, e.g. MAPE (scale 100)', () => {
    const rows = [
      row({ metric: 'MAPE', case_id: 'case_a', method: 'CatBoost', value: 0.0002 }),
      row({ metric: 'MAPE', case_id: 'case_a', method: 'LightGBM', value: 5 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAPE', EMPTY_SELECTION, false, false, false),
    );
    const heatmap = result.current!.data[0] as {
      colorbar: { ticktext: string[] };
    };
    expect(heatmap.colorbar.ticktext).toEqual([
      '0.02',
      '0.05',
      '0.1',
      '0.2',
      '0.5',
      '1',
      '2',
      '5',
      '10',
      '20',
      '50',
      '100',
      '200',
      '500',
    ]);
  });

  it('does not apply a log color scale to non-log metrics like MASE', () => {
    const rows = [
      row({ metric: 'MASE', case_id: 'case_a', method: 'CatBoost', value: 0.5 }),
      row({ metric: 'MASE', case_id: 'case_a', method: 'LightGBM', value: 5 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MASE', EMPTY_SELECTION, false, false, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[0.5, 5]]);
  });
});

describe('useMatrixFigure — training-time overlay', () => {
  const rows = [
    row({ case_id: 'case_a', method: 'LightGBM', value: 1.0, seconds: 4.32 }),
    row({ case_id: 'case_a', method: 'CatBoost', value: 2.0, seconds: 0.5 }),
  ];

  it('colors cells by log10(seconds + 1), not the raw seconds', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    // colOrder sorts alphabetically within a family: CatBoost before LightGBM.
    expect(heatmap.z).toEqual([[Math.log10(1.5), Math.log10(5.32)]]);
  });

  it('formats cell text as a plain whole number and skips fastest-method bolding', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const texts = result.current!.layout.annotations!.map((a) => String(a.text));
    expect(texts).toContain('4');
    expect(texts).toContain('1');
    expect(texts).not.toContain('<b>4</b>');
    expect(texts).not.toContain('<b>1</b>');
  });

  it('shows a short training-time colorbar title', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const heatmap = result.current!.data[0] as {
      colorbar: { title: { text: string } };
    };
    expect(heatmap.colorbar.title.text).toBe('Seconds');
  });

  it('labels the colorbar with round-number seconds ticks in log space', () => {
    const heavyTail = [
      row({ case_id: 'case_a', method: 'LightGBM', seconds: 1 }),
      row({ case_id: 'case_a', method: 'CatBoost', seconds: 900 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(heavyTail, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const heatmap = result.current!.data[0] as {
      colorbar: { tickmode: string; tickvals: number[]; ticktext: string[] };
    };
    expect(heatmap.colorbar.tickmode).toBe('array');
    expect(heatmap.colorbar.ticktext).toEqual(['0', '1', '10', '100', '1000']);
    expect(heatmap.colorbar.tickvals).toEqual([
      Math.log10(1),
      Math.log10(2),
      Math.log10(11),
      Math.log10(101),
      Math.log10(1001),
    ]);
  });

  it('pivots the showGlobal swap-in cell on seconds too, still log-colored', () => {
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
    expect(heatmap.z).toEqual([[Math.log10(9.64)]]);
  });

  it('keeps the plot area pixel-identical to the normal metric view', () => {
    const metricResult = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    ).result;
    const overlayResult = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    ).result;
    expect(overlayResult.current!.layout.width).toBe(metricResult.current!.layout.width);
    expect(overlayResult.current!.layout.height).toBe(metricResult.current!.layout.height);
    expect(overlayResult.current!.layout.margin).toEqual(metricResult.current!.layout.margin);
  });
});
