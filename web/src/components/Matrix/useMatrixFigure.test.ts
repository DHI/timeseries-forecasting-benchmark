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
    fit_seconds: 0.08,
    predict_seconds: 0.02,
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

// Both swaps write the same cell — the +exo sibling has no column of its own — so the
// exogenous swap, which runs second, used to overwrite the extended-history value with
// the sibling's local one and leave the full-history checkbox with nothing to change.
describe('useMatrixFigure — extended-history and exogenous swaps together', () => {
  const globalExo = (overrides: Partial<ScoreRow>) =>
    row({ is_global: true, case_exo_tier: 'necessary', ...overrides });

  const rows = [
    globalExo({ method: 'LightGBM', used_extended_history: false, value: 1.0 }),
    globalExo({ method: 'LightGBM', used_extended_history: true, value: 9.0 }),
    globalExo({ method: 'LightGBM+exo', used_extended_history: false, value: 0.5 }),
    globalExo({ method: 'LightGBM+exo', used_extended_history: true, value: 0.25 }),
  ];

  const fillcolors = (result: { layout: { shapes?: Array<{ fillcolor?: unknown }> } }) =>
    (result.layout.shapes ?? []).map((s) => s.fillcolor);

  it('shows the +exo sibling’s extended value and marks both swaps', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, true, true, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[Math.log10(0.25)]]);
    expect(fillcolors(result.current!)).toEqual(expect.arrayContaining(['#FF1493', '#0B7285']));
  });

  it('falls back to the +exo local value and drops the pink marker when the sibling has no extended row', () => {
    const noExoExtended = rows.filter(
      (r) => !(r.method === 'LightGBM+exo' && r.used_extended_history),
    );
    const { result } = renderHook(() =>
      useMatrixFigure(noExoExtended, 'MAE', EMPTY_SELECTION, true, true, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[Math.log10(0.5)]]);
    expect(fillcolors(result.current!)).toEqual(['#0B7285']);
  });

  it('leaves the base method’s extended value in place for a case with no exogenous data', () => {
    const noExoCase = rows.map((r) => ({ ...r, case_exo_tier: 'none' }));
    const { result } = renderHook(() =>
      useMatrixFigure(noExoCase, 'MAE', EMPTY_SELECTION, true, true, false),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[Math.log10(9.0)]]);
    expect(fillcolors(result.current!)).toEqual(['#FF1493']);
  });

  it('pivots the combined swap-in cell on training time too', () => {
    const timed = rows.map((r) =>
      r.method === 'LightGBM+exo' && r.used_extended_history
        ? { ...r, fit_seconds: 8.6, predict_seconds: 0.4 }
        : r,
    );
    const { result } = renderHook(() =>
      useMatrixFigure(timed, 'MAE', EMPTY_SELECTION, true, true, true),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    expect(heatmap.z).toEqual([[Math.log10(10.0)]]);
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

describe('useMatrixFigure — colour range adapts to the filtered/visible cells', () => {
  function heatmapRange(result: { current: { data: unknown[] } | null }) {
    return result.current!.data[0] as { zmin: number; zmax: number; zmid: number };
  }

  it('shrinks MASE’s colour range to the visible data instead of its full [0, 2] cap', () => {
    const rows = [
      row({ metric: 'MASE', case_id: 'case_a', method: 'CatBoost', value: 0.8 }),
      row({ metric: 'MASE', case_id: 'case_a', method: 'LightGBM', value: 1.2 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MASE', EMPTY_SELECTION, false, false, false),
    );
    const { zmin, zmax, zmid } = heatmapRange(result);
    expect(zmin).toBeCloseTo(0.8);
    expect(zmax).toBeCloseTo(1.2);
    expect(zmid).toBe(1.0);
  });

  it('still caps MASE’s colour range at [0, 2] when a catastrophic outlier is visible', () => {
    const rows = [
      row({ metric: 'MASE', case_id: 'case_a', method: 'CatBoost', value: 1.0 }),
      row({ metric: 'MASE', case_id: 'case_a', method: 'LightGBM', value: 10 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MASE', EMPTY_SELECTION, false, false, false),
    );
    const { zmin, zmax } = heatmapRange(result);
    expect(zmin).toBeCloseTo(1.0);
    expect(zmax).toBe(2);
  });

  it('keeps the centre anchored even when every visible value sits on one side of it', () => {
    const rows = [
      row({ metric: 'Skill_vs_Persistence', case_id: 'case_a', method: 'CatBoost', value: 0.3 }),
      row({ metric: 'Skill_vs_Persistence', case_id: 'case_a', method: 'LightGBM', value: 0.6 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'Skill_vs_Persistence', EMPTY_SELECTION, false, false, false),
    );
    const { zmin, zmax, zmid } = heatmapRange(result);
    expect(zmid).toBe(0);
    expect(zmin).toBe(0);
    expect(zmax).toBeCloseTo(0.6);
  });
});

describe('useMatrixFigure — training-time overlay', () => {
  const rows = [
    row({
      case_id: 'case_a',
      method: 'LightGBM',
      value: 1.0,
      fit_seconds: 4.32,
      predict_seconds: 0,
    }),
    row({
      case_id: 'case_a',
      method: 'CatBoost',
      value: 2.0,
      fit_seconds: 0.5,
      predict_seconds: 0,
    }),
  ];

  it('colors cells by log10(fit_seconds + predict_seconds + 1), not the raw value', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const heatmap = result.current!.data[0] as { z: number[][] };
    // colOrder sorts alphabetically within a family: CatBoost before LightGBM.
    expect(heatmap.z).toEqual([[Math.log10(1.5), Math.log10(5.32)]]);
  });

  it('formats cell text as a plain one-decimal number and skips fastest-method bolding', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const texts = result.current!.layout.annotations!.map((a) => String(a.text));
    expect(texts).toContain('4.3');
    expect(texts).toContain('0.5');
    expect(texts).not.toContain('<b>4.3</b>');
    expect(texts).not.toContain('<b>0.5</b>');
  });

  it('shows a short training-time colorbar title', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    const heatmap = result.current!.data[0] as {
      colorbar: { title: { text: string } };
    };
    expect(heatmap.colorbar.title.text).toBe('Time per<br>forecast<br>[s]');
  });

  it('labels the colorbar with round-number seconds ticks in log space', () => {
    const heavyTail = [
      row({ case_id: 'case_a', method: 'LightGBM', fit_seconds: 1, predict_seconds: 0 }),
      row({ case_id: 'case_a', method: 'CatBoost', fit_seconds: 900, predict_seconds: 0 }),
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

  it('pivots the showGlobal swap-in cell on training time too, still log-colored', () => {
    const globalRows = [
      row({
        case_id: 'case_a',
        method: 'LightGBM',
        used_extended_history: false,
        is_global: true,
        value: 1.0,
        fit_seconds: 4.32,
        predict_seconds: 0,
      }),
      row({
        case_id: 'case_a',
        method: 'LightGBM',
        used_extended_history: true,
        is_global: true,
        value: 9.0,
        fit_seconds: 8.64,
        predict_seconds: 0,
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

  it('exposes the fit/predict split per case+method for the hover breakdown', () => {
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    expect(result.current!.trainingBreakdownOf.get('case_a|LightGBM')).toEqual({
      fitSeconds: 4.32,
      predictSeconds: 0,
    });
    expect(result.current!.trainingBreakdownOf.get('case_a|CatBoost')).toEqual({
      fitSeconds: 0.5,
      predictSeconds: 0,
    });
    expect(result.current!.trainingBreakdownOf.get('case_a|Nonexistent')).toBeUndefined();
  });
});

describe('useMatrixFigure — training-time overlay cell text color', () => {
  function annotationFontColor(
    result: { layout: { annotations?: unknown[] } },
    text: string,
  ): string | undefined {
    const match = result.layout.annotations!.find((a) => (a as { text: string }).text === text);
    return (match as { font: { color?: string } }).font.color;
  }

  it('switches to light text on the darkest (slowest) cell but not the lightest (fastest) one', () => {
    const rows = [
      row({
        case_id: 'case_a',
        method: 'CatBoost',
        value: 1.0,
        fit_seconds: 0.01,
        predict_seconds: 0,
      }),
      row({
        case_id: 'case_a',
        method: 'LightGBM',
        value: 1.0,
        fit_seconds: 10000,
        predict_seconds: 0,
      }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, true),
    );
    expect(annotationFontColor(result.current!, '0.0')).toBe('#000000');
    expect(annotationFontColor(result.current!, '10000.0')).toBe('#f5f5f5');
  });

  it('does not override text color for non-greyscale metric views', () => {
    const rows = [
      row({ case_id: 'case_a', method: 'CatBoost', value: 0.01 }),
      row({ case_id: 'case_a', method: 'LightGBM', value: 10000 }),
    ];
    const { result } = renderHook(() =>
      useMatrixFigure(rows, 'MAE', EMPTY_SELECTION, false, false, false),
    );
    // MAE's RdYlGn_r colormap gets just as saturated at high values, but only the
    // training-time overlay's Greys config opts into the readability fix — CatBoost (0.01)
    // is this row's best (lowest) value and so is bolded, LightGBM (10000) is the dark cell.
    expect(annotationFontColor(result.current!, '<b>0.01</b>')).toBe('#000000');
    expect(annotationFontColor(result.current!, '10000.00')).toBe('#000000');
  });
});
