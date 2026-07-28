import { describe, expect, it } from 'vitest';
import {
  caseHasExtendedHistory,
  casesEligibleForMetric,
  domainOptions,
  domainsEligibleForMetric,
  effectiveMethod,
  familyOptions,
  groupBoundaries,
  groupLabels,
  metricOptions,
  orderedCases,
  orderedMethods,
  pivotValues,
  rowBestMethods,
} from './selectors';
import type { ScoreRow } from '../types/data';

function row(overrides: Partial<ScoreRow>): ScoreRow {
  return {
    case_id: 'case_a',
    domain: 'water_distribution',
    case_exo_tier: 'none',
    method: 'Mean',
    family: 'baseline',
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

describe('orderedCases', () => {
  it('sorts by domain then case_id', () => {
    const rows = [
      row({ case_id: 'z_case', domain: 'a_domain' }),
      row({ case_id: 'b_case', domain: 'b_domain' }),
      row({ case_id: 'a_case', domain: 'a_domain' }),
    ];
    expect(orderedCases(rows)).toEqual(['a_case', 'z_case', 'b_case']);
  });
});

describe('orderedMethods', () => {
  it('sorts by family rank then name and excludes +exo variants', () => {
    const rows = [
      row({ method: 'LightGBM', family: 'ml' }),
      row({ method: 'Persistence', family: 'baseline' }),
      row({ method: 'LightGBM+exo', family: 'ml' }),
      row({ method: 'AutoARIMA', family: 'classical' }),
    ];
    expect(orderedMethods(rows)).toEqual(['Persistence', 'AutoARIMA', 'LightGBM']);
  });
});

describe('casesEligibleForMetric', () => {
  const domainMetrics = new Map([
    ['flow_case', ['NSE', 'KGE']],
    ['wave_case', []],
  ]);
  const restricted = new Set(['NSE', 'KGE']);

  it('is a no-op for unrestricted metrics', () => {
    expect(
      casesEligibleForMetric('MASE', ['flow_case', 'wave_case'], domainMetrics, restricted),
    ).toEqual(['flow_case', 'wave_case']);
  });

  it('filters to declaring cases for restricted metrics', () => {
    expect(
      casesEligibleForMetric('NSE', ['flow_case', 'wave_case'], domainMetrics, restricted),
    ).toEqual(['flow_case']);
  });
});

describe('domainsEligibleForMetric', () => {
  const rows = [
    row({ case_id: 'flow_case', domain: 'flow' }),
    row({ case_id: 'wave_case', domain: 'waves' }),
  ];
  const domainMetrics = new Map([
    ['flow_case', ['NSE', 'KGE']],
    ['wave_case', []],
  ]);
  const restricted = new Set(['NSE', 'KGE']);

  it('is a no-op for unrestricted metrics', () => {
    expect(domainsEligibleForMetric('MASE', rows, domainMetrics, restricted)).toEqual(
      new Set(['flow', 'waves']),
    );
  });

  it('restricts to domains with at least one declaring case', () => {
    expect(domainsEligibleForMetric('NSE', rows, domainMetrics, restricted)).toEqual(
      new Set(['flow']),
    );
  });
});

describe('metricOptions', () => {
  it('follows METRIC_CONFIG order, restricted to metrics present in the data', () => {
    const rows = [row({ metric: 'RMSE' }), row({ metric: 'MASE' }), row({ metric: 'NSE' })];
    expect(metricOptions(rows)).toEqual(['MASE', 'RMSE', 'NSE']);
  });
});

describe('domainOptions', () => {
  it('returns unique domains, alphabetical', () => {
    const rows = [
      row({ domain: 'z_domain' }),
      row({ domain: 'a_domain' }),
      row({ domain: 'a_domain' }),
    ];
    expect(domainOptions(rows)).toEqual(['a_domain', 'z_domain']);
  });
});

describe('familyOptions', () => {
  it('orders by FAMILY_ORDER then leftover families alphabetically', () => {
    const rows = [
      row({ family: 'dl' }),
      row({ family: 'baseline' }),
      row({ family: 'unclassified' }),
      row({ family: 'ml' }),
    ];
    expect(familyOptions(rows)).toEqual(['baseline', 'ml', 'dl', 'unclassified']);
  });
});

describe('caseHasExtendedHistory', () => {
  it('is true when some row for the case has used_extended_history', () => {
    const rows = [
      row({ case_id: 'c1', used_extended_history: false }),
      row({ case_id: 'c1', method: 'LightGBM', used_extended_history: true }),
    ];
    expect(caseHasExtendedHistory(rows, 'c1')).toBe(true);
  });

  it('is false when no row for the case has used_extended_history', () => {
    const rows = [row({ case_id: 'c1', used_extended_history: false })];
    expect(caseHasExtendedHistory(rows, 'c1')).toBe(false);
  });

  it('is false for an unknown case', () => {
    const rows = [row({ case_id: 'c1', used_extended_history: true })];
    expect(caseHasExtendedHistory(rows, 'c2')).toBe(false);
  });
});

describe('effectiveMethod', () => {
  it('returns the base method when the toggle is off', () => {
    const rows = [row({ method: 'LightGBM+exo', case_exo_tier: 'necessary' })];
    expect(effectiveMethod(rows, 'case_a', 'LightGBM', false)).toBe('LightGBM');
  });

  it('returns the base method when there is no +exo sibling', () => {
    const rows = [row({ method: 'LightGBM', case_exo_tier: 'necessary' })];
    expect(effectiveMethod(rows, 'case_a', 'LightGBM', true)).toBe('LightGBM');
  });

  it('returns the base method when the case has no exogenous data', () => {
    const rows = [
      row({ method: 'LightGBM', case_exo_tier: 'none' }),
      row({ method: 'LightGBM+exo', case_exo_tier: 'none' }),
    ];
    expect(effectiveMethod(rows, 'case_a', 'LightGBM', true)).toBe('LightGBM');
  });

  it('swaps to the +exo sibling when the toggle is on and the case has exogenous data', () => {
    const rows = [
      row({ method: 'LightGBM', case_exo_tier: 'necessary' }),
      row({ method: 'LightGBM+exo', case_exo_tier: 'necessary' }),
    ];
    expect(effectiveMethod(rows, 'case_a', 'LightGBM', true)).toBe('LightGBM+exo');
  });
});

describe('pivotValues', () => {
  it('places values at their case/method position and NaN elsewhere', () => {
    const rows = [
      row({ case_id: 'c1', method: 'm1', value: 1.5 }),
      row({ case_id: 'c2', method: 'm2', value: 2.5 }),
    ];
    const grid = pivotValues(rows, ['c1', 'c2'], ['m1', 'm2']);
    expect(grid).toEqual([
      [1.5, NaN],
      [NaN, 2.5],
    ]);
  });

  it('pivots on a custom field when a value accessor is given', () => {
    const rows = [row({ case_id: 'c1', method: 'm1', value: 1.5, fit_seconds: 4.32 })];
    const grid = pivotValues(rows, ['c1'], ['m1'], (r) => r.fit_seconds);
    expect(grid).toEqual([[4.32]]);
  });
});

describe('groupBoundaries', () => {
  it('returns a boundary at each group change', () => {
    const keys = ['a', 'a', 'b', 'b', 'b', 'c'];
    const keyOf = new Map(keys.map((k, i) => [`k${i}`, k]));
    const orderedKeys = keys.map((_, i) => `k${i}`);
    expect(groupBoundaries(orderedKeys, keyOf)).toEqual([1.5, 4.5]);
  });

  it('returns no boundaries for a single group', () => {
    const orderedKeys = ['k0', 'k1', 'k2'];
    const keyOf = new Map(orderedKeys.map((k) => [k, 'only']));
    expect(groupBoundaries(orderedKeys, keyOf)).toEqual([]);
  });
});

describe('groupLabels', () => {
  it('returns the center position and label of each contiguous run', () => {
    const keys = ['a', 'a', 'b', 'b', 'b'];
    const keyOf = new Map(keys.map((k, i) => [`k${i}`, k]));
    const orderedKeys = keys.map((_, i) => `k${i}`);
    expect(groupLabels(orderedKeys, keyOf)).toEqual([
      [0.5, 'a'],
      [3, 'b'],
    ]);
  });
});

describe('rowBestMethods', () => {
  it('picks the minimum for a lower-is-better metric', () => {
    const values = [[3, 1, 2]];
    expect(rowBestMethods(['c1'], ['m1', 'm2', 'm3'], values, true)).toEqual(
      new Map([['c1', 'm2']]),
    );
  });

  it('picks the maximum for a higher-is-better metric', () => {
    const values = [[3, 1, 2]];
    expect(rowBestMethods(['c1'], ['m1', 'm2', 'm3'], values, false)).toEqual(
      new Map([['c1', 'm1']]),
    );
  });

  it('omits rows with no finite value', () => {
    const values = [[NaN, NaN]];
    expect(rowBestMethods(['c1'], ['m1', 'm2'], values, true)).toEqual(new Map());
  });
});
