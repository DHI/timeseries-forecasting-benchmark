import { describe, expect, it } from 'vitest';
import { methodInfo, methodInfoDetail } from './methodText';
import type { CaseMeta, MethodMeta } from '../types/data';

function caseMeta(overrides: Partial<CaseMeta> = {}): CaseMeta {
  return {
    domain: 'water_distribution',
    target: 'inlet_flow',
    target_unit: 'L/s',
    freq: '1h',
    horizon_primary: '24h',
    exogenous: [],
    exogenous_known_future: [],
    necessary_exogenous: [],
    exo_tier: 'none',
    domain_metrics: [],
    train_tail: { ds: [], y: [] },
    train_tail_extended: { ds: [], y: [] },
    test_split: { ds: [], y: [] },
    split_counts: { train: 0, val: 0, test: 0 },
    split_counts_extended: { train: 0, val: 0, test: 0 },
    distribution: { x_grid: [], train_density: null, test_density: null },
    distribution_extended: { x_grid: [], train_density: null, test_density: null },
    seasonal_patterns: [],
    seasonal_patterns_extended: [],
    source: { name: 'Test', url: null },
    notes: '',
    ...overrides,
  };
}

function methodMeta(overrides: Partial<MethodMeta> = {}): MethodMeta {
  return {
    family: 'ml',
    description: 'Gradient boosting.',
    uses_exogenous: false,
    library: '',
    compute_profile: '',
    limitations: '',
    code_snippet: '',
    capabilities: {
      trend: false,
      seasonality: false,
      multiple_seasonality: false,
      exogenous: false,
      gaps: false,
    },
    scorecard: {
      history_requirements: 'Low',
      interpretability: 'High',
      fit_cost: 'Low',
      inference_cost: 'Low',
      tuning_effort: 'Low',
      robustness_to_noise: 'Medium',
    },
    forecast_strategy: 'Recursive',
    ...overrides,
  };
}

describe('methodInfo', () => {
  it('reports "none" when the method does not use exogenous inputs', () => {
    const info = methodInfo(caseMeta(), methodMeta({ uses_exogenous: false }));
    expect(info).toContain('**Exogenous inputs used**: none');
  });

  it('lists exogenous columns, flagging necessary ones', () => {
    const meta = caseMeta({ exogenous: ['rain', 'temp'], necessary_exogenous: ['rain'] });
    const info = methodInfo(meta, methodMeta({ uses_exogenous: true }));
    expect(info).toContain('rain (necessary), temp');
  });

  it('falls back to a placeholder description when unknown', () => {
    const info = methodInfo(caseMeta(), undefined);
    expect(info).toContain('No description yet.');
    expect(info).not.toContain('Family');
  });
});

describe('methodInfoDetail', () => {
  it('omits blank fields', () => {
    const detail = methodInfoDetail(methodMeta({ library: 'statsforecast' }));
    expect(detail).toBe('**Library**: statsforecast');
  });

  it('renders a fenced code block for the snippet', () => {
    const detail = methodInfoDetail(methodMeta({ code_snippet: 'model.fit(x)' }));
    expect(detail).toContain('```python\nmodel.fit(x)\n```');
  });

  it('falls back to a placeholder when every field is blank', () => {
    expect(methodInfoDetail(methodMeta())).toBe('No implementation detail yet.');
  });
});
