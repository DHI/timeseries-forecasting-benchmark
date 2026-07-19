import { describe, expect, it } from 'vitest';
import { caseSummary, caseTrainingStats, methodTrainingSeconds } from './caseText';
import type { CaseMeta, ScoreRow } from '../types/data';

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
    split_counts: { train: 12264, val: 2628, test: 2628 },
    split_counts_extended: { train: 12264, val: 2628, test: 2628 },
    distribution: { x_grid: [], train_density: null, test_density: null },
    distribution_extended: { x_grid: [], train_density: null, test_density: null },
    source: { name: 'BattLeDIM L-Town', url: 'https://battledim.ucy.ac.cy/' },
    notes: '',
    ...overrides,
  };
}

function row(overrides: Partial<ScoreRow>): ScoreRow {
  return {
    case_id: 'demand_ltown',
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
    seconds: 4.32,
    frac_finite_test: 1.0,
    status: 'ok',
    n_origins: 40,
    ...overrides,
  };
}

describe('caseSummary', () => {
  it('renders a link when the source has a URL', () => {
    const summary = caseSummary(caseMeta());
    expect(summary).toContain('[BattLeDIM L-Town](https://battledim.ucy.ac.cy/)');
    expect(summary).toContain('**Domain**: water distribution');
    expect(summary).toContain('**Exogenous inputs**: none');
  });

  it('renders plain source text with no URL', () => {
    const summary = caseSummary(caseMeta({ source: { name: 'Internal dataset', url: null } }));
    expect(summary).toContain('**Source**: Internal dataset');
  });

  it('reports the exogenous count and tier when present', () => {
    const summary = caseSummary(caseMeta({ exogenous: ['rain', 'temp'], exo_tier: 'necessary' }));
    expect(summary).toContain('**Exogenous inputs**: 2 (necessary)');
  });

  it('appends notes as a separate paragraph when present', () => {
    const summary = caseSummary(caseMeta({ notes: 'Some caveat.' }));
    expect(summary.endsWith('Some caveat.')).toBe(true);
  });
});

describe('methodTrainingSeconds', () => {
  it('returns the matching local row', () => {
    const scores = [row({ method: 'LightGBM', seconds: 4.32, used_extended_history: false })];
    expect(methodTrainingSeconds(scores, 'demand_ltown', 'LightGBM')).toBe(4.32);
  });

  it('falls back to the local row when the extended variant is missing', () => {
    const scores = [row({ method: 'LightGBM', seconds: 4.32, used_extended_history: false })];
    expect(methodTrainingSeconds(scores, 'demand_ltown', 'LightGBM', true)).toBe(4.32);
  });

  it('prefers the extended row when it exists and extended is requested', () => {
    const scores = [
      row({ method: 'LightGBM', seconds: 4.32, used_extended_history: false }),
      row({ method: 'LightGBM', seconds: 9.87, used_extended_history: true }),
    ];
    expect(methodTrainingSeconds(scores, 'demand_ltown', 'LightGBM', true)).toBe(9.87);
  });

  it('returns null for an unknown method', () => {
    const scores = [row({ method: 'LightGBM' })];
    expect(methodTrainingSeconds(scores, 'demand_ltown', 'CatBoost')).toBeNull();
  });
});

describe('caseTrainingStats', () => {
  it('shows only the samples line with no methods selected', () => {
    const stats = caseTrainingStats([], caseMeta(), 'demand_ltown', []);
    expect(stats).toContain('train 12,264');
    expect(stats).not.toContain('Training time');
  });

  it('adds a training-time line per selected method, with an em-dash when unavailable', () => {
    const scores = [row({ method: 'LightGBM', seconds: 4.32 })];
    const stats = caseTrainingStats(scores, caseMeta(), 'demand_ltown', ['LightGBM', 'CatBoost']);
    expect(stats).toContain('LightGBM 4.32s');
    expect(stats).toContain('CatBoost —');
  });
});
