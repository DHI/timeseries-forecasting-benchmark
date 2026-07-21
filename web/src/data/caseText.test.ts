import { describe, expect, it } from 'vitest';
import { caseSummary, caseTrainingStats } from './caseText';
import type { CaseMeta } from '../types/data';

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
    seasonal_patterns: [],
    seasonal_patterns_extended: [],
    source: { name: 'BattLeDIM L-Town', url: 'https://battledim.ucy.ac.cy/' },
    notes: '',
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

describe('caseTrainingStats', () => {
  it('renders the samples line', () => {
    const stats = caseTrainingStats(caseMeta());
    expect(stats).toContain('train 12,264');
    expect(stats).not.toContain('Training time');
  });
});
