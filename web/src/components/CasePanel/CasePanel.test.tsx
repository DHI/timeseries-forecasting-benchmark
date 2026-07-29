import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { CasePanel } from './CasePanel';
import { ThemeProvider } from '../../theme/ThemeContext';
import type { CaseMeta, ScoreRow } from '../../types/data';

// Plotly.js depends on browser APIs jsdom doesn't fully implement (canvas, ResizeObserver);
// stub the Plot component here so this test exercises CasePanel's own logic, not Plotly's
// own rendering (that's covered by manual browser verification instead).
vi.mock('../../plotly', () => ({
  default: () => <div data-testid="plot-stub" />,
}));

const caseMeta: CaseMeta = {
  domain: 'water_distribution',
  target: 'flow',
  target_unit: 'm3/s',
  freq: 'D',
  horizon_primary: '24h',
  exogenous: [],
  exogenous_known_future: [],
  necessary_exogenous: [],
  exo_tier: 'none',
  domain_metrics: [],
  train_tail: { ds: [], y: [] },
  train_tail_extended: { ds: [], y: [] },
  test_split: { ds: [], y: [] },
  split_counts: { train: 70, val: 15, test: 15 },
  split_counts_extended: { train: 70, val: 15, test: 15 },
  distribution: { x_grid: [], train_density: null, test_density: null },
  distribution_extended: { x_grid: [], train_density: null, test_density: null },
  seasonal_patterns: [],
  seasonal_patterns_extended: [],
  source: { name: 'Test source', url: null },
  notes: '',
};

function scoreRow(method: string, family: string, isGlobal = family === 'fm'): ScoreRow {
  return {
    case_id: 'demand_ltown',
    domain: 'water_distribution',
    case_exo_tier: 'none',
    method,
    family,
    uses_exogenous: false,
    is_global: isGlobal,
    used_extended_history: false,
    horizon: '24h',
    metric: 'MASE',
    value: 1.0,
    fit_seconds: 0.005,
    predict_seconds: 0.005,
    frac_finite_test: 1.0,
    status: 'ok',
    n_origins: 40,
  };
}

// Local-vs-global has to be read off `is_global` rather than inferred from `family`: every
// real classical method (AutoARIMA, AutoETS, Theta, MSTL, TBATS) fits once today, but the
// panel's logic must still key off the flag itself in case that ever splits again.
// `MockLocalClassical` is a synthetic fixture for that scenario, not a real forecaster.
const scores: ScoreRow[] = [
  scoreRow('Persistence', 'baseline'),
  scoreRow('TimesFM-2.5', 'fm'),
  scoreRow('AutoARIMA', 'classical', true),
  scoreRow('MockLocalClassical', 'classical', false),
];

describe('CasePanel', () => {
  it('shows no foundation-model note when the selection has none', () => {
    render(
      <ThemeProvider>
        <CasePanel
          caseId="demand_ltown"
          caseMeta={caseMeta}
          scores={scores}
          showGlobal={false}
          selectedMethods={['Persistence']}
        />
      </ThemeProvider>,
    );

    expect(screen.queryByText(/zero-shot foundation model/)).not.toBeInTheDocument();
  });

  it('notes that a selected foundation model does not use the train/val split', () => {
    render(
      <ThemeProvider>
        <CasePanel
          caseId="demand_ltown"
          caseMeta={caseMeta}
          scores={scores}
          showGlobal={false}
          selectedMethods={['Persistence', 'TimesFM-2.5']}
        />
      </ThemeProvider>,
    );

    expect(screen.getByText(/is a zero-shot foundation model/)).toBeInTheDocument();
    expect(screen.getByText('TimesFM-2.5')).toBeInTheDocument();
  });

  it('notes that a selected local method refits at every origin', () => {
    render(
      <ThemeProvider>
        <CasePanel
          caseId="demand_ltown"
          caseMeta={caseMeta}
          scores={scores}
          showGlobal={false}
          selectedMethods={['Persistence']}
        />
      </ThemeProvider>,
    );

    expect(screen.getByText(/is a local method/)).toBeInTheDocument();
    expect(screen.getByText('Persistence')).toBeInTheDocument();
  });

  it('does not call a fit-once classical method local', () => {
    render(
      <ThemeProvider>
        <CasePanel
          caseId="demand_ltown"
          caseMeta={caseMeta}
          scores={scores}
          showGlobal={false}
          selectedMethods={['AutoARIMA']}
        />
      </ThemeProvider>,
    );

    expect(screen.queryByText(/is a local method/)).not.toBeInTheDocument();
  });

  it('calls a refit-per-origin classical method local even though its family is mixed', () => {
    render(
      <ThemeProvider>
        <CasePanel
          caseId="demand_ltown"
          caseMeta={caseMeta}
          scores={scores}
          showGlobal={false}
          selectedMethods={['AutoARIMA', 'MockLocalClassical']}
        />
      </ThemeProvider>,
    );

    expect(screen.getByText(/is a local method/)).toBeInTheDocument();
    expect(screen.getByText('MockLocalClassical')).toBeInTheDocument();
    expect(screen.queryByText('AutoARIMA')).not.toBeInTheDocument();
  });

  it('shows both notes when local and foundation methods are selected together', () => {
    render(
      <ThemeProvider>
        <CasePanel
          caseId="demand_ltown"
          caseMeta={caseMeta}
          scores={scores}
          showGlobal={false}
          selectedMethods={['Persistence', 'TimesFM-2.5']}
        />
      </ThemeProvider>,
    );

    expect(screen.getByText(/is a zero-shot foundation model/)).toBeInTheDocument();
    expect(screen.getByText(/is a local method/)).toBeInTheDocument();
  });
});
