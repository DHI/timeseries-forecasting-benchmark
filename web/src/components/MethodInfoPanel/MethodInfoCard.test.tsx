import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { MethodInfoCard } from './MethodInfoCard';
import type { CaseMeta, MethodMeta } from '../../types/data';

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
  split_counts: { train: 0, val: 0, test: 0 },
  split_counts_extended: { train: 0, val: 0, test: 0 },
  distribution: { x_grid: [], train_density: null, test_density: null },
  distribution_extended: { x_grid: [], train_density: null, test_density: null },
  seasonal_patterns: [],
  seasonal_patterns_extended: [],
  source: { name: 'Test source', url: null },
  notes: '',
};

const methodMeta: MethodMeta = {
  family: 'ml',
  description: 'A test method.',
  uses_exogenous: false,
  library: 'test-lib',
  compute_profile: 'cpu',
  limitations: 'None.',
  code_snippet: 'fit(x, y)',
  capabilities: {
    trend: true,
    seasonality: true,
    multiple_seasonality: false,
    exogenous: true,
    gaps: false,
  },
  scorecard: {
    history_requirements: 'High',
    interpretability: 'Medium',
    fit_cost: 'Low',
    inference_cost: 'Low',
    tuning_effort: 'Medium',
    robustness_to_noise: 'Medium',
  },
};

describe('MethodInfoCard', () => {
  it('shows its content and a resize handle by default, and hides both when collapsed', async () => {
    const user = userEvent.setup();
    render(
      <MethodInfoCard method="LightGBM" index={0} caseMeta={caseMeta} methodMeta={methodMeta} />,
    );

    expect(screen.getByText('A test method.')).toBeInTheDocument();
    expect(screen.getByTitle('Collapse (drag to resize)')).toBeInTheDocument();

    await user.click(screen.getByTitle('Collapse (drag to resize)'));

    expect(screen.queryByText('A test method.')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Collapse (drag to resize)')).not.toBeInTheDocument();
    expect(screen.getByText('LightGBM')).toBeInTheDocument();

    await user.click(screen.getByTitle('Expand'));
    expect(screen.getByText('A test method.')).toBeInTheDocument();
  });

  it('shows the capability row and the scorecard meters', () => {
    render(
      <MethodInfoCard method="LightGBM" index={0} caseMeta={caseMeta} methodMeta={methodMeta} />,
    );

    expect(screen.getByText('Handles')).toBeInTheDocument();
    expect(screen.getByLabelText('Exogenous: yes')).toBeInTheDocument();
    expect(screen.getByText('Scorecard')).toBeInTheDocument();
    expect(screen.getByLabelText('History needed: High')).toBeInTheDocument();
    expect(screen.getByTitle('How to read the scorecard')).toBeInTheDocument();
  });
});
