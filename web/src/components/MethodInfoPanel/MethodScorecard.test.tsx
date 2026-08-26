import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { CapabilityRow, ForecastStrategyBadge, ScorecardMeters } from './MethodScorecard';
import type { Capabilities, Scorecard } from '../../types/data';

const capabilities: Capabilities = {
  trend: true,
  seasonality: true,
  multiple_seasonality: false,
  exogenous: false,
  gaps: false,
};

const scorecard: Scorecard = {
  history_requirements: 'Low',
  interpretability: 'Low',
  running_cost: { level: 'High', growth: 'superlinear' },
  tuning_effort: 'Medium',
};

describe('CapabilityRow', () => {
  it('lists unsupported capabilities rather than omitting them', () => {
    render(<CapabilityRow capabilities={capabilities} />);

    expect(screen.getByLabelText('Trend: yes')).toBeInTheDocument();
    expect(screen.getByLabelText('Multiple seasonality: no')).toBeInTheDocument();
    expect(screen.getByLabelText('Exogenous: no')).toBeInTheDocument();
    expect(screen.getByLabelText('Gaps: no')).toBeInTheDocument();
  });
});

describe('ScorecardMeters', () => {
  it('sets bar length from the level regardless of polarity', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    // Low -> 1 of 3 filled, High -> 3, Medium -> 2. Length is magnitude, not goodness, so a
    // Low history requirement draws the same single segment as a Low interpretability.
    expect(screen.getAllByTestId('segment-history_requirements-filled')).toHaveLength(1);
    expect(screen.getAllByTestId('segment-interpretability-filled')).toHaveLength(1);
    expect(screen.getAllByTestId('segment-running_cost-filled')).toHaveLength(3);
    expect(screen.getAllByTestId('segment-tuning_effort-filled')).toHaveLength(2);
  });

  it('colours by goodness, so the same level reads differently per field', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    const toneOf = (key: keyof Scorecard): string => {
      const [first] = screen.getAllByTestId(`segment-${key}-filled`);
      return first!.style.background;
    };

    // history_requirements is lowerIsBetter, interpretability is not — both are 'Low' here,
    // so the two must not share a colour. This is the encoding the whole scorecard rests on.
    expect(toneOf('history_requirements')).toBe('var(--level-good)');
    expect(toneOf('interpretability')).toBe('var(--level-poor)');

    // And a High running cost is the bad direction, unlike a High interpretability would be.
    expect(toneOf('running_cost')).toBe('var(--level-poor)');
  });

  it('prints the growth shape beside the running-cost bar and nowhere else', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    // The level is read at one reference record size, so the bar alone can't separate a cost
    // that ignores record length from one that outruns it.
    expect(screen.getByTestId('growth-running_cost')).toHaveTextContent('superlinear');
    expect(screen.queryByTestId('growth-tuning_effort')).not.toBeInTheDocument();
  });

  it('does not spell the level out as text — the bar carries it', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    for (const level of ['Low', 'Medium', 'High']) {
      expect(screen.queryByText(level)).not.toBeInTheDocument();
    }
  });

  it('labels each meter with its field and level for screen readers', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    expect(screen.getByLabelText('Running cost: High')).toBeInTheDocument();
    expect(screen.getByLabelText('Tuning effort: Medium')).toBeInTheDocument();
  });
});

describe('ForecastStrategyBadge', () => {
  it('renders the strategy value and a help title per value', () => {
    render(<ForecastStrategyBadge strategy="Direct" />);

    expect(screen.getByText('Direct')).toBeInTheDocument();
    expect(screen.getByLabelText('Forecast strategy: Direct')).toBeInTheDocument();
  });
});
