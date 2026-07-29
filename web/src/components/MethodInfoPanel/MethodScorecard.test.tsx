import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { CapabilityRow, ScorecardMeters } from './MethodScorecard';
import type { Capabilities, Scorecard } from '../../types/data';

const capabilities: Capabilities = {
  trend: true,
  seasonality: true,
  multiple_seasonality: false,
  exogenous: false,
};

const scorecard: Scorecard = {
  history_requirements: 'Low',
  interpretability: 'Low',
  fit_cost: 'Low',
  inference_cost: 'High',
  tuning_effort: 'Medium',
  robustness_to_noise: 'High',
};

describe('CapabilityRow', () => {
  it('lists unsupported capabilities rather than omitting them', () => {
    render(<CapabilityRow capabilities={capabilities} />);

    expect(screen.getByLabelText('Trend: yes')).toBeInTheDocument();
    expect(screen.getByLabelText('Multiple seasonality: no')).toBeInTheDocument();
    expect(screen.getByLabelText('Exogenous: no')).toBeInTheDocument();
  });
});

describe('ScorecardMeters', () => {
  it('sets bar length from the level regardless of polarity', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    // Low -> 1 of 3 filled, High -> 3, Medium -> 2. Length is magnitude, not goodness, so a
    // Low fit cost draws the same single segment as a Low interpretability.
    expect(screen.getAllByTestId('segment-fit_cost-filled')).toHaveLength(1);
    expect(screen.getAllByTestId('segment-interpretability-filled')).toHaveLength(1);
    expect(screen.getAllByTestId('segment-inference_cost-filled')).toHaveLength(3);
    expect(screen.getAllByTestId('segment-tuning_effort-filled')).toHaveLength(2);
  });

  it('colours by goodness, so the same level reads differently per field', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    const toneOf = (key: keyof Scorecard): string => {
      const [first] = screen.getAllByTestId(`segment-${key}-filled`);
      return first!.style.background;
    };

    // fit_cost is lowerIsBetter, interpretability is not — both are 'Low' here, so the two
    // must not share a colour. This is the encoding the whole scorecard rests on.
    expect(toneOf('fit_cost')).toBe('var(--level-good)');
    expect(toneOf('interpretability')).toBe('var(--level-poor)');

    // And a High inference cost is the bad direction, unlike a High robustness.
    expect(toneOf('inference_cost')).toBe('var(--level-poor)');
    expect(toneOf('robustness_to_noise')).toBe('var(--level-good)');
  });

  it('does not spell the level out as text — the bar carries it', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    for (const level of ['Low', 'Medium', 'High']) {
      expect(screen.queryByText(level)).not.toBeInTheDocument();
    }
  });

  it('labels each meter with its field and level for screen readers', () => {
    render(<ScorecardMeters scorecard={scorecard} />);

    expect(screen.getByLabelText('Inference cost: High')).toBeInTheDocument();
    expect(screen.getByLabelText('Tuning effort: Medium')).toBeInTheDocument();
  });
});
