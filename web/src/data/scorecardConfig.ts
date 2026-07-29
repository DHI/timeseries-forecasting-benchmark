import type { Capabilities, Scorecard, ScorecardLevel } from '../types/data';

/** Display config for the method card's Handles row and Scorecard meters: labels, order,
 * polarity, and the tooltip text explaining each field. Same split of concerns as
 * metricConfig.ts — the Python models carry the values, this file carries how they read.
 *
 * The field names are duplicated from src/tsforecastbench/bench/methods/scorecard.py.
 * tests/test_method_scorecards.py pins them on the Python side so a rename there fails
 * loudly instead of silently dropping a row here. */

export type CapabilityField = {
  key: keyof Capabilities;
  label: string;
  help: string;
};

export const CAPABILITY_FIELDS: readonly CapabilityField[] = [
  {
    key: 'trend',
    label: 'Trend',
    help: 'Has a mechanism that extrapolates level drift, rather than reverting to the mean of the history it saw.',
  },
  {
    key: 'seasonality',
    label: 'Seasonality',
    help: 'Has a mechanism for a repeating within-period shape, whether an explicit seasonal term or a window long enough to learn one.',
  },
  {
    key: 'multiple_seasonality',
    label: 'Multiple seasonality',
    help: 'Handles several periods at once, e.g. a daily and a weekly cycle together.',
  },
  {
    key: 'exogenous',
    label: 'Exogenous',
    help: 'Can consume covariates in principle. This describes the method, not this column — a method shown here without covariates may still support them (that is what its +exo column is).',
  },
  {
    key: 'gaps',
    label: 'Gaps',
    help: 'Has a fit path that pools history from before a gap in the record. This only shows a difference in the extended-history matrix — the shared local matrix is always gap-free by the time any method sees it.',
  },
];

export type ScorecardField = {
  key: keyof Scorecard;
  label: string;
  /** True where a low rating is the good one (cost and effort), false where a high one is. */
  lowerIsBetter: boolean;
  help: string;
};

export const SCORECARD_FIELDS: readonly ScorecardField[] = [
  {
    key: 'history_requirements',
    label: 'History needed',
    lowerIsBetter: true,
    help: 'How much history before the method is worth using. Low: usable from a handful of points, or needs no training data at all. High: needs thousands of points before it beats a cheap baseline.',
  },
  {
    key: 'interpretability',
    label: 'Interpretability',
    lowerIsBetter: false,
    help: 'How readable the forecast’s reasoning is. High: it follows from a rule or a few named parameters. Medium: inspectable structure. Low: a black box.',
  },
  {
    key: 'fit_cost',
    label: 'Fit cost',
    lowerIsBetter: true,
    help: 'Cost of the one-off fit, anchored to the fit_seconds this benchmark measures. Low: seconds on CPU. High: many minutes, or needs a GPU to be practical.',
  },
  {
    key: 'inference_cost',
    label: 'Inference cost',
    lowerIsBetter: true,
    help: 'Cost per forecast origin, anchored to the measured predict_seconds. Low: milliseconds. High: seconds or more per origin, which dominates the total time over a rolling run.',
  },
  {
    key: 'tuning_effort',
    label: 'Tuning effort',
    lowerIsBetter: true,
    help: 'How many choices you must get right before it works. Low: nothing that matters, or auto-selected. High: several interacting hyperparameters where a bad pick quietly costs accuracy.',
  },
  {
    key: 'robustness_to_noise',
    label: 'Robustness to noise',
    lowerIsBetter: false,
    help: 'Tolerance to noise, outliers and gaps. High: barely moves for one bad point. Low: fits the noise, or lets a single outlier propagate across the horizon.',
  },
];

/** Bar length: the magnitude of the rating, independent of whether that magnitude is good. */
export const LEVEL_SEGMENTS: Record<ScorecardLevel, number> = { Low: 1, Medium: 2, High: 3 };

export const SEGMENT_COUNT = 3;

export type Tone = 'good' | 'mid' | 'poor';

/** Bar colour: whether the rating is a good one for this field. A Low fit cost and a High
 * interpretability are both 'good', which is why polarity lives on the field and not on the
 * level. */
export function levelTone(level: ScorecardLevel, lowerIsBetter: boolean): Tone {
  if (level === 'Medium') return 'mid';
  const isBest = lowerIsBetter ? level === 'Low' : level === 'High';
  return isBest ? 'good' : 'poor';
}

export const TONE_VAR: Record<Tone, string> = {
  good: 'var(--level-good)',
  mid: 'var(--level-mid)',
  poor: 'var(--level-poor)',
};
