import type {
  Capabilities,
  ForecastStrategy,
  GrowthShape,
  Scorecard,
  ScorecardLevel,
} from '../types/data';

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
    help: 'Has a mechanism for a repeating within-period shape, whether an explicit seasonal term or a window long enough to learn one. Unlike Exogenous, this describes the method as configured in this benchmark — AutoETS and AutoARIMA run non-seasonally here (season_length=1), which is a choice about search cost, not a property of exponential smoothing.',
  },
  {
    key: 'multiple_seasonality',
    label: 'Multiple seasonality',
    help: 'Handles several periods at once, e.g. a daily and a weekly cycle together. Like Seasonality, this describes the method as configured here.',
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
    key: 'running_cost',
    label: 'Running cost',
    lowerIsBetter: true,
    help: 'What it costs to actually use the method: one fit plus 1,000 forecasts on a 100,000-point record, read off the fit and predict times this benchmark measures. Low: under a minute. Medium: one to fifteen minutes. High: beyond that. One field rather than a fit/inference pair because whichever term is larger dominates — an expensive fit amortises over many forecasts, an expensive forecast does not.',
  },
  {
    key: 'tuning_effort',
    label: 'Tuning effort',
    lowerIsBetter: true,
    help: 'How many choices you must get right before it works. Low: nothing that matters, or auto-selected. High: several interacting hyperparameters where a bad pick quietly costs accuracy.',
  },
];

/** Per-value help text for the growth shape shown beside the running-cost meter. The level
 * is read at one reference record size, so on its own it cannot tell a cost that ignores
 * record length from one that outruns it — this is the word that does. */
export const GROWTH_SHAPE_HELP: Record<GrowthShape, string> = {
  flat: 'Cost barely moves with record length — these train to a step ceiling or run one pass over a fixed context, so a short record costs about what a long one does.',
  linear: 'Cost tracks the record: roughly twice the data, roughly twice the time.',
  superlinear:
    'Cost outruns the record — doubling the data more than doubles the time, so a method that is free on a short record can dominate your run on a long one.',
};

/** Per-value help text for the Forecast strategy badge — see `ForecastStrategy` in
 * types/data.ts and the Python rubric it mirrors (scorecard.py). */
export const FORECAST_STRATEGY_HELP: Record<ForecastStrategy, string> = {
  Recursive:
    'Each step of the forecast feeds forward into computing a later step: mlforecast’s default lag-feedback strategy, or a classical state-space recursion substituting its own prior forecast for an unobserved future value.',
  Direct:
    'One model call produces the whole horizon at once — nothing computed for one step feeds into computing another.',
  'N/A':
    'The distinction carries no information for this method — a naive baseline that looks up or repeats a value independently per timestamp, or a composite of several methods.',
};

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
