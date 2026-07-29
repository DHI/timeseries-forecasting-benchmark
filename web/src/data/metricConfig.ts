// Port of METRIC_CONFIG from src/tsforecastbench/bench/viz/matrix.py. `vlo`/`vhi` clamp
// only the colour scale, not the printed value — a single catastrophic cell shouldn't
// compress the whole colormap so every meaningful difference looks identical.
export type MetricConfig = {
  cmap: 'RdYlGn_r' | 'RdYlGn' | 'Greys';
  center: number | null;
  vlo?: number;
  vhi?: number;
  label: string;
  /** d3-format spec (not the Python "{:.2f}"-style spec the source config uses), applied to
   * (value * (scale ?? 1)). */
  format: string;
  /** Multiply the value by this before formatting — e.g. 100 for a percent metric whose cell
   * text is a plain number (unit shown once, on the colorbar) rather than d3-format's `%` type.
   * Default 1. */
  scale?: number;
  /** 'log' pivots cell *color* (not cell text) on log10(value) — for heavy-tailed metrics
   * where a linear scale would leave almost every cell looking identical. Default 'linear'. */
  colorScale?: 'linear' | 'log';
  /** Colorbar title's bracketed unit, e.g. 'unitless', '%', 'target unit'. */
  unit: string;
  lowerIsBetter: boolean;
};

export const METRIC_CONFIG: Record<string, MetricConfig> = {
  MASE: {
    cmap: 'RdYlGn_r',
    center: 1.0,
    vlo: 0.0,
    vhi: 2.0,
    label: 'MASE (lower=better; 1 = seasonal naive; colour clamped to [0, 2])',
    format: '.2f',
    unit: 'unitless',
    lowerIsBetter: true,
  },
  Skill_vs_Persistence: {
    cmap: 'RdYlGn',
    center: 0.0,
    vlo: -1.0,
    vhi: 1.0,
    label: 'Skill vs persistence (higher=better; 0 = no better; colour clamped to [-1, 1])',
    format: '+.2f',
    unit: 'unitless',
    lowerIsBetter: false,
  },
  Skill_vs_Mean: {
    cmap: 'RdYlGn',
    center: 0.0,
    vlo: -1.0,
    vhi: 1.0,
    label: 'Skill vs mean (higher=better; 0 = no better; colour clamped to [-1, 1])',
    format: '+.2f',
    unit: 'unitless',
    lowerIsBetter: false,
  },
  MAE: {
    cmap: 'RdYlGn_r',
    center: null,
    label: 'MAE (lower is better)',
    format: '.2f',
    colorScale: 'log',
    unit: 'target unit',
    lowerIsBetter: true,
  },
  RMSE: {
    cmap: 'RdYlGn_r',
    center: null,
    label: 'RMSE (lower is better)',
    format: '.2f',
    colorScale: 'log',
    unit: 'target unit',
    lowerIsBetter: true,
  },
  MAPE: {
    cmap: 'RdYlGn_r',
    center: null,
    label: 'MAPE (lower is better)',
    format: '.0f',
    scale: 100,
    colorScale: 'log',
    unit: '%',
    lowerIsBetter: true,
  },
  sMAPE: {
    cmap: 'RdYlGn_r',
    center: null,
    label: 'sMAPE (lower is better)',
    format: '.0f',
    scale: 100,
    colorScale: 'log',
    unit: '%',
    lowerIsBetter: true,
  },
  // NSE/KGE are domain-restricted (Case.domain_metrics): listed here so they're
  // selectable, but the matrix rows are additionally filtered to only the cases that
  // declare eligibility (see selectors.ts's `casesEligibleForMetric`).
  NSE: {
    cmap: 'RdYlGn',
    center: 0.0,
    vlo: -1.0,
    vhi: 1.0,
    label: 'NSE (higher=better; 1=perfect; 0=no better than the mean; colour clamped to [-1, 1])',
    format: '.2f',
    unit: 'unitless',
    lowerIsBetter: false,
  },
  KGE: {
    cmap: 'RdYlGn',
    center: 0.0,
    vlo: -1.0,
    vhi: 1.0,
    label: 'KGE (higher=better; 1=perfect; colour clamped to [-1, 1])',
    format: '.2f',
    unit: 'unitless',
    lowerIsBetter: false,
  },
};

/** Display config for the training-time heatmap overlay — greyscale (to visually set it
 * apart from the Reds/RdYlGn metric views) /no-center/lower-is-better. Kept separate from
 * METRIC_CONFIG since training time isn't a selectable metric and must not appear in
 * metricOptions(). `unit` is unused: the training-time colorbar title is a hardcoded
 * string in useMatrixFigure.ts.
 *
 * The cell value is `fit_seconds + predict_seconds`, each already averaged per rolling
 * origin (see ScoreRow / eval.rolling) — for a method that refits at every origin
 * (baseline/ensemble, and TBATS), `fit_seconds` is the mean cost of one such refit; for a
 * method that fits once (ML/DL, the other classical ones, zero-shot foundation models),
 * it's that single fit's cost, and for zero-shot methods specifically it's ~0 since they
 * have no case-specific training. `predict_seconds` is not always the small half: Theta and
 * MSTL re-derive a seasonal decomposition over the whole history at every origin, so on a
 * long record their predict cost far exceeds their fit cost. So this reads as "cost to
 * produce one forecast with this method", not a training-time-only number and not something
 * that scales with how many origins a run used. Hovering a cell while this overlay is held
 * shows the fit/predict split (see MatrixGraph's breakdown tooltip). */
export const TRAINING_TIME_CONFIG: MetricConfig = {
  cmap: 'Greys',
  center: null,
  label:
    'Time to produce one forecast (fit, if the method needs it, plus predict — hover a cell for the fit/predict split)',
  format: '.0f',
  unit: 's',
  lowerIsBetter: true,
};

const FALLBACK_CONFIG: MetricConfig = {
  cmap: 'RdYlGn_r',
  center: null,
  label: '',
  format: '.3~g',
  unit: 'target unit',
  lowerIsBetter: true,
};

export function metricConfig(metric: string): MetricConfig {
  return METRIC_CONFIG[metric] ?? { ...FALLBACK_CONFIG, label: metric };
}

/** Display order for method families (column grouping) and case domains (row grouping).
 * Anything not listed sorts after, alphabetically. Mirrors app/data.py's FAMILY_ORDER. */
export const FAMILY_ORDER = ['baseline', 'ensemble', 'classical', 'ml', 'dl', 'fm'];

export const DOMAIN_RESTRICTED_METRICS = new Set(['NSE', 'KGE']);
