// Port of METRIC_CONFIG from src/tsforecastbench/bench/viz/matrix.py. `vlo`/`vhi` clamp
// only the colour scale, not the printed value — a single catastrophic cell shouldn't
// compress the whole colormap so every meaningful difference looks identical.
export type MetricConfig = {
  cmap: 'RdYlGn_r' | 'RdYlGn' | 'Reds' | 'Greys';
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
    cmap: 'Reds',
    center: null,
    label: 'MAE (lower is better)',
    format: '.2f',
    colorScale: 'log',
    unit: 'target unit',
    lowerIsBetter: true,
  },
  RMSE: {
    cmap: 'Reds',
    center: null,
    label: 'RMSE (lower is better)',
    format: '.2f',
    colorScale: 'log',
    unit: 'target unit',
    lowerIsBetter: true,
  },
  MAPE: {
    cmap: 'Reds',
    center: null,
    label: 'MAPE (lower is better)',
    format: '.0f',
    scale: 100,
    colorScale: 'log',
    unit: '%',
    lowerIsBetter: true,
  },
  sMAPE: {
    cmap: 'Reds',
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
 * 'Train<br>time<br>[seconds]' string in useMatrixFigure.ts. */
export const TRAINING_TIME_CONFIG: MetricConfig = {
  cmap: 'Greys',
  center: null,
  label: 'Training time (fit + predict, pooled across rolling origins)',
  format: '.0f',
  unit: 's',
  lowerIsBetter: true,
};

const FALLBACK_CONFIG: MetricConfig = {
  cmap: 'Reds',
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
