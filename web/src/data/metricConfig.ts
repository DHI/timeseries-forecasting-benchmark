// Port of METRIC_CONFIG from src/tsforecastbench/bench/viz/matrix.py. `vlo`/`vhi` clamp
// only the colour scale, not the printed value — a single catastrophic cell shouldn't
// compress the whole colormap so every meaningful difference looks identical.
export type MetricConfig = {
  cmap: 'RdYlGn_r' | 'RdYlGn' | 'Reds';
  center: number | null;
  vlo?: number;
  vhi?: number;
  label: string;
  /** d3-format spec (not the Python "{:.2f}"-style spec the source config uses). */
  format: string;
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
    lowerIsBetter: true,
  },
  Skill_vs_Persistence: {
    cmap: 'RdYlGn',
    center: 0.0,
    vlo: -1.0,
    vhi: 1.0,
    label: 'Skill vs persistence (higher=better; 0 = no better; colour clamped to [-1, 1])',
    format: '+.2f',
    lowerIsBetter: false,
  },
  Skill_vs_Mean: {
    cmap: 'RdYlGn',
    center: 0.0,
    vlo: -1.0,
    vhi: 1.0,
    label: 'Skill vs mean (higher=better; 0 = no better; colour clamped to [-1, 1])',
    format: '+.2f',
    lowerIsBetter: false,
  },
  MAE: {
    cmap: 'Reds',
    center: null,
    label: 'MAE (lower is better)',
    format: '.3~g',
    lowerIsBetter: true,
  },
  RMSE: {
    cmap: 'Reds',
    center: null,
    label: 'RMSE (lower is better)',
    format: '.3~g',
    lowerIsBetter: true,
  },
  MAPE: {
    cmap: 'Reds',
    center: null,
    label: 'MAPE (lower is better)',
    format: '.1%',
    lowerIsBetter: true,
  },
  sMAPE: {
    cmap: 'Reds',
    center: null,
    label: 'sMAPE (lower is better)',
    format: '.1%',
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
    lowerIsBetter: false,
  },
  KGE: {
    cmap: 'RdYlGn',
    center: 0.0,
    vlo: -1.0,
    vhi: 1.0,
    label: 'KGE (higher=better; 1=perfect; colour clamped to [-1, 1])',
    format: '.2f',
    lowerIsBetter: false,
  },
};

/** Display config for the training-time heatmap overlay — Reds/no-center/lower-is-better,
 * mirroring MAE/RMSE. Kept separate from METRIC_CONFIG since training time isn't a
 * selectable metric and must not appear in metricOptions(). */
export const TRAINING_TIME_CONFIG: MetricConfig = {
  cmap: 'Reds',
  center: null,
  label: 'Training time (fit + predict, pooled across rolling origins)',
  format: '.0f',
  lowerIsBetter: true,
};

const FALLBACK_CONFIG: MetricConfig = {
  cmap: 'Reds',
  center: null,
  label: '',
  format: '.3~g',
  lowerIsBetter: true,
};

export function metricConfig(metric: string): MetricConfig {
  return METRIC_CONFIG[metric] ?? { ...FALLBACK_CONFIG, label: metric };
}

/** Display order for method families (column grouping) and case domains (row grouping).
 * Anything not listed sorts after, alphabetically. Mirrors app/data.py's FAMILY_ORDER. */
export const FAMILY_ORDER = ['baseline', 'ensemble', 'classical', 'ml', 'dl', 'fm'];

export const DOMAIN_RESTRICTED_METRICS = new Set(['NSE', 'KGE']);
