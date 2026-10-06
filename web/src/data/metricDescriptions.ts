/** Short, plain-language descriptions for the sidebar's metric cheatsheet (MetricInfoModal).
 * `key` matches METRIC_CONFIG in metricConfig.ts and the literal text shown in the metric
 * dropdown; order matches METRIC_CONFIG's own definition order. Kept as a separate file since
 * these are prose, not display config. */
export const METRIC_DESCRIPTIONS: { key: string; description: string }[] = [
  {
    key: 'MASE',
    description:
      "Mean Absolute Scaled Error: test-window error divided by the error a seasonal-naive forecast makes on the case's own history, at the period pinned in case.toml. Below 1 beats that reference. The SeasonalNaive row is not fixed at 1.0 — it is scored on the test window, against its error on history, and the two differ. Lower is better.",
  },
  {
    key: 'Skill_vs_Persistence',
    description:
      'Relative improvement over a naive "tomorrow = today" forecast. 0 means no better, 1 means perfect. Higher is better.',
  },
  {
    key: 'Skill_vs_Mean',
    description:
      "Relative improvement over predicting the case's historical mean. 0 means no better, 1 means perfect. Higher is better.",
  },
  {
    key: 'MAE',
    description:
      "Mean Absolute Error, in the target's own units: the average absolute difference between forecast and actual. Lower is better.",
  },
  {
    key: 'RMSE',
    description:
      'Root Mean Squared Error, in the target unit: like MAE but squares errors first, so large misses count more. Lower is better.',
  },
  {
    key: 'MAPE',
    description:
      'Mean Absolute Percentage Error: average absolute error as a percentage of the actual value. Lower is better.',
  },
  {
    key: 'sMAPE',
    description:
      "Symmetric MAPE: percentage error scaled by the average of forecast and actual instead of just the actual, so it doesn't blow up when actual values are near zero. Lower is better.",
  },
  {
    key: 'NSE',
    description:
      'Nash-Sutcliffe Efficiency, a hydrology-standard fit measure. 1 = perfect, 0 = no better than predicting the mean, negative = worse than the mean. Higher is better. Only applies to cases whose case.toml declares it (flow/level domains) — selecting it narrows the matrix rows to just those cases.',
  },
  {
    key: 'KGE',
    description:
      'Kling-Gupta Efficiency: splits error into correlation, bias, and variability components. 1 = perfect. Higher is better. Only applies to cases whose case.toml declares it (flow/level domains) — selecting it narrows the matrix rows to just those cases.',
  },
];
