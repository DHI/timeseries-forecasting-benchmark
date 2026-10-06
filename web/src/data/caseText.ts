import type { CaseMeta } from '../types/data';

const NUMBER_FORMAT = new Intl.NumberFormat('en-US');

/** Markdown summary of a case's static metadata: domain, target, horizon, exogenous
 * inputs, and data source — the config in case.toml, not anything measured. Port of
 * app/data.py's `case_summary`. */
export function caseSummary(caseMeta: CaseMeta): string {
  const exo =
    caseMeta.exogenous.length > 0 ? `${caseMeta.exogenous.length} (${caseMeta.exo_tier})` : 'none';
  const source = caseMeta.source.url
    ? `[${caseMeta.source.name}](${caseMeta.source.url})`
    : caseMeta.source.name;

  const lines = [
    `**Domain**: ${caseMeta.domain.replace(/_/g, ' ')}`,
    `**Target**: \`${caseMeta.target}\`` +
      (caseMeta.target_unit ? ` (${caseMeta.target_unit})` : ''),
    `**Frequency**: \`${caseMeta.freq}\`  ·  **Horizon**: \`${caseMeta.horizon_primary}\``,
    `**Exogenous inputs**: ${exo}`,
    `**Source**: ${source}`,
  ];
  if (caseMeta.notes) lines.push('', caseMeta.notes);
  return lines.join('\n\n');
}

/** Markdown line for the case-summary panel, below the split-proportions bar: the
 * actual observed sample counts behind that bar's percentages. Port of app/data.py's
 * `case_training_stats` (minus its training-time line, now shown as a heatmap overlay
 * instead — see the "Training time" control in the sidebar). */
export function caseTrainingStats(caseMeta: CaseMeta, extended = false): string {
  const counts = extended ? caseMeta.split_counts_extended : caseMeta.split_counts;
  return (
    `Number of samples: train ${NUMBER_FORMAT.format(counts.train)} · ` +
    `val ${NUMBER_FORMAT.format(counts.val)} · test ${NUMBER_FORMAT.format(counts.test)} ` +
    '\n\nParameters are estimated on train; val is the early-stopping holdout for the ' +
    'iteratively trained deep-learning methods and is never trained on; test is scored.'
  );
}
