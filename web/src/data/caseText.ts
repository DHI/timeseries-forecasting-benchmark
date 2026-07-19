import type { CaseMeta, ScoreRow } from '../types/data';

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

/** The persisted `seconds` timing (fit + predict, pooled across rolling origins) for this
 * case x method run. Falls back to the local (non-extended) fit's timing when `extended`
 * is requested but this method never produced an extended-history variant. Port of
 * app/data.py's `method_training_seconds`. */
export function methodTrainingSeconds(
  scores: ScoreRow[],
  caseId: string,
  method: string,
  extended = false,
): number | null {
  const forMethod = scores.filter((r) => r.case_id === caseId && r.method === method);
  let rows = forMethod.filter((r) => r.used_extended_history === extended);
  if (rows.length === 0 && extended) {
    rows = forMethod.filter((r) => !r.used_extended_history);
  }
  return rows.length > 0 ? (rows[0]!.seconds ?? null) : null;
}

/** Markdown lines for the case-summary panel, below the split-proportions bar: the
 * actual observed sample counts behind that bar's percentages, and the measured
 * training time for each currently-selected method. Port of app/data.py's
 * `case_training_stats`. */
export function caseTrainingStats(
  scores: ScoreRow[],
  caseMeta: CaseMeta,
  caseId: string,
  methods: string[],
  extended = false,
): string {
  const counts = extended ? caseMeta.split_counts_extended : caseMeta.split_counts;
  const total = counts.train + counts.val + counts.test;
  const samplesLine =
    `**Samples used**: train ${NUMBER_FORMAT.format(counts.train)} · ` +
    `val ${NUMBER_FORMAT.format(counts.val)} · test ${NUMBER_FORMAT.format(counts.test)} ` +
    `(total ${NUMBER_FORMAT.format(total)})`;
  if (methods.length === 0) return samplesLine;

  const times = methods.map((method) => {
    const seconds = methodTrainingSeconds(scores, caseId, method, extended);
    return seconds !== null ? `${method} ${seconds.toFixed(2)}s` : `${method} —`;
  });
  const timeLine =
    '**Training time** (fit + predict, pooled across rolling origins): ' + times.join(' · ');
  return `${samplesLine}\n\n${timeLine}`;
}
