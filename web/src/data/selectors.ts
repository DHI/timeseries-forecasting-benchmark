import { FAMILY_ORDER, METRIC_CONFIG } from './metricConfig';
import type { ScoreRow } from '../types/data';

/** Whether the pilot run recorded an extended-history fit for this case — i.e. some
 * `is_global` method found real history beyond the curated span. Port of app/data.py's
 * `case_has_extended_history`. */
export function caseHasExtendedHistory(scores: ScoreRow[], caseId: string): boolean {
  return scores.some((r) => r.case_id === caseId && r.used_extended_history);
}

/** The method that actually produced this cell's score — `method` unless the exogenous
 * toggle swaps in its "+exo" sibling's score, which happens only for cases with real
 * exogenous data. Port of app/app.py's `_effective_method`. */
export function effectiveMethod(
  scores: ScoreRow[],
  caseId: string,
  method: string,
  showExo: boolean,
): string {
  if (!showExo) return method;
  const exoMethod = `${method}+exo`;
  if (!scores.some((r) => r.method === exoMethod)) return method;
  const tier = scores.find((r) => r.case_id === caseId)?.case_exo_tier;
  if (!tier || tier === 'none') return method;
  return exoMethod;
}

function rank(order: string[]): (value: string) => number {
  return (value) => {
    const i = order.indexOf(value);
    return i === -1 ? order.length : i;
  };
}

/** Case ids sorted by domain (alphabetical — FAMILY_ORDER-style grouping isn't
 * meaningful for domains), then case id, for stable row grouping with separators.
 * Port of app/data.py's `ordered_cases`. */
export function orderedCases(rows: ScoreRow[]): string[] {
  const domainOf = new Map<string, string>();
  for (const row of rows) domainOf.set(row.case_id, row.domain);
  return [...domainOf.keys()].sort((a, b) => {
    const byDomain = (domainOf.get(a) ?? '').localeCompare(domainOf.get(b) ?? '');
    return byDomain !== 0 ? byDomain : a.localeCompare(b);
  });
}

/** Method names sorted by family (FAMILY_ORDER), then name, for stable column grouping.
 * Excludes "+exo" variants — they're folded into their base method's column (see the
 * exogenous-toggle swap, added in a later milestone), not shown as separate columns.
 * Port of app/data.py's `ordered_methods`. */
export function orderedMethods(rows: ScoreRow[]): string[] {
  const familyOf = new Map<string, string>();
  for (const row of rows) {
    if (!row.method.endsWith('+exo')) familyOf.set(row.method, row.family);
  }
  const familyRank = rank(FAMILY_ORDER);
  return [...familyOf.keys()].sort((a, b) => {
    const byFamily = familyRank(familyOf.get(a) ?? '') - familyRank(familyOf.get(b) ?? '');
    return byFamily !== 0 ? byFamily : a.localeCompare(b);
  });
}

/** `caseIds` restricted to those declaring `metric` in their case.toml `domain_metrics`
 * — a no-op for every metric outside the domain-restricted set (MASE, MAPE, etc. apply
 * to every case). Port of app/data.py's `cases_eligible_for_metric`; unlike that
 * function, `domainMetrics` must be passed in explicitly (from cases.json) since there's
 * no case.toml to read client-side. */
export function casesEligibleForMetric(
  metric: string,
  caseIds: string[],
  domainMetrics: Map<string, string[]>,
  domainRestrictedMetrics: ReadonlySet<string>,
): string[] {
  if (!domainRestrictedMetrics.has(metric)) return caseIds;
  return caseIds.filter((c) => (domainMetrics.get(c) ?? []).includes(metric));
}

/** Metrics selectable in the dropdown: METRIC_CONFIG's own order, restricted to metrics
 * actually present in the data. Port of app/app.py's `_metric_options`. */
export function metricOptions(rows: ScoreRow[]): string[] {
  const present = new Set(rows.map((r) => r.metric));
  return Object.keys(METRIC_CONFIG).filter((m) => present.has(m));
}

/** Domains selectable in the sidebar checklist, alphabetical. Port of app/app.py's
 * `_domain_options` (minus the display-label capitalization, done at render time). */
export function domainOptions(rows: ScoreRow[]): string[] {
  return [...new Set(rows.map((r) => r.domain))].sort();
}

/** Method families selectable in the sidebar multi-select: FAMILY_ORDER's own order,
 * restricted to families present in the data, then any unlisted family alphabetically.
 * Port of app/app.py's `_family_options`. */
export function familyOptions(rows: ScoreRow[]): string[] {
  const present = new Set(rows.map((r) => r.family));
  const ordered = FAMILY_ORDER.filter((f) => present.has(f));
  const leftover = [...present].filter((f) => !FAMILY_ORDER.includes(f)).sort();
  return [...ordered, ...leftover];
}

/** case_id/method -> value, for one metric, restricted to the given row/column order. */
export function pivotValues(rows: ScoreRow[], rowOrder: string[], colOrder: string[]): number[][] {
  const rowIndex = new Map(rowOrder.map((id, i) => [id, i]));
  const colIndex = new Map(colOrder.map((id, i) => [id, i]));
  const grid: number[][] = rowOrder.map(() => colOrder.map(() => NaN));
  for (const row of rows) {
    const r = rowIndex.get(row.case_id);
    const c = colIndex.get(row.method);
    if (r === undefined || c === undefined) continue;
    if (row.value !== null) grid[r]![c] = row.value;
  }
  return grid;
}

/** Grid positions (between row/col i-1 and i) where the group (domain/family) changes.
 * Port of app/app.py's `_group_boundaries`. */
export function groupBoundaries(orderedKeys: string[], keyOf: Map<string, string>): number[] {
  const boundaries: number[] = [];
  for (let i = 1; i < orderedKeys.length; i++) {
    if (keyOf.get(orderedKeys[i]!) !== keyOf.get(orderedKeys[i - 1]!)) {
      boundaries.push(i - 0.5);
    }
  }
  return boundaries;
}

/** (center position, label) for each contiguous run of the same group. Port of
 * app/app.py's `_group_labels`. */
export function groupLabels(
  orderedKeys: string[],
  keyOf: Map<string, string>,
): Array<[number, string]> {
  const labels: Array<[number, string]> = [];
  let start = 0;
  for (let i = 1; i <= orderedKeys.length; i++) {
    const atEnd = i === orderedKeys.length;
    if (atEnd || keyOf.get(orderedKeys[i]!) !== keyOf.get(orderedKeys[start]!)) {
      labels.push([(start + i - 1) / 2, keyOf.get(orderedKeys[start]!) ?? '']);
      start = i;
    }
  }
  return labels;
}

/** The best-scoring method per row (case), by the displayed values. Rows with no finite
 * value are omitted. Port of app/app.py's `_row_best_methods`. */
export function rowBestMethods(
  rowOrder: string[],
  colOrder: string[],
  values: number[][],
  lowerIsBetter: boolean,
): Map<string, string> {
  const best = new Map<string, string>();
  rowOrder.forEach((caseId, r) => {
    let bestCol: number | null = null;
    for (let c = 0; c < colOrder.length; c++) {
      const v = values[r]![c]!;
      if (Number.isNaN(v)) continue;
      if (bestCol === null) {
        bestCol = c;
        continue;
      }
      const better = lowerIsBetter ? v < values[r]![bestCol]! : v > values[r]![bestCol]!;
      if (better) bestCol = c;
    }
    if (bestCol !== null) best.set(caseId, colOrder[bestCol]!);
  });
  return best;
}
