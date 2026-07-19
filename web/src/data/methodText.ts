import type { CaseMeta, MethodMeta } from '../types/data';

/** Markdown blurb for the method-info panel: family, a brief description drawn from the
 * Forecaster subclass's docstring, and which of the case's exogenous inputs (if any) this
 * method actually consumes. Port of app/data.py's `method_info`. */
export function methodInfo(caseMeta: CaseMeta, methodMeta: MethodMeta | undefined): string {
  const family = methodMeta?.family ?? 'unclassified';
  const description = methodMeta?.description ?? 'No description yet.';

  let exoLine = 'none';
  if (methodMeta?.uses_exogenous) {
    const exoCols = [...new Set([...caseMeta.exogenous, ...caseMeta.exogenous_known_future])];
    if (exoCols.length > 0) {
      const necessary = new Set(caseMeta.necessary_exogenous);
      exoLine = exoCols.map((c) => (necessary.has(c) ? `${c} (necessary)` : c)).join(', ');
    }
  }

  return [
    `**Family**: ${family.replace(/_/g, ' ')}`,
    description,
    `**Exogenous inputs used**: ${exoLine}`,
  ].join('\n\n');
}

/** Markdown for the method-info panel's collapsed "Implementation detail" section:
 * library, compute-cost profile, known limitations, and a code excerpt. Blank fields are
 * omitted rather than shown as an empty heading. Port of app/data.py's
 * `method_info_detail`. */
export function methodInfoDetail(methodMeta: MethodMeta | undefined): string {
  const lines: string[] = [];
  if (methodMeta?.library) lines.push(`**Library**: ${methodMeta.library}`);
  if (methodMeta?.compute_profile) lines.push(`**Compute profile**: ${methodMeta.compute_profile}`);
  if (methodMeta?.limitations) lines.push(`**Limitations**: ${methodMeta.limitations}`);
  if (methodMeta?.code_snippet) lines.push(`\`\`\`python\n${methodMeta.code_snippet}\n\`\`\``);
  return lines.length > 0 ? lines.join('\n\n') : 'No implementation detail yet.';
}
