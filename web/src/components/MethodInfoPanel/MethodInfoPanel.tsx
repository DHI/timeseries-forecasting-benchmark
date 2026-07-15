import { effectiveMethod } from '../../data/selectors';
import { MethodInfoCard } from './MethodInfoCard';
import type { CaseMeta, MethodRegistry, ScoreRow } from '../../types/data';

const PANEL_STYLE = {
  position: 'sticky' as const,
  bottom: 0,
  display: 'flex',
  flexWrap: 'wrap' as const,
  gap: 16,
  padding: '12px 16px',
  background: 'var(--bg)',
  borderTop: '1px solid var(--border)',
  boxShadow: '0 -2px 6px rgba(0,0,0,0.08)',
  boxSizing: 'border-box' as const,
};

type Props = {
  caseId: string | null;
  caseMeta: CaseMeta | undefined;
  methodsRegistry: MethodRegistry;
  scores: ScoreRow[];
  methods: string[];
  showExogenous: boolean;
};

/** Strip docked to the bottom of the viewport: one card per selected method, colour-matched
 * to that method's chip/forecast line. Rendered as the last child of the matrix column so its
 * width always matches that column. Port of app/app.py's `method-info-panel` /
 * `render_method_info_panel`. */
export function MethodInfoPanel({
  caseId,
  caseMeta,
  methodsRegistry,
  scores,
  methods,
  showExogenous,
}: Props) {
  if (!caseId || !caseMeta || methods.length === 0) return null;

  const effectiveMethods = methods.map((m) => effectiveMethod(scores, caseId, m, showExogenous));

  return (
    <div style={PANEL_STYLE}>
      {effectiveMethods.map((method, i) => (
        <MethodInfoCard
          key={method}
          method={method}
          index={i}
          caseMeta={caseMeta}
          methodMeta={methodsRegistry[method]}
        />
      ))}
    </div>
  );
}
