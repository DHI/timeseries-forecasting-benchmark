import { effectiveMethod } from '../../data/selectors';
import { MethodInfoCard } from './MethodInfoCard';
import type { CaseMeta, MethodRegistry, ScoreRow } from '../../types/data';

const PANEL_STYLE = {
  position: 'sticky' as const,
  bottom: 0,
  zIndex: 20,
  display: 'flex',
  alignItems: 'flex-end' as const,
  gap: 12,
  overflowX: 'auto' as const,
  padding: '0 16px',
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

/** Dock at the bottom of the viewport: one independently collapsible and height-resizable
 * card per selected method, colour-matched to that method's chip/forecast line. The cards
 * share the row's width evenly, so together they span the full matrix column between the
 * sidebar and the case detail panel. Rendered as the last child of the matrix column. Port
 * of app/app.py's `method-info-panel` / `render_method_info_panel`. */
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
