import { memo, useMemo } from 'react';
import { effectiveMethod } from '../../data/selectors';
import { MethodInfoCard } from './MethodInfoCard';
import type { CaseMeta, MethodRegistry, ScoreRow } from '../../types/data';

type Props = {
  caseId: string | null;
  caseMeta: CaseMeta | undefined;
  methodsRegistry: MethodRegistry;
  scores: ScoreRow[];
  methods: string[];
  showExogenous: boolean;
  leftInset: number;
  rightInset: number;
};

/** Fixed overlay docked to the bottom of the viewport: one independently collapsible and
 * height-resizable card per selected method, colour-matched to that method's chip/forecast
 * line. `leftInset`/`rightInset` align the row with the matrix column between the sidebar and
 * the case detail panel. Fixed (not sticky) so the cards never affect the matrix column's own
 * scroll height — they float on top of its content instead. Port of app/app.py's
 * `method-info-panel` / `render_method_info_panel`. */
function MethodInfoPanelComponent({
  caseId,
  caseMeta,
  methodsRegistry,
  scores,
  methods,
  showExogenous,
  leftInset,
  rightInset,
}: Props) {
  const effectiveMethods = useMemo(
    () => (caseId ? methods.map((m) => effectiveMethod(scores, caseId, m, showExogenous)) : []),
    [scores, caseId, methods, showExogenous],
  );

  if (!caseId || !caseMeta || methods.length === 0) return null;

  return (
    <div
      style={{
        position: 'fixed',
        left: leftInset,
        right: rightInset,
        bottom: 0,
        zIndex: 20,
        display: 'flex',
        alignItems: 'flex-end',
        gap: 12,
        overflowX: 'auto',
        padding: '0 16px',
        boxSizing: 'border-box',
        // This box has no background and is taller/wider than the cards it lays out (padding,
        // gaps, and any leftover row space beyond the last card) — without this, it silently
        // blocks clicks on whatever sits underneath (e.g. the Getting Started button) anywhere
        // in that transparent area. Cards opt back in below.
        pointerEvents: 'none',
      }}
    >
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

export const MethodInfoPanel = memo(MethodInfoPanelComponent);
