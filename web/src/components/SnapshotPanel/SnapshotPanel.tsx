import { memo, useMemo } from 'react';
import { useJsonData } from '../../data/loadData';
import { effectiveMethod } from '../../data/selectors';
import { MethodChips } from './MethodChips';
import { SnapshotGrid } from './SnapshotGrid';
import type { CaseMeta, ScoreRow, SnapshotBundle } from '../../types/data';

const SUMMARY_STYLE = { cursor: 'pointer', fontWeight: 'bold' as const, fontSize: '1em' };
const CAPTION_STYLE = { color: 'var(--text-muted)', fontSize: '0.85em', margin: '0 0 8px 0' };

type Props = {
  caseId: string;
  caseMeta: CaseMeta;
  scores: ScoreRow[];
  methods: string[];
  metric: string;
  showGlobal: boolean;
  showExogenous: boolean;
  onRemoveMethod: (method: string) => void;
  onClear: () => void;
};

/** Below the case panel: the curated forecast-snapshot grid for the selected case and up
 * to 3 selected methods. Port of app/app.py's `render_panel`. */
function SnapshotPanelComponent({
  caseId,
  caseMeta,
  scores,
  methods,
  metric,
  showGlobal,
  showExogenous,
  onRemoveMethod,
  onClear,
}: Props) {
  const bundleState = useJsonData<SnapshotBundle>(`data/snapshots/${caseId}.json`);
  const effectiveMethods = useMemo(
    () => methods.map((m) => effectiveMethod(scores, caseId, m, showExogenous)),
    [scores, caseId, methods, showExogenous],
  );

  if (methods.length === 0) {
    return (
      <p style={{ color: 'var(--text-muted)' }}>
        Click a matrix cell to see real forecast snapshots: the best, median, and worst performing
        origins, by the metric selected above, for that method. Click more cells in the same row to
        overlay other methods on those same origins.
      </p>
    );
  }

  if (bundleState.status === 'loading') return <p>Loading forecast snapshots…</p>;
  if (bundleState.status === 'error') {
    return (
      <p role="alert" style={{ color: 'crimson' }}>
        Failed to load snapshot data: {bundleState.error.message}
      </p>
    );
  }

  return (
    <div>
      <details open>
        <summary style={SUMMARY_STYLE}>Prediction snapshots</summary>
        <p style={CAPTION_STYLE}>
          The following snapshots belong to the test split and were selected for exhibiting the
          best/median/worst{' '}
          <b>
            {effectiveMethods[0]}&apos;s {metric}
          </b>
          . The black line represents the actual target values.
        </p>
        <div
          style={{
            display: 'flex',
            gap: 8,
            alignItems: 'center',
            flexWrap: 'wrap',
            marginBottom: 8,
          }}
        >
          <MethodChips methods={methods} onRemove={onRemoveMethod} />
          <button type="button" onClick={onClear} style={{ marginLeft: 'auto' }}>
            Clear
          </button>
        </div>
        <SnapshotGrid
          caseId={caseId}
          caseMeta={caseMeta}
          bundle={bundleState.data}
          effectiveMethods={effectiveMethods}
          metric={metric}
          showGlobal={showGlobal}
        />
      </details>
    </div>
  );
}

export const SnapshotPanel = memo(SnapshotPanelComponent);
