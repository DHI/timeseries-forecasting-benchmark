import { useJsonData } from '../../data/loadData';
import { effectiveMethod } from '../../data/selectors';
import { MethodChips } from './MethodChips';
import { MethodInfoCard } from './MethodInfoCard';
import { SnapshotGrid } from './SnapshotGrid';
import type { CaseMeta, MethodRegistry, ScoreRow, SnapshotBundle } from '../../types/data';

const SUMMARY_STYLE = { cursor: 'pointer', fontWeight: 'bold' as const, fontSize: '1em' };
const CAPTION_STYLE = { color: '#666', fontSize: '0.85em', margin: '0 0 8px 0' };

type Props = {
  caseId: string;
  caseMeta: CaseMeta;
  methodsRegistry: MethodRegistry;
  scores: ScoreRow[];
  methods: string[];
  metric: string;
  showGlobal: boolean;
  showExogenous: boolean;
  onRemoveMethod: (method: string) => void;
  onClear: () => void;
};

/** Below the case panel: the curated forecast-snapshot grid for the selected case and up
 * to 3 selected methods, plus a method-info card per method. Port of app/app.py's
 * `render_panel` / `render_method_info_panel`. */
export function SnapshotPanel({
  caseId,
  caseMeta,
  methodsRegistry,
  scores,
  methods,
  metric,
  showGlobal,
  showExogenous,
  onRemoveMethod,
  onClear,
}: Props) {
  const bundleState = useJsonData<SnapshotBundle>(`data/snapshots/${caseId}.json`);

  if (methods.length === 0) {
    return (
      <p style={{ color: '#666' }}>
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

  const effectiveMethods = methods.map((m) => effectiveMethod(scores, caseId, m, showExogenous));

  return (
    <div>
      <details open>
        <summary style={SUMMARY_STYLE}>[Test] Prediction Snapshots</summary>
        <p style={CAPTION_STYLE}>
          Origins curated by {effectiveMethods[0]}&apos;s {metric} (best / median / worst).
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
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 16,
          marginTop: 16,
          paddingTop: 12,
          borderTop: '1px solid var(--border)',
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
    </div>
  );
}
