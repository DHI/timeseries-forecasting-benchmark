import { memo, useMemo } from 'react';
import Markdown from 'react-markdown';
import { caseHasExtendedHistory } from '../../data/selectors';
import { TRAINING_TAIL_N } from '../../data/constants';
import { caseTrainingStats } from '../../data/caseText';
import { CaseSummary } from './CaseSummary';
import { DistributionPlot } from './DistributionPlot';
import { ProportionsBar } from './ProportionsBar';
import { SeasonalPatternPlot } from './SeasonalPatternPlot';
import { TestSplitPlot } from './TestSplitPlot';
import { TrainingTailPlot } from './TrainingTailPlot';
import type { CaseMeta, ScoreRow } from '../../types/data';

const CAPTION_STYLE = { color: 'var(--text-muted)', fontSize: '0.9em', margin: '12px 0 4px 0' };
const SUBCAPTION_STYLE = {
  color: 'var(--text-muted)',
  fontSize: '0.85em',
  margin: '10px 0 2px 0',
};
const SUMMARY_STYLE = { cursor: 'pointer', fontWeight: 'bold' as const, fontSize: '1em' };
const FOUNDATION_NOTE_STYLE = {
  display: 'flex',
  alignItems: 'flex-start',
  gap: 8,
  border: '1px solid var(--warning-text)',
  borderRadius: 4,
  padding: '6px 10px',
  margin: '4px 0 8px 0',
};

type Props = {
  caseId: string;
  caseMeta: CaseMeta;
  scores: ScoreRow[];
  showGlobal: boolean;
  selectedMethods: string[];
};

/** Right-hand detail panel: training/test snapshots, a train-vs-test distribution, a
 * train/val/test proportions bar, a seasonal-pattern figure, and case metadata for the
 * selected case. When `showGlobal` reveals extended history for this case, the training
 * snapshot, distribution, proportions bar, seasonal patterns, and sample counts all swap
 * in the extended-history train split. Port of app/app.py's `render_case_panel`. */
function CasePanelComponent({ caseId, caseMeta, scores, showGlobal, selectedMethods }: Props) {
  const extended = showGlobal && caseHasExtendedHistory(scores, caseId);
  const trainTail = extended ? caseMeta.train_tail_extended : caseMeta.train_tail;
  const splitCounts = extended ? caseMeta.split_counts_extended : caseMeta.split_counts;
  const distribution = extended ? caseMeta.distribution_extended : caseMeta.distribution;
  const seasonalPatterns = extended
    ? caseMeta.seasonal_patterns_extended
    : caseMeta.seasonal_patterns;
  const caption =
    `Training data (last ${TRAINING_TAIL_N.toLocaleString('en-US')} points` +
    (extended ? ', including extended history)' : ')');

  const familyByMethod = useMemo(() => {
    const map = new Map<string, string>();
    for (const row of scores) {
      if (row.case_id === caseId) map.set(row.method, row.family);
    }
    return map;
  }, [scores, caseId]);

  // Foundation models forecast zero-shot, from recent context, so the train/val/test
  // split below doesn't describe how they were fit the way it does for other methods.
  const foundationMethods = useMemo(
    () => selectedMethods.filter((m) => familyByMethod.get(m) === 'fm'),
    [familyByMethod, selectedMethods],
  );
  const foundationNote =
    foundationMethods.length === 1
      ? "is a zero-shot foundation model: it forecasts from recent context only and never fits on this case's train/val split."
      : "are zero-shot foundation models: they forecast from recent context only and never fit on this case's train/val split.";

  return (
    <div>
      <div style={{ fontWeight: 'bold', fontSize: '1.1em', margin: '0 0 4px 0' }}>
        {caseId.replace(/_/g, ' ').toUpperCase()}
      </div>
      <details key={caseId}>
        <summary style={SUMMARY_STYLE}>Case details</summary>
        <div style={{ marginTop: 4 }}>
          <CaseSummary caseMeta={caseMeta} />
        </div>
      </details>
      <details open>
        <summary style={SUMMARY_STYLE}>Target data overview</summary>
        {foundationMethods.length > 0 && (
          <div style={FOUNDATION_NOTE_STYLE}>
            <span aria-hidden="true" style={{ color: 'var(--warning-text)' }}>
              ⚠
            </span>
            <p style={{ ...SUBCAPTION_STYLE, margin: 0 }}>
              <strong>{foundationMethods.join(', ')}</strong> {foundationNote} The split below still
              applies to the other methods and to how the test points are scored.
            </p>
          </div>
        )}
        <p style={{ ...CAPTION_STYLE, margin: '0 0 4px 0' }}>{caption}</p>
        <TrainingTailPlot series={trainTail} caseId={caseId} />
        <p style={CAPTION_STYLE}>Test data (full test split)</p>
        <TestSplitPlot series={caseMeta.test_split} caseId={caseId} />
        <p style={CAPTION_STYLE}>Train vs test value distribution</p>
        <DistributionPlot distribution={distribution} caseId={caseId} />
        <p style={CAPTION_STYLE}>Train / val / test split</p>
        <ProportionsBar counts={splitCounts} caseId={caseId} />
        <Markdown>{caseTrainingStats(caseMeta, extended)}</Markdown>
        {seasonalPatterns.length > 0 && (
          <>
            <p style={CAPTION_STYLE}>Seasonal pattern (detrended)</p>
            <p style={SUBCAPTION_STYLE}>
              Grey dots are individual training points; the black line is a smoothed curve showing
              the typical shape of the cycle. Zoom in (click and drag on a subplot) to see the
              pattern more clearly.
            </p>
            <SeasonalPatternPlot patterns={seasonalPatterns} />
          </>
        )}
      </details>
    </div>
  );
}

export const CasePanel = memo(CasePanelComponent);
