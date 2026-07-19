import { caseHasExtendedHistory } from '../../data/selectors';
import { TRAINING_TAIL_N } from '../../data/constants';
import { CaseSummary } from './CaseSummary';
import { DistributionPlot } from './DistributionPlot';
import { ProportionsBar } from './ProportionsBar';
import { TestSplitPlot } from './TestSplitPlot';
import { TrainingStats } from './TrainingStats';
import { TrainingTailPlot } from './TrainingTailPlot';
import type { CaseMeta, ScoreRow } from '../../types/data';

const CAPTION_STYLE = { color: '#666', fontSize: '0.85em', margin: '12px 0 4px 0' };
const SUMMARY_STYLE = { cursor: 'pointer', fontWeight: 'bold' as const, fontSize: '1em' };

type Props = {
  caseId: string;
  caseMeta: CaseMeta;
  scores: ScoreRow[];
  methods: string[];
  showGlobal: boolean;
};

/** Right-hand detail panel: training/test snapshots, a train-vs-test distribution, a
 * train/val/test proportions bar, and case metadata for the selected case. When
 * `showGlobal` reveals extended history for this case, the training snapshot,
 * distribution, proportions bar, and sample counts all swap in the extended-history
 * train split. Port of app/app.py's `render_case_panel`. */
export function CasePanel({ caseId, caseMeta, scores, methods, showGlobal }: Props) {
  const extended = showGlobal && caseHasExtendedHistory(scores, caseId);
  const trainTail = extended ? caseMeta.train_tail_extended : caseMeta.train_tail;
  const splitCounts = extended ? caseMeta.split_counts_extended : caseMeta.split_counts;
  const distribution = extended ? caseMeta.distribution_extended : caseMeta.distribution;
  const caption =
    `Training data (last ${TRAINING_TAIL_N.toLocaleString('en-US')} points` +
    (extended ? ', including extended history)' : ')');

  return (
    <div>
      <div style={{ fontWeight: 'bold', fontSize: '1.1em', margin: '0 0 4px 0' }}>
        {caseId.replace(/_/g, ' ').toUpperCase()}
      </div>
      <details open>
        <summary style={SUMMARY_STYLE}>Training & test data</summary>
        <p style={{ ...CAPTION_STYLE, margin: '0 0 4px 0' }}>{caption}</p>
        <TrainingTailPlot series={trainTail} caseId={caseId} />
        <p style={CAPTION_STYLE}>Test data (full test split)</p>
        <TestSplitPlot series={caseMeta.test_split} caseId={caseId} />
        <p style={CAPTION_STYLE}>Train vs test value distribution</p>
        <DistributionPlot distribution={distribution} caseId={caseId} />
        <p style={CAPTION_STYLE}>Train / val / test split</p>
        <ProportionsBar counts={splitCounts} caseId={caseId} />
        <TrainingStats
          scores={scores}
          caseMeta={caseMeta}
          caseId={caseId}
          methods={methods}
          extended={extended}
        />
      </details>
      <hr />
      <details open>
        <summary style={SUMMARY_STYLE}>Case summary</summary>
        <div style={{ marginTop: 4 }}>
          <CaseSummary caseMeta={caseMeta} />
        </div>
      </details>
    </div>
  );
}
