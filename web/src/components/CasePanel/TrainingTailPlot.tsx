import { LineSnapshot } from './LineSnapshot';
import type { Series } from '../../types/data';

/** Quick-look line plot of the case's last N training-split points. Port of
 * app/snapshot.py's `render_training_snapshot`. */
export function TrainingTailPlot({ series, caseId }: { series: Series; caseId: string }) {
  return (
    <LineSnapshot series={series} emptyMessage={`No training data available for ${caseId}.`} />
  );
}
