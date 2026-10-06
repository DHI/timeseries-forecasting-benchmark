import { LineSnapshot } from './LineSnapshot';
import type { Series } from '../../types/data';

/** Quick-look line plot of the case's full test split. Port of app/snapshot.py's
 * `render_test_snapshot`. */
export function TestSplitPlot({ series, caseId }: { series: Series; caseId: string }) {
  return <LineSnapshot series={series} emptyMessage={`No test data available for ${caseId}.`} />;
}
