import { METRIC_DESCRIPTIONS } from '../../data/metricDescriptions';
import { InfoModal } from '../InfoModal/InfoModal';

/** Info icon next to the sidebar's "Metric" label; opens a cheatsheet briefly describing each
 * metric the dropdown offers. */
export function MetricInfoModal() {
  return (
    <InfoModal
      buttonTitle="Metric cheatsheet"
      ariaLabel="Show metric cheatsheet"
      heading="Metric cheatsheet"
      testId="metric-cheatsheet-backdrop"
    >
      <dl style={{ color: 'var(--text-secondary)' }}>
        {METRIC_DESCRIPTIONS.map(({ key, description }) => (
          <div key={key} style={{ marginBottom: 10 }}>
            <dt style={{ fontWeight: 'bold' }}>{key}</dt>
            <dd style={{ margin: 0 }}>{description}</dd>
          </div>
        ))}
      </dl>
    </InfoModal>
  );
}
