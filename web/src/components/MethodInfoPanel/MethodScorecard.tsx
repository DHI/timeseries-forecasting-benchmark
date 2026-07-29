import { Fragment, memo } from 'react';
import {
  CAPABILITY_FIELDS,
  LEVEL_SEGMENTS,
  SCORECARD_FIELDS,
  SEGMENT_COUNT,
  TONE_VAR,
  levelTone,
} from '../../data/scorecardConfig';
import type { Capabilities, Scorecard } from '../../types/data';

const ROW_LABEL_STYLE = {
  color: 'var(--text-secondary)',
  whiteSpace: 'nowrap',
} as const;

const CAPABILITY_ROW_STYLE = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: '2px 12px',
  fontSize: '0.85em',
} as const;

const SEGMENT_BASE_STYLE = {
  width: 10,
  height: 8,
  flex: '0 0 auto',
} as const;

/** Always lists every capability, marked supported or not, so a method with no seasonal
 * mechanism reads as an explicit "no" rather than as an absence you have to notice. */
function CapabilityRowComponent({ capabilities }: { capabilities: Capabilities }) {
  return (
    <div style={CAPABILITY_ROW_STYLE}>
      {CAPABILITY_FIELDS.map(({ key, label, help }) => {
        const supported = capabilities[key];
        return (
          <span
            key={key}
            title={help}
            aria-label={`${label}: ${supported ? 'yes' : 'no'}`}
            style={{
              display: 'inline-flex',
              alignItems: 'baseline',
              gap: 4,
              color: supported ? 'var(--text-secondary)' : 'var(--text-muted)',
            }}
          >
            <span aria-hidden="true" style={{ color: supported ? 'var(--level-good)' : 'inherit' }}>
              {supported ? '✓' : '—'}
            </span>
            {label}
          </span>
        );
      })}
    </div>
  );
}

/** One row per rating: label, a three-segment bar, the level in words. Bar length is the
 * rating's magnitude and bar colour is whether that magnitude is good for this field (see
 * `levelTone`), so a Low fit cost is a short green bar. */
function ScorecardMetersComponent({ scorecard }: { scorecard: Scorecard }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr auto',
        alignItems: 'center',
        columnGap: 8,
        rowGap: 3,
        fontSize: '0.85em',
      }}
    >
      {SCORECARD_FIELDS.map(({ key, label, lowerIsBetter, help }) => {
        const level = scorecard[key];
        const filled = LEVEL_SEGMENTS[level];
        const color = TONE_VAR[levelTone(level, lowerIsBetter)];
        // Cells are siblings of the grid rather than wrapped per row, so the three columns
        // stay aligned across every field without needing subgrid.
        return (
          <Fragment key={key}>
            <span title={help} style={ROW_LABEL_STYLE}>
              {label}
            </span>
            <span
              role="img"
              aria-label={`${label}: ${level}`}
              style={{ display: 'inline-flex', gap: 2 }}
            >
              {Array.from({ length: SEGMENT_COUNT }, (_, i) => (
                <span
                  key={i}
                  aria-hidden="true"
                  data-testid={`segment-${key}-${i < filled ? 'filled' : 'empty'}`}
                  style={{
                    ...SEGMENT_BASE_STYLE,
                    background: i < filled ? color : 'var(--level-empty)',
                  }}
                />
              ))}
            </span>
            <span style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{level}</span>
          </Fragment>
        );
      })}
    </div>
  );
}

export const CapabilityRow = memo(CapabilityRowComponent);
export const ScorecardMeters = memo(ScorecardMetersComponent);
