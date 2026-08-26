import { Fragment, memo } from 'react';
import {
  CAPABILITY_FIELDS,
  FORECAST_STRATEGY_HELP,
  GROWTH_SHAPE_HELP,
  LEVEL_SEGMENTS,
  SCORECARD_FIELDS,
  SEGMENT_COUNT,
  TONE_VAR,
  levelTone,
} from '../../data/scorecardConfig';
import type {
  Capabilities,
  ForecastStrategy,
  GrowthShape,
  Scorecard,
  ScorecardLevel,
} from '../../types/data';

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

const GROWTH_STYLE = {
  color: 'var(--text-muted)',
  fontSize: '0.9em',
  whiteSpace: 'nowrap',
} as const;

/** `running_cost` carries a level plus a growth shape; every other field is a bare level.
 * Reading them apart here keeps the meter loop uniform. */
function readField(scorecard: Scorecard, key: keyof Scorecard) {
  const value = scorecard[key];
  return typeof value === 'string'
    ? { level: value as ScorecardLevel, growth: undefined }
    : { level: value.level, growth: value.growth as GrowthShape };
}

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

/** One row per rating: label, then a three-segment bar. Bar length is the rating's magnitude
 * and bar colour is whether that magnitude is good for this field (see `levelTone`), so a Low
 * running cost is a short green bar. The level isn't spelled out — length already distinguishes
 * the three values on its own, so it stays legible without colour, and each bar carries the
 * level in its `aria-label` for screen readers.
 *
 * Running cost also prints its growth shape as a word after the bar. The bar is read at one
 * reference record size, so without the word it can't separate a cost that ignores record
 * length from one that outruns it. */
function ScorecardMetersComponent({ scorecard }: { scorecard: Scorecard }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr',
        alignItems: 'center',
        columnGap: 8,
        rowGap: 3,
        fontSize: '0.85em',
      }}
    >
      {SCORECARD_FIELDS.map(({ key, label, lowerIsBetter, help }) => {
        const { level, growth } = readField(scorecard, key);
        const filled = LEVEL_SEGMENTS[level];
        const color = TONE_VAR[levelTone(level, lowerIsBetter)];
        // Cells are siblings of the grid rather than wrapped per row, so the two columns stay
        // aligned across every field without needing subgrid.
        return (
          <Fragment key={key}>
            <span title={help} style={ROW_LABEL_STYLE}>
              {label}
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span
                role="img"
                aria-label={`${label}: ${level}`}
                title={level}
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
              {growth && (
                <span
                  data-testid={`growth-${key}`}
                  title={GROWTH_SHAPE_HELP[growth]}
                  style={GROWTH_STYLE}
                >
                  {growth}
                </span>
              )}
            </span>
          </Fragment>
        );
      })}
    </div>
  );
}

/** A single value, not a row of options — this is a one-of-three category, not a boolean
 * (CapabilityRow) or a graded rating (ScorecardMeters), so it gets its own small badge rather
 * than being forced into either of those shapes. */
function ForecastStrategyBadgeComponent({ strategy }: { strategy: ForecastStrategy }) {
  return (
    <span
      title={FORECAST_STRATEGY_HELP[strategy]}
      aria-label={`Forecast strategy: ${strategy}`}
      style={{
        display: 'inline-block',
        fontSize: '0.85em',
        color: 'var(--text-secondary)',
        border: '1px solid var(--border)',
        borderRadius: 3,
        padding: '0 6px',
      }}
    >
      {strategy}
    </span>
  );
}

export const CapabilityRow = memo(CapabilityRowComponent);
export const ScorecardMeters = memo(ScorecardMetersComponent);
export const ForecastStrategyBadge = memo(ForecastStrategyBadgeComponent);
