import { memo, useState } from 'react';
import Markdown from 'react-markdown';
import { METHOD_COLORS, TITLE_BAR_HEIGHT } from '../../data/constants';
import { useEdgeResize } from '../../hooks/useEdgeResize';
import { methodInfo, methodInfoDetail } from '../../data/methodText';
import { InfoModal } from '../InfoModal/InfoModal';
import { CapabilityRow, ForecastStrategyBadge, ScorecardMeters } from './MethodScorecard';
import type { CaseMeta, MethodMeta } from '../../types/data';

const CARD_MIN_WIDTH = 220;
// Tall enough for the description plus the Handles row and six meters without scrolling on
// first open; the title bar is a drag handle, so a user can still resize from here.
const DEFAULT_CONTENT_HEIGHT = 300;

const MIN_CONTENT_HEIGHT = 120;
const MAX_CONTENT_HEIGHT = 800;

const BLOCK_HEADING_STYLE = {
  fontSize: '0.85em',
  fontWeight: 'bold',
  marginTop: 8,
  marginBottom: 2,
} as const;

type Props = {
  method: string;
  index: number;
  caseMeta: CaseMeta;
  methodMeta: MethodMeta | undefined;
};

/** One card per selected method, docked to the bottom of the viewport next to its siblings and
 * sharing the row's width evenly with them, so together they span the full matrix column. The
 * title bar sits at the top when expanded and can be collapsed down to just that bar (which then
 * sits alone at the bottom edge); expanded, its content height is user-adjustable by dragging the
 * title bar. Colour-matched to that method's chip/forecast line. Port of app/app.py's
 * `_method_info_cards`. */
function MethodInfoCardComponent({ method, index, caseMeta, methodMeta }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [contentHeight, setContentHeight] = useState(DEFAULT_CONTENT_HEIGHT);
  const color = METHOD_COLORS[index % METHOD_COLORS.length];

  const resize = useEdgeResize('y', contentHeight, (height) =>
    setContentHeight(Math.min(MAX_CONTENT_HEIGHT, Math.max(MIN_CONTENT_HEIGHT, height))),
  );

  const titleBar = (
    <button
      type="button"
      onMouseDown={resize.onMouseDown}
      onClick={() => {
        if (resize.moved.current) return;
        setCollapsed((c) => !c);
      }}
      title={collapsed ? 'Expand' : 'Collapse (drag to resize)'}
      style={{
        flex: '0 0 auto',
        height: TITLE_BAR_HEIGHT,
        boxSizing: 'border-box',
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '6px 10px',
        font: 'inherit',
        fontWeight: 'bold',
        textAlign: 'left',
        background: 'none',
        border: 'none',
        borderLeft: `6px solid ${color}`,
        ...(collapsed
          ? { borderTop: '1px solid var(--border)' }
          : { borderBottom: '1px solid var(--border)' }),
        cursor: 'pointer',
      }}
    >
      <span>{collapsed ? '▲' : '▼'}</span>
      <span>{method}</span>
    </button>
  );

  return (
    <div
      style={{
        flex: `1 1 ${CARD_MIN_WIDTH}px`,
        minWidth: CARD_MIN_WIDTH,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg)',
        border: '2px solid var(--border)',
        borderBottom: 'none',
        boxSizing: 'border-box',
        pointerEvents: 'auto',
      }}
    >
      {!collapsed && titleBar}
      {!collapsed && (
        <div
          style={{
            height: contentHeight,
            overflowY: 'auto',
            padding: '0 10px 8px',
            boxSizing: 'border-box',
          }}
        >
          <div style={{ fontSize: '0.85em' }}>
            <Markdown>{methodInfo(caseMeta, methodMeta)}</Markdown>
          </div>
          {methodMeta && (
            <>
              <div style={BLOCK_HEADING_STYLE}>Handles</div>
              <CapabilityRow capabilities={methodMeta.capabilities} />
              <div
                style={{ ...BLOCK_HEADING_STYLE, display: 'flex', alignItems: 'center', gap: 4 }}
              >
                Forecast strategy
                <InfoModal
                  buttonTitle="How to read forecast strategy"
                  ariaLabel="How to read forecast strategy"
                  heading="How to read forecast strategy"
                  testId={`forecast-strategy-info-${method}`}
                  size={16}
                >
                  <p>
                    How this method turns one fit into a multi-step forecast — a different question
                    from whether it&rsquo;s <em>well-behaved</em> doing so, which the
                    Scorecard&rsquo;s robustness rating already covers.
                  </p>
                  <p>
                    <strong>Recursive</strong>: each step&rsquo;s forecast feeds forward into
                    computing a later step. mlforecast&rsquo;s default lag-feedback strategy and a
                    classical ARIMA/ETS/TBATS-style state-space recursion both work this way — the
                    model substitutes its own prior forecast for the unobserved future value at each
                    step.
                  </p>
                  <p>
                    <strong>Direct</strong>: one model call produces the whole horizon at once, with
                    nothing feeding into a later step. For the foundation models this describes only
                    this benchmark&rsquo;s calling interface (one call, full horizon requested) —
                    not whether the pretrained decoder itself generates internally autoregressively,
                    which isn&rsquo;t verified here.
                  </p>
                  <p>
                    <strong>N/A</strong>: the distinction carries no information — a naive baseline
                    that looks up or repeats a value independently per timestamp, or a composite of
                    several methods.
                  </p>
                </InfoModal>
              </div>
              <ForecastStrategyBadge strategy={methodMeta.forecast_strategy} />
              <div
                style={{ ...BLOCK_HEADING_STYLE, display: 'flex', alignItems: 'center', gap: 4 }}
              >
                Scorecard
                <InfoModal
                  buttonTitle="How to read the scorecard"
                  ariaLabel="How to read the scorecard"
                  heading="How to read the scorecard"
                  testId={`scorecard-info-${method}`}
                  size={16}
                >
                  <p>
                    These ratings are the benchmark authors&rsquo; qualitative judgement of each
                    method, not measurements taken from the results matrix. They are meant to narrow
                    a shortlist before you read the matrix, not to stand in for it. The two cost
                    ratings are anchored to the fit and predict times the benchmark actually
                    records, so they should not contradict the training-time overlay.
                  </p>
                  <p>
                    A bar&rsquo;s <strong>length</strong> is the size of the rating and its{' '}
                    <strong>colour</strong> is whether that size is good for the field. Some fields
                    are better when low: a low fit cost draws one short green bar, while a low
                    interpretability draws one short red bar.
                  </p>
                  <p>
                    <strong>Handles</strong> above is a separate question — whether the method has a
                    mechanism for that structure at all. Exogenous there describes the method rather
                    than this column, so a method shown without covariates may still support them.
                    Gaps is similar: it only shows a difference in the extended-history matrix,
                    since the shared local matrix is always gap-free by the time any method sees it.
                  </p>
                </InfoModal>
              </div>
              <ScorecardMeters scorecard={methodMeta.scorecard} />
            </>
          )}
          <details style={{ marginTop: 6 }}>
            <summary style={{ fontSize: '0.85em', cursor: 'pointer' }}>
              Implementation detail
            </summary>
            <div style={{ fontSize: '0.85em', marginTop: 4 }}>
              <Markdown
                components={{
                  pre: ({ children }) => (
                    <pre
                      style={{
                        overflowX: 'auto',
                        maxWidth: '100%',
                        background: '#161b22',
                        color: '#e6edf3',
                        padding: '8px 10px',
                        borderRadius: 4,
                        boxSizing: 'border-box',
                      }}
                    >
                      {children}
                    </pre>
                  ),
                }}
              >
                {methodInfoDetail(methodMeta)}
              </Markdown>
            </div>
          </details>
        </div>
      )}
      {collapsed && titleBar}
    </div>
  );
}

export const MethodInfoCard = memo(MethodInfoCardComponent);
