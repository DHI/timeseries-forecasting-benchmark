import { memo, useState } from 'react';
import Markdown from 'react-markdown';
import { METHOD_COLORS, TITLE_BAR_HEIGHT } from '../../data/constants';
import { useEdgeResize } from '../../hooks/useEdgeResize';
import { methodInfo, methodInfoDetail } from '../../data/methodText';
import type { CaseMeta, MethodMeta } from '../../types/data';

const CARD_MIN_WIDTH = 220;
const DEFAULT_CONTENT_HEIGHT = 220;
const MIN_CONTENT_HEIGHT = 120;
const MAX_CONTENT_HEIGHT = 800;

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
