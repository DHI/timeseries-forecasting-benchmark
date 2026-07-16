import { useState } from 'react';
import Markdown from 'react-markdown';
import { METHOD_COLORS } from '../../data/constants';
import { useEdgeResize } from '../../hooks/useEdgeResize';
import { methodInfo, methodInfoDetail } from '../../data/methodText';
import type { CaseMeta, MethodMeta } from '../../types/data';

const CARD_MIN_WIDTH = 220;
const DEFAULT_CONTENT_HEIGHT = 220;
const MIN_CONTENT_HEIGHT = 120;
const MAX_CONTENT_HEIGHT = 520;

type Props = {
  method: string;
  index: number;
  caseMeta: CaseMeta;
  methodMeta: MethodMeta | undefined;
};

/** One card per selected method, docked to the bottom of the viewport next to its siblings and
 * sharing the row's width evenly with them, so together they span the full matrix column. The
 * title bar stays put at the bottom edge and can be collapsed down to just that bar; expanded,
 * its content height is user-adjustable via a drag handle along its top edge. Colour-matched to
 * that method's chip/forecast line. Port of app/app.py's `_method_info_cards`. */
export function MethodInfoCard({ method, index, caseMeta, methodMeta }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [contentHeight, setContentHeight] = useState(DEFAULT_CONTENT_HEIGHT);
  const color = METHOD_COLORS[index % METHOD_COLORS.length];

  const resize = useEdgeResize('y', contentHeight, (height) =>
    setContentHeight(Math.min(MAX_CONTENT_HEIGHT, Math.max(MIN_CONTENT_HEIGHT, height))),
  );

  return (
    <div
      style={{
        flex: `1 1 ${CARD_MIN_WIDTH}px`,
        minWidth: CARD_MIN_WIDTH,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg)',
        border: '1px solid var(--border)',
        borderBottom: 'none',
        boxSizing: 'border-box',
      }}
    >
      {!collapsed && (
        <>
          <div
            onMouseDown={resize.onMouseDown}
            title="Drag to resize"
            style={{ flex: '0 0 6px', cursor: 'row-resize' }}
          />
          <div
            style={{
              height: contentHeight,
              overflowY: 'auto',
              padding: '0 10px 8px',
              borderLeft: `3px solid ${color}`,
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
                      <pre style={{ overflowX: 'auto', maxWidth: '100%' }}>{children}</pre>
                    ),
                  }}
                >
                  {methodInfoDetail(methodMeta)}
                </Markdown>
              </div>
            </details>
          </div>
        </>
      )}
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        title={collapsed ? 'Expand' : 'Collapse'}
        style={{
          flex: '0 0 auto',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          padding: '6px 10px',
          font: 'inherit',
          fontWeight: 'bold',
          textAlign: 'left',
          background: 'none',
          border: 'none',
          borderLeft: `3px solid ${color}`,
          borderTop: '1px solid var(--border)',
          cursor: 'pointer',
        }}
      >
        <span>{collapsed ? '▲' : '▼'}</span>
        <span>{method}</span>
      </button>
    </div>
  );
}
