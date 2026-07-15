import Markdown from 'react-markdown';
import { METHOD_COLORS } from '../../data/constants';
import { methodInfo, methodInfoDetail } from '../../data/methodText';
import type { CaseMeta, MethodMeta } from '../../types/data';

type Props = {
  method: string;
  index: number;
  caseMeta: CaseMeta;
  methodMeta: MethodMeta | undefined;
};

/** One card per selected method: a brief description and its exogenous-input usage for
 * this case, plus a collapsed "Implementation detail" section. Colour-matched to that
 * method's chip/forecast line. Port of app/app.py's `_method_info_cards`. */
export function MethodInfoCard({ method, index, caseMeta, methodMeta }: Props) {
  const color = METHOD_COLORS[index % METHOD_COLORS.length];
  return (
    <div
      style={{
        flex: '1 1 320px',
        minWidth: 320,
        paddingLeft: 10,
        borderLeft: `3px solid ${color}`,
      }}
    >
      <div style={{ fontWeight: 'bold', marginBottom: 4 }}>{method}</div>
      <div style={{ fontSize: '0.85em' }}>
        <Markdown>{methodInfo(caseMeta, methodMeta)}</Markdown>
      </div>
      <details style={{ marginTop: 6 }}>
        <summary style={{ fontSize: '0.85em', cursor: 'pointer' }}>Implementation detail</summary>
        <div style={{ fontSize: '0.85em', marginTop: 4 }}>
          <Markdown>{methodInfoDetail(methodMeta)}</Markdown>
        </div>
      </details>
    </div>
  );
}
