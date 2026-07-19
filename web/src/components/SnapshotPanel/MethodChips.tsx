import { METHOD_COLORS } from '../../data/constants';

type Props = {
  methods: string[];
  onRemove: (method: string) => void;
};

/** One removable chip per selected method, colour-matched to its forecast line. Port of
 * app/app.py's `_method_chips`. */
export function MethodChips({ methods, onRemove }: Props) {
  return (
    <>
      {methods.map((method, i) => {
        const color = METHOD_COLORS[i % METHOD_COLORS.length];
        return (
          <button
            key={method}
            type="button"
            title="Remove from overlay"
            onClick={() => onRemove(method)}
            style={{
              border: `1px solid ${color}`,
              borderRadius: 14,
              padding: '2px 10px',
              background: 'white',
              cursor: 'pointer',
              fontSize: '0.85em',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span
              style={{
                display: 'inline-block',
                width: 10,
                height: 10,
                borderRadius: '50%',
                backgroundColor: color,
              }}
            />
            {method} ✕
          </button>
        );
      })}
    </>
  );
}
