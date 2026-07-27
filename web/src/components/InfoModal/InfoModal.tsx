import { useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

type Props = {
  buttonTitle: string;
  ariaLabel: string;
  heading: string;
  testId: string;
  /** Button diameter in px. Default 22 (the page-title icon's size); pass a smaller value for
   * tighter contexts like a sidebar label row. */
  size?: number;
  children: ReactNode;
};

/** Thin grey info icon that opens a centered, scrollable modal with arbitrary content. Styled
 * like the app's other panels (white background / black border via the `--bg` / `--border`
 * tokens from index.css) rather than as a one-off. Shared chrome for the page-title's
 * "Benchmark Design" icon (InstructionsModal.tsx) and the sidebar's metric cheatsheet icon
 * (MetricInfoModal.tsx). The modal itself is portaled to `document.body`: several ancestors in
 * this app (e.g. the sidebar's `position: sticky` wrapper) establish their own stacking context,
 * which would otherwise trap a `position: fixed` modal underneath unrelated later-in-DOM content
 * like the matrix, regardless of its own z-index. */
export function InfoModal({ buttonTitle, ariaLabel, heading, testId, size = 22, children }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        title={buttonTitle}
        aria-label={ariaLabel}
        onClick={() => setOpen(true)}
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          border: '1.5px solid var(--text-muted)',
          background: 'var(--panel-bg)',
          color: 'var(--text-muted)',
          fontSize: size * 0.55,
          fontWeight: 'bold',
          lineHeight: 1,
          cursor: 'pointer',
          padding: 0,
          flex: '0 0 auto',
        }}
      >
        i
      </button>
      {open &&
        createPortal(
          <div
            onClick={() => setOpen(false)}
            data-testid={testId}
            style={{
              position: 'fixed',
              inset: 0,
              zIndex: 100,
              background: 'rgba(0, 0, 0, 0.5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <div
              onClick={(e) => e.stopPropagation()}
              style={{
                position: 'relative',
                background: 'var(--bg)',
                border: '2px solid var(--border)',
                maxWidth: 700,
                width: '90%',
                maxHeight: '80vh',
                overflowY: 'auto',
                boxSizing: 'border-box',
                padding: '16px 20px',
              }}
            >
              <div
                style={{
                  position: 'sticky',
                  top: 0,
                  background: 'var(--bg)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                }}
              >
                <button
                  type="button"
                  title="Close"
                  aria-label="Close"
                  onClick={() => setOpen(false)}
                  style={{
                    background: 'none',
                    border: 'none',
                    font: 'inherit',
                    fontSize: '1.2em',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    padding: 0,
                  }}
                >
                  ✕
                </button>
                <h2 style={{ margin: 0 }}>{heading}</h2>
              </div>
              {children}
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
