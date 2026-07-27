import { useState } from 'react';
import Markdown from 'react-markdown';
import { BENCHMARK_DESIGN } from '../../data/benchmarkDesign';

/** Thin grey info icon next to the page title; clicking it opens a centered, scrollable modal
 * with the benchmark's design/methodology writeup. Styled like the app's other panels (white
 * background / black border via the `--bg` / `--border` tokens from index.css) rather than as
 * a one-off. */
export function InstructionsModal() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        title="Benchmark Design"
        aria-label="Show benchmark design details"
        onClick={() => setOpen(true)}
        style={{
          width: 22,
          height: 22,
          borderRadius: '50%',
          border: '1.5px solid var(--text-muted)',
          background: 'var(--panel-bg)',
          color: 'var(--text-muted)',
          fontSize: '0.8em',
          fontWeight: 'bold',
          lineHeight: 1,
          cursor: 'pointer',
          padding: 0,
        }}
      >
        i
      </button>
      {open && (
        <div
          onClick={() => setOpen(false)}
          data-testid="instructions-backdrop"
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
                aria-label="Close instructions"
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
              <h2 style={{ margin: 0 }}>Benchmark Design</h2>
            </div>
            <Markdown>{BENCHMARK_DESIGN}</Markdown>
          </div>
        </div>
      )}
    </>
  );
}
