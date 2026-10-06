import { describe, expect, it } from 'vitest';
import { snapshotPanelLayout } from './snapshotLayout';

describe('snapshotPanelLayout', () => {
  it('stacks panels in a single column when maxCols is 1', () => {
    const panels = snapshotPanelLayout(3, 1);
    // Every panel gets its own axis pair (x/x2/x3), same suffix on both — Plotly has no
    // "shared x-axis" without opting in explicitly, so each subplot is independent.
    expect(panels.map((p) => p.xaxisRef)).toEqual(['x', 'x2', 'x3']);
    expect(panels.map((p) => p.yaxisRef)).toEqual(['y', 'y2', 'y3']);
    // All panels span (most of) the full width in a single-column layout.
    for (const p of panels) expect(p.xDomain[1] - p.xDomain[0]).toBeGreaterThan(0.9);
    // Panel 0 is at the top (highest y), panel 2 at the bottom (lowest y).
    expect(panels[0]!.yDomain[0]).toBeGreaterThan(panels[2]!.yDomain[0]);
  });

  it('assigns distinct axis keys per panel beyond the first', () => {
    const panels = snapshotPanelLayout(2, 2);
    expect(panels[0]!.xaxisKey).toBe('xaxis');
    expect(panels[1]!.xaxisKey).toBe('xaxis2');
    expect(panels[0]!.yaxisKey).toBe('yaxis');
    expect(panels[1]!.yaxisKey).toBe('yaxis2');
  });

  it('wraps into rows when panelCount exceeds maxCols', () => {
    const panels = snapshotPanelLayout(3, 2);
    // 3 panels, 2 columns -> 2 rows: panel 2 wraps to row 1, col 0.
    expect(panels[0]!.xDomain[0]).toBeLessThan(panels[1]!.xDomain[0]);
    expect(panels[2]!.xDomain).toEqual(panels[0]!.xDomain);
    expect(panels[2]!.yDomain[1]).toBeLessThan(panels[0]!.yDomain[1]);
  });

  it('never exceeds the requested column count even with a single panel', () => {
    const panels = snapshotPanelLayout(1, 3);
    expect(panels).toHaveLength(1);
    expect(panels[0]!.xDomain[0]).toBeGreaterThanOrEqual(0);
    expect(panels[0]!.xDomain[1]).toBeLessThanOrEqual(1);
  });
});
