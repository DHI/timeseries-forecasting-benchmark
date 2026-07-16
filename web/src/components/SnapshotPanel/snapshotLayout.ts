export type PanelAxes = {
  /** Plotly trace `xaxis`/`yaxis` property value: "x"/"y" for the first panel, "x2"/"y2"
   * for the second, etc. */
  xaxisRef: string;
  yaxisRef: string;
  /** Layout key for this panel's axis config object: "xaxis"/"yaxis" for the first panel,
   * "xaxis2"/"yaxis2" for the second, etc. */
  xaxisKey: string;
  yaxisKey: string;
  xDomain: [number, number];
  yDomain: [number, number];
  /** Center x position (domain fraction) for this panel's title annotation. */
  titleX: number;
  titleY: number;
};

const GAP = 0.05;

/** Per-panel axis domains for a `maxCols`-wide grid of `panelCount` subplots, stacking
 * rows top-to-bottom (panel 0 at the top-left) — Plotly's own multi-axis layout has no
 * `make_subplots`-style helper in JS, so this is a small hand-written equivalent for a
 * fixed grid. In practice `maxCols` is always 1 here (the snapshot panel is a narrow
 * column, not a wide page), but this isn't hard-coded to that case. */
export function snapshotPanelLayout(panelCount: number, maxCols: number): PanelAxes[] {
  const nCols = Math.max(1, Math.min(maxCols, panelCount));
  const nRows = Math.ceil(panelCount / nCols);

  return Array.from({ length: panelCount }, (_, i) => {
    const row = Math.floor(i / nCols);
    const col = i % nCols;
    const xLo = col / nCols;
    const xHi = (col + 1) / nCols;
    const yHi = 1 - row / nRows;
    const yLo = 1 - (row + 1) / nRows;
    const suffix = i === 0 ? '' : String(i + 1);
    const xDomain: [number, number] = [xLo + GAP / 2, xHi - GAP / 2];
    const yDomain: [number, number] = [yLo + GAP, yHi - GAP];
    return {
      xaxisRef: `x${suffix}`,
      yaxisRef: `y${suffix}`,
      xaxisKey: `xaxis${suffix}`,
      yaxisKey: `yaxis${suffix}`,
      xDomain,
      yDomain,
      titleX: (xDomain[0] + xDomain[1]) / 2,
      titleY: yDomain[1],
    };
  });
}
