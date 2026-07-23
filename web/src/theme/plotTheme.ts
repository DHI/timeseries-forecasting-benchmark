export type PlotTheme = {
  paperBg: string;
  plotBg: string;
  /** Domain/family boundary lines are drawn the same color as the plot background, so they
   * "blend in" as pure grid separators rather than a visible color choice. */
  boundaryLineColor: string;
  axisLineColor: string;
  labelText: string;
  mutedLineColor: string;
  actualLineColor: string;
  /** Translucent scatter-dot fill (e.g. the seasonal-pattern cloud) — needs its own alpha
   * against each background rather than reusing an opaque token. */
  scatterDotColor: string;
};

const LIGHT: PlotTheme = {
  paperBg: '#ffffff',
  plotBg: '#ffffff',
  boundaryLineColor: '#ffffff',
  axisLineColor: '#000000',
  labelText: '#1f2328',
  mutedLineColor: '#888888',
  actualLineColor: '#000000',
  scatterDotColor: 'rgba(0, 0, 0, 0.15)',
};

const DARK: PlotTheme = {
  paperBg: '#0d1117',
  plotBg: '#0d1117',
  boundaryLineColor: '#0d1117',
  axisLineColor: '#8b949e',
  labelText: '#e6edf3',
  mutedLineColor: '#8b949e',
  actualLineColor: '#e6edf3',
  scatterDotColor: 'rgba(230, 237, 243, 0.25)',
};

export function getPlotTheme(isDark: boolean): PlotTheme {
  return isDark ? DARK : LIGHT;
}
