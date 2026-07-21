import { interpolateGreys, interpolateRdYlGn, interpolateReds } from 'd3-scale-chromatic';
import type { ColorScale } from 'plotly.js';
import type { MetricConfig } from './metricConfig';

const N_STOPS = 17;

/** matplotlib colormap name -> a (0..1) -> color interpolator, oriented the same way
 * matplotlib orients it (low value -> first color, high value -> last color). */
function interpolatorFor(cmap: MetricConfig['cmap']): (t: number) => string {
  if (cmap === 'RdYlGn') return interpolateRdYlGn;
  if (cmap === 'RdYlGn_r') return (t: number) => interpolateRdYlGn(1 - t);
  if (cmap === 'Greys') return interpolateGreys;
  return interpolateReds;
}

/** A Plotly `colorscale` array (`[[stop, "rgb(...)"] , ...]`, stop in [0, 1]) equivalent
 * to this metric's matplotlib colormap, sampled at a fixed resolution. Plotly heatmaps
 * accept this directly in place of a named colorscale string. */
export function colorscaleFor(cfg: MetricConfig): ColorScale {
  const interpolate = interpolatorFor(cfg.cmap);
  return Array.from({ length: N_STOPS }, (_, i) => {
    const stop = i / (N_STOPS - 1);
    return [stop, interpolate(stop)] as [number, string];
  });
}
