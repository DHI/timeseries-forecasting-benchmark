// Pixel size of one heatmap cell, and the fixed margin (title/colorbar/labels) around it
// — together they size the heatmap figure so cells render square. Port of app/app.py's
// CELL / MATRIX_MARGIN.
export const CELL = 42;
export const MATRIX_MARGIN = 120;

/** Short row-label abbreviation per domain, so the vertical domain label column stays
 * narrow. Port of app/app.py's DOMAIN_KEY_DICT. */
export const DOMAIN_KEY_DICT: Record<string, string> = {
  groundwater: 'GrW',
  hydrology: 'Hyd',
  spectral_waves: 'SpW',
  storm_surge: 'StS',
  urban_drainage: 'UrD',
  wastewater_treatment: 'WaT',
  water_distribution: 'WaD',
  water_quality: 'WaQ',
};

/** Colour-blind-safe palette (Wong 2011, dropping its low-contrast yellow), used to
 * colour-match a selected method's chip/outline/forecast line. Port of
 * app/snapshot.py's METHOD_COLORS. */
export const METHOD_COLORS = [
  '#0072B2',
  '#D55E00',
  '#009E73',
  '#CC79A7',
  '#56B4E9',
  '#E69F00',
  '#8B008B',
];
