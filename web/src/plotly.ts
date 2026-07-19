import Plotly from 'plotly.js-cartesian-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

// The default `react-plotly.js` import pulls in the full ~2MB-minified plotly.js bundle
// (3D, maps, finance charts, WebGL — none of which this app uses). Everything here is a
// 2D cartesian trace (Heatmap, Scatter, Bar), so the cartesian-only bundle is enough.
const Plot = createPlotlyComponent(Plotly);

export default Plot;
