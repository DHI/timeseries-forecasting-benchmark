// This partial bundle has no types of its own; `react-plotly.js/factory`'s
// `createPlotlyComponent` accepts its Plotly argument as `unknown`, so a minimal ambient
// declaration is all that's needed to satisfy the import. Use `import type { ... } from
// 'plotly.js'` elsewhere for real figure types — @types/plotly.js provides those without
// needing the full `plotly.js` package installed at runtime.
declare module 'plotly.js-cartesian-dist-min' {
  const Plotly: unknown;
  export default Plotly;
}
