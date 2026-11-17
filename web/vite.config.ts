import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The repo is public under DHI, so GitHub Pages serves it as a project page at
// https://dhi.github.io/timeseries-forecasting-benchmark/ — assets must resolve under that
// sub-path, not the domain root. Update this if the repo is renamed or moved.
//
// The AdaLab mirror (see ../Containerfile) sets APP_BASE to its own /apps/<app_url>/ path.
// https://vite.dev/config/
export default defineConfig({
  base: process.env.APP_BASE || '/timeseries-forecasting-benchmark/',
  plugins: [react()],
});
