import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Served from a GitHub Pages project URL (https://<user>.github.io/timeseries-forecasting-benchmark/),
// not the domain root, so every asset/data-fetch path must be aware of this base.
// https://vite.dev/config/
export default defineConfig({
  base: '/timeseries-forecasting-benchmark/',
  plugins: [react()],
});
