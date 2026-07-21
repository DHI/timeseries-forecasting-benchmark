import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// This repo's GitHub Enterprise Pages site is private, so it gets a randomly-named
// subdomain (e.g. https://scaling-spork-6238k13.pages.github.io/) served at the domain
// root — not the org.github.io/<repo>/ project-page path public GitHub.com uses. Check
// `gh api repos/DHI-Internal/timeseries-forecasting-benchmark/pages` before changing this.
// https://vite.dev/config/
export default defineConfig({
  base: '/',
  plugins: [react()],
});
