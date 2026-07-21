import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// `globals: false` in vitest.config.ts means Testing Library's own auto-cleanup (which
// detects a global `afterEach`) never registers, so renders accumulate across tests
// within a file unless unmounted here explicitly.
afterEach(() => {
  cleanup();
});
