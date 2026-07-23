import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// `globals: false` in vitest.config.ts means Testing Library's own auto-cleanup (which
// detects a global `afterEach`) never registers, so renders accumulate across tests
// within a file unless unmounted here explicitly.
afterEach(() => {
  cleanup();
});

// jsdom doesn't implement matchMedia; ThemeContext reads it to detect the OS light/dark
// preference, so tests need a stub. Always reports "no preference" (light).
window.matchMedia ??= (query: string) =>
  ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }) as MediaQueryList;
