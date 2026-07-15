import { describe, expect, it } from 'vitest';
import { appReducer, initialAppState } from './AppStateContext';

describe('appReducer', () => {
  it('sets the metric', () => {
    const state = appReducer(initialAppState(), { type: 'SET_METRIC', metric: 'MASE' });
    expect(state.metric).toBe('MASE');
  });

  it('sets domains and families independently', () => {
    let state = initialAppState();
    state = appReducer(state, { type: 'SET_DOMAINS', domains: ['hydrology'] });
    state = appReducer(state, { type: 'SET_FAMILIES', families: ['ml', 'dl'] });
    expect(state.domains).toEqual(['hydrology']);
    expect(state.families).toEqual(['ml', 'dl']);
  });

  it('toggles the sidebar', () => {
    const collapsed = appReducer(initialAppState({ sidebarCollapsed: false }), {
      type: 'TOGGLE_SIDEBAR',
    });
    expect(collapsed.sidebarCollapsed).toBe(true);
    const expanded = appReducer(collapsed, { type: 'TOGGLE_SIDEBAR' });
    expect(expanded.sidebarCollapsed).toBe(false);
  });
});
