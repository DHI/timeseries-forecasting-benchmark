import { describe, expect, it } from 'vitest';
import {
  appReducer,
  CASE_PANEL_MAX_WIDTH,
  CASE_PANEL_MIN_WIDTH,
  GETTING_STARTED_MAX_HEIGHT,
  GETTING_STARTED_MAX_WIDTH,
  GETTING_STARTED_MIN_HEIGHT,
  GETTING_STARTED_MIN_WIDTH,
  initialAppState,
} from './AppStateContext';

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

  it('toggles the global-history and exogenous toggles independently', () => {
    let state = initialAppState();
    state = appReducer(state, { type: 'TOGGLE_GLOBAL' });
    expect(state.showGlobal).toBe(true);
    expect(state.showExogenous).toBe(false);
    state = appReducer(state, { type: 'TOGGLE_EXOGENOUS' });
    expect(state.showGlobal).toBe(true);
    expect(state.showExogenous).toBe(true);
  });

  it('sets the training-time overlay', () => {
    const state = appReducer(initialAppState(), {
      type: 'SET_TRAINING_TIME_OVERLAY',
      active: true,
    });
    expect(state.trainingTimeOverlay).toBe(true);
    const cleared = appReducer(state, { type: 'SET_TRAINING_TIME_OVERLAY', active: false });
    expect(cleared.trainingTimeOverlay).toBe(false);
  });

  it('sets the case panel width, clamped to the allowed range', () => {
    const state = appReducer(initialAppState(), { type: 'SET_CASE_PANEL_WIDTH', width: 600 });
    expect(state.casePanelWidth).toBe(600);

    const clampedLow = appReducer(initialAppState(), {
      type: 'SET_CASE_PANEL_WIDTH',
      width: CASE_PANEL_MIN_WIDTH - 50,
    });
    expect(clampedLow.casePanelWidth).toBe(CASE_PANEL_MIN_WIDTH);

    const clampedHigh = appReducer(initialAppState(), {
      type: 'SET_CASE_PANEL_WIDTH',
      width: CASE_PANEL_MAX_WIDTH + 50,
    });
    expect(clampedHigh.casePanelWidth).toBe(CASE_PANEL_MAX_WIDTH);
  });

  it('toggles the getting-started window', () => {
    const opened = appReducer(initialAppState(), { type: 'TOGGLE_GETTING_STARTED' });
    expect(opened.gettingStartedOpen).toBe(true);
    const closed = appReducer(opened, { type: 'TOGGLE_GETTING_STARTED' });
    expect(closed.gettingStartedOpen).toBe(false);
  });

  it('sets the getting-started position', () => {
    const state = appReducer(initialAppState(), {
      type: 'SET_GETTING_STARTED_POSITION',
      x: 120,
      y: 80,
    });
    expect(state.gettingStartedX).toBe(120);
    expect(state.gettingStartedY).toBe(80);
  });

  it('sets the getting-started size, clamped to the allowed range', () => {
    const state = appReducer(initialAppState(), {
      type: 'SET_GETTING_STARTED_SIZE',
      width: 500,
      height: 600,
    });
    expect(state.gettingStartedWidth).toBe(500);
    expect(state.gettingStartedHeight).toBe(600);

    const clampedLow = appReducer(initialAppState(), {
      type: 'SET_GETTING_STARTED_SIZE',
      width: GETTING_STARTED_MIN_WIDTH - 50,
      height: GETTING_STARTED_MIN_HEIGHT - 50,
    });
    expect(clampedLow.gettingStartedWidth).toBe(GETTING_STARTED_MIN_WIDTH);
    expect(clampedLow.gettingStartedHeight).toBe(GETTING_STARTED_MIN_HEIGHT);

    const clampedHigh = appReducer(initialAppState(), {
      type: 'SET_GETTING_STARTED_SIZE',
      width: GETTING_STARTED_MAX_WIDTH + 50,
      height: GETTING_STARTED_MAX_HEIGHT + 50,
    });
    expect(clampedHigh.gettingStartedWidth).toBe(GETTING_STARTED_MAX_WIDTH);
    expect(clampedHigh.gettingStartedHeight).toBe(GETTING_STARTED_MAX_HEIGHT);
  });

  it('delegates selection actions to nextSelection', () => {
    const state = appReducer(initialAppState(), {
      type: 'CELL_CLICK',
      case_id: 'case_a',
      method: 'LightGBM',
    });
    expect(state.selection).toEqual({ case_id: 'case_a', methods: ['LightGBM'] });
  });
});
