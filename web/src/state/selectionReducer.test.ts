import { describe, expect, it } from 'vitest';
import { EMPTY_SELECTION, nextSelection } from './selectionReducer';
import type { SelectionState } from './selectionReducer';

describe('nextSelection', () => {
  it('starts a new selection on the first click', () => {
    const sel = nextSelection(EMPTY_SELECTION, {
      type: 'CELL_CLICK',
      case_id: 'case_a',
      method: 'LightGBM',
    });
    expect(sel).toEqual({ case_id: 'case_a', methods: ['LightGBM'] });
  });

  it('accumulates methods clicked in the same row', () => {
    let sel = nextSelection(EMPTY_SELECTION, {
      type: 'CELL_CLICK',
      case_id: 'case_a',
      method: 'LightGBM',
    });
    sel = nextSelection(sel, { type: 'CELL_CLICK', case_id: 'case_a', method: 'CatBoost' });
    expect(sel).toEqual({ case_id: 'case_a', methods: ['LightGBM', 'CatBoost'] });
  });

  it('evicts the oldest method beyond MAX_OVERLAY_METHODS (3)', () => {
    let sel: SelectionState = { case_id: 'case_a', methods: ['m1', 'm2', 'm3'] };
    sel = nextSelection(sel, { type: 'CELL_CLICK', case_id: 'case_a', method: 'm4' });
    expect(sel.methods).toEqual(['m2', 'm3', 'm4']);
  });

  it('toggles a method off on re-click', () => {
    const sel: SelectionState = { case_id: 'case_a', methods: ['m1', 'm2'] };
    const next = nextSelection(sel, { type: 'CELL_CLICK', case_id: 'case_a', method: 'm1' });
    expect(next.methods).toEqual(['m2']);
  });

  it('clears the selection when the last method is toggled off', () => {
    const sel: SelectionState = { case_id: 'case_a', methods: ['m1'] };
    const next = nextSelection(sel, { type: 'CELL_CLICK', case_id: 'case_a', method: 'm1' });
    expect(next).toEqual(EMPTY_SELECTION);
  });

  it('resets to the clicked case when a different row is clicked', () => {
    const sel: SelectionState = { case_id: 'case_a', methods: ['m1', 'm2'] };
    const next = nextSelection(sel, { type: 'CELL_CLICK', case_id: 'case_b', method: 'm3' });
    expect(next).toEqual({ case_id: 'case_b', methods: ['m3'] });
  });

  it('removes a method via its chip', () => {
    const sel: SelectionState = { case_id: 'case_a', methods: ['m1', 'm2'] };
    const next = nextSelection(sel, { type: 'REMOVE_CHIP', method: 'm1' });
    expect(next).toEqual({ case_id: 'case_a', methods: ['m2'] });
  });

  it('clears the selection when the last chip is removed', () => {
    const sel: SelectionState = { case_id: 'case_a', methods: ['m1'] };
    const next = nextSelection(sel, { type: 'REMOVE_CHIP', method: 'm1' });
    expect(next).toEqual(EMPTY_SELECTION);
  });

  it('clears the selection on CLEAR_SELECTION', () => {
    const sel: SelectionState = { case_id: 'case_a', methods: ['m1', 'm2'] };
    expect(nextSelection(sel, { type: 'CLEAR_SELECTION' })).toEqual(EMPTY_SELECTION);
  });
});
