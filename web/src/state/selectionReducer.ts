export type SelectionState = {
  case_id: string | null;
  methods: string[];
};

export const EMPTY_SELECTION: SelectionState = { case_id: null, methods: [] };

// Cap on methods overlaid per case: the snapshot graph and the method-info panel both lay
// their per-method content out side by side, which stops being readable much past this.
// Adding a 4th distinct method evicts the oldest rather than refusing the click. Port of
// app/app.py's MAX_OVERLAY_METHODS.
export const MAX_OVERLAY_METHODS = 3;

export type SelectionAction =
  | { type: 'CELL_CLICK'; case_id: string; method: string }
  | { type: 'REMOVE_CHIP'; method: string }
  | { type: 'CLEAR_SELECTION' };

/** Pure transition for the case x method overlay selection. Port of app/app.py's
 * `next_selection` — same-row clicks accumulate methods (oldest evicted beyond
 * MAX_OVERLAY_METHODS, re-click removes); a click in a different row resets to that case;
 * removing the last method clears the selection entirely.
 *
 * The Dash original also handled two purely Dash-specific cases that don't apply here: a
 * chip-rebuild firing its callback with no real click (`trigger_value` falsy) and the
 * matrix's `clickData` being empty before any click ever happens. Neither is reachable
 * from a real React event handler, so this port omits them rather than porting dead
 * branches. */
export function nextSelection(sel: SelectionState, action: SelectionAction): SelectionState {
  switch (action.type) {
    case 'CLEAR_SELECTION':
      return EMPTY_SELECTION;

    case 'REMOVE_CHIP': {
      const methods = sel.methods.filter((m) => m !== action.method);
      return methods.length ? { case_id: sel.case_id, methods } : EMPTY_SELECTION;
    }

    case 'CELL_CLICK': {
      if (action.case_id !== sel.case_id) {
        return { case_id: action.case_id, methods: [action.method] };
      }
      const methods = sel.methods.includes(action.method)
        ? sel.methods.filter((m) => m !== action.method)
        : [...sel.methods, action.method].slice(-MAX_OVERLAY_METHODS);
      return methods.length ? { case_id: action.case_id, methods } : EMPTY_SELECTION;
    }

    default:
      return sel;
  }
}
