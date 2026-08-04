import { createContext, useContext, useReducer } from 'react';
import type { Dispatch, ReactNode } from 'react';
import { EMPTY_SELECTION, nextSelection } from './selectionReducer';
import type { SelectionAction, SelectionState } from './selectionReducer';

export const CASE_PANEL_MIN_WIDTH = 320;
export const CASE_PANEL_MAX_WIDTH = 900;
export const CASE_PANEL_DEFAULT_WIDTH = 460;

export const GETTING_STARTED_MIN_WIDTH = 280;
export const GETTING_STARTED_MAX_WIDTH = 700;
export const GETTING_STARTED_DEFAULT_WIDTH = 380;

function clampCasePanelWidth(width: number): number {
  return Math.min(CASE_PANEL_MAX_WIDTH, Math.max(CASE_PANEL_MIN_WIDTH, width));
}

function clampGettingStartedWidth(width: number): number {
  return Math.min(GETTING_STARTED_MAX_WIDTH, Math.max(GETTING_STARTED_MIN_WIDTH, width));
}

export type AppState = {
  metric: string;
  domains: string[];
  families: string[];
  sidebarCollapsed: boolean;
  showGlobal: boolean;
  showExogenous: boolean;
  trainingTimeOverlay: boolean;
  casePanelWidth: number;
  gettingStartedOpen: boolean;
  gettingStartedWidth: number;
  selection: SelectionState;
};

export type AppAction =
  | { type: 'SET_METRIC'; metric: string }
  | { type: 'SET_DOMAINS'; domains: string[] }
  | { type: 'SET_FAMILIES'; families: string[] }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'TOGGLE_GLOBAL' }
  | { type: 'TOGGLE_EXOGENOUS' }
  | { type: 'SET_TRAINING_TIME_OVERLAY'; active: boolean }
  | { type: 'SET_CASE_PANEL_WIDTH'; width: number }
  | { type: 'TOGGLE_GETTING_STARTED' }
  | { type: 'SET_GETTING_STARTED_WIDTH'; width: number }
  | SelectionAction;

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_METRIC':
      return { ...state, metric: action.metric };
    case 'SET_DOMAINS':
      return { ...state, domains: action.domains };
    case 'SET_FAMILIES':
      return { ...state, families: action.families };
    case 'TOGGLE_SIDEBAR':
      return { ...state, sidebarCollapsed: !state.sidebarCollapsed };
    case 'TOGGLE_GLOBAL':
      return { ...state, showGlobal: !state.showGlobal };
    case 'TOGGLE_EXOGENOUS':
      return { ...state, showExogenous: !state.showExogenous };
    case 'SET_TRAINING_TIME_OVERLAY':
      return { ...state, trainingTimeOverlay: action.active };
    case 'SET_CASE_PANEL_WIDTH':
      return { ...state, casePanelWidth: clampCasePanelWidth(action.width) };
    case 'TOGGLE_GETTING_STARTED':
      return { ...state, gettingStartedOpen: !state.gettingStartedOpen };
    case 'SET_GETTING_STARTED_WIDTH':
      return { ...state, gettingStartedWidth: clampGettingStartedWidth(action.width) };
    case 'CELL_CLICK':
    case 'REMOVE_CHIP':
    case 'CLEAR_SELECTION':
      return { ...state, selection: nextSelection(state.selection, action) };
    default:
      return state;
  }
}

export function initialAppState(overrides: Partial<AppState> = {}): AppState {
  return {
    metric: '',
    domains: [],
    families: [],
    sidebarCollapsed: true,
    showGlobal: false,
    showExogenous: false,
    trainingTimeOverlay: false,
    casePanelWidth: CASE_PANEL_DEFAULT_WIDTH,
    gettingStartedOpen: false,
    gettingStartedWidth: GETTING_STARTED_DEFAULT_WIDTH,
    selection: EMPTY_SELECTION,
    ...overrides,
  };
}

const StateContext = createContext<AppState | null>(null);
const DispatchContext = createContext<Dispatch<AppAction> | null>(null);

export function AppStateProvider({
  initial,
  children,
}: {
  initial: AppState;
  children: ReactNode;
}) {
  const [state, dispatch] = useReducer(appReducer, initial);
  return (
    <StateContext.Provider value={state}>
      <DispatchContext.Provider value={dispatch}>{children}</DispatchContext.Provider>
    </StateContext.Provider>
  );
}

export function useAppState(): AppState {
  const state = useContext(StateContext);
  if (!state) throw new Error('useAppState must be used within AppStateProvider');
  return state;
}

export function useAppDispatch(): Dispatch<AppAction> {
  const dispatch = useContext(DispatchContext);
  if (!dispatch) throw new Error('useAppDispatch must be used within AppStateProvider');
  return dispatch;
}
