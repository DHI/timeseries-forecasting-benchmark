import { createContext, useContext, useReducer } from 'react';
import type { Dispatch, ReactNode } from 'react';

export type AppState = {
  metric: string;
  domains: string[];
  families: string[];
  sidebarCollapsed: boolean;
};

export type AppAction =
  | { type: 'SET_METRIC'; metric: string }
  | { type: 'SET_DOMAINS'; domains: string[] }
  | { type: 'SET_FAMILIES'; families: string[] }
  | { type: 'TOGGLE_SIDEBAR' };

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
    default:
      return state;
  }
}

export function initialAppState(overrides: Partial<AppState> = {}): AppState {
  return {
    metric: '',
    domains: [],
    families: [],
    sidebarCollapsed: false,
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
