import { useCallback, useEffect, useMemo } from 'react';
import { DomainChecklist } from './components/Sidebar/DomainChecklist';
import { FamilyMultiSelect } from './components/Sidebar/FamilyMultiSelect';
import { HoldTrainingTimeButton } from './components/Sidebar/HoldTrainingTimeButton';
import { MarkerSwatch } from './components/Sidebar/MarkerSwatch';
import { MetricDropdown } from './components/Sidebar/MetricDropdown';
import { ToggleExogenous } from './components/Sidebar/ToggleExogenous';
import { ToggleGlobal } from './components/Sidebar/ToggleGlobal';
import { MatrixGraph } from './components/Matrix/MatrixGraph';
import { BenchmarkDesign } from './components/BenchmarkDesign/BenchmarkDesign';
import { CasePanel } from './components/CasePanel/CasePanel';
import { MethodInfoPanel } from './components/MethodInfoPanel/MethodInfoPanel';
import { SnapshotPanel } from './components/SnapshotPanel/SnapshotPanel';
import { useEdgeResize } from './hooks/useEdgeResize';
import { DOMAIN_RESTRICTED_METRICS } from './data/metricConfig';
import {
  casesEligibleForMetric,
  domainOptions,
  familyOptions,
  metricOptions,
} from './data/selectors';
import { useJsonData } from './data/loadData';
import {
  AppStateProvider,
  initialAppState,
  useAppDispatch,
  useAppState,
} from './state/AppStateContext';
import type { CaseRegistry, MethodRegistry, ScoreRow } from './types/data';

function App() {
  const scores = useJsonData<ScoreRow[]>('data/scores.json');
  const methods = useJsonData<MethodRegistry>('data/methods.json');
  const cases = useJsonData<CaseRegistry>('data/cases.json');

  if (scores.status === 'error') return <ErrorBanner error={scores.error} />;
  if (methods.status === 'error') return <ErrorBanner error={methods.error} />;
  if (cases.status === 'error') return <ErrorBanner error={cases.error} />;
  if (scores.status !== 'ready' || methods.status !== 'ready' || cases.status !== 'ready') {
    return <p style={{ padding: 16 }}>Loading benchmark results…</p>;
  }

  return <AppRoot scores={scores.data} methods={methods.data} cases={cases.data} />;
}

function ErrorBanner({ error }: { error: Error }) {
  return (
    <p role="alert" style={{ padding: 16, color: 'crimson' }}>
      Failed to load benchmark data: {error.message}
    </p>
  );
}

function AppRoot({
  scores,
  methods,
  cases,
}: {
  scores: ScoreRow[];
  methods: MethodRegistry;
  cases: CaseRegistry;
}) {
  const metricOpts = useMemo(() => metricOptions(scores), [scores]);
  const domainOpts = useMemo(() => domainOptions(scores), [scores]);
  const familyOpts = useMemo(() => familyOptions(scores), [scores]);

  const initial = useMemo(
    () =>
      initialAppState({
        metric: metricOpts[0] ?? '',
        domains: domainOpts,
        families: familyOpts,
      }),
    // Only the first-load option sets should seed the initial state; user changes after
    // that shouldn't be clobbered by a later options recompute.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const domainMetrics = useMemo(
    () => new Map(Object.entries(cases).map(([id, meta]) => [id, meta.domain_metrics])),
    [cases],
  );

  return (
    <AppStateProvider initial={initial}>
      <MainLayout
        scores={scores}
        methods={methods}
        cases={cases}
        domainOpts={domainOpts}
        familyOpts={familyOpts}
        metricOpts={metricOpts}
        domainMetrics={domainMetrics}
      />
    </AppStateProvider>
  );
}

function MainLayout({
  scores,
  methods,
  cases,
  domainOpts,
  familyOpts,
  metricOpts,
  domainMetrics,
}: {
  scores: ScoreRow[];
  methods: MethodRegistry;
  cases: CaseRegistry;
  domainOpts: string[];
  familyOpts: string[];
  metricOpts: string[];
  domainMetrics: Map<string, string[]>;
}) {
  const state = useAppState();
  const dispatch = useAppDispatch();

  const handleCellClick = useCallback(
    (caseId: string, method: string) => dispatch({ type: 'CELL_CLICK', case_id: caseId, method }),
    [dispatch],
  );
  const handleRemoveMethod = useCallback(
    (method: string) => dispatch({ type: 'REMOVE_CHIP', method }),
    [dispatch],
  );
  const handleClear = useCallback(() => dispatch({ type: 'CLEAR_SELECTION' }), [dispatch]);
  const handleCasePanelResize = useCallback(
    (width: number) => dispatch({ type: 'SET_CASE_PANEL_WIDTH', width }),
    [dispatch],
  );
  const casePanelResize = useEdgeResize('x', state.casePanelWidth, handleCasePanelResize);

  // Plotly's `useResizeHandler` only re-measures a plot's width on a native `window`
  // "resize" event; dragging the case panel's edge only changes React state, so nudge
  // Plotly's existing listener manually to keep the panel's plots in sync with its width.
  useEffect(() => {
    window.dispatchEvent(new Event('resize'));
  }, [state.casePanelWidth]);

  const filteredRows = useMemo(() => {
    if (!state.metric) return [];
    const domainSet = new Set(state.domains);
    const familySet = new Set(state.families);
    let caseIds = [...new Set(scores.filter((r) => domainSet.has(r.domain)).map((r) => r.case_id))];
    caseIds = casesEligibleForMetric(
      state.metric,
      caseIds,
      domainMetrics,
      DOMAIN_RESTRICTED_METRICS,
    );
    const caseIdSet = new Set(caseIds);
    const methodSet = new Set(scores.filter((r) => familySet.has(r.family)).map((r) => r.method));
    return scores.filter(
      (r) => r.metric === state.metric && caseIdSet.has(r.case_id) && methodSet.has(r.method),
    );
  }, [scores, state.metric, state.domains, state.families, domainMetrics]);

  const sidebarWidth = state.sidebarCollapsed ? 0 : 260;
  const selectedCase = state.selection.case_id ? cases[state.selection.case_id] : undefined;
  const casePanelWidth = selectedCase ? state.casePanelWidth : 0;

  return (
    <div style={{ display: 'flex' }}>
      <div
        style={{
          flex: `0 0 ${sidebarWidth}px`,
          position: 'sticky',
          top: 0,
          maxHeight: '100vh',
        }}
      >
        <aside
          style={{
            width: sidebarWidth,
            height: '100%',
            background: 'var(--panel-bg)',
            borderRight: state.sidebarCollapsed ? 'none' : '2px solid var(--border)',
            overflow: 'hidden',
            boxSizing: 'border-box',
          }}
        >
          {!state.sidebarCollapsed && (
            <div
              style={{
                boxSizing: 'border-box',
                maxHeight: '100vh',
                padding: 16,
                overflowY: 'auto',
              }}
            >
              <h3>Controls</h3>
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>Metric</label>
              <MetricDropdown
                options={metricOpts}
                value={state.metric}
                onChange={(metric) => dispatch({ type: 'SET_METRIC', metric })}
              />
              <p style={{ color: '#666', fontSize: '0.8em', marginTop: 4 }}>
                NSE and KGE only apply to cases whose case.toml declares them (flow/level domains) —
                the matrix rows narrow to just those cases when one is selected.
              </p>
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>Domains</label>
              <DomainChecklist
                options={domainOpts}
                value={state.domains}
                onChange={(domains) => dispatch({ type: 'SET_DOMAINS', domains })}
              />
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>
                Method families
              </label>
              <FamilyMultiSelect
                options={familyOpts}
                value={state.families}
                onChange={(families) => dispatch({ type: 'SET_FAMILIES', families })}
              />
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>
                Global methods
              </label>
              <ToggleGlobal
                checked={state.showGlobal}
                onChange={() => dispatch({ type: 'TOGGLE_GLOBAL' })}
              />
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 4 }}>
                <MarkerSwatch corner="top-right" color="#FF1493" />
                <p style={{ color: '#666', fontSize: '0.8em', margin: 0 }}>
                  Some global models have been trained with longer history.
                </p>
              </div>
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>
                Exogenous inputs
              </label>
              <ToggleExogenous
                checked={state.showExogenous}
                onChange={() => dispatch({ type: 'TOGGLE_EXOGENOUS' })}
              />
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 4 }}>
                <MarkerSwatch corner="bottom-left" color="#0B7285" />
                <p style={{ color: '#666', fontSize: '0.8em', margin: 0 }}>
                  Where a method has an exogenous-aware variant and the case provides exogenous
                  data, this swaps in that variant's score for the cell. Cases or methods without an
                  exogenous option are unaffected.
                </p>
              </div>
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>
                Training time
              </label>
              <HoldTrainingTimeButton
                onChange={(active) => dispatch({ type: 'SET_TRAINING_TIME_OVERLAY', active })}
              />
              <p style={{ color: '#666', fontSize: '0.8em', marginTop: 4 }}>
                Press and hold to show each cell's measured training time instead of its score. You
                can still hover around matrix cells while holding the button.
              </p>
            </div>
          )}
        </aside>
        <button
          type="button"
          title="Collapse/expand controls"
          onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
          style={{
            position: 'absolute',
            top: 16,
            left: '100%',
            transform: 'translateX(-50%)',
            background: 'var(--bg)',
            cursor: 'pointer',
          }}
        >
          {state.sidebarCollapsed ? '▶' : '◀'}
        </button>
      </div>
      <main style={{ flex: 1, padding: 16, minWidth: 0 }}>
        <h1>Timeseries Forecasting Benchmark</h1>
        <p style={{ color: '#444', maxWidth: 1000 }}>
          The following matrix compiles the forecasting results of multiple methods across a variety
          of timeseries in different water domains. Each cell represents one case x method run,
          coloured by the metric chosen in the sidebar.
        </p>
        <MatrixGraph
          rows={filteredRows}
          metric={state.metric}
          selection={state.selection}
          showGlobal={state.showGlobal}
          showExogenous={state.showExogenous}
          showTrainingTime={state.trainingTimeOverlay}
          onCellClick={handleCellClick}
        />
        <BenchmarkDesign />
      </main>
      <MethodInfoPanel
        caseId={state.selection.case_id}
        caseMeta={selectedCase}
        methodsRegistry={methods}
        scores={scores}
        methods={state.selection.methods}
        showExogenous={state.showExogenous}
        leftInset={sidebarWidth}
        rightInset={casePanelWidth}
      />
      {selectedCase && (
        <div
          style={{
            flex: `0 0 ${state.casePanelWidth}px`,
            width: state.casePanelWidth,
            boxSizing: 'border-box',
            position: 'sticky',
            top: 0,
            maxHeight: '100vh',
            borderLeft: '2px solid var(--border)',
            background: 'var(--bg)',
          }}
        >
          {/* The drag handle lives outside the scrollable div below so it stays anchored to
              the panel's full visible height instead of scrolling away with its content. */}
          <div
            onMouseDown={casePanelResize.onMouseDown}
            title="Drag to resize"
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: 6,
              cursor: 'col-resize',
            }}
          />
          <div
            style={{
              boxSizing: 'border-box',
              maxHeight: '100vh',
              padding: 16,
              overflowY: 'auto',
            }}
          >
            <CasePanel
              caseId={state.selection.case_id!}
              caseMeta={selectedCase}
              scores={scores}
              showGlobal={state.showGlobal}
            />
            <SnapshotPanel
              caseId={state.selection.case_id!}
              caseMeta={selectedCase}
              scores={scores}
              methods={state.selection.methods}
              metric={state.metric}
              showGlobal={state.showGlobal}
              showExogenous={state.showExogenous}
              onRemoveMethod={handleRemoveMethod}
              onClear={handleClear}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
