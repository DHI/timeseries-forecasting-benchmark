import { useCallback, useMemo } from 'react';
import { DomainChecklist } from './components/Sidebar/DomainChecklist';
import { FamilyMultiSelect } from './components/Sidebar/FamilyMultiSelect';
import { MetricDropdown } from './components/Sidebar/MetricDropdown';
import { ToggleExogenous } from './components/Sidebar/ToggleExogenous';
import { ToggleGlobal } from './components/Sidebar/ToggleGlobal';
import { MatrixGraph } from './components/Matrix/MatrixGraph';
import { BenchmarkDesign } from './components/BenchmarkDesign/BenchmarkDesign';
import { CasePanel } from './components/CasePanel/CasePanel';
import { MethodInfoPanel } from './components/MethodInfoPanel/MethodInfoPanel';
import { SnapshotPanel } from './components/SnapshotPanel/SnapshotPanel';
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

  return (
    <div style={{ display: 'flex' }}>
      <aside
        style={{
          width: state.sidebarCollapsed ? 0 : 260,
          flex: `0 0 ${state.sidebarCollapsed ? 0 : 260}px`,
          padding: state.sidebarCollapsed ? 0 : 16,
          borderRight: state.sidebarCollapsed ? 'none' : '1px solid var(--border)',
          overflow: 'hidden',
          boxSizing: 'border-box',
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
          NSE and KGE only apply to cases whose case.toml declares them (flow/level domains) — the
          matrix rows narrow to just those cases when one is selected.
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
        <p style={{ color: '#666', fontSize: '0.8em', marginTop: 4 }}>
          Some global models have been trained with longer history, marked with a corner triangle.
        </p>
        <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>
          Exogenous inputs
        </label>
        <ToggleExogenous
          checked={state.showExogenous}
          onChange={() => dispatch({ type: 'TOGGLE_EXOGENOUS' })}
        />
        <p style={{ color: '#666', fontSize: '0.8em', marginTop: 4 }}>
          Where a method has an exogenous-aware variant and the case provides exogenous data, this
          swaps in that variant's score for the cell and marks it with a teal corner. Cases or
          methods without an exogenous option are unaffected.
        </p>
      </aside>
      <button
        type="button"
        title="Collapse/expand controls"
        onClick={() => dispatch({ type: 'TOGGLE_SIDEBAR' })}
        style={{ flex: '0 0 auto', alignSelf: 'flex-start', marginTop: 16, cursor: 'pointer' }}
      >
        {state.sidebarCollapsed ? '▶' : '◀'}
      </button>
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
          onCellClick={handleCellClick}
        />
        <BenchmarkDesign />
        <MethodInfoPanel
          caseId={state.selection.case_id}
          caseMeta={state.selection.case_id ? cases[state.selection.case_id] : undefined}
          methodsRegistry={methods}
          scores={scores}
          methods={state.selection.methods}
          showExogenous={state.showExogenous}
        />
      </main>
      {state.selection.case_id && cases[state.selection.case_id] && (
        <div
          style={{
            flex: '0 0 460px',
            width: 460,
            boxSizing: 'border-box',
            position: 'sticky',
            top: 0,
            maxHeight: '100vh',
            padding: 16,
            borderLeft: '1px solid var(--border)',
            overflowY: 'auto',
          }}
        >
          <CasePanel
            caseId={state.selection.case_id}
            caseMeta={cases[state.selection.case_id]!}
            scores={scores}
            methods={state.selection.methods}
            showGlobal={state.showGlobal}
          />
          <hr />
          <SnapshotPanel
            caseId={state.selection.case_id}
            caseMeta={cases[state.selection.case_id]!}
            scores={scores}
            methods={state.selection.methods}
            metric={state.metric}
            showGlobal={state.showGlobal}
            showExogenous={state.showExogenous}
            onRemoveMethod={handleRemoveMethod}
            onClear={handleClear}
          />
        </div>
      )}
    </div>
  );
}

export default App;
