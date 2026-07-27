import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { DomainChecklist } from './components/Sidebar/DomainChecklist';
import { FamilyMultiSelect } from './components/Sidebar/FamilyMultiSelect';
import { HoldTrainingTimeButton } from './components/Sidebar/HoldTrainingTimeButton';
import { MarkerSwatch } from './components/Sidebar/MarkerSwatch';
import { MetricDropdown } from './components/Sidebar/MetricDropdown';
import { MetricInfoModal } from './components/Sidebar/MetricInfoModal';
import { SidebarToggleHint } from './components/Sidebar/SidebarToggleHint';
import { ToggleExogenous } from './components/Sidebar/ToggleExogenous';
import { ToggleGlobal } from './components/Sidebar/ToggleGlobal';
import { MatrixGraph } from './components/Matrix/MatrixGraph';
import { SelectionHint } from './components/Matrix/SelectionHint';
import { AppInstructions } from './components/AppInstructions/AppInstructions';
import { InstructionsModal } from './components/InstructionsModal/InstructionsModal';
import { useEdgeResize } from './hooks/useEdgeResize';
import { DOMAIN_RESTRICTED_METRICS } from './data/metricConfig';
import { TEXT_WIDTH_OVERHANG, TITLE_BAR_HEIGHT, TRAINING_TIME_MIN_ROWS } from './data/constants';
import {
  casesEligibleForMetric,
  domainOptions,
  domainsEligibleForMetric,
  familyOptions,
  metricOptions,
  orderedCases,
} from './data/selectors';
import { useJsonData } from './data/loadData';
import { ThemeProvider, useTheme } from './theme/ThemeContext';
import {
  AppStateProvider,
  initialAppState,
  useAppDispatch,
  useAppState,
} from './state/AppStateContext';
import type { CaseRegistry, MethodRegistry, ScoreRow } from './types/data';

// Only rendered once a matrix cell is selected, and each pulls in its own Plotly traces
// and/or react-markdown — split into their own chunks so the always-visible matrix view
// doesn't pay for them upfront.
const CasePanel = lazy(() =>
  import('./components/CasePanel/CasePanel').then((m) => ({ default: m.CasePanel })),
);
const SnapshotPanel = lazy(() =>
  import('./components/SnapshotPanel/SnapshotPanel').then((m) => ({ default: m.SnapshotPanel })),
);
const MethodInfoPanel = lazy(() =>
  import('./components/MethodInfoPanel/MethodInfoPanel').then((m) => ({
    default: m.MethodInfoPanel,
  })),
);

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

function AppContent() {
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

  // Undefined until MatrixGraph reports its first rendered width; the intro/Benchmark
  // Design text falls back to its old fixed width until then.
  const [matrixWidth, setMatrixWidth] = useState<number | undefined>(undefined);
  const textMaxWidth = matrixWidth ? matrixWidth + TEXT_WIDTH_OVERHANG : 1000;

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

  // The open case panel shows one specific case's row; if switching to a domain-restricted
  // metric (NSE/KGE) makes that case ineligible, its row is about to vanish from the matrix,
  // so close the panel along with it rather than leaving it pointing at a dropped row.
  useEffect(() => {
    const caseId = state.selection.case_id;
    if (!caseId) return;
    const stillEligible = casesEligibleForMetric(
      state.metric,
      [caseId],
      domainMetrics,
      DOMAIN_RESTRICTED_METRICS,
    ).includes(caseId);
    if (!stillEligible) dispatch({ type: 'CLEAR_SELECTION' });
  }, [state.metric, state.selection.case_id, domainMetrics, dispatch]);

  const eligibleDomains = useMemo(
    () => domainsEligibleForMetric(state.metric, scores, domainMetrics, DOMAIN_RESTRICTED_METRICS),
    [scores, state.metric, domainMetrics],
  );
  const disabledDomains = useMemo(
    () => new Set(domainOpts.filter((d) => !eligibleDomains.has(d))),
    [domainOpts, eligibleDomains],
  );

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

  // Mirrors how useMatrixFigure derives the matrix's own row count (case IDs, excluding
  // extended-history duplicate rows) — needed here to gate the training-time button before
  // the matrix ever tries to render it. See TRAINING_TIME_MIN_ROWS for why.
  const matrixRowCount = useMemo(
    () => orderedCases(filteredRows.filter((r) => !r.used_extended_history)).length,
    [filteredRows],
  );
  const trainingTimeDisabled = matrixRowCount > 0 && matrixRowCount < TRAINING_TIME_MIN_ROWS;
  // Plotly.js fails to repaint the heatmap when a react() call fires while the matrix is
  // this short (see TRAINING_TIME_MIN_ROWS) — NSE/KGE always narrow the matrix into that
  // zone, so leaving/re-entering it (or switching metrics within it) needs a full remount
  // rather than relying on Plotly's in-place update. Keying on the metric only while short
  // avoids remounting on every ordinary (>=4-row) metric switch, where react() already works.
  const matrixKey = matrixRowCount < TRAINING_TIME_MIN_ROWS ? `${state.metric}-short` : 'full';
  const handleTrainingTimeChange = useCallback(
    (active: boolean) => {
      // Only block turning the overlay *on* for a too-short matrix; always let an
      // already-active hold end normally so it can't get stuck on mid-release.
      if (active && trainingTimeDisabled) return;
      dispatch({ type: 'SET_TRAINING_TIME_OVERLAY', active });
    },
    [trainingTimeDisabled, dispatch],
  );

  const { theme, toggle: toggleTheme } = useTheme();

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
            background: 'var(--bg)',
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
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  marginTop: 16,
                  marginBottom: 4,
                }}
              >
                <label style={{ fontWeight: 'bold' }}>Metric</label>
                <MetricInfoModal />
              </div>
              <MetricDropdown
                options={metricOpts}
                value={state.metric}
                onChange={(metric) => dispatch({ type: 'SET_METRIC', metric })}
              />
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>Domains</label>
              <DomainChecklist
                options={domainOpts}
                value={state.domains}
                onChange={(domains) => dispatch({ type: 'SET_DOMAINS', domains })}
                disabledOptions={disabledDomains}
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
                <p style={{ color: 'var(--text-muted)', fontSize: '0.8em', margin: 0 }}>
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
                <p style={{ color: 'var(--text-muted)', fontSize: '0.8em', margin: 0 }}>
                  Where a method has an exogenous-aware variant and the case provides exogenous
                  data, this swaps in that variant's score for the cell. Cases or methods without an
                  exogenous option are unaffected.
                </p>
              </div>
              <label style={{ fontWeight: 'bold', display: 'block', marginTop: 16 }}>
                Training time
              </label>
              <HoldTrainingTimeButton
                onChange={handleTrainingTimeChange}
                disabled={trainingTimeDisabled}
                title={
                  trainingTimeDisabled
                    ? 'Disabled for this short a matrix — see the note below'
                    : undefined
                }
              />
              <p style={{ color: 'var(--text-muted)', fontSize: '0.8em', marginTop: 4 }}>
                Press and hold to show each cell's measured training time instead of its score. You
                can still hover around matrix cells while holding the button. You can also hold the
                space bar to do the same thing without clicking.
              </p>
              {trainingTimeDisabled && (
                <p style={{ color: 'var(--warning-text)', fontSize: '0.8em', marginTop: 4 }}>
                  Training time can't be shown for a matrix this short — a Plotly rendering bug
                  leaves the cells blank instead of updating. Select more domains or a
                  less-restrictive metric to check training time.
                </p>
              )}
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
            left: state.sidebarCollapsed ? 4 : sidebarWidth,
            transform: state.sidebarCollapsed ? 'none' : 'translateX(-50%)',
            background: 'var(--bg)',
            cursor: 'pointer',
          }}
        >
          {state.sidebarCollapsed ? '▶' : '◀'}
        </button>
        <button
          type="button"
          title="Toggle dark mode"
          onClick={toggleTheme}
          style={{
            position: 'absolute',
            top: 16,
            left: (state.sidebarCollapsed ? 4 : sidebarWidth) + 32,
            background: 'var(--bg)',
            cursor: 'pointer',
          }}
        >
          {theme === 'dark' ? '☀' : '🌙'}
        </button>
        {state.sidebarCollapsed && <SidebarToggleHint />}
      </div>
      <main
        style={{
          flex: 1,
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '100vh',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            flexShrink: 0,
            padding: '16px 16px 0',
            overflow: 'auto',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <h1 style={{ margin: 0 }}>Timeseries Forecasting Benchmark</h1>
            <InstructionsModal />
          </div>
          <div style={{ position: 'relative' }}>
            <MatrixGraph
              key={matrixKey}
              rows={filteredRows}
              metric={state.metric}
              selection={state.selection}
              showGlobal={state.showGlobal}
              showExogenous={state.showExogenous}
              showTrainingTime={state.trainingTimeOverlay}
              onCellClick={handleCellClick}
              onWidthChange={setMatrixWidth}
            />
            {!selectedCase && <SelectionHint />}
          </div>
          <h1>Getting started</h1>
        </div>
        <div
          style={{
            flex: 1,
            minHeight: 0,
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            padding: '0 16px 16px',
          }}
        >
          <AppInstructions maxWidth={textMaxWidth} />
          <div style={{ height: TITLE_BAR_HEIGHT }} />
        </div>
      </main>
      <Suspense fallback={null}>
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
      </Suspense>
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
            <Suspense fallback={<p>Loading case details…</p>}>
              <CasePanel
                caseId={state.selection.case_id!}
                caseMeta={selectedCase}
                scores={scores}
                showGlobal={state.showGlobal}
                selectedMethods={state.selection.methods}
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
            </Suspense>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
