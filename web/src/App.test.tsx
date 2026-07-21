import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { clearJsonDataCache } from './data/loadData';

// Plotly.js depends on browser APIs jsdom doesn't fully implement (canvas, ResizeObserver);
// stub the Plot component here so these tests exercise the app's data flow and controls,
// not Plotly's own rendering (that's covered by manual browser verification instead).
vi.mock('./plotly', () => ({
  default: () => <div data-testid="plot-stub" />,
}));

const SCORES = [
  {
    case_id: 'demand_ltown',
    domain: 'water_distribution',
    case_exo_tier: 'none',
    method: 'Persistence',
    family: 'baseline',
    uses_exogenous: false,
    is_global: false,
    used_extended_history: false,
    horizon: '24h',
    metric: 'MASE',
    value: 1.42,
    seconds: 0.01,
    frac_finite_test: 1.0,
    status: 'ok',
    n_origins: 40,
  },
];
const METHODS = {
  Persistence: {
    family: 'baseline',
    description: '',
    uses_exogenous: false,
    library: '',
    compute_profile: '',
    limitations: '',
    code_snippet: '',
  },
};
const CASES = {
  demand_ltown: {
    domain: 'water_distribution',
    target: 'inlet_flow',
    target_unit: 'L/s',
    freq: '1h',
    horizon_primary: '24h',
    exogenous: [],
    exogenous_known_future: [],
    necessary_exogenous: [],
    exo_tier: 'none',
    domain_metrics: [],
    source: { name: 'Test', url: null },
    notes: '',
  },
};

function mockFetchByPath(byPath: Record<string, unknown>) {
  return vi.fn((url: string) => {
    const match = Object.keys(byPath).find((path) => url.endsWith(path));
    if (!match) return Promise.reject(new Error(`unexpected fetch: ${url}`));
    return Promise.resolve({ ok: true, json: async () => byPath[match] } as Response);
  });
}

describe('App', () => {
  beforeEach(() => {
    clearJsonDataCache();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows a loading state before data resolves', () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise(() => {})),
    );
    render(<App />);
    expect(screen.getByText(/Loading benchmark results/)).toBeInTheDocument();
  });

  it('shows an error banner when a fetch fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve({ ok: false, status: 500, statusText: 'error' } as Response)),
    );
    render(<App />);
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
  });

  it('renders the sidebar and matrix once data loads', async () => {
    const user = userEvent.setup();
    vi.stubGlobal(
      'fetch',
      mockFetchByPath({
        'data/scores.json': SCORES,
        'data/methods.json': METHODS,
        'data/cases.json': CASES,
      }),
    );
    render(<App />);
    await waitFor(() => expect(screen.getByTestId('plot-stub')).toBeInTheDocument());
    expect(screen.getByText('Timeseries Forecasting Benchmark')).toBeInTheDocument();

    // The control panel starts collapsed; open it to reach the sidebar controls.
    await user.click(screen.getByTitle('Collapse/expand controls'));

    expect(screen.getByRole('option', { name: 'MASE' })).toBeInTheDocument();
    expect(screen.getByLabelText('WaD')).toBeInTheDocument();
    expect(screen.getByLabelText('Baseline')).toBeInTheDocument();
  });

  it('greys out domain checkboxes ineligible for a domain-restricted metric', async () => {
    const user = userEvent.setup();
    const scores = [
      ...SCORES,
      {
        ...SCORES[0],
        case_id: 'flow_case',
        domain: 'hydrology',
        metric: 'NSE',
        value: 0.8,
      },
    ];
    const cases = {
      ...CASES,
      flow_case: { ...CASES.demand_ltown, domain: 'hydrology', domain_metrics: ['NSE', 'KGE'] },
    };
    vi.stubGlobal(
      'fetch',
      mockFetchByPath({
        'data/scores.json': scores,
        'data/methods.json': METHODS,
        'data/cases.json': cases,
      }),
    );
    render(<App />);
    await waitFor(() => expect(screen.getByTestId('plot-stub')).toBeInTheDocument());
    await user.click(screen.getByTitle('Collapse/expand controls'));

    const waterDistributionCheckbox = screen.getByLabelText('WaD') as HTMLInputElement;
    const hydrologyCheckbox = screen.getByLabelText('Hyd') as HTMLInputElement;
    expect(waterDistributionCheckbox.disabled).toBe(false);
    expect(hydrologyCheckbox.disabled).toBe(false);

    await user.selectOptions(screen.getByRole('combobox'), 'NSE');

    expect(waterDistributionCheckbox.disabled).toBe(true);
    expect(waterDistributionCheckbox.checked).toBe(false);
    expect(hydrologyCheckbox.disabled).toBe(false);
    expect(hydrologyCheckbox.checked).toBe(true);
  });
});

describe('mockFetchByPath', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetchByPath({ 'x.json': { a: 1 } }));
  });

  afterEach(() => vi.unstubAllGlobals());

  it('rejects on an unregistered path', async () => {
    await expect(fetch('/base/other.json')).rejects.toThrow('unexpected fetch');
  });
});
