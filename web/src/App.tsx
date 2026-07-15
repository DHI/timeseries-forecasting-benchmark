import { useJsonData } from './data/loadData';
import type { ScoreRow } from './types/data';

function App() {
  const fixture = useJsonData<ScoreRow[]>('data/fixture.json');

  return (
    <main style={{ padding: 16 }}>
      <h1>Timeseries Forecasting Benchmark</h1>
      <p>Scaffolding placeholder — verifies the static data-fetch pipeline works end to end.</p>
      {fixture.status === 'loading' && <p>Loading fixture data…</p>}
      {fixture.status === 'error' && <p role="alert">Failed to load: {fixture.error.message}</p>}
      {fixture.status === 'ready' && (
        <table>
          <thead>
            <tr>
              <th>Case</th>
              <th>Method</th>
              <th>Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {fixture.data.map((row) => (
              <tr key={`${row.case_id}-${row.method}-${row.metric}`}>
                <td>{row.case_id}</td>
                <td>{row.method}</td>
                <td>{row.metric}</td>
                <td>{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}

export default App;
