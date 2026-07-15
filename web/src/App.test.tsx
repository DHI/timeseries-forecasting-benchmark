import { render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

describe('App', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => [
          {
            case_id: 'demand_ltown',
            method: 'Persistence',
            metric: 'MASE',
            value: 1.42,
          },
        ],
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('fetches and renders fixture data', async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText('demand_ltown')).toBeInTheDocument());
    expect(screen.getByText('Persistence')).toBeInTheDocument();
  });
});
