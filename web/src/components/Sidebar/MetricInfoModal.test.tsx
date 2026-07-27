import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { MetricInfoModal } from './MetricInfoModal';

describe('MetricInfoModal', () => {
  it('opens on click and closes via the X or the backdrop', async () => {
    const user = userEvent.setup();
    render(<MetricInfoModal />);

    expect(screen.queryByText(/Nash-Sutcliffe Efficiency/)).not.toBeInTheDocument();

    await user.click(screen.getByTitle('Metric cheatsheet'));
    expect(screen.getByText(/Nash-Sutcliffe Efficiency/)).toBeInTheDocument();

    await user.click(screen.getByTitle('Close'));
    expect(screen.queryByText(/Nash-Sutcliffe Efficiency/)).not.toBeInTheDocument();

    await user.click(screen.getByTitle('Metric cheatsheet'));
    expect(screen.getByText(/Nash-Sutcliffe Efficiency/)).toBeInTheDocument();

    await user.click(screen.getByTestId('metric-cheatsheet-backdrop'));
    expect(screen.queryByText(/Nash-Sutcliffe Efficiency/)).not.toBeInTheDocument();
  });
});
