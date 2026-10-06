import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { InstructionsModal } from './InstructionsModal';

describe('InstructionsModal', () => {
  it('opens on click and closes via the X or the backdrop', async () => {
    const user = userEvent.setup();
    render(<InstructionsModal />);

    expect(screen.queryByText(/Curation enforces minimum lengths/)).not.toBeInTheDocument();

    await user.click(screen.getByTitle('Benchmark Design'));
    expect(screen.getByText(/Curation enforces minimum lengths/)).toBeInTheDocument();

    await user.click(screen.getByTitle('Close'));
    expect(screen.queryByText(/Curation enforces minimum lengths/)).not.toBeInTheDocument();

    await user.click(screen.getByTitle('Benchmark Design'));
    expect(screen.getByText(/Curation enforces minimum lengths/)).toBeInTheDocument();

    await user.click(screen.getByTestId('instructions-backdrop'));
    expect(screen.queryByText(/Curation enforces minimum lengths/)).not.toBeInTheDocument();
  });
});
