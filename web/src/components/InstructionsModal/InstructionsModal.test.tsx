import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { InstructionsModal } from './InstructionsModal';

describe('InstructionsModal', () => {
  it('opens on click and closes via the X or the backdrop', async () => {
    const user = userEvent.setup();
    render(<InstructionsModal />);

    expect(screen.queryByText(/Selecting methods in the matrix/)).not.toBeInTheDocument();

    await user.click(screen.getByTitle('App instructions'));
    expect(screen.getByText(/Selecting methods in the matrix/)).toBeInTheDocument();

    await user.click(screen.getByTitle('Close'));
    expect(screen.queryByText(/Selecting methods in the matrix/)).not.toBeInTheDocument();

    await user.click(screen.getByTitle('App instructions'));
    expect(screen.getByText(/Selecting methods in the matrix/)).toBeInTheDocument();

    await user.click(screen.getByTestId('instructions-backdrop'));
    expect(screen.queryByText(/Selecting methods in the matrix/)).not.toBeInTheDocument();
  });
});
