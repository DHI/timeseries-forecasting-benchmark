import type { MouseEvent as ReactMouseEvent } from 'react';

type Props = {
  onMouseDown: (event: ReactMouseEvent) => void;
  disabled?: boolean;
  title?: string;
};

export function HoldTrainingTimeButton({ onMouseDown, disabled, title }: Props) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      onMouseDown={onMouseDown}
      style={{ display: 'block', opacity: disabled ? 0.5 : 1, cursor: disabled ? 'not-allowed' : 'pointer' }}
    >
      Show training time
    </button>
  );
}
