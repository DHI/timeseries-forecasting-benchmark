import { useHoldButton } from '../../hooks/useHoldButton';

type Props = {
  onChange: (active: boolean) => void;
  disabled?: boolean;
  title?: string;
};

export function HoldTrainingTimeButton({ onChange, disabled, title }: Props) {
  const hold = useHoldButton(onChange);
  return (
    <button
      type="button"
      disabled={disabled}
      title={title}
      onMouseDown={hold.onMouseDown}
      style={{ display: 'block', opacity: disabled ? 0.5 : 1, cursor: disabled ? 'not-allowed' : 'pointer' }}
    >
      Show training time
    </button>
  );
}
