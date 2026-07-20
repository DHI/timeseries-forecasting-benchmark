import { useHoldButton } from '../../hooks/useHoldButton';

type Props = {
  onChange: (active: boolean) => void;
};

export function HoldTrainingTimeButton({ onChange }: Props) {
  const hold = useHoldButton(onChange);
  return (
    <button type="button" onMouseDown={hold.onMouseDown} style={{ display: 'block' }}>
      Hold to show training time
    </button>
  );
}
