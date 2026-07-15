type Props = {
  checked: boolean;
  onChange: () => void;
};

export function ToggleGlobal({ checked, onChange }: Props) {
  return (
    <label style={{ display: 'block' }}>
      <input type="checkbox" checked={checked} onChange={onChange} /> Score on full training
      history.
    </label>
  );
}
