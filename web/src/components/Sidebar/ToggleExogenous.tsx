type Props = {
  checked: boolean;
  onChange: () => void;
};

export function ToggleExogenous({ checked, onChange }: Props) {
  return (
    <label style={{ display: 'block' }}>
      <input type="checkbox" checked={checked} onChange={onChange} /> Use exogenous inputs where
      available.
    </label>
  );
}
