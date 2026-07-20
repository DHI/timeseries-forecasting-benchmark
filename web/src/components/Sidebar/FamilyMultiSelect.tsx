import { formatFamilyLabel } from '../../data/selectors';

type Props = {
  options: string[];
  value: string[];
  onChange: (families: string[]) => void;
};

export function FamilyMultiSelect({ options, value, onChange }: Props) {
  const selected = new Set(value);

  function toggle(family: string) {
    const next = new Set(selected);
    if (next.has(family)) next.delete(family);
    else next.add(family);
    onChange(options.filter((f) => next.has(f)));
  }

  return (
    <div>
      {options.map((family) => (
        <label key={family} style={{ display: 'block' }}>
          <input type="checkbox" checked={selected.has(family)} onChange={() => toggle(family)} />{' '}
          {formatFamilyLabel(family)}
        </label>
      ))}
    </div>
  );
}
