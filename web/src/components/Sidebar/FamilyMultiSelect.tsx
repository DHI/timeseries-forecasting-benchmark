import { FAMILY_FULL_NAMES } from '../../data/constants';
import { formatSnakeCaseLabel } from '../../data/selectors';

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
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 1px' }}>
      {options.map((family) => (
        <label key={family} style={{ whiteSpace: 'nowrap' }}>
          <input type="checkbox" checked={selected.has(family)} onChange={() => toggle(family)} />{' '}
          {FAMILY_FULL_NAMES[family] ?? formatSnakeCaseLabel(family)}
        </label>
      ))}
    </div>
  );
}
