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
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
      {options.map((family) => (
        <label key={family} title={FAMILY_FULL_NAMES[family] ?? family}>
          <input type="checkbox" checked={selected.has(family)} onChange={() => toggle(family)} />{' '}
          {formatSnakeCaseLabel(family)}
        </label>
      ))}
    </div>
  );
}
