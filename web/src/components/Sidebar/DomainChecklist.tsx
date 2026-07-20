import { DOMAIN_KEY_DICT } from '../../data/constants';
import { formatSnakeCaseLabel } from '../../data/selectors';

type Props = {
  options: string[];
  value: string[];
  onChange: (domains: string[]) => void;
};

export function DomainChecklist({ options, value, onChange }: Props) {
  const selected = new Set(value);

  function toggle(domain: string) {
    const next = new Set(selected);
    if (next.has(domain)) next.delete(domain);
    else next.add(domain);
    onChange(options.filter((d) => next.has(d)));
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
      {options.map((domain) => (
        <label key={domain} title={formatSnakeCaseLabel(domain)}>
          <input type="checkbox" checked={selected.has(domain)} onChange={() => toggle(domain)} />{' '}
          {DOMAIN_KEY_DICT[domain] ?? domain}
        </label>
      ))}
    </div>
  );
}
