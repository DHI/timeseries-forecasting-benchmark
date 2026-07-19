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
    <div>
      {options.map((domain) => (
        <label key={domain} style={{ display: 'block' }}>
          <input type="checkbox" checked={selected.has(domain)} onChange={() => toggle(domain)} />{' '}
          {domain.replace(/_/g, ' ')}
        </label>
      ))}
    </div>
  );
}
