import { DOMAIN_KEY_DICT } from '../../data/constants';
import { formatSnakeCaseLabel } from '../../data/selectors';

type Props = {
  options: string[];
  value: string[];
  onChange: (domains: string[]) => void;
  disabledOptions?: ReadonlySet<string>;
};

export function DomainChecklist({ options, value, onChange, disabledOptions }: Props) {
  const selected = new Set(value);

  function toggle(domain: string) {
    const next = new Set(selected);
    if (next.has(domain)) next.delete(domain);
    else next.add(domain);
    onChange(options.filter((d) => next.has(d)));
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
      {options.map((domain) => {
        const disabled = disabledOptions?.has(domain) ?? false;
        return (
          <label
            key={domain}
            title={formatSnakeCaseLabel(domain)}
            style={{ opacity: disabled ? 0.4 : 1 }}
          >
            <input
              type="checkbox"
              checked={!disabled && selected.has(domain)}
              disabled={disabled}
              onChange={() => toggle(domain)}
            />{' '}
            {DOMAIN_KEY_DICT[domain] ?? domain}
          </label>
        );
      })}
    </div>
  );
}
