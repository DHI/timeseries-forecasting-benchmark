type Props = {
  options: string[];
  value: string;
  onChange: (metric: string) => void;
};

export function MetricDropdown({ options, value, onChange }: Props) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((metric) => (
        <option key={metric} value={metric}>
          {metric}
        </option>
      ))}
    </select>
  );
}
