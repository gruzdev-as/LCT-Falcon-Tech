import { MAX_TOP_K, MIN_TOP_K } from "../lib/constants";

interface Props {
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export function TopKControl({ value, onChange, disabled }: Props) {
  return (
    <label className="flex items-center gap-3 text-sm">
      <span className="text-ink-300">Кандидатов</span>
      <input
        type="range"
        min={MIN_TOP_K}
        max={MAX_TOP_K}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1 w-40 cursor-pointer appearance-none rounded-full bg-surface-700 accent-brand-500 disabled:cursor-not-allowed disabled:opacity-50"
      />
      <span className="w-8 text-right font-medium tabular-nums text-ink-100">{value}</span>
    </label>
  );
}
