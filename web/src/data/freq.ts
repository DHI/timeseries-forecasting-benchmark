const UNIT_MS: Record<string, number> = {
  min: 60_000,
  h: 3_600_000,
  D: 86_400_000,
};

/** A case's `freq` (a pandas-style offset alias, e.g. "1h", "15min", "1D") as a
 * millisecond step — just the handful of units this benchmark's cases actually use, not
 * a general pandas-offset parser. Used to reconstruct a snapshot origin's window
 * timestamps client-side, since the export step drops them to save space. */
export function freqToMs(freq: string): number {
  const match = /^(\d*)(min|h|D)$/.exec(freq);
  if (!match) throw new Error(`Unsupported freq "${freq}"`);
  const [, countStr, unit] = match;
  const count = countStr ? Number(countStr) : 1;
  return count * UNIT_MS[unit!]!;
}
