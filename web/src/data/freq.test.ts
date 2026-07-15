import { describe, expect, it } from 'vitest';
import { freqToMs } from './freq';

describe('freqToMs', () => {
  it('parses hours', () => {
    expect(freqToMs('1h')).toBe(3_600_000);
    expect(freqToMs('24h')).toBe(24 * 3_600_000);
  });

  it('parses minutes', () => {
    expect(freqToMs('15min')).toBe(15 * 60_000);
  });

  it('parses days', () => {
    expect(freqToMs('1D')).toBe(86_400_000);
  });

  it('throws on an unsupported unit', () => {
    expect(() => freqToMs('1W')).toThrow();
  });
});
