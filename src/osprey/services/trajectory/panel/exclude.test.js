// @ts-check
import { describe, expect, it } from 'vitest';

import { formatExclude, indicesOf, parseExclude, toggleName } from './exclude.js';

const FAMILY = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'];

describe('parseExclude', () => {
  it('reads an empty list as nothing excluded', () => {
    expect(parseExclude('', FAMILY)).toEqual({ ok: true, indices: [], names: [] });
    expect(parseExclude('  [ ] ', FAMILY)).toEqual({ ok: true, indices: [], names: [] });
  });

  it('reads space and comma separators, in any mix', () => {
    for (const text of ['2 4 7', '2,4,7', '2, 4 ,7', '2 ,4,  7']) {
      expect(parseExclude(text, FAMILY)).toEqual({ ok: true, indices: [2, 4, 7], names: ['B2', 'B4', 'B7'] });
    }
  });

  it('accepts optional brackets', () => {
    expect(parseExclude('[1 3]', FAMILY)).toEqual(parseExclude('1 3', FAMILY));
  });

  it('expands an inclusive range and merges duplicates in index order', () => {
    expect(parseExclude('[6, 3:5 4]', FAMILY)).toEqual({
      ok: true,
      indices: [3, 4, 5, 6],
      names: ['B3', 'B4', 'B5', 'B6'],
    });
    expect(parseExclude('2:2', FAMILY)).toMatchObject({ ok: true, indices: [2] });
  });

  it('refuses an index outside the family, naming it and the family', () => {
    const parsed = parseExclude('3 9', FAMILY);
    expect(parsed.ok).toBe(false);
    expect(parsed).toHaveProperty('error', 'BPM 9 is not in the BPM family (1-8)');
    expect(parseExclude('0', FAMILY)).toHaveProperty('error', 'BPM 0 is not in the BPM family (1-8)');
    expect(parseExclude('7:12', FAMILY, 'HCM')).toHaveProperty('error', 'HCM 12 is not in the HCM family (1-8)');
  });

  it('refuses a backwards range, an unknown token and unbalanced brackets', () => {
    expect(parseExclude('5:3', FAMILY)).toHaveProperty('error', 'range 5:3 runs backwards');
    expect(parseExclude('2 B3', FAMILY)).toHaveProperty('error', '"B3" is not an index or a range like 3:5');
    expect(parseExclude('-1', FAMILY).ok).toBe(false);
    expect(parseExclude('1.5', FAMILY).ok).toBe(false);
    expect(parseExclude('[1 2', FAMILY).ok).toBe(false);
  });
});

describe('formatExclude', () => {
  it('writes runs of three or more as ranges, the rest one by one', () => {
    expect(formatExclude([])).toBe('');
    expect(formatExclude([7, 1, 2, 4, 5, 6])).toBe('1 2 4:7');
    expect(formatExclude([1, 2, 5])).toBe('1 2 5');
  });

  it('round-trips through parseExclude', () => {
    for (const indices of [[1], [2, 3, 4, 8], [1, 3, 5, 6, 7]]) {
      expect(parseExclude(formatExclude(indices), FAMILY)).toMatchObject({ ok: true, indices });
    }
  });
});

describe('names and indices', () => {
  it('maps names to indices, dropping names outside the family', () => {
    expect(indicesOf(['B5', 'X', 'B2', 'B5'], FAMILY)).toEqual([2, 5]);
  });

  it('toggles one name in and out, in family order', () => {
    expect(toggleName(['B5'], 'B2', FAMILY)).toEqual(['B2', 'B5']);
    expect(toggleName(['B2', 'B5'], 'B2', FAMILY)).toEqual(['B5']);
  });
});
