// @ts-check
/**
 * Unit tests for channel-matcher.js — the pure ranking core behind the
 * plan-form channel typeahead. Pins the load-bearing pool rule:
 *
 *   - case-insensitive substring hits are the candidate pool when there are
 *     at least MIN_POOL of them; below that the pool widens to every catalog
 *     entry fuzzyMatch accepts (subsequence matching)
 *   - both modes rank by fuzzyMatch score (a substring is a subsequence, so
 *     substring hits always fuzzy-match), tie-broken by value ascending
 *   - `total` is the honest pool size; `results` is capped at RENDER_CAP
 *   - empty/whitespace query: pool = whole catalog, first RENDER_CAP entries
 *     in catalog order
 *
 * Pure module, no DOM:
 *   npx vitest run tests/interfaces/design_system/channel-matcher.test.mjs
 */

import { describe, it, expect } from 'vitest';

import {
  matchChannels,
  MIN_POOL,
  RENDER_CAP,
} from '/design-system/js/channel-matcher.js';
import { fuzzyMatch } from '/design-system/js/fuzzy.js';

/**
 * Build `n` distinct channel names containing `stem`, e.g. "SR01:HCM:SP".
 * @param {number} n
 * @param {string} stem
 * @returns {string[]}
 */
function namesContaining(n, stem) {
  return Array.from({ length: n }, (_, i) => `SR${String(i).padStart(2, '0')}:${stem}:SP`);
}

describe('matchChannels', () => {
  it('POOL CONTAINMENT: substring mode — every result contains the query case-insensitively', () => {
    // MIN_POOL substring hits trigger substring mode; the fuzzy-only entry
    // (matches "HCM" only as a subsequence) must be excluded from the pool.
    const fuzzyOnly = 'SR:H7:CTRL:M1';
    expect(fuzzyOnly.toLowerCase().includes('hcm')).toBe(false);
    expect(fuzzyMatch('HCM', fuzzyOnly)).not.toBeNull();

    const catalog = [...namesContaining(MIN_POOL, 'HCM'), fuzzyOnly].sort();
    const { results, total } = matchChannels('hcm', catalog);

    expect(total).toBe(MIN_POOL);
    expect(results.length).toBe(MIN_POOL);
    for (const { value } of results) {
      expect(value.toLowerCase()).toContain('hcm');
    }
  });

  it('POOL CONTAINMENT: fuzzy mode — every result fuzzy-matches the query', () => {
    // Fewer than MIN_POOL substring hits: the pool is all fuzzyMatch matches,
    // and non-matching entries stay out.
    const catalog = [
      ...namesContaining(MIN_POOL - 1, 'HCM'),
      'SR:H7:CTRL:M1', // fuzzy-only match for "hcm"
      'SR:VBP:01:RB', // no h at all — must not appear
    ].sort();
    const { results, total } = matchChannels('hcm', catalog);

    expect(total).toBe(MIN_POOL);
    for (const { value } of results) {
      expect(fuzzyMatch('hcm', value)).not.toBeNull();
    }
    expect(results.map((r) => r.value)).not.toContain('SR:VBP:01:RB');
  });

  it('FLIP BOUNDARY: dropping below MIN_POOL substring hits widens the pool to a superset', () => {
    const substringHits = namesContaining(MIN_POOL, 'HCM');
    const fuzzyOnly = 'SR:H7:CTRL:M1';

    // Exactly MIN_POOL substring hits: substring mode, fuzzy-only excluded.
    const atBoundary = matchChannels('hcm', [...substringHits, fuzzyOnly].sort());
    expect(atBoundary.total).toBe(MIN_POOL);
    expect(atBoundary.results.map((r) => r.value)).not.toContain(fuzzyOnly);

    // Remove one hit (MIN_POOL - 1 remain): fuzzy mode. The pool must be a
    // strict superset of the remaining substring hits — all still present,
    // plus the fuzzy-only entry.
    const remaining = substringHits.slice(1);
    const below = matchChannels('hcm', [...remaining, fuzzyOnly].sort());
    const belowValues = below.results.map((r) => r.value);
    for (const hit of remaining) {
      expect(belowValues).toContain(hit);
    }
    expect(belowValues).toContain(fuzzyOnly);
    expect(below.total).toBe(remaining.length + 1);
  });

  it('SUBSEQUENCE FALL-THROUGH: "SRHCM" has zero substring hits but matches SR:HCM names', () => {
    const catalog = ['SR:HCM:01:SP', 'SR:HCM:02:SP', 'SR:VCM:01:SP'];
    for (const value of catalog) {
      expect(value.toLowerCase().includes('srhcm')).toBe(false);
    }

    const { results, total } = matchChannels('SRHCM', catalog);
    expect(total).toBe(2);
    expect(results.map((r) => r.value).sort()).toEqual(['SR:HCM:01:SP', 'SR:HCM:02:SP']);
  });

  it('RENDER CAP: results capped at RENDER_CAP, total stays honest, results ⊆ pool', () => {
    const poolSize = RENDER_CAP + 10;
    const catalog = [...namesContaining(poolSize, 'HCM'), 'SR:VBP:01:RB'].sort();
    const { results, total } = matchChannels('hcm', catalog);

    expect(results.length).toBe(RENDER_CAP);
    expect(total).toBe(poolSize);
    for (const { value } of results) {
      // Substring mode: every rendered entry is a member of the substring pool.
      expect(value.toLowerCase()).toContain('hcm');
    }
  });

  it('spans come from fuzzyMatch (half-open [start, end) pairs into the value)', () => {
    const catalog = namesContaining(MIN_POOL, 'HCM');
    const { results } = matchChannels('hcm', catalog);
    const first = results[0];
    if (first === undefined) throw new Error('expected at least one result');

    const expected = fuzzyMatch('hcm', first.value);
    if (expected === null) throw new Error('substring hit must fuzzy-match');
    expect(first.spans).toEqual(expected.spans);
  });

  it('ties rank by value ascending so ordering is deterministic', () => {
    // Identical shapes score identically; the tie-break must not depend on
    // catalog order.
    const catalog = ['SR02:HCM:SP', 'SR00:HCM:SP', 'SR01:HCM:SP'];
    const scores = catalog.map((v) => {
      const hit = fuzzyMatch('hcm', v);
      if (hit === null) throw new Error('expected a match');
      return hit.score;
    });
    expect(new Set(scores).size).toBe(1);

    // 3 substring hits < MIN_POOL: fuzzy mode, same ranking rule applies.
    const { results } = matchChannels('hcm', catalog);
    expect(results.map((r) => r.value)).toEqual(['SR00:HCM:SP', 'SR01:HCM:SP', 'SR02:HCM:SP']);
  });

  it('empty/whitespace query: whole catalog is the pool, first RENDER_CAP in catalog order', () => {
    const catalog = namesContaining(RENDER_CAP + 5, 'BPM');
    for (const query of ['', '   ']) {
      const { results, total } = matchChannels(query, catalog);
      expect(total).toBe(catalog.length);
      expect(results.length).toBe(RENDER_CAP);
      expect(results.map((r) => r.value)).toEqual(catalog.slice(0, RENDER_CAP));
      for (const { spans } of results) {
        expect(spans).toEqual([]);
      }
    }
  });
});
