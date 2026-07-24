/**
 * Unit tests for fuzzy.js — the pure command-palette scorer. Pins the
 * load-bearing contract that a naive subsequence matcher fails:
 *
 *   - the query is tokenized on whitespace and EVERY token must independently
 *     subsequence-match the candidate (all-tokens-AND), so "wrt vrf" matches
 *     "control_system.write_verification" via two separate tokens
 *   - a token that cannot match anywhere yields null (no match)
 *   - boundary hits (. _ - / camelCase / index 0) and consecutive runs score
 *     higher; scores sum across tokens
 *   - matched ranges merge into sorted, non-overlapping half-open [start, end)
 *     spans, and an empty query matches everything with score 0
 *
 * Pure module, no DOM:
 *   npx vitest run tests/interfaces/web_terminal/fuzzy.test.mjs
 */

import { describe, it, expect } from 'vitest';

import { fuzzyMatch } from '../../../src/osprey/interfaces/web_terminal/static/js/fuzzy.js';

/** Reconstruct the matched substring set by slicing the candidate by spans.
 * @param {string} candidate
 * @param {Array<[number, number]>} spans
 * @returns {string}
 */
function sliceBySpans(candidate, spans) {
  return spans.map(([s, e]) => candidate.slice(s, e)).join('|');
}

describe('fuzzyMatch', () => {
  it('FLAGSHIP: two space-separated tokens each subsequence-match the candidate', () => {
    const hit = fuzzyMatch('wrt vrf', 'control_system.write_verification');
    expect(hit).not.toBeNull();
    // Devil's-advocate case: the whole point is that this non-null hit exists.
    if (hit === null) throw new Error('expected a match');  // narrows for checkJs

    // read_verification has no "w", so token "wrt" fails -> null. Non-null wins.
    const read = fuzzyMatch('wrt vrf', 'control_system.read_verification');
    expect(read).toBeNull();

    // Incidental candidate with no "w" anywhere: far lower / no match.
    const incidental = fuzzyMatch('wrt vrf', 'approval.tools.archiver_read');
    expect(incidental).toBeNull();

    // And an explicit "clearly outscores" against a candidate that DOES match
    // both tokens but only incidentally (scattered, no boundaries).
    const scattered = fuzzyMatch('wrt vrf', 'wandering_river_of_verbose_fragments');
    if (scattered !== null) {
      expect(hit.score).toBeGreaterThan(scattered.score);
    }
  });

  it('ALL-TOKENS-REQUIRED: one unmatchable token makes the whole query fail', () => {
    expect(fuzzyMatch('write zzzz', 'control_system.write_verification')).toBeNull();
    // The matchable token alone still matches, proving it is the zzzz that fails.
    expect(fuzzyMatch('write', 'control_system.write_verification')).not.toBeNull();
  });

  it('BOUNDARY BONUS: a boundary match outscores the same letters mid-word', () => {
    const boundary = fuzzyMatch('ver', 'x.verify'); // v after "." boundary
    const midWord = fuzzyMatch('ver', 'observer'); // v mid-word, still contiguous
    expect(boundary).not.toBeNull();
    expect(midWord).not.toBeNull();
    if (boundary && midWord) {
      expect(boundary.score).toBeGreaterThan(midWord.score);
    }
  });

  it('CONSECUTIVE-RUN BONUS: a contiguous match outscores a scattered one', () => {
    const contiguous = fuzzyMatch('abc', 'abcxx'); // a,b,c adjacent
    const scattered = fuzzyMatch('abc', 'axbxc'); // same letters, gaps
    expect(contiguous).not.toBeNull();
    expect(scattered).not.toBeNull();
    if (contiguous && scattered) {
      expect(contiguous.score).toBeGreaterThan(scattered.score);
    }
  });

  it('SPAN MERGING: spans are sorted, non-overlapping, half-open, and merge adjacencies', () => {
    const candidate = 'abcd';
    const hit = fuzzyMatch('ab cd', candidate); // two tokens whose hits abut
    expect(hit).not.toBeNull();
    if (hit) {
      // Sorted, non-overlapping, well-formed [start, end).
      for (let i = 0; i < hit.spans.length; i++) {
        const [s, e] = hit.spans[i];
        expect(e).toBeGreaterThan(s);
        if (i > 0) {
          expect(s).toBeGreaterThan(hit.spans[i - 1][1]);
        }
      }
      // The abutting token hits merge into a single [0,4) span covering "abcd".
      expect(hit.spans).toEqual([[0, 4]]);
      expect(sliceBySpans(candidate, hit.spans)).toBe('abcd');
    }

    // The flagship spans actually cover the matched characters.
    const flagship = fuzzyMatch('wrt vrf', 'control_system.write_verification');
    expect(flagship).not.toBeNull();
    if (flagship) {
      // Every matched span slice is a substring of the candidate; concatenation
      // yields the highlighted characters w,r,t (from write) and v,r,f (verif).
      const covered = sliceBySpans('control_system.write_verification', flagship.spans);
      expect(covered).toBe('wr|t|v|r|f');
      // Sorted + non-overlapping invariant holds here too.
      for (let i = 1; i < flagship.spans.length; i++) {
        expect(flagship.spans[i][0]).toBeGreaterThan(flagship.spans[i - 1][1]);
      }
    }
  });

  it('EMPTY QUERY: empty or whitespace-only matches everything with score 0', () => {
    expect(fuzzyMatch('', 'anything')).toEqual({ score: 0, spans: [] });
    expect(fuzzyMatch('   ', 'anything')).toEqual({ score: 0, spans: [] });
  });

  it('CASE INSENSITIVITY: query and candidate casing do not affect the match', () => {
    const lowerQ = fuzzyMatch('wrt', 'control_system.write_verification');
    const upperQ = fuzzyMatch('WRT', 'control_system.write_verification');
    expect(lowerQ).not.toBeNull();
    expect(upperQ).not.toBeNull();
    if (lowerQ && upperQ) {
      expect(upperQ.score).toBe(lowerQ.score);
      expect(upperQ.spans).toEqual(lowerQ.spans);
    }
    // Uppercase candidate, lowercase query still matches.
    expect(fuzzyMatch('write', 'CONTROL_SYSTEM.WRITE')).not.toBeNull();
  });
});
