/**
 * Unit tests for fuzzy.js — the design system's pure fuzzy scorer (shared by
 * the command palette and the channel-catalog matcher). Pins the load-bearing
 * contract that a naive subsequence matcher fails:
 *
 *   - the query is tokenized on whitespace and EVERY token must independently
 *     subsequence-match the candidate (all-tokens-AND), so "ctl lim" matches
 *     "control_system.limits_checking" via two separate tokens
 *   - a token that cannot match anywhere yields null (no match)
 *   - boundary hits (. _ - / camelCase / index 0) and consecutive runs score
 *     higher; scores sum across tokens
 *   - matched ranges merge into sorted, non-overlapping half-open [start, end)
 *     spans, and an empty query matches everything with score 0
 *
 * Pure module, no DOM:
 *   npx vitest run tests/interfaces/design_system/fuzzy.test.mjs
 */

import { describe, it, expect } from 'vitest';

import { fuzzyMatch } from '/design-system/js/fuzzy.js';

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
    const hit = fuzzyMatch('ctl lim', 'control_system.limits_checking');
    expect(hit).not.toBeNull();
    // Devil's-advocate case: the whole point is that this non-null hit exists.
    if (hit === null) throw new Error('expected a match');  // narrows for checkJs

    // approval.limits_checking has no "t" after its only "c", so token "ctl"
    // fails -> null. Non-null wins.
    const sibling = fuzzyMatch('ctl lim', 'approval.limits_checking');
    expect(sibling).toBeNull();

    // Incidental candidate that cannot serve either token: no match.
    const incidental = fuzzyMatch('ctl lim', 'approval.tools.archiver_read');
    expect(incidental).toBeNull();

    // And an explicit "clearly outscores" against a candidate that DOES match
    // both tokens but only incidentally (scattered, no boundaries).
    const scattered = fuzzyMatch('ctl lim', 'marching_octopus_tulips_limped');
    if (scattered !== null) {
      expect(hit.score).toBeGreaterThan(scattered.score);
    }
  });

  it('ALL-TOKENS-REQUIRED: one unmatchable token makes the whole query fail', () => {
    expect(fuzzyMatch('limits zzzz', 'control_system.limits_checking')).toBeNull();
    // The matchable token alone still matches, proving it is the zzzz that fails.
    expect(fuzzyMatch('limits', 'control_system.limits_checking')).not.toBeNull();
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
    const flagship = fuzzyMatch('ctl lim', 'control_system.limits_checking');
    expect(flagship).not.toBeNull();
    if (flagship) {
      // Every matched span slice is a substring of the candidate; concatenation
      // yields the highlighted characters c,t,l (from control) and i,m (limits;
      // its l is the one already covered by the first token).
      const covered = sliceBySpans('control_system.limits_checking', flagship.spans);
      expect(covered).toBe('c|t|l|im');
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
    const lowerQ = fuzzyMatch('ctl', 'control_system.limits_checking');
    const upperQ = fuzzyMatch('CTL', 'control_system.limits_checking');
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
