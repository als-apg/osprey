/**
 * Unit tests for the pure preset-diff core.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-presets.test.mjs
 *
 * computePresetDiff resolves a preset into an EXCLUSIVE show/hide diff against
 * the live visible set; applyPreset orchestrates show-before-hide + local focus.
 * The DOM/popover behavior is covered end-to-end by the Playwright suite
 * (test_panels_browser.py); here we pin the pure contract.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe } from 'vitest';

import {
  computePresetDiff,
  applyPreset,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-presets.js';

describe('computePresetDiff', () => {
  test('exclusive diff: show missing members, hide visible non-members', () => {
    const known = new Set(['a', 'b', 'c', 'd']);
    const visible = new Set(['b', 'c', 'd']);
    const diff = computePresetDiff(['a', 'b'], visible, known);
    expect(diff.toShow).toEqual(['a']); // b already visible
    expect(diff.toHide.sort()).toEqual(['c', 'd']); // visible non-members
    expect(diff.focus).toBe('a');
  });

  test('members are filtered to the known set (typo/disabled ids dropped)', () => {
    const known = new Set(['a', 'b']);
    const diff = computePresetDiff(['a', 'ghost'], new Set(), known);
    expect(diff.toShow).toEqual(['a']); // ghost is not known → skipped
    expect(diff.toHide).toEqual([]);
    expect(diff.focus).toBe('a');
  });

  test('focus is the first FILTERED member (leading unknowns skipped)', () => {
    const known = new Set(['a', 'b']);
    const diff = computePresetDiff(['ghost', 'a', 'b'], new Set(['a', 'b']), known);
    expect(diff.focus).toBe('a');
  });

  test('empty guard: all members unknown → focus null, no ops', () => {
    const known = new Set(['a', 'b']);
    const diff = computePresetDiff(['x', 'y'], new Set(['a']), known);
    expect(diff).toEqual({ toShow: [], toHide: [], focus: null });
  });
});

describe('applyPreset', () => {
  /**
   * Build a recording harness over applyPreset's injected deps.
   * @param {string[]} members
   * @param {{visible: string[], known: string[], healthy?: string[]}} state
   * @returns {[string, string][]}
   */
  function harness(members, { visible, known, healthy }) {
    // Default: every known panel is healthy (the common all-reachable case).
    const healthySet = new Set(healthy ?? known);
    /** @type {[string, string][]} */
    const calls = [];
    applyPreset(members, {
      getVisible: () => new Set(visible),
      getKnown: () => new Set(known),
      isHealthy: (id) => healthySet.has(id),
      setVisibility: (id, v) => calls.push([v ? 'show' : 'hide', id]),
      focus: (id) => calls.push(['focus', id]),
    });
    return calls;
  }

  test('shows every member first, then focuses, then hides non-members', () => {
    const calls = harness(['a', 'b'], { visible: ['c'], known: ['a', 'b', 'c'] });
    // a,b shown (both missing), focus a, then c hidden — show-before-hide ordering.
    expect(calls).toEqual([
      ['show', 'a'],
      ['show', 'b'],
      ['focus', 'a'],
      ['hide', 'c'],
    ]);
  });

  test('focus fires before any hide (no transient all-hidden flash)', () => {
    const calls = harness(['a'], { visible: ['a', 'b'], known: ['a', 'b'] });
    // a already visible (no show); focus a happens BEFORE hiding b.
    expect(calls).toEqual([
      ['focus', 'a'],
      ['hide', 'b'],
    ]);
  });

  test('no-op when the diff yields a null focus (all members unknown)', () => {
    const calls = harness(['ghost'], { visible: ['a'], known: ['a', 'b'] });
    expect(calls).toEqual([]);
  });

  test('focuses the first HEALTHY member when the primary is unhealthy', () => {
    // a is the primary but offline → focus falls through to b (first healthy member).
    const calls = harness(['a', 'b'], { visible: ['c'], known: ['a', 'b', 'c'], healthy: ['b', 'c'] });
    expect(calls).toEqual([
      ['show', 'a'],
      ['show', 'b'],
      ['focus', 'b'],
      ['hide', 'c'],
    ]);
  });

  test('no focus call when no member is healthy (SSE fallback lands the view)', () => {
    // both members already visible (no show), neither healthy → no local focus.
    const calls = harness(['a', 'b'], { visible: ['a', 'b'], known: ['a', 'b'], healthy: [] });
    expect(calls).toEqual([]);
  });
});
