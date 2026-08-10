/**
 * Unit tests for the preset core.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-presets.test.mjs
 *
 * computePresetDiff resolves a preset into an EXCLUSIVE show/hide diff against
 * the live visible set. It is the executable statement of what "apply a layout"
 * means — exactly the members open, every non-member closed, unknown ids
 * dropped fail-safe — and the resolution /api/panel-arrange performs
 * server-side mirrors it, so these cases stay the reference for both.
 *
 * applyPreset itself no longer orchestrates anything locally: it sends the
 * preset NAME to the arrange endpoint and the panel_arrange SSE echo applies
 * the result on every client (panel-placement.js), which is what makes a human
 * "Layouts" click and an agent arrange_workspace(preset=...) call one
 * operation. What is pinned here is that request; the applied DOM behavior is
 * covered by panel-manager.test.mjs and the Playwright suite.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

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

describe('applyPreset — one arrange request, no local orchestration', () => {
  /** @type {{url: string, opts: any}[]} */
  let calls = [];

  beforeEach(() => {
    delete window.__OSPREY_PREFIX__;
    calls = [];
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url, /** @type {any} */ opts) => {
      calls.push({ url, opts });
      return { ok: true, status: 200, json: async () => ({ status: 'ok' }) };
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test('POSTs the preset NAME to /api/panel-arrange', () => {
    applyPreset('Machine setup');

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe('/api/panel-arrange');
    expect(calls[0].opts).toMatchObject({ method: 'POST' });
    // preset, never a resolved panel list: the server owns resolution, and it is
    // the preset path that sets prune_rail — a tiles request would silently lose
    // the exclusive ("and the rest close") half of the semantics.
    expect(JSON.parse(calls[0].opts.body)).toEqual({ preset: 'Machine setup' });
  });

  test('sends nothing else — no visibility or focus POST rides along', () => {
    applyPreset('Logbook review');

    expect(calls.map((c) => c.url)).toEqual(['/api/panel-arrange']);
  });

  test('the request is prefixed under a multi-user deployment', () => {
    window.__OSPREY_PREFIX__ = '/u/alice';

    applyPreset('Machine setup');

    expect(calls[0].url).toBe('/u/alice/api/panel-arrange');
  });

  test('a rejected request is swallowed — a failed layout click never throws', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new Error('offline');
    }));

    expect(() => applyPreset('Machine setup')).not.toThrow();
    // Let the rejected promise settle: an unhandled rejection would fail the run.
    await new Promise((r) => setTimeout(r, 0));
  });
});
