/**
 * Unit tests for applying a preset ("Layout").
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-presets.test.mjs
 *
 * applyPreset orchestrates nothing locally: it sends the preset NAME to the
 * arrange endpoint, the server resolves and filters the members (pinned by
 * test_panel_arrange_route.py), and the panel_arrange SSE echo applies the
 * result on every client (panel-placement.js). That is what makes a human
 * "Layouts" click and an agent arrange_workspace(preset=...) call one
 * operation. What is pinned here is that request; the applied DOM behavior is
 * covered by panel-manager.test.mjs and the Playwright suite.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { applyPreset } from '../../../src/osprey/interfaces/web_terminal/static/js/panel-presets.js';

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
