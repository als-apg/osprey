/**
 * Unit tests for applying a preset ("Layout").
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-presets.test.mjs
 *
 * applyPreset orchestrates nothing locally: it sends the preset NAME to the
 * open-arrangement op route — the route an agent's arrange_workspace uses — the
 * server resolves the members and layout, relays one panel_op frame to the
 * current dock client, and that client's op runner (panel-placement.js)
 * applies it. That is what makes a human "Layouts" click and an agent
 * arrange_workspace(preset=...) call one operation. What is pinned here is that
 * request; the applied layout is covered by panel-placement.test.mjs.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { applyPreset } from '../../../src/osprey/interfaces/web_terminal/static/js/panel-presets.js';

describe('applyPreset — one open-arrangement op, no local orchestration', () => {
  /** @type {{url: string, opts: any}[]} */
  let calls = [];

  beforeEach(() => {
    delete window.__OSPREY_PREFIX__;
    calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url, /** @type {any} */ opts) => {
        calls.push({ url, opts });
        return { ok: true, status: 200, json: async () => ({ status: 'ok' }) };
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test('POSTs the preset NAME to /api/panel-op/open-arrangement with a request id', () => {
    applyPreset('Machine setup');

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe('/api/panel-op/open-arrangement');
    expect(calls[0].opts).toMatchObject({ method: 'POST' });
    // The name, never a resolved panel list or layout: the server owns
    // resolution, and the preset path is what prunes the rail to its members.
    const body = JSON.parse(calls[0].opts.body);
    expect(Object.keys(body).sort()).toEqual(['name', 'request_id']);
    expect(body.name).toBe('Machine setup');
    expect(typeof body.request_id).toBe('string');
    expect(body.request_id.length).toBeGreaterThan(0);
  });

  test('each click is its own request', () => {
    applyPreset('Machine setup');
    applyPreset('Machine setup');

    const ids = calls.map((c) => JSON.parse(c.opts.body).request_id);
    expect(new Set(ids).size).toBe(2);
  });

  test('the retired /api/panel-arrange route is never called', () => {
    applyPreset('Machine setup');

    expect(calls.some((c) => c.url.includes('/api/panel-arrange'))).toBe(false);
  });

  test('the request is prefixed under a multi-user deployment', () => {
    window.__OSPREY_PREFIX__ = '/u/alice';

    applyPreset('Machine setup');

    expect(calls[0].url).toBe('/u/alice/api/panel-op/open-arrangement');
  });

  test('a refused op is swallowed — a refused layout click never throws', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 409,
        json: async () => ({ status: 'refused', reason: 'no_dock_client' }),
      }))
    );

    expect(() => applyPreset('Machine setup')).not.toThrow();
    await new Promise((r) => setTimeout(r, 0));
  });

  test('a rejected request is swallowed — a failed layout click never throws', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('offline');
      })
    );

    expect(() => applyPreset('Machine setup')).not.toThrow();
    // Let the rejected promise settle: an unhandled rejection would fail the run.
    await new Promise((r) => setTimeout(r, 0));
  });
});
