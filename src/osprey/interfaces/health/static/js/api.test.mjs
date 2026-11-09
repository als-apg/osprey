// @ts-check
/**
 * The panel's HTTP surface (api.js):
 *   npx vitest run src/osprey/interfaces/health/static/js/api.test.mjs
 */

import { test, expect, describe, vi, afterEach } from 'vitest';

import { call, makeApi } from './api.js';

/** @param {number} status @param {any} body */
function reply(status, body) {
  return Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('call', () => {
  test('GET without a body, POST JSON with one', async () => {
    const f = vi.fn(() => reply(200, { ok: 1 }));
    vi.stubGlobal('fetch', f);
    await call('/checks');
    await call('/reprobe', { checks: ['a'] });
    expect(f.mock.calls[0]).toEqual(['/checks', {}]);
    const [, init] = /** @type {any} */ (f.mock.calls[1]);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ checks: ['a'] });
  });

  test('409 surfaces the server detail; other errors their status', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => reply(409, { detail: 'refresh suppressed', suppressed: true }))
    );
    await expect(call('/reprobe', {})).rejects.toThrow('refresh suppressed');
    vi.stubGlobal(
      'fetch',
      vi.fn(() => reply(500, {}))
    );
    await expect(call('/checks')).rejects.toThrow('HTTP 500');
  });
});

describe('makeApi', () => {
  test('routes each verb to its path and body', async () => {
    const f = vi.fn(() => reply(200, {}));
    vi.stubGlobal('fetch', f);
    const api = makeApi();
    await api.checks();
    await api.mcp();
    await api.mcpProbe('controls');
    await api.mcpProbe();
    await api.reprobe({ component: 'hooks' });
    await api.ping('als-apg');
    const calls = f.mock.calls.map((/** @type {any} */ c) => [
      c[0],
      c[1].body ? JSON.parse(c[1].body) : null,
    ]);
    expect(calls).toEqual([
      ['/checks', null],
      ['/mcp', null],
      ['/mcp/probe', { server: 'controls' }],
      ['/mcp/probe', {}],
      ['/reprobe', { component: 'hooks' }],
      ['/ping', { provider: 'als-apg' }],
    ]);
  });
});
