// @ts-check
/**
 * Unit tests for the OSPREY Web Terminal's connection helpers (api.js):
 *   npx vitest run tests/interfaces/web_terminal/api.test.mjs
 *
 * Covers the browser-free surface of api.js:
 *   - wsUrl(path): scheme derivation (wss:// under TLS, ws:// otherwise)
 *   - fetchJSON(url): 2xx -> parsed JSON; non-2xx -> throws `HTTP <s>: <t>`
 *   - onConnectionStateChange / getConnectionState: initial shape and that a
 *     registered listener is stored and fires on the next state transition
 *
 * Module isolation: api.js keeps `wsState`/`sseState`/`stateListeners` as
 * module-private state that no init() resets. `vi.resetModules()` plus a fresh
 * dynamic `import()` per test gives each test a never-before-touched module
 * instance, so there is no shared state to leak by construction.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/api.js')} */
let api;

beforeEach(async () => {
  vi.resetModules();
  api = await import('../../../src/osprey/interfaces/web_terminal/static/js/api.js');
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('wsUrl: scheme derivation from location.protocol', () => {
  test('returns a wss:// URL when the page is served over https', () => {
    vi.stubGlobal('location', { protocol: 'https:', host: 'example.org:8443' });
    expect(api.wsUrl('/ws/terminal')).toBe('wss://example.org:8443/ws/terminal');
  });

  test('returns a ws:// URL when the page is not served over https', () => {
    vi.stubGlobal('location', { protocol: 'http:', host: 'localhost:5000' });
    expect(api.wsUrl('/ws/terminal')).toBe('ws://localhost:5000/ws/terminal');
  });
});

describe('fetchJSON: success and error paths', () => {
  test('a 2xx response resolves to the parsed JSON body', async () => {
    const body = { ok: true, items: [1, 2, 3] };
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => body,
    }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.fetchJSON('/api/state');
    expect(result).toEqual(body);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/api/state', { cache: 'no-store' });
  });

  test('a non-2xx response throws `HTTP <status>: <statusText>`', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => ({ never: 'read' }),
      }))
    );

    await expect(api.fetchJSON('/api/state')).rejects.toThrow(
      'HTTP 503: Service Unavailable'
    );
  });
});

describe('connection state: initial shape and listener registration', () => {
  test('getConnectionState reports both channels disconnected before any connection', () => {
    expect(api.getConnectionState()).toEqual({ ws: 'disconnected', sse: 'disconnected' });
  });

  test('a listener registered via onConnectionStateChange fires on the next state transition with the current state', () => {
    const listener = vi.fn();
    api.onConnectionStateChange(listener);
    // No transition has happened yet, so the listener has not been invoked.
    expect(listener).not.toHaveBeenCalled();

    // createEventSource's connect() runs synchronously: it flips sseState to
    // 'connecting' and drives notifyStateChange -> the registered listener,
    // then constructs `new EventSource(url)`. happy-dom does not provide an
    // EventSource, so stub a minimal, side-effect-free constructor that lets
    // connect() finish; the notification we assert on has already fired by
    // then. `close` is what the returned handle's stop() calls.
    vi.stubGlobal(
      'EventSource',
      class {
        close() {}
      }
    );
    const source = api.createEventSource('/events');
    try {
      expect(listener).toHaveBeenCalled();
      expect(listener).toHaveBeenLastCalledWith({ ws: 'disconnected', sse: 'connecting' });
      expect(api.getConnectionState()).toEqual({ ws: 'disconnected', sse: 'connecting' });
    } finally {
      source.stop();
    }
  });
});
