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
 *   - per-user URL prefix: wsUrl/fetchJSON/createEventSource read
 *     `window.__OSPREY_PREFIX__` (the multi-user prefix contract) and
 *     prepend it to root-absolute paths only, are a no-op when the prefix is
 *     empty/absent, and never double-prefix or touch already-absolute URLs
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
  delete window.__OSPREY_PREFIX__;
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

describe('URL prefix: window.__OSPREY_PREFIX__ (multi-user prefix contract)', () => {
  /** A stand-in for `fetch` that always resolves 2xx with an empty body. */
  function stubFetchOk() {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({}),
    }));
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  test('wsUrl prepends the prefix to the path before scheme+host assembly', () => {
    vi.stubGlobal('location', { protocol: 'https:', host: 'example.org:8443' });
    window.__OSPREY_PREFIX__ = '/u/alice';
    expect(api.wsUrl('/ws/terminal')).toBe('wss://example.org:8443/u/alice/ws/terminal');
  });

  test('wsUrl is byte-identical to the unprefixed result when the prefix is empty', () => {
    vi.stubGlobal('location', { protocol: 'http:', host: 'localhost:5000' });
    window.__OSPREY_PREFIX__ = '';
    expect(api.wsUrl('/ws/terminal')).toBe('ws://localhost:5000/ws/terminal');
  });

  test('wsUrl is byte-identical to the unprefixed result when the prefix is absent', () => {
    vi.stubGlobal('location', { protocol: 'http:', host: 'localhost:5000' });
    expect(api.wsUrl('/ws/terminal')).toBe('ws://localhost:5000/ws/terminal');
  });

  test('fetchJSON prepends the prefix to a root-absolute URL', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = stubFetchOk();
    await api.fetchJSON('/api/state');
    expect(fetchMock).toHaveBeenCalledWith('/u/alice/api/state', { cache: 'no-store' });
  });

  test('fetchJSON is a no-op when the prefix is empty', async () => {
    window.__OSPREY_PREFIX__ = '';
    const fetchMock = stubFetchOk();
    await api.fetchJSON('/api/state');
    expect(fetchMock).toHaveBeenCalledWith('/api/state', { cache: 'no-store' });
  });

  test.each([
    'https://other.example/api',
    'http://other.example/api',
    '//cdn.example/api',
  ])('fetchJSON leaves the already-absolute URL %s untouched', async (url) => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = stubFetchOk();
    await api.fetchJSON(url);
    expect(fetchMock).toHaveBeenCalledWith(url, { cache: 'no-store' });
  });

  test('fetchJSON does not double-prefix a path that already carries the prefix', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = stubFetchOk();
    await api.fetchJSON('/u/alice/api/state');
    expect(fetchMock).toHaveBeenCalledWith('/u/alice/api/state', { cache: 'no-store' });
  });

  test('createEventSource prepends the prefix to a root-absolute path', () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    /** @type {string[]} */
    const constructedUrls = [];
    vi.stubGlobal(
      'EventSource',
      class {
        /** @param {string} url */
        constructor(url) {
          constructedUrls.push(url);
        }
        close() {}
      }
    );

    const source = api.createEventSource('/events');
    try {
      expect(constructedUrls).toEqual(['/u/alice/events']);
    } finally {
      source.stop();
    }
  });

  test('createEventSource is a no-op when the prefix is absent', () => {
    /** @type {string[]} */
    const constructedUrls = [];
    vi.stubGlobal(
      'EventSource',
      class {
        /** @param {string} url */
        constructor(url) {
          constructedUrls.push(url);
        }
        close() {}
      }
    );

    const source = api.createEventSource('/events');
    try {
      expect(constructedUrls).toEqual(['/events']);
    } finally {
      source.stop();
    }
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
