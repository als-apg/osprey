// @ts-check
/**
 * Unit tests for the OSPREY Web Terminal's connection helpers (api.js):
 *   npx vitest run tests/interfaces/web_terminal/api.test.mjs
 *
 * Covers the browser-free surface of api.js:
 *   - wsUrl(path): scheme derivation (wss:// under TLS, ws:// otherwise)
 *   - fetchJSON(url): 2xx -> parsed JSON; non-2xx -> throws `HTTP <s>: <t>`
 *   - apiRequest(url, opts): mutating-verb helper -- json body wiring, server
 *     `detail` extraction on error, errorPrefix fallback, null on empty body
 *   - onConnectionStateChange / getConnectionState: initial shape and that a
 *     registered listener is stored and fires on the next state transition
 *   - per-user URL prefix: wsUrl/fetchJSON/createEventSource read
 *     `window.__OSPREY_PREFIX__` (the multi-user prefix contract) and
 *     prepend it to root-absolute paths only, are a no-op when the prefix is
 *     empty/absent, and never double-prefix or touch already-absolute URLs
 *   - reload-on-401 probe: a closed WS/SSE channel asks the app's own
 *     prefixed health route whether the session survived, and only a definite
 *     401 stops the reconnect loop and reloads
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

describe('apiRequest: mutating-verb helper (method/body wiring and detail extraction)', () => {
  test('serializes `json` as the request body with the JSON content type', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ saved: true }),
    }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.apiRequest('/api/config', {
      method: 'PATCH',
      json: { updates: { a: 1 } },
    });
    expect(result).toEqual({ saved: true });
    expect(fetchMock).toHaveBeenCalledWith('/api/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates: { a: 1 } }),
    });
  });

  test('omits headers and body when no `json` payload is given', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({}),
    }));
    vi.stubGlobal('fetch', fetchMock);

    await api.apiRequest('/api/scaffold/x/claim', { method: 'POST' });
    expect(fetchMock).toHaveBeenCalledWith('/api/scaffold/x/claim', { method: 'POST' });
  });

  test('a non-OK response throws the server `detail` message when present', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'already claimed' }),
      }))
    );

    await expect(
      api.apiRequest('/api/scaffold/x/claim', { method: 'POST', errorPrefix: 'Scaffold failed' })
    ).rejects.toThrow('already claimed');
  });

  test('a non-OK response without a JSON body falls back to `<errorPrefix> (HTTP <status>)`', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 502,
        json: async () => { throw new SyntaxError('not JSON'); },
      }))
    );

    await expect(
      api.apiRequest('/api/config', { method: 'PUT', errorPrefix: 'Save failed' })
    ).rejects.toThrow('Save failed (HTTP 502)');
  });

  test('an OK response without a JSON body (e.g. empty DELETE) resolves to null', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 204,
        json: async () => { throw new SyntaxError('empty body'); },
      }))
    );

    await expect(api.apiRequest('/api/thing', { method: 'DELETE' })).resolves.toBeNull();
  });

  test('routes the URL through the withPrefix chokepoint', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({}),
    }));
    vi.stubGlobal('fetch', fetchMock);

    await api.apiRequest('/api/terminal/restart', { method: 'POST' });
    expect(fetchMock).toHaveBeenCalledWith('/u/alice/api/terminal/restart', { method: 'POST' });
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

describe('reconnect: session-expiry probe (reload-on-401)', () => {
  /**
   * Drain the probe's promise chain (fetch -> then -> catch -> finally -> the
   * reload decision). A fixed number of microtask ticks, so this stays
   * deterministic under fake timers, which only fake the macrotask queue.
   */
  async function flushProbe() {
    for (let i = 0; i < 10; i++) await Promise.resolve();
  }

  /**
   * Stub `WebSocket` with a constructor that records every instance, so a test
   * can drive `onclose` by hand and count reconnect attempts.
   * @returns {any[]} the constructed instances, in order
   */
  function stubWebSocket() {
    /** @type {any[]} */
    const instances = [];
    vi.stubGlobal(
      'WebSocket',
      class {
        constructor() {
          instances.push(this);
        }
        close() {}
      }
    );
    return instances;
  }

  /**
   * Stub `EventSource` the same way, recording instances and `close()` calls.
   * @returns {any[]}
   */
  function stubEventSource() {
    /** @type {any[]} */
    const instances = [];
    vi.stubGlobal(
      'EventSource',
      class {
        constructor() {
          this.closed = false;
          instances.push(this);
        }
        close() {
          this.closed = true;
        }
      }
    );
    return instances;
  }

  /**
   * Stub `location` with a reload spy (and the fields wsUrl reads).
   * @returns {any}
   */
  function stubLocation() {
    const loc = { protocol: 'http:', host: 'localhost:5000', reload: vi.fn() };
    vi.stubGlobal('location', loc);
    return loc;
  }

  /** @param {number} status */
  function stubFetchStatus(status) {
    const fetchMock = vi.fn(async () => ({ ok: status < 400, status }));
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  beforeEach(() => {
    window.__OSPREY_PREFIX__ = '/u/alice';
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('a closed WebSocket probes the app\'s own prefixed health route with a JSON Accept header', async () => {
    stubLocation();
    const sockets = stubWebSocket();
    const fetchMock = stubFetchStatus(401);

    api.createWebSocket('ws://localhost:5000/u/alice/ws/terminal');
    sockets[0].onclose({ code: 1006 });
    await flushProbe();

    // Never the sidecar's unauthenticated /health, never a root-absolute
    // /health: only /u/<user>/health sits behind the user's auth gate.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/u/alice/health', {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    });
  });

  test('a 401 probe stops the reconnect loop and reloads', async () => {
    vi.useFakeTimers();
    const loc = stubLocation();
    const sockets = stubWebSocket();
    stubFetchStatus(401);

    api.createWebSocket('ws://localhost:5000/u/alice/ws/terminal');
    sockets[0].onclose({ code: 1006 });
    await flushProbe();

    expect(loc.reload).toHaveBeenCalledTimes(1);
    // Past the whole backoff ceiling: no further connection is attempted.
    vi.advanceTimersByTime(60000);
    expect(sockets).toHaveLength(1);
  });

  test('a probe that fails with a network error keeps the existing reconnect/backoff and never reloads', async () => {
    vi.useFakeTimers();
    const loc = stubLocation();
    const sockets = stubWebSocket();
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));

    api.createWebSocket('ws://localhost:5000/u/alice/ws/terminal');
    sockets[0].onclose({ code: 1006 });
    await flushProbe();

    expect(loc.reload).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(2);
  });

  test('a healthy 200 probe keeps the existing reconnect/backoff and never reloads', async () => {
    vi.useFakeTimers();
    const loc = stubLocation();
    const sockets = stubWebSocket();
    stubFetchStatus(200);

    api.createWebSocket('ws://localhost:5000/u/alice/ws/terminal');
    sockets[0].onclose({ code: 1006 });
    await flushProbe();

    expect(loc.reload).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(2);
  });

  test('a 503 probe (server restarting, session intact) keeps reconnecting', async () => {
    vi.useFakeTimers();
    const loc = stubLocation();
    const sockets = stubWebSocket();
    stubFetchStatus(503);

    api.createWebSocket('ws://localhost:5000/u/alice/ws/terminal');
    sockets[0].onclose({ code: 1006 });
    await flushProbe();

    expect(loc.reload).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(2);
  });

  test('concurrent closes share a single in-flight probe', async () => {
    stubLocation();
    const sockets = stubWebSocket();
    const fetchMock = stubFetchStatus(200);

    api.createWebSocket('ws://localhost:5000/u/alice/ws/terminal');
    api.createWebSocket('ws://localhost:5000/u/alice/ws/other');
    sockets[0].onclose({ code: 1006 });
    sockets[1].onclose({ code: 1006 });
    await flushProbe();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('reloads at most once no matter how many channels close afterwards', async () => {
    const loc = stubLocation();
    const sockets = stubWebSocket();
    const fetchMock = stubFetchStatus(401);

    api.createWebSocket('ws://localhost:5000/u/alice/ws/terminal');
    const second = api.createWebSocket('ws://localhost:5000/u/alice/ws/other');
    sockets[0].onclose({ code: 1006 });
    await flushProbe();
    expect(loc.reload).toHaveBeenCalledTimes(1);

    sockets[1].onclose({ code: 1006 });
    await flushProbe();

    expect(loc.reload).toHaveBeenCalledTimes(1);
    // The expiry is already known, so no second request goes out.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    second.stop();
  });

  test('an EventSource error on a 401 closes the source and reloads', async () => {
    const loc = stubLocation();
    const sources = stubEventSource();
    const fetchMock = stubFetchStatus(401);

    api.createEventSource('/api/files/events');
    sources[0].onerror();
    await flushProbe();

    expect(fetchMock).toHaveBeenCalledWith('/u/alice/health', {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    });
    expect(sources[0].closed).toBe(true);
    expect(loc.reload).toHaveBeenCalledTimes(1);
  });

  test('an EventSource error with the session intact neither closes the source nor reloads', async () => {
    const loc = stubLocation();
    const sources = stubEventSource();
    stubFetchStatus(200);

    const source = api.createEventSource('/api/files/events');
    sources[0].onerror();
    await flushProbe();

    expect(sources[0].closed).toBe(false);
    expect(loc.reload).not.toHaveBeenCalled();
    source.stop();
  });

  test('probes the unprefixed /health when no per-user prefix is set (single-origin/dev)', async () => {
    delete window.__OSPREY_PREFIX__;
    const loc = stubLocation();
    const sockets = stubWebSocket();
    const fetchMock = stubFetchStatus(200);

    api.createWebSocket('ws://localhost:5000/ws/terminal');
    sockets[0].onclose({ code: 1006 });
    await flushProbe();

    expect(fetchMock).toHaveBeenCalledWith('/health', {
      cache: 'no-store',
      headers: { Accept: 'application/json' },
    });
    expect(loc.reload).not.toHaveBeenCalled();
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
