// @ts-check
/**
 * Unit tests for panel-iframe-factory.js: the two sandboxes it frames a
 * service panel under (OPAQUE_SANDBOX until the hub has answered, loosened to
 * FIRST_PARTY_SANDBOX for a panel the hub calls first-party; both carry
 * `allow-downloads`, so a panel can hand the operator a file from inside its
 * tile), the element createPanelIframe builds, and the framing path
 * (framePanel): every navigation of a service panel — first load, a
 * re-navigation, a recovered panel's held address, a cap renewal — goes through
 * it, so an opaque panel is always framed through a freshly minted cap and a
 * first-party one keeps its plain URL. The whole-hub wiring (navigatePanel,
 * health recovery, the panel left open past its cap) is pinned in
 * panel-manager.test.mjs; this file pins the factory's own rules: the sandbox
 * token sets, the expiry it reads off a cap, when a renewal fires, and which
 * answers it drops.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-iframe-factory.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

// The dock adapter is stubbed at the module boundary: what createPanelIframe
// owes it is one adoptIframe call with the element it built, which a spy states
// directly; where the adapter then mounts the element is dock-iframe's own
// contract. The factory is the only module on this graph that imports it.
const adopted = vi.hoisted(() => vi.fn());
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  adoptIframe: adopted,
  setPanelReachable: vi.fn(),
}));

const FACTORY_PATH =
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-factory.js';
const FIRST_PARTY =
  'allow-scripts allow-same-origin allow-popups allow-forms allow-modals allow-downloads';
const OPAQUE = 'allow-scripts allow-popups allow-modals allow-downloads';
const HOUR_MS = 60 * 60 * 1000;

/** @param {any} body */
function jsonOk(body) {
  return { ok: true, status: 200, statusText: 'OK', json: async () => body };
}

/** @param {number} ms epoch ms @returns {string} the cap token's expiry segment */
const hexExpiry = (ms) => Math.floor(ms / 1000).toString(16);

/**
 * Stub the hub's cap mint. `answer(n)` is the n-th (1-based) response.
 * @param {(n: number) => any} answer
 */
function stubCaps(answer) {
  /** @type {string[]} */
  const calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      calls.push(url);
      return answer(calls.length);
    })
  );
  return calls;
}

/** A capped answer expiring `ttlMs` from now, numbered `n`. */
const capAnswer = (/** @type {number} */ n, ttlMs = 12 * HOUR_MS) =>
  jsonOk({ cap_url: `/panel/p/~c/${hexExpiry(Date.now() + ttlMs)}.T${n}.S/` });

/** @returns {HTMLIFrameElement} an iframe attached to the document, sandboxed opaque */
function attachedFrame() {
  const el = document.createElement('iframe');
  el.sandbox = OPAQUE;
  document.body.appendChild(el);
  return el;
}

/** @param {HTMLIFrameElement} el */
const path = (el) => new URL(el.src).pathname;

/** @param {string} sandbox @returns {string[]} the sandbox's tokens, sorted */
const tokens = (sandbox) => sandbox.split(/\s+/).filter(Boolean).sort();

/** @param {HTMLIFrameElement} el @returns {string[]} the element's sandbox tokens, sorted */
const sandboxTokens = (el) => tokens(el.getAttribute('sandbox') ?? '');

/** Let the resolve promise chain run under fake timers. */
const flush = () => vi.advanceTimersByTimeAsync(0);

/** @returns {Promise<any>} */
async function load() {
  vi.resetModules();
  return import(FACTORY_PATH);
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
  vi.setSystemTime(new Date(2026, 0, 5, 9, 0, 0));
  window.__OSPREY_PREFIX__ = '';
  document.body.innerHTML = '';
  adopted.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('the sandbox constants', () => {
  test('a first-party panel gets exactly the granted tokens, downloads included', async () => {
    const { FIRST_PARTY_SANDBOX } = await load();
    expect(tokens(FIRST_PARTY_SANDBOX)).toEqual([
      'allow-downloads',
      'allow-forms',
      'allow-modals',
      'allow-popups',
      'allow-same-origin',
      'allow-scripts',
    ]);
    expect(FIRST_PARTY_SANDBOX).toBe(FIRST_PARTY);
  });

  test('an opaque panel loses the origin and forms, keeps downloads', async () => {
    const { OPAQUE_SANDBOX } = await load();
    expect(tokens(OPAQUE_SANDBOX)).toEqual([
      'allow-downloads',
      'allow-modals',
      'allow-popups',
      'allow-scripts',
    ]);
    expect(OPAQUE_SANDBOX).toBe(OPAQUE);
  });
});

describe('createPanelIframe', () => {
  /** @type {{url: string | null, pendingUrl: string | null, iframe: HTMLIFrameElement | null}} */
  let state;
  beforeEach(() => {
    state = { url: '/svc/tuning', pendingUrl: null, iframe: null };
  });

  test('frames the tile opaque until the hub answers, then first-party with downloads', async () => {
    stubCaps(() => ({ ok: false, status: 404, statusText: 'Not Found', json: async () => ({}) }));
    const { createPanelIframe } = await load();
    createPanelIframe(
      { id: 'tuning', label: 'Tuning' },
      'tuning',
      state,
      document.createElement('div')
    );

    const iframe = /** @type {HTMLIFrameElement} */ (state.iframe);
    expect(adopted).toHaveBeenCalledWith('tuning', iframe, { title: 'Tuning' });
    // Restrictive before the hub has said what the panel is, and no src yet.
    expect(iframe.dataset.panelFrame).toBe('pending');
    expect(sandboxTokens(iframe)).toEqual(tokens(OPAQUE));
    expect(iframe.getAttribute('src')).toBeNull();

    await flush();
    expect(iframe.dataset.panelFrame).toBe('first-party');
    expect(sandboxTokens(iframe)).toEqual(tokens(FIRST_PARTY));
    expect(sandboxTokens(iframe)).toContain('allow-downloads');
    expect(path(iframe)).toBe('/svc/tuning');
  });

  test('a panel the hub calls opaque stays under the opaque sandbox, downloads included', async () => {
    stubCaps((n) => capAnswer(n));
    const { createPanelIframe } = await load();
    createPanelIframe(
      { id: 'tuning', label: 'Tuning' },
      'tuning',
      state,
      document.createElement('div')
    );
    await flush();
    const iframe = /** @type {HTMLIFrameElement} */ (state.iframe);
    expect(iframe.dataset.panelFrame).toBe('opaque');
    expect(sandboxTokens(iframe)).toEqual(tokens(OPAQUE));
    expect(sandboxTokens(iframe)).toContain('allow-downloads');
    expect(sandboxTokens(iframe)).not.toContain('allow-same-origin');
  });

  test('builds no iframe before the panel has a URL', async () => {
    const { createPanelIframe } = await load();
    state.url = null;
    createPanelIframe(undefined, 'tuning', state, document.createElement('div'));
    expect(state.iframe).toBeNull();
    expect(adopted).not.toHaveBeenCalled();
  });
});

describe('capExpiryMs reads the expiry the hub minted into the cap', () => {
  test('the first token segment is epoch seconds in hex', async () => {
    const { capExpiryMs } = await load();
    expect(capExpiryMs('/panel/p/~c/6a0b1c2d.TAG.SIG/')).toBe(0x6a0b1c2d * 1000);
    expect(capExpiryMs('/u/alice/panel-static/b/~c/ff.T.S/')).toBe(255_000);
  });

  test('a URL with no cap segment has no expiry', async () => {
    const { capExpiryMs } = await load();
    expect(capExpiryMs('/panel/p/')).toBeNull();
    expect(capExpiryMs('/panel/p/~c/not-hex/')).toBeNull();
  });
});

describe('framePanel', () => {
  test('a catalog panel is framed at the plain URL with no cap request and no renewal', async () => {
    const calls = stubCaps(() => jsonOk({}));
    const { framePanel } = await load();
    const el = attachedFrame();
    framePanel(
      { id: 'artifacts', configEndpoint: '/api/artifact-server' },
      'artifacts',
      el,
      '/panel/artifacts/x',
      '/panel/artifacts'
    );
    expect(path(el)).toBe('/panel/artifacts/x');
    expect(el.getAttribute('sandbox')).toBe(FIRST_PARTY);
    await vi.advanceTimersByTimeAsync(48 * HOUR_MS);
    expect(calls).toEqual([]);
  });

  test('a panel the hub calls first-party (404) keeps the plain URL and is not renewed', async () => {
    const calls = stubCaps(() => ({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({}),
    }));
    const { framePanel } = await load();
    const el = attachedFrame();
    framePanel({ id: 'results' }, 'results', el, '/panel/results/', '/panel/results');
    await flush();
    expect(el.dataset.panelFrame).toBe('first-party');
    expect(el.getAttribute('sandbox')).toBe(FIRST_PARTY);
    expect(path(el)).toBe('/panel/results/');
    await vi.advanceTimersByTimeAsync(48 * HOUR_MS);
    expect(calls).toHaveLength(1);
  });

  test('an opaque panel is framed under its cap and renews it before it expires', async () => {
    const calls = stubCaps((n) => capAnswer(n));
    const { framePanel, CAP_RENEW_MARGIN_MS } = await load();
    const el = attachedFrame();
    framePanel({ id: 'p' }, 'p', el, '/panel/p/ui/', '/panel/p');
    await flush();
    expect(el.dataset.panelFrame).toBe('opaque');
    expect(path(el)).toMatch(/^\/panel\/p\/~c\/[0-9a-f]+\.T1\.S\/ui\/$/);
    const expiry = parseInt(path(el).split('~c/')[1], 16) * 1000;

    await vi.advanceTimersByTimeAsync(expiry - CAP_RENEW_MARGIN_MS - Date.now() - 1);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await flush();
    expect(calls).toHaveLength(2);
    // The same target, rebased onto the new cap, before the old one lapsed.
    expect(path(el)).toMatch(/\.T2\.S\/ui\/$/);
    expect(Date.now()).toBeLessThan(expiry);
    expect(el.getAttribute('sandbox')).toBe(OPAQUE);
  });

  test('a cap that already reads as expired renews after the floor, not in a loop', async () => {
    const calls = stubCaps((n) => capAnswer(n, -HOUR_MS));
    const { framePanel, CAP_RENEW_FLOOR_MS } = await load();
    const el = attachedFrame();
    framePanel({ id: 'p' }, 'p', el, '/panel/p/', '/panel/p');
    await flush();
    await vi.advanceTimersByTimeAsync(CAP_RENEW_FLOOR_MS - 1);
    expect(calls).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    await flush();
    expect(calls).toHaveLength(2);
  });

  test('a re-navigation supersedes the pending renewal and any late answer', async () => {
    /** @type {((v: any) => void)[]} */
    const pending = [];
    const calls = stubCaps(
      (n) =>
        new Promise((resolve) => {
          pending.push(() => resolve(capAnswer(n)));
        })
    );
    const { framePanel } = await load();
    const el = attachedFrame();
    framePanel({ id: 'p' }, 'p', el, '/panel/p/old', '/panel/p');
    framePanel({ id: 'p' }, 'p', el, '/panel/p/new', '/panel/p');
    expect(calls).toHaveLength(2);
    // The newer answer lands first, then the stale one: the stale one is dropped.
    pending[1](undefined);
    await flush();
    pending[0](undefined);
    await flush();
    expect(path(el)).toMatch(/\.T2\.S\/new$/);
  });

  test('a frame that left the document stops renewing', async () => {
    const calls = stubCaps((n) => capAnswer(n));
    const { framePanel } = await load();
    const el = attachedFrame();
    framePanel({ id: 'p' }, 'p', el, '/panel/p/', '/panel/p');
    await flush();
    el.remove();
    await vi.advanceTimersByTimeAsync(48 * HOUR_MS);
    expect(calls).toHaveLength(1);
  });
});
