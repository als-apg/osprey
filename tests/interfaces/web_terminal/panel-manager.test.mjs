// @ts-check
/**
 * Unit tests for panel-manager.js's per-user URL prefix awareness
 * (window.__OSPREY_PREFIX__, the multi-user prefix contract — see
 * api.test.mjs for the api.js helpers this module builds on). Covers:
 *
 *   - initPanel()'s PANELS[].configEndpoint fetches and the /api/panels
 *     fetch, both via fetchJSON (prefixed internally)
 *   - the /api/files/events EventSource, via createEventSource (prefixed
 *     internally)
 *   - the /api/panel-focus POST on a user-initiated rail switch (via
 *     panel-commands.js, prefixed with withPrefix)
 *   - the iframe-src builders in navigatePanel()/createIframe(): state.url
 *     arrives from the server ALREADY prefixed (routes/panels.py's
 *     compute_url_prefix()), so `new URL(path, origin)` must preserve it
 *     as-is, never re-strip or double-add window.__OSPREY_PREFIX__
 *
 * Every prefix case is paired with an empty-prefix case asserting
 * byte-identical (unprefixed) behavior, per the prefix contract.
 *
 * Module isolation: panel-manager.js keeps PANELS/panelState/visiblePanels
 * as module-private state mutated in place by initPanelManager(), so each
 * test does vi.resetModules() + a fresh dynamic import (same pattern as
 * api.test.mjs) to avoid cross-test leakage.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-manager.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** Minimal ok-JSON fetch Response stand-in. @param {any} body */
function jsonOk(body) {
  return { ok: true, status: 200, statusText: 'OK', json: async () => body };
}

/** Renders the DOM initPanelManager expects: a container with #panel-content, and a sibling #panel-rail. */
function renderContainer() {
  document.body.innerHTML = `
    <nav id="panel-rail"></nav>
    <div id="panel-manager"><div id="panel-content"></div></div>
  `;
}

/**
 * A no-op EventSource stub that records constructed URLs and exposes `emit`
 * to inject server frames through the live onmessage handler — the same
 * dispatch seam real SSE frames arrive on (api.js's createEventSource
 * JSON-parses e.data before invoking panel-manager's handler).
 * @returns {{ urls: string[], emit: (frame: object) => void }}
 */
function stubEventSource() {
  /** @type {string[]} */
  const urls = [];
  /** @type {{ onmessage?: ((e: { data: string }) => void) | null }[]} */
  const sources = [];
  class FakeEventSource {
    /** @param {string} url */
    constructor(url) {
      urls.push(url);
      /** @type {((e: { data: string }) => void) | null} */
      this.onmessage = null;
      sources.push(this);
    }
    close() {}
  }
  vi.stubGlobal('EventSource', FakeEventSource);
  return {
    urls,
    emit: (frame) => {
      for (const s of sources) s.onmessage?.({ data: JSON.stringify(frame) });
    },
  };
}

/** @returns {Promise<typeof import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js')>} */
async function freshImport() {
  vi.resetModules();
  return import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js');
}

beforeEach(() => {
  delete window.__OSPREY_PREFIX__;
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('config fetches: /api/panels and PANELS[].configEndpoint (via fetchJSON)', () => {
  test('prepend window.__OSPREY_PREFIX__ when set', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    renderContainer();
    /** @type {string[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      calls.push(url);
      if (url === '/u/alice/api/panels') {
        return jsonOk({ enabled: ['artifacts'], custom: [], default: null, visible: ['artifacts'], active: null, labels: {} });
      }
      if (url === '/u/alice/api/artifact-server') {
        return jsonOk({ url: '/u/alice/panel/artifacts', available: true });
      }
      return jsonOk({});
    }));
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    expect(calls).toContain('/u/alice/api/panels');
    expect(calls).toContain('/u/alice/api/artifact-server');
  });

  test('empty prefix ⇒ byte-identical (unprefixed) URLs', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    /** @type {string[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      calls.push(url);
      if (url === '/api/panels') {
        return jsonOk({ enabled: ['artifacts'], custom: [], default: null, visible: ['artifacts'], active: null, labels: {} });
      }
      if (url === '/api/artifact-server') {
        return jsonOk({ url: '/panel/artifacts', available: true });
      }
      return jsonOk({});
    }));
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    expect(calls).toContain('/api/panels');
    expect(calls).toContain('/api/artifact-server');
  });
});

describe('/api/files/events EventSource (via createEventSource)', () => {
  /**
   * @param {string|undefined} prefix
   * @param {string} expectedUrl
   */
  async function assertEventSourceUrl(prefix, expectedUrl) {
    if (prefix !== undefined) window.__OSPREY_PREFIX__ = prefix;
    renderContainer();
    vi.stubGlobal('fetch', vi.fn(async () =>
      jsonOk({ enabled: [], custom: [], default: null, visible: [], active: null, labels: {} })
    ));
    const { urls } = stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    expect(urls).toEqual([expectedUrl]);
  }

  test('prepends the prefix when set', async () => {
    await assertEventSourceUrl('/u/alice', '/u/alice/api/files/events');
  });

  test('is a no-op when the prefix is empty', async () => {
    await assertEventSourceUrl('', '/api/files/events');
  });
});

describe('/api/panel-focus POST on a user-initiated rail switch', () => {
  /**
   * @param {string|undefined} prefix
   * @param {string} expectedUrl
   */
  async function assertPanelFocusUrl(prefix, expectedUrl) {
    if (prefix !== undefined) window.__OSPREY_PREFIX__ = prefix;
    renderContainer();
    const artifactsUrl = `${prefix || ''}/panel/artifacts`;
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url, /** @type {any} */ opts) => {
      calls.push({ url, opts });
      if (url.endsWith('/api/panels')) {
        return jsonOk({ enabled: ['artifacts'], custom: [], default: null, visible: ['artifacts'], active: null, labels: {} });
      }
      if (url.endsWith('/api/artifact-server')) {
        return jsonOk({ url: artifactsUrl, available: true });
      }
      return jsonOk({ status: 'ok' });
    }));
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    const tab = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]'));
    await vi.waitFor(() => expect(tab.classList.contains('disabled')).toBe(false));

    // Isolate the click's own request from the config/panels fetches above.
    calls.length = 0;
    tab.click();

    await vi.waitFor(() => expect(calls.some(c => c.url === expectedUrl)).toBe(true));
    const focusCall = calls.find(c => c.url === expectedUrl);
    if (!focusCall) throw new Error('expected a panel-focus fetch call');
    expect(focusCall.opts).toMatchObject({ method: 'POST' });
    expect(JSON.parse(focusCall.opts.body)).toEqual({ panel: 'artifacts' });
  }

  test('prepends the prefix when set', async () => {
    await assertPanelFocusUrl('/u/alice', '/u/alice/api/panel-focus');
  });

  test('is a no-op when the prefix is empty', async () => {
    await assertPanelFocusUrl('', '/api/panel-focus');
  });
});

describe('iframe src: state.url arrives already server-prefixed (2.2) and must not be re-stripped/double-prefixed', () => {
  /**
   * @param {string|undefined} prefix
   * @param {string} expectedPath
   */
  async function assertIframeSrc(prefix, expectedPath) {
    if (prefix !== undefined) window.__OSPREY_PREFIX__ = prefix;
    renderContainer();
    const serverUrl = `${prefix || ''}/panel/artifacts`;
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url.endsWith('/api/panels')) {
        return jsonOk({ enabled: ['artifacts'], custom: [], default: null, visible: ['artifacts'], active: null, labels: {} });
      }
      if (url.endsWith('/api/artifact-server')) {
        return jsonOk({ url: serverUrl, available: true });
      }
      return jsonOk({});
    }));
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    await vi.waitFor(() => {
      expect(document.querySelector('iframe[data-panel-id="artifacts"]')).not.toBeNull();
    });
    const iframe = document.querySelector('iframe[data-panel-id="artifacts"]');
    if (!(iframe instanceof HTMLIFrameElement)) throw new Error('expected an iframe to be created');
    const parsed = new URL(iframe.src);
    expect(parsed.origin + parsed.pathname).toBe(`${window.location.origin}${expectedPath}`);
    expect(parsed.searchParams.get('embedded')).toBe('true');
  }

  test('preserves the /u/<user>/panel/<id> prefix (multi-user)', async () => {
    await assertIframeSrc('/u/alice', '/u/alice/panel/artifacts');
  });

  test('resolves to the unprefixed /panel/<id> when the prefix is empty', async () => {
    await assertIframeSrc('', '/panel/artifacts');
  });
});

describe('rail LEDs for custom panels without a health endpoint', () => {
  test('a null-healthEndpoint panel renders a healthy LED, not the offline default', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({
          enabled: [],
          custom: [
            { id: 'results', label: 'RESULTS', url: '/panel/results', healthEndpoint: null, path: '/results/' },
          ],
          default: null,
          visible: ['results'],
          active: null,
          labels: {},
        });
      }
      return jsonOk({});
    }));
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    const entry = document.querySelector('[data-panel-id="results"]');
    if (!(entry instanceof HTMLElement)) throw new Error('expected a results rail entry');
    expect(entry.classList.contains('disabled')).toBe(false);
    const led = entry.querySelector('.panel-rail-led');
    if (!(led instanceof HTMLElement)) throw new Error('expected a rail LED');
    expect(led.className).toBe('panel-rail-led healthy');
  });
});

describe('agent activity: rail badge/glow + the activity-strip seam', () => {
  /**
   * Boot the manager with a healthy 'artifacts' panel (no health endpoint, so
   * it enables synchronously) and an unhealthy 'ariel' panel (config endpoint
   * returns no url, so its entry stays disabled). Returns the SSE `emit`
   * injector, the fresh module, and the artifacts rail entry.
   */
  async function bootWithSSE() {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({ enabled: ['artifacts', 'ariel'], custom: [], default: null, visible: ['artifacts', 'ariel'], active: null, labels: {} });
      }
      if (url === '/api/artifact-server') {
        return jsonOk({ url: '/panel/artifacts', available: true });
      }
      // /api/ariel-server (and any POST): no url ⇒ ariel stays unhealthy
      return jsonOk({});
    }));
    const { emit } = stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');

    const artifacts = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]'));
    await vi.waitFor(() => expect(artifacts.classList.contains('disabled')).toBe(false));
    return { emit, mod, artifacts };
  }

  test("agent_activity kind:'panel' with a rail entry sets badge + flash, no strip fallback", async () => {
    const { emit, mod, artifacts } = await bootWithSSE();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'artifacts' }, ts: 1 });

    expect(artifacts.classList.contains('agent-attention')).toBe(true);
    expect(artifacts.classList.contains('agent-flash')).toBe(true);
    expect(strip).not.toHaveBeenCalled();
  });

  test("agent_activity kind:'panel' with an unknown id falls back to the strip handler", async () => {
    const { emit, mod } = await bootWithSSE();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    const frame = { type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'no-such-panel' }, ts: 2 };
    emit(frame);

    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip).toHaveBeenCalledWith(frame);
  });

  test("agent_activity kind:'channel' goes to the strip handler and leaves the rail alone", async () => {
    const { emit, mod } = await bootWithSSE();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    const frame = { type: 'agent_activity', tool: 'read_channel', target: { kind: 'channel', detail: 'SR01C:BPM1:X' }, ts: 3 };
    emit(frame);

    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip).toHaveBeenCalledWith(frame);
    expect(document.querySelector('.agent-attention')).toBeNull();
    expect(document.querySelector('.agent-flash')).toBeNull();
  });

  test("panel_focus with source:'agent' glows transiently (no badge); untagged has no agent styling", async () => {
    const { emit, artifacts } = await bootWithSSE();

    emit({ type: 'panel_focus', panel: 'artifacts', source: 'agent' });
    expect(artifacts.classList.contains('agent-flash')).toBe(true);
    expect(artifacts.classList.contains('agent-attention')).toBe(false);

    // Same frame without the tag: no agent styling at all.
    artifacts.classList.remove('agent-flash');
    emit({ type: 'panel_focus', panel: 'artifacts' });
    expect(artifacts.classList.contains('agent-flash')).toBe(false);
    expect(artifacts.classList.contains('agent-attention')).toBe(false);
  });

  test('activateTab clears the badge when the panel surfaces (agent-driven focus)', async () => {
    const { emit, artifacts } = await bootWithSSE();

    emit({ type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'artifacts' }, ts: 4 });
    expect(artifacts.classList.contains('agent-attention')).toBe(true);

    emit({ type: 'panel_focus', panel: 'artifacts', source: 'agent' });
    expect(artifacts.classList.contains('agent-attention')).toBe(false);
  });

  test('an unhealthy-panel activation early-returns and keeps the badge', async () => {
    const { emit } = await bootWithSSE();
    const ariel = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="ariel"]'));
    expect(ariel.classList.contains('disabled')).toBe(true); // never became healthy

    emit({ type: 'agent_activity', tool: 'search_logbook', target: { kind: 'panel', panel: 'ariel' }, ts: 5 });
    expect(ariel.classList.contains('agent-attention')).toBe(true);

    emit({ type: 'panel_focus', panel: 'ariel' }); // activateTab bails on !healthy
    expect(ariel.classList.contains('agent-attention')).toBe(true);
  });

  test('getActivePanel returns the surfaced panel id', async () => {
    const { mod } = await bootWithSSE();
    await vi.waitFor(() => expect(mod.getActivePanel()).toBe('artifacts'));
  });
});
