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
 *   - the /api/panel-focus POST on a user-initiated tab switch (prefixed
 *     inline — fetchJSON doesn't support POST bodies)
 *   - the iframe-src builders in navigatePanel()/createIframe(): state.url
 *     arrives from the server ALREADY prefixed (routes/panels.py's
 *     _url_prefix()), so `new URL(path, origin)` must preserve it as-is,
 *     never re-strip or double-add window.__OSPREY_PREFIX__
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

/** Renders the DOM initPanelManager expects: a container with #panel-content, and a sibling #header-tabs. */
function renderContainer() {
  document.body.innerHTML = `
    <div id="panel-manager"><div id="panel-content"></div></div>
    <div id="header-tabs"></div>
  `;
}

/** A no-op EventSource stub that records constructed URLs. @returns {string[]} */
function stubEventSource() {
  /** @type {string[]} */
  const urls = [];
  class FakeEventSource {
    /** @param {string} url */
    constructor(url) {
      urls.push(url);
    }
    close() {}
  }
  vi.stubGlobal('EventSource', FakeEventSource);
  return urls;
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
    const urls = stubEventSource();

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

describe('/api/panel-focus POST on a user-initiated tab switch', () => {
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
