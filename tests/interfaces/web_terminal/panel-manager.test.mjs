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

describe('rail state for custom panels without a health endpoint', () => {
  test('a null-healthEndpoint panel is enabled, not left inert at the disabled default', async () => {
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
    // The rail reports liveness ONLY as .disabled — no per-entry LED. Backend
    // health is surfaced by the SYSTEM panel's `web_panels` category instead.
    expect(entry.querySelector('.panel-rail-led')).toBeNull();
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

describe('simple-UX chat-only first boot (workspace suppression)', () => {
  afterEach(() => {
    document.documentElement.removeAttribute('data-ui-mode');
  });

  /**
   * Boot the manager under a given html[data-ui-mode] with a healthy
   * 'artifacts' panel and a server-reported workspace_has_artifacts flag.
   * Resolves once the artifacts rail entry is enabled (healthy), i.e. past
   * the point where auto-activation would have fired.
   * @param {{ mode: 'simple'|'expert', hasArtifacts: boolean }} opts
   */
  async function boot({ mode, hasArtifacts }) {
    document.documentElement.setAttribute('data-ui-mode', mode);
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({
          enabled: ['artifacts'],
          custom: [],
          default: null,
          visible: ['artifacts'],
          active: null,
          labels: {},
          workspace_has_artifacts: hasArtifacts,
        });
      }
      if (url === '/api/artifact-server') {
        return jsonOk({ url: '/panel/artifacts', available: true });
      }
      return jsonOk({ status: 'ok' });
    }));
    const { emit } = stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');

    const entry = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]'));
    await vi.waitFor(() => expect(entry.classList.contains('disabled')).toBe(false));
    return { emit, mod };
  }

  /** The activation observables: the workspace iframe and the active stamp. */
  function workspaceOpen() {
    const container = /** @type {HTMLElement} */ (document.getElementById('panel-manager'));
    return {
      iframe: document.querySelector('iframe[data-panel-id="artifacts"]'),
      active: container.dataset.activePanel ?? null,
    };
  }

  test('simple mode + empty workspace boots chat-only (no auto-activation)', async () => {
    await boot({ mode: 'simple', hasArtifacts: false });
    // Give any (wrong) deferred activation a chance to land before asserting.
    await new Promise((r) => setTimeout(r, 25));
    expect(workspaceOpen()).toEqual({ iframe: null, active: null });
  });

  test('simple mode with pre-existing artifacts activates the workspace as before', async () => {
    await boot({ mode: 'simple', hasArtifacts: true });
    await vi.waitFor(() => expect(workspaceOpen().iframe).not.toBeNull());
    expect(workspaceOpen().active).toBe('artifacts');
  });

  test('expert mode is untouched by an empty workspace', async () => {
    await boot({ mode: 'expert', hasArtifacts: false });
    await vi.waitFor(() => expect(workspaceOpen().iframe).not.toBeNull());
    expect(workspaceOpen().active).toBe('artifacts');
  });

  test("agent show_panel (panel_visibility) reveals the workspace on that panel", async () => {
    const { emit } = await boot({ mode: 'simple', hasArtifacts: false });
    expect(workspaceOpen().iframe).toBeNull();

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: true, source: 'agent' });

    await vi.waitFor(() => expect(workspaceOpen().iframe).not.toBeNull());
    expect(workspaceOpen().active).toBe('artifacts');
  });

  test('agent switch_panel (panel_focus) reveals the workspace', async () => {
    const { emit } = await boot({ mode: 'simple', hasArtifacts: false });
    expect(workspaceOpen().iframe).toBeNull();

    emit({ type: 'panel_focus', panel: 'artifacts', source: 'agent' });

    await vi.waitFor(() => expect(workspaceOpen().iframe).not.toBeNull());
    expect(workspaceOpen().active).toBe('artifacts');
  });

  test('hiding a panel never reveals the workspace', async () => {
    const { emit } = await boot({ mode: 'simple', hasArtifacts: false });

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false, source: 'agent' });

    await new Promise((r) => setTimeout(r, 25));
    expect(workspaceOpen()).toEqual({ iframe: null, active: null });
  });

  test('flipping to expert ends the suppression and fills the empty slot', async () => {
    const { mod } = await boot({ mode: 'simple', hasArtifacts: false });
    expect(workspaceOpen().iframe).toBeNull();

    document.documentElement.setAttribute('data-ui-mode', 'expert');
    mod.handleUiModeFlip('expert');

    await vi.waitFor(() => expect(workspaceOpen().iframe).not.toBeNull());
    expect(workspaceOpen().active).toBe('artifacts');
  });
});

describe('rail membership (launcher model: entry ⇔ member, never dimmed)', () => {
  /**
   * Boot with two enabled panels but only 'artifacts' a member (visible), so
   * 'ariel' starts in the "+" catalog with no rail entry.
   */
  async function bootMembership() {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({
          enabled: ['artifacts', 'ariel'],
          custom: [],
          default: null,
          visible: ['artifacts'],
          active: null,
          labels: {},
        });
      }
      if (url === '/api/artifact-server') {
        return jsonOk({ url: '/panel/artifacts', available: true });
      }
      if (url === '/api/ariel-server') {
        return jsonOk({ url: '/panel/ariel', available: true });
      }
      return jsonOk({ status: 'ok' });
    }));
    const { emit } = stubEventSource();
    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    return { emit, mod };
  }

  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);

  test('only members render entries; non-members are in the hidden (catalog) list', async () => {
    const { mod } = await bootMembership();
    expect(entry('terminal')).not.toBeNull();
    expect(entry('artifacts')).not.toBeNull();
    expect(entry('ariel')).toBeNull();
    expect(mod.getHiddenPanels()).toEqual([{ id: 'ariel', label: 'ARIEL' }]);
  });

  test('no entry ever carries the retired dimmed/closed class', async () => {
    const { emit } = await bootMembership();
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });
    expect(document.querySelector('.panel-rail-closed')).toBeNull();
  });

  test('a panel_visibility show APPENDS the entry with its live health state', async () => {
    const { emit } = await bootMembership();
    // ariel's config resolved at init; its no-endpoint health settle may still
    // be pending — wait for artifacts' enable as the settle barrier.
    await vi.waitFor(() =>
      expect(entry('artifacts')?.classList.contains('disabled')).toBe(false));

    emit({ type: 'panel_visibility', panel: 'ariel', visible: true });

    const ariel = entry('ariel');
    expect(ariel).not.toBeNull();
    await vi.waitFor(() => expect(ariel?.classList.contains('disabled')).toBe(false));
  });

  test('a panel_visibility hide REMOVES the entry and returns it to the catalog', async () => {
    const { emit, mod } = await bootMembership();

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });

    expect(entry('artifacts')).toBeNull();
    expect(mod.getHiddenPanels().map((p) => p.id)).toContain('artifacts');
    // Re-show rebuilds the entry.
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: true });
    expect(entry('artifacts')).not.toBeNull();
  });

  test('the SESSION (terminal) entry is always present and enabled', async () => {
    await bootMembership();
    const term = entry('terminal');
    expect(term).not.toBeNull();
    expect(term?.classList.contains('disabled')).toBe(false);
  });
});
