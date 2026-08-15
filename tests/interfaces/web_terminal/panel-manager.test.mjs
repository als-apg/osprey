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
 * Beyond the prefix contract this file also pins the SSE-driven behavior the
 * rail and the dock share — agent activity styling, rail membership, the
 * simple-UX chat-only boot, and (with a hand-built DockviewApi published
 * through the mocked dock-workspace module) the placement an agent open_panel
 * produces: focus the panel's own tile, or open one BESIDE, never evicting the
 * tile the operator is watching.
 *
 * Module isolation: panel-manager.js keeps PANELS/panelState/visiblePanels
 * as module-private state mutated in place by initPanelManager(), so each
 * test does vi.resetModules() + a fresh dynamic import (same pattern as
 * api.test.mjs) to avoid cross-test leakage.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-manager.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

// dock-workspace.js is stubbed at the module boundary so a test can publish a
// hand-built DockviewApi (see makeDockApi) and exercise the real placement
// engine in dock-iframe.js / dock-sync.js. `dockState.api` stays null by
// default, which is exactly what the real getDockApi returns with no dockview
// shell — every test that does not opt in keeps running in fallback mode.
const { getDockApi, openTerminalPanel, closeTerminalPanel, dockState } = vi.hoisted(() => ({
  dockState: { api: /** @type {any} */ (null) },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
}));
getDockApi.mockImplementation(() => dockState.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  openTerminalPanel,
  closeTerminalPanel,
  // Pulled in by dock-iframe.js on the transitive import chain; a mock must
  // supply them or they resolve to undefined.
  defaultServiceWidth: () => 600,
  setServiceRedock: () => {},
  onDragGesture: () => [],
}));

// dock-iframe.js keeps its REAL placement engine — only the tile-glow entry
// point is spied on. What the glow looks like (an overlay rectangle measured a
// frame later) is dock-glow.test.mjs's and the browser suite's contract; what
// this file pins is WHICH call sites fire it, which a spy states directly and a
// class assertion on a happy-dom-unlaid-out overlay could not.
//
// Registered per import (see freshImport) rather than through a hoisted
// vi.mock: the mock registry survives vi.resetModules(), so a hoisted factory
// would hand every later test the FIRST test's dock-iframe instance — an
// adapter still holding a removed overlay and a stale managed set. Re-running
// the factory after each reset keeps the adapter as fresh as the rest of the
// graph, which is the isolation every suite in this file depends on.
const DOCK_IFRAME_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js';
const { glowPanelSpy } = vi.hoisted(() => ({ glowPanelSpy: vi.fn() }));

/** @param {() => Promise<unknown>} importOriginal */
async function dockIframeWithGlowSpy(importOriginal) {
  const actual = /** @type {Record<string, unknown>} */ (await importOriginal());
  return { ...actual, glowPanel: glowPanelSpy };
}

/** The adapter's live-follow observer; geometry itself is browser-suite turf. */
class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

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
  vi.doMock(DOCK_IFRAME_PATH, dockIframeWithGlowSpy);
  return import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js');
}

beforeEach(() => {
  delete window.__OSPREY_PREFIX__;
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => dockState.api);
  dockState.api = null;
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

    // Boot surfaces the only visible panel, so its entry starts active — and a
    // click on the ACTIVE entry is the retire-tile toggle, not a switch. Retire
    // first so the click under test is a genuine activation.
    await vi.waitFor(() => expect(tab.classList.contains('active')).toBe(true));
    tab.click();
    await vi.waitFor(() => expect(tab.classList.contains('active')).toBe(false));

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

  test('re-clicking the ACTIVE entry retires its tile locally — no POST, entry stays', async () => {
    renderContainer();
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url, /** @type {any} */ opts) => {
      calls.push({ url, opts });
      if (url.endsWith('/api/panels')) {
        return jsonOk({ enabled: ['artifacts'], custom: [], default: null, visible: ['artifacts'], active: null, labels: {} });
      }
      if (url.endsWith('/api/artifact-server')) return jsonOk({ url: '/panel/artifacts', available: true });
      return jsonOk({ status: 'ok' });
    }));
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    const tab = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]'));
    await vi.waitFor(() => expect(tab.classList.contains('active')).toBe(true));

    calls.length = 0;
    tab.click();
    await vi.waitFor(() => expect(tab.classList.contains('active')).toBe(false));

    // Retiring a tile is LOCAL layout state: nothing is sent to the server,
    // and the panel keeps its rail membership so a second click brings it back.
    expect(calls.filter((c) => c.opts?.method === 'POST')).toEqual([]);
    expect(document.querySelector('[data-panel-id="artifacts"]')).toBe(tab);
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

  test("agent add_panel_to_rail (panel_visibility) reveals the workspace on that panel", async () => {
    const { emit } = await boot({ mode: 'simple', hasArtifacts: false });
    expect(workspaceOpen().iframe).toBeNull();

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: true, source: 'agent' });

    await vi.waitFor(() => expect(workspaceOpen().iframe).not.toBeNull());
    expect(workspaceOpen().active).toBe('artifacts');
  });

  test('agent open_panel (panel_focus) reveals the workspace', async () => {
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

describe('getPanelStandaloneUrl — the rail popout corner\'s target (railOptions onPopout)', () => {
  test('is null before config resolves a url, then the resolved url once it has', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({ enabled: ['ariel'], custom: [], default: null, visible: ['ariel'], active: null, labels: {} });
      }
      if (url === '/api/ariel-server') {
        return jsonOk({ url: '/panel/ariel', available: true });
      }
      return jsonOk({});
    }));
    stubEventSource();

    const mod = await freshImport();
    expect(mod.getPanelStandaloneUrl('ariel')).toBeNull();

    await mod.initPanelManager('panel-manager');
    // initPanel()'s configEndpoint fetch settles after initPanelManager's own
    // returned promise (fire-and-forget, like the rail's disabled → enabled
    // transition other tests in this file wait on).
    const entry = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="ariel"]'));
    await vi.waitFor(() => expect(entry.classList.contains('disabled')).toBe(false));

    expect(mod.getPanelStandaloneUrl('ariel')).toBe('/panel/ariel');
  });

  test('unknown panel id resolves to null', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal('fetch', vi.fn(async () =>
      jsonOk({ enabled: [], custom: [], default: null, visible: [], active: null, labels: {} })
    ));
    stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');

    expect(mod.getPanelStandaloneUrl('no-such-panel')).toBeNull();
  });

  test('suffixes a custom panel\'s catalog path onto its resolved url', async () => {
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

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');

    expect(mod.getPanelStandaloneUrl('results')).toBe('/panel/results/results/');
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
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });

    // Removal stays synchronous with the frame — the entry is gone by the time
    // the emit returns, and an UNTAGGED (human-origin) change stays off the
    // activity strip, which only ever reports the agent.
    expect(entry('artifacts')).toBeNull();
    expect(mod.getHiddenPanels().map((p) => p.id)).toContain('artifacts');
    // Re-show rebuilds the entry.
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: true });
    expect(entry('artifacts')).not.toBeNull();
    expect(strip).not.toHaveBeenCalled();
  });

  test("an agent hide reports itself on the strip; the entry is still removed", async () => {
    const { emit, mod } = await bootMembership();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false, source: 'agent' });

    expect(entry('artifacts')).toBeNull(); // the rail glow had nothing to land on
    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip.mock.calls[0][0]).toMatchObject({
      type: 'agent_activity',
      tool: 'remove_panel_from_rail',
      target: { kind: 'panel', panel: 'artifacts' },
    });
  });

  test('an agent show keeps the rail glow AND reports itself on the strip', async () => {
    const { emit, mod } = await bootMembership();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });

    expect(entry('ariel')?.classList.contains('agent-flash')).toBe(true);
    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip.mock.calls[0][0]).toMatchObject({
      type: 'agent_activity',
      tool: 'add_panel_to_rail',
      target: { kind: 'panel', panel: 'ariel' },
    });
  });

  test('an agent close takes the tile but LEAVES the rail entry', async () => {
    // The whole reason the on-screen axis is its own frame: the operator must
    // still be able to bring the panel back in one click.
    const { emit, mod } = await bootMembership();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_close', panel: 'artifacts', source: 'agent' });

    expect(entry('artifacts')).not.toBeNull();
    expect(mod.getHiddenPanels().map((p) => p.id)).not.toContain('artifacts');
    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip.mock.calls[0][0]).toMatchObject({
      type: 'agent_activity',
      tool: 'close_panel',
      target: { kind: 'panel', panel: 'artifacts' },
    });
  });

  test('a human close is applied without reporting agent activity', async () => {
    const { emit, mod } = await bootMembership();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_close', panel: 'artifacts' });

    expect(entry('artifacts')).not.toBeNull();
    expect(strip).not.toHaveBeenCalled();
  });

  test('the SESSION (terminal) entry is always present and enabled', async () => {
    await bootMembership();
    const term = entry('terminal');
    expect(term).not.toBeNull();
    expect(term?.classList.contains('disabled')).toBe(false);
  });
});

/**
 * A hand-built DockviewApi stand-in modeling the group bookkeeping the placement
 * engine relies on: an add with `direction: 'within'` joins the reference group
 * (dockview's stacking — the evicting 'replace' placement), anything else opens
 * a fresh group. Activation is tracked on both `activePanel` and `activeGroup`
 * (the anchor dockPanelBesideActive splits against) and fires the active-panel
 * listeners synchronously, as dockview does — that synchronicity is what makes
 * the echo guard work.
 * @returns {any}
 */
function makeDockApi() {
  let groupSeq = 0;
  /** @type {any} */
  const api = {
    activePanel: null,
    activeGroup: null,
    groups: /** @type {any[]} */ ([]),
    panels: /** @type {any[]} */ ([]),
    _added: /** @type {any[]} */ ([]),
    _activeCbs: /** @type {(() => void)[]} */ ([]),
    onDidLayoutChange: vi.fn(() => ({ dispose() {} })),
    // rail-drag.js wires these two; HTML5 rail drags are not under test here.
    onUnhandledDragOver: vi.fn(() => ({ dispose() {} })),
    onDidDrop: vi.fn(() => ({ dispose() {} })),
    onDidActivePanelChange: vi.fn((/** @type {() => void} */ cb) => {
      api._activeCbs.push(cb);
      return { dispose() {} };
    }),
    getPanel: (/** @type {string} */ id) =>
      api.panels.find((/** @type {any} */ p) => p.id === id) ?? null,
    addPanel: (/** @type {any} */ opts) => {
      api._added.push(opts);
      const group = opts.position?.referenceGroup && opts.position.direction === 'within'
        ? opts.position.referenceGroup
        : makeGroup();
      /** @type {any} */
      const panel = { id: opts.id, title: opts.title, group };
      panel.api = { setActive: () => activate(panel) };
      group.panels.push(panel);
      api.panels.push(panel);
      activate(panel);
      return panel;
    },
    removePanel: (/** @type {any} */ panel) => {
      api.panels = api.panels.filter((/** @type {any} */ p) => p !== panel);
      const group = panel.group;
      group.panels = group.panels.filter((/** @type {any} */ p) => p !== panel);
      if (group.panels.length === 0) {
        api.groups = api.groups.filter((/** @type {any} */ g) => g !== group);
      } else if (group.activePanel === panel) {
        group.activePanel = group.panels[0];
      }
      if (api.activePanel === panel) {
        const next = group.panels[0] ?? api.panels[0] ?? null;
        api.activePanel = next;
        api.activeGroup = next?.group ?? null;
        if (next) for (const cb of api._activeCbs) cb();
      }
    },
    /** The dock's own view of its layout — what serializeOpenTiles walks. */
    toJSON: () => ({
      grid: {
        root: {
          type: 'branch',
          data: api.groups.map((/** @type {any} */ g) => ({
            type: 'leaf',
            data: { views: g.panels.map((/** @type {any} */ p) => p.id) },
          })),
        },
      },
    }),
  };
  /** @param {any} panel */
  function activate(panel) {
    panel.group.activePanel = panel;
    api.activePanel = panel;
    api.activeGroup = panel.group;
    for (const cb of api._activeCbs) cb();
  }
  function makeGroup() {
    const element = document.createElement('div');
    const content = document.createElement('div');
    content.className = 'dv-content-container';
    element.appendChild(content);
    /** @type {any} */
    const group = { id: `group-${++groupSeq}`, panels: [], activePanel: null, element };
    api.groups.push(group);
    return group;
  }
  /** Seed the native terminal card in its own group (the first split's anchor). */
  api._addTerminal = () => {
    const group = makeGroup();
    /** @type {any} */
    const terminal = { id: 'terminal', group };
    terminal.api = { setActive: () => activate(terminal) };
    group.panels.push(terminal);
    api.panels.push(terminal);
    activate(terminal);
    return terminal;
  };
  return api;
}

/** Catalog config endpoints for the panels these suites boot (panel-catalog.js).
 *  @type {Record<string, string>} */
const CONFIG_ENDPOINT = {
  artifacts: '/api/artifact-server',
  ariel: '/api/ariel-server',
  'channel-finder': '/api/channel-finder-server',
  lattice: '/api/lattice-server',
};

/**
 * The service ids holding a dock tile, in the order the fake api added them —
 * a set membership check, NOT spatial order. Left-to-right placement is pinned
 * by the `position` an add carried, which is what dockview actually lays out on.
 */
const dockedTiles = (/** @type {any} */ api) =>
  api.panels
    .filter((/** @type {any} */ p) => p.id.startsWith('iframe:'))
    .map((/** @type {any} */ p) => p.id.slice('iframe:'.length));

/**
 * Boot the manager against a live dock shell. Every listed panel is enabled and
 * (unless named in `unhealthy`) resolves a url, which is what marks these
 * endpoint-less panels healthy; `visible` is the server-owned rail membership,
 * defaulting to all of them. The FIRST listed panel is the catalog default, so
 * boot docks its tile. Resolves once every panel's config has settled and that
 * first tile exists, i.e. past every activation the boot itself performs.
 * @param {{ panels?: string[], visible?: string[], unhealthy?: string[],
 *           mode?: 'simple'|'expert' }} [opts]
 */
async function bootWorkspace({ panels = ['artifacts', 'ariel'], visible, unhealthy = [], mode } = {}) {
  if (mode) document.documentElement.setAttribute('data-ui-mode', mode);
  window.__OSPREY_PREFIX__ = '';
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  // The overlay layer mounts in .main-container — without it the adapter
  // silently stays in fallback mode and no placeholder is ever created.
  // (No #dock-root: dock-sync's reverse half — a human dock gesture becoming a
  // POST — is its own module's contract, and leaving it unwired keeps the POST
  // assertions in these suites about panel-manager's own reporting.)
  document.body.innerHTML = `
    <div class="main-container">
      <nav id="panel-rail"></nav>
      <div id="panel-manager"><div id="panel-content"></div></div>
    </div>
  `;
  const members = visible ?? panels;
  /** @type {{url: string, opts: any}[]} */
  const calls = [];
  vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url, /** @type {any} */ o) => {
    calls.push({ url, opts: o });
    if (url === '/api/panels') {
      return jsonOk({
        enabled: panels,
        custom: [],
        default: null,
        visible: members,
        active: null,
        labels: {},
        // Simple mode boots chat-only while the workspace is empty; these
        // suites are about placement, so start past that onboarding state.
        workspace_has_artifacts: true,
      });
    }
    const id = panels.find((p) => CONFIG_ENDPOINT[p] === url);
    if (id) return jsonOk(unhealthy.includes(id) ? {} : { url: `/panel/${id}`, available: true });
    return jsonOk({ status: 'ok' });
  }));
  const { emit } = stubEventSource();

  const api = makeDockApi();
  api._addTerminal();
  dockState.api = api;

  const mod = await freshImport();
  await mod.initPanelManager('panel-manager');
  // The configEndpoint fetches settle after initPanelManager's own promise: a
  // healthy panel resolves its url on that settle, an unhealthy one never does
  // (wait on the fetch itself so both land before the frame under test).
  for (const id of panels) {
    if (unhealthy.includes(id)) {
      await vi.waitFor(() => expect(calls.some((c) => c.url === CONFIG_ENDPOINT[id])).toBe(true));
    } else {
      await vi.waitFor(() => expect(mod.getPanelStandaloneUrl(id)).toBe(`/panel/${id}`));
    }
  }
  await vi.waitFor(() => expect(api.getPanel(`iframe:${panels[0]}`)).not.toBeNull());
  calls.length = 0;
  return { api, emit, mod, calls };
}

/** POSTs the client sent — an applied SSE frame must never re-report focus. */
const posts = (/** @type {{url: string, opts: any}[]} */ calls) =>
  calls.filter((c) => c.opts?.method === 'POST').map((c) => c.url);

describe("agent open_panel (panel_focus source:'agent') — focus or open BESIDE, never evict", () => {
  afterEach(() => {
    document.documentElement.removeAttribute('data-ui-mode');
  });

  test('an undocked rail member opens as a NEW tile beside the active one — nothing is evicted', async () => {
    const { api, emit, calls } = await bootWorkspace();
    const artifactsTile = api.getPanel('iframe:artifacts');
    const artifactsGroup = artifactsTile.group;

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    // The operator's tile survives, untouched (same panel object, same group).
    expect(api.getPanel('iframe:artifacts')).toBe(artifactsTile);
    expect(artifactsTile.group).toBe(artifactsGroup);
    // ariel arrived as its own tile, split off the active group.
    const added = api._added.filter((/** @type {any} */ o) => o.id === 'iframe:ariel');
    expect(added).toHaveLength(1);
    expect(added[0].position).toMatchObject({ referenceGroup: artifactsGroup, direction: 'right' });
    expect(api.getPanel('iframe:ariel').group).not.toBe(artifactsGroup);
    // ...and holds the focus, without echoing it back to the server.
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('ariel');
    expect(posts(calls)).toEqual([]);
  });

  test('a panel that already holds a tile is only focused — no second tile, no move', async () => {
    const { api, emit, calls } = await bootWorkspace();
    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });
    const arielTile = api.getPanel('iframe:ariel');
    const artifactsTile = api.getPanel('iframe:artifacts');
    const addsBefore = api._added.length;

    emit({ type: 'panel_focus', panel: 'artifacts', source: 'agent' });

    expect(api._added).toHaveLength(addsBefore);
    expect(api.getPanel('iframe:artifacts')).toBe(artifactsTile);
    expect(api.getPanel('iframe:ariel')).toBe(arielTile);
    expect(api.activePanel).toBe(artifactsTile);
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('artifacts');
    expect(posts(calls)).toEqual([]);
  });

  test('a NON-member gains its rail entry first, then opens beside', async () => {
    const { api, emit, mod, calls } = await bootWorkspace({ visible: ['artifacts'] });
    expect(document.querySelector('.panel-rail-button[data-panel-id="ariel"]')).toBeNull();
    const artifactsTile = api.getPanel('iframe:artifacts');

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    const entry = document.querySelector('.panel-rail-button[data-panel-id="ariel"]');
    expect(entry).not.toBeNull();
    expect(entry?.classList.contains('disabled')).toBe(false);
    expect(mod.getHiddenPanels().map((p) => p.id)).not.toContain('ariel');
    expect(api.getPanel('iframe:artifacts')).toBe(artifactsTile);
    expect(api.getPanel('iframe:ariel')).not.toBeNull();
    // Membership is APPLIED locally, never reported: this is the optimistic
    // half of a decision the server makes too, and its visibility echo lands on
    // the entry already here (see the idempotency case below).
    expect(posts(calls)).toEqual([]);
  });

  test('an UNHEALTHY target opens no tile — a tile it could never fill would linger empty', async () => {
    const { api, emit } = await bootWorkspace({ unhealthy: ['ariel'] });
    const addsBefore = api._added.length;

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    expect(api.getPanel('iframe:ariel')).toBeNull();
    expect(api._added).toHaveLength(addsBefore);
    // The activation refuses too, exactly as before — artifacts keeps the focus.
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('artifacts');
  });

  test('an unknown panel id touches neither the rail nor the grid', async () => {
    const { api, emit } = await bootWorkspace();
    const addsBefore = api._added.length;

    emit({ type: 'panel_focus', panel: 'no-such-panel', source: 'agent' });

    expect(api._added).toHaveLength(addsBefore);
    expect(api.getPanel('iframe:no-such-panel')).toBeNull();
    expect(document.querySelector('.panel-rail-button[data-panel-id="no-such-panel"]')).toBeNull();
  });

  test('a layout that cannot be serialized still opens BESIDE — never a takeover', async () => {
    const { api, emit } = await bootWorkspace();
    const artifactsTile = api.getPanel('iframe:artifacts');
    const artifactsGroup = artifactsTile.group;
    // Mid-rebuild, or an api that refuses toJSON: occupancy is unreadable from
    // the serialized layout even though a tile is plainly on screen. Reading it
    // as "no tiles" would send this switch down the replacing path.
    api.toJSON = () => {
      throw new Error('layout mid-rebuild');
    };

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    expect(api.getPanel('iframe:artifacts')).toBe(artifactsTile);
    expect(artifactsTile.group).toBe(artifactsGroup);
    expect(api.getPanel('iframe:ariel')).not.toBeNull();
    expect(api.getPanel('iframe:ariel').group).not.toBe(artifactsGroup);
  });

  test('with EVERY service tile retired, the switch re-anchors left of the terminal', async () => {
    const { api, emit } = await bootWorkspace({ visible: ['artifacts'] });
    // Retire the only service tile the way a hide does, leaving the dock holding
    // just the terminal — the state where "beside the active group" would mean
    // "beside the terminal".
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });
    expect(dockedTiles(api)).toEqual([]);
    expect(api.activeGroup).toBe(api.getPanel('terminal').group);

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    // The first tile rule wins: left of the terminal at the classic width, not
    // a default-width split off the terminal's own group.
    const added = api._added.filter((/** @type {any} */ o) => o.id === 'iframe:ariel');
    expect(added).toHaveLength(1);
    expect(added[0].position).toMatchObject({ referencePanel: 'terminal', direction: 'left' });
    expect(added[0].initialWidth).toBeGreaterThan(0);
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('ariel');
  });

  test("the switch's membership add is idempotent with the server's own visibility echo", async () => {
    const { emit } = await bootWorkspace({ visible: ['artifacts'] });

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });
    const added = document.querySelector('.panel-rail-button[data-panel-id="ariel"]');
    expect(added).not.toBeNull();

    // The focus route also appends the non-member server-side and broadcasts a
    // visibility frame; arriving after the optimistic local add, it must leave
    // the very same entry in place rather than rebuild or duplicate it.
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });

    const entries = document.querySelectorAll('.panel-rail-button[data-panel-id="ariel"]');
    expect(entries).toHaveLength(1);
    expect(entries[0]).toBe(added);
    expect(entries[0].classList.contains('disabled')).toBe(false);
  });

  test('an untagged (human-origin) focus echo keeps the evicting takeover', async () => {
    const { api, emit } = await bootWorkspace();
    const artifactsTile = api.getPanel('iframe:artifacts');
    const artifactsGroup = artifactsTile.group;

    emit({ type: 'panel_focus', panel: 'ariel' });

    // Rail-click semantics: ariel takes the tile over and artifacts is evicted.
    expect(api.getPanel('iframe:artifacts')).toBeNull();
    expect(api.getPanel('iframe:ariel').group).toBe(artifactsGroup);
  });

  test('simple mode keeps the single-tile takeover (the placement verb no-ops there)', async () => {
    const { api, emit } = await bootWorkspace({ mode: 'simple' });
    const artifactsGroup = api.getPanel('iframe:artifacts').group;

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    expect(api.getPanel('iframe:artifacts')).toBeNull();
    expect(api.getPanel('iframe:ariel').group).toBe(artifactsGroup);
    expect(api.panels.filter((/** @type {any} */ p) => p.id.startsWith('iframe:'))).toHaveLength(1);
  });

  test('fallback mode (no dock shell) activates exactly as before — no focus re-POST', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url, /** @type {any} */ o) => {
      calls.push({ url, opts: o });
      if (url === '/api/panels') {
        return jsonOk({
          enabled: ['artifacts', 'ariel'], custom: [], default: null,
          visible: ['artifacts'], active: null, labels: {},
        });
      }
      if (url === '/api/artifact-server') return jsonOk({ url: '/panel/artifacts', available: true });
      if (url === '/api/ariel-server') return jsonOk({ url: '/panel/ariel', available: true });
      return jsonOk({ status: 'ok' });
    }));
    const { emit } = stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    await vi.waitFor(() => expect(mod.getPanelStandaloneUrl('ariel')).toBe('/panel/ariel'));
    calls.length = 0;

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('ariel');
    expect(document.querySelector('.panel-rail-button[data-panel-id="ariel"]')).not.toBeNull();
    expect(posts(calls)).toEqual([]);
  });
});

describe('panel_arrange — the declarative whole-workspace rebuild', () => {
  afterEach(() => {
    document.documentElement.removeAttribute('data-ui-mode');
  });

  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);
  const activeStamp = () => document.getElementById('panel-manager')?.dataset.activePanel ?? null;

  const THREE = ['artifacts', 'ariel', 'channel-finder'];

  test('converges to exactly the requested tiles, left to right', async () => {
    const { api, emit } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'] });

    expect(dockedTiles(api)).toEqual(['channel-finder', 'ariel']);
    // The first tile splits against the grid's left edge (services sit left of
    // the terminal); each next one lands to the RIGHT of the one before it.
    const adds = api._added.filter((/** @type {any} */ o) => o.id.startsWith('iframe:'));
    const cf = adds.at(-2);
    const ariel = adds.at(-1);
    expect(cf).toMatchObject({ id: 'iframe:channel-finder', position: { direction: 'left' } });
    expect(ariel.position).toMatchObject({
      referenceGroup: api.getPanel('iframe:channel-finder').group,
      direction: 'right',
    });
  });

  test('converges from a DIFFERENT start layout to the same end state', async () => {
    const { api, emit } = await bootWorkspace({ panels: THREE });
    // Start with all three open in a different order than the arrangement asks
    // for, so the rebuild has both a tile to drop and tiles to reorder.
    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });
    emit({ type: 'panel_focus', panel: 'channel-finder', source: 'agent' });
    expect(dockedTiles(api).sort()).toEqual(['ariel', 'artifacts', 'channel-finder']);

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'] });

    expect(dockedTiles(api)).toEqual(['channel-finder', 'ariel']);
    // Same end state as the clean-start case, positions included: the region is
    // rebuilt, not diffed, so a differing start layout cannot leave order behind.
    expect(api._added.at(-1).position).toMatchObject({
      referenceGroup: api.getPanel('iframe:channel-finder').group,
      direction: 'right',
    });
    // The dropped tile keeps its rail membership — a tile close is not a hide.
    expect(entry('artifacts')).not.toBeNull();
  });

  test('the requested focus wins when healthy; the tiles all get their iframes', async () => {
    const { api, emit } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'], focus: 'ariel' });

    expect(activeStamp()).toBe('ariel');
    expect(api.getPanel('iframe:ariel').group.activePanel.id).toBe('iframe:ariel');
    // Every arranged tile is surfaced, not just the focused one: a docked tile
    // whose panel was never activated would show an empty placeholder.
    for (const id of ['channel-finder', 'ariel']) {
      expect(document.querySelector(`iframe[data-panel-id="${id}"]`)).not.toBeNull();
    }
  });

  test('an unhealthy focus target falls back to the first healthy listed tile', async () => {
    const { emit } = await bootWorkspace({ panels: THREE, unhealthy: ['channel-finder'] });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'], focus: 'channel-finder' });

    expect(activeStamp()).toBe('ariel');
  });

  test('no healthy tile at all leaves focus unchanged rather than stranding it', async () => {
    const { emit } = await bootWorkspace({ panels: THREE, unhealthy: ['ariel', 'channel-finder'] });
    expect(activeStamp()).toBe('artifacts');

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'], focus: 'channel-finder' });

    // artifacts lost its tile to the rebuild, so the accent it held is dropped;
    // nothing healthy remains to take it, and no unhealthy panel is surfaced.
    expect(activeStamp()).toBeNull();
    expect(document.querySelector('.panel-rail-button.active')).toBeNull();
    // ...and the pane is painted empty, the same strand-proof ending the
    // visibility channel gives when its last usable panel goes away — a
    // cleared accent over a stale surface would be the worse half of both.
    expect(document.querySelector('.artifacts-empty-state')).not.toBeNull();
  });

  test('an UNHEALTHY listed panel gets no tile — only its rail entry', async () => {
    const { api, emit } = await bootWorkspace({ panels: THREE, unhealthy: ['channel-finder'] });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'] });

    // A tile it could never fill would sit empty for as long as anything else
    // holds focus, and nothing comes back for it.
    expect(api.getPanel('iframe:channel-finder')).toBeNull();
    expect(api._added.filter((/** @type {any} */ o) => o.id === 'iframe:channel-finder')).toHaveLength(0);
    expect(dockedTiles(api)).toEqual(['ariel']);
    // Membership still applied, so it stays one rail click away.
    expect(entry('channel-finder')).not.toBeNull();
  });

  test('a skipped tile never becomes the anchor the next one splits against', async () => {
    const { api, emit } = await bootWorkspace({ panels: THREE, unhealthy: ['channel-finder'] });

    // The unhealthy panel is listed FIRST, so a naive walk would leave `prev`
    // pointing at a tile that does not exist and split ariel off the terminal.
    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'] });

    const added = api._added.filter((/** @type {any} */ o) => o.id === 'iframe:ariel').at(-1);
    expect(added.position).toEqual({ direction: 'left' });
    expect(api.getPanel('iframe:ariel').group).not.toBe(api.getPanel('terminal').group);
  });

  test('a tile still on screen is never painted over as an empty workspace', async () => {
    // The one reachable route to "docked but UNHEALTHY": the rail's ⊞ corner
    // docks a tile regardless of health (its activation then refuses), which is
    // the state the rebuild deliberately leaves on screen.
    const { api, emit } = await bootWorkspace({
      panels: ['artifacts', 'ariel'],
      unhealthy: ['ariel'],
    });
    const beside = /** @type {HTMLElement} */ (
      document.querySelector('[data-panel-id="ariel"] .panel-rail-beside')
    );
    beside.click();
    expect(dockedTiles(api).sort()).toEqual(['ariel', 'artifacts']);
    expect(activeStamp()).toBe('artifacts'); // the unhealthy panel took no focus

    // Arrange onto the unhealthy panel alone: nothing listed can be focused, and
    // the panel that held the view (artifacts) is not listed.
    emit({ type: 'panel_arrange', tiles: ['ariel'] });

    // Its existing tile survives the rebuild...
    expect(dockedTiles(api)).toEqual(['ariel']);
    // ...so the accent clears, but the pane is NOT painted empty over it.
    expect(activeStamp()).toBeNull();
    expect(document.querySelector('.artifacts-empty-state')).toBeNull();
  });

  test('a listed NON-member is added to the rail; a plain tiles request removes nobody', async () => {
    const { emit, mod } = await bootWorkspace({ panels: THREE, visible: ['artifacts'] });
    expect(entry('ariel')).toBeNull();

    emit({ type: 'panel_arrange', tiles: ['ariel'] });

    expect(entry('ariel')).not.toBeNull();
    // Membership is only ADDED: artifacts is not in the arrangement but keeps
    // its rail entry (the precedence split against panel_visibility).
    expect(entry('artifacts')).not.toBeNull();
    expect(mod.getHiddenPanels().map((p) => p.id)).toEqual(['channel-finder']);
  });

  test('prune_rail (the preset path) makes membership exactly the arranged tiles', async () => {
    const { api, emit, mod } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['ariel'], prune_rail: true });

    expect(entry('ariel')).not.toBeNull();
    expect(entry('artifacts')).toBeNull();
    expect(entry('channel-finder')).toBeNull();
    expect(mod.getHiddenPanels().map((p) => p.id).sort()).toEqual(['artifacts', 'channel-finder']);
    expect(dockedTiles(api)).toEqual(['ariel']);
  });

  test('applying an arrangement reports nothing back to the server', async () => {
    const { emit, calls } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'], focus: 'ariel', source: 'agent' });

    expect(posts(calls)).toEqual([]);
  });

  test("source:'agent' glows the arranged entries transiently, without the badge", async () => {
    const { emit } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'], source: 'agent' });

    expect(entry('ariel')?.classList.contains('agent-flash')).toBe(true);
    expect(entry('channel-finder')?.classList.contains('agent-flash')).toBe(true);
    expect(entry('ariel')?.classList.contains('agent-attention')).toBe(false);
    expect(entry('artifacts')?.classList.contains('agent-flash')).toBe(false);
  });

  test('unknown ids are dropped rather than arranged as ghost tiles', async () => {
    const { api, emit } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['ariel', 'no-such-panel'] });

    expect(dockedTiles(api)).toEqual(['ariel']);
    expect(entry('no-such-panel')).toBeNull();
  });

  test('simple mode skips the rebuild and takes the single tile over', async () => {
    const { api, emit } = await bootWorkspace({ panels: THREE, mode: 'simple' });
    expect(dockedTiles(api)).toEqual(['artifacts']);

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'], focus: 'ariel' });

    // One service tile by construction: the focus target takes it over, and no
    // second tile is opened beside it.
    expect(dockedTiles(api)).toEqual(['ariel']);
    expect(activeStamp()).toBe('ariel');
    // Membership still applies in simple mode — the rail is not a dock feature.
    expect(entry('channel-finder')).not.toBeNull();
  });

  test('fallback mode (no dock shell) applies membership and focus only', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url, /** @type {any} */ o) => {
      calls.push({ url, opts: o });
      if (url === '/api/panels') {
        return jsonOk({
          enabled: ['artifacts', 'ariel'], custom: [], default: null,
          visible: ['artifacts'], active: null, labels: {},
        });
      }
      if (url === '/api/artifact-server') return jsonOk({ url: '/panel/artifacts', available: true });
      if (url === '/api/ariel-server') return jsonOk({ url: '/panel/ariel', available: true });
      return jsonOk({ status: 'ok' });
    }));
    const { emit } = stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    await vi.waitFor(() => expect(mod.getPanelStandaloneUrl('ariel')).toBe('/panel/ariel'));
    calls.length = 0;

    emit({ type: 'panel_arrange', tiles: ['ariel'], focus: 'ariel', prune_rail: true });

    expect(entry('ariel')).not.toBeNull();
    expect(entry('artifacts')).toBeNull();  // prune_rail holds without a dock
    expect(activeStamp()).toBe('ariel');
    expect(posts(calls)).toEqual([]);
  });
});

describe('SSE reconnect resync — membership re-converges from /api/panels', () => {
  /**
   * Boot against a MUTABLE server state and an EventSource stub whose
   * `open()` drives the onopen handler — the reconnect seam. SSE has no
   * replay, so a frame published while a client was disconnected is simply
   * gone; on every open panel-manager must re-fetch /api/panels and apply
   * the authoritative visible set as a delta.
   */
  async function bootResync() {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    const server = { visible: ['artifacts', 'ariel'] };
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({
          enabled: ['artifacts', 'ariel'],
          custom: [],
          default: null,
          visible: [...server.visible],
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

    /** @type {{ onopen?: (() => void) | null }[]} */
    const sources = [];
    class FakeEventSource {
      constructor() {
        /** @type {(() => void) | null} */
        this.onopen = null;
        sources.push(this);
      }
      close() {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    return { mod, server, open: () => { for (const s of sources) s.onopen?.(); } };
  }

  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);

  test('a hide missed while disconnected is applied on reconnect', async () => {
    const { mod, server, open } = await bootResync();
    expect(entry('ariel')).not.toBeNull();

    // The agent hid ariel while this client's SSE stream was down: the
    // panel_visibility frame never arrived. The stream then reconnects.
    server.visible = ['artifacts'];
    open();

    await vi.waitFor(() => expect(entry('ariel')).toBeNull());
    expect(mod.getHiddenPanels().map((p) => p.id)).toContain('ariel');
  });

  test('a show missed while disconnected is applied on reconnect', async () => {
    const { server, open } = await bootResync();
    server.visible = ['artifacts'];
    open();
    await vi.waitFor(() => expect(entry('ariel')).toBeNull());

    server.visible = ['artifacts', 'ariel'];
    open();

    await vi.waitFor(() => expect(entry('ariel')).not.toBeNull());
  });

  test('an in-sync reconnect changes nothing', async () => {
    const { open } = await bootResync();
    const before = [...document.querySelectorAll('.panel-rail-button')].map(
      (b) => b.getAttribute('data-panel-id'),
    );

    open();
    await new Promise((r) => setTimeout(r, 0));

    const after = [...document.querySelectorAll('.panel-rail-button')].map(
      (b) => b.getAttribute('data-panel-id'),
    );
    expect(after).toEqual(before);
  });
});

describe('tile-body glow — fired only where a tile visibly changed', () => {
  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);
  /** The panel ids handed to glowPanel, in call order. */
  const glowed = () => glowPanelSpy.mock.calls.map((/** @type {any[]} */ c) => c[0]);
  const THREE = ['artifacts', 'ariel', 'channel-finder'];

  test("an agent panel_focus glows the switched panel's tile", async () => {
    const { emit } = await bootWorkspace();

    emit({ type: 'panel_focus', panel: 'ariel', source: 'agent' });

    expect(glowed()).toEqual(['ariel']);
  });

  test('a human (untagged) panel_focus glows nothing — the operator did it', async () => {
    const { emit } = await bootWorkspace();

    emit({ type: 'panel_focus', panel: 'ariel' });

    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  test('an agent panel_arrange glows every arranged tile', async () => {
    const { emit } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'], source: 'agent' });

    expect(glowed().sort()).toEqual(['ariel', 'channel-finder']);
    // A panel the arrangement did not list keeps its tile out of the story.
    expect(glowed()).not.toContain('artifacts');
  });

  test('an untagged panel_arrange (human Layouts click) glows nothing', async () => {
    const { emit } = await bootWorkspace({ panels: THREE });

    emit({ type: 'panel_arrange', tiles: ['channel-finder', 'ariel'] });

    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  test('an agent SHOW glows the rail entry only — no tile exists to attribute to', async () => {
    const { emit } = await bootWorkspace({ visible: ['artifacts'] });

    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });

    expect(entry('ariel')?.classList.contains('agent-flash')).toBe(true);
    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  test("an agent HIDE glows nothing — the tile is on its way out", async () => {
    const { emit, mod } = await bootWorkspace();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false, source: 'agent' });

    expect(strip).toHaveBeenCalledTimes(1); // the agent branch DID run
    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  test('an agent panel_register glows the new rail entry only — it opens no tile', async () => {
    const { emit } = await bootWorkspace();

    emit({
      type: 'panel_register', id: 'scan', label: 'SCAN', url: '/panel/scan',
      healthEndpoint: null, path: '/', source: 'agent',
    });

    expect(entry('scan')?.classList.contains('agent-flash')).toBe(true);
    expect(glowPanelSpy).not.toHaveBeenCalled();
  });
});

describe('agent-attention badges survive a reload — acknowledged by server ts', () => {
  const ACK = 'agent-ack:';

  // The ack store is real localStorage and outlives a module reset, so every
  // case starts from a page that has acknowledged nothing.
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  /**
   * Boot with a healthy 'artifacts' panel, an unhealthy-but-present 'ariel'
   * entry, and a MUTABLE history ring behind /api/agent-activity/recent. Both
   * SSE seams are exposed: `emit` for frames, and `open` for the hook that
   * re-reads the ring (a reload's first open and every reconnect run it).
   * @param {{events?: any[]}} [opts]
   */
  async function bootAck({ events = [] } = {}) {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    const ring = { events };
    const reads = { recent: 0 };
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({ enabled: ['artifacts', 'ariel'], custom: [], default: null, visible: ['artifacts', 'ariel'], active: null, labels: {} });
      }
      if (url === '/api/artifact-server') return jsonOk({ url: '/panel/artifacts', available: true });
      if (url === '/api/agent-activity/recent') {
        reads.recent += 1;
        return jsonOk({ events: [...ring.events] });
      }
      return jsonOk({ status: 'ok' });  // ariel: no url ⇒ entry present, disabled
    }));

    /** @type {{ onmessage?: ((e: {data: string}) => void) | null, onopen?: (() => void) | null }[]} */
    const sources = [];
    class FakeEventSource {
      constructor() {
        /** @type {((e: {data: string}) => void) | null} */
        this.onmessage = null;
        /** @type {(() => void) | null} */
        this.onopen = null;
        sources.push(this);
      }
      close() {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    const artifacts = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]'));
    await vi.waitFor(() => expect(artifacts.classList.contains('disabled')).toBe(false));
    return {
      mod, ring, artifacts,
      /** @param {object} frame */
      emit: (frame) => { for (const s of sources) s.onmessage?.({ data: JSON.stringify(frame) }); },
      open: () => { for (const s of sources) s.onopen?.(); },
      // Wait until the open() hook has actually READ the ring, then flush the
      // handler that consumes it. A "no badge appeared" assertion is only
      // evidence once the restore has genuinely run.
      settle: async () => {
        await vi.waitFor(() => expect(reads.recent).toBeGreaterThan(0));
        await new Promise((r) => setTimeout(r, 0));
      },
    };
  }

  /** @param {string} id */
  const badged = (id) =>
    !!document.querySelector(`[data-panel-id="${id}"]`)?.classList.contains('agent-attention');

  test("surfacing a badged panel persists that badge's SERVER ts as the ack", async () => {
    const { emit } = await bootAck();

    emit({ type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'artifacts' }, ts: 1000.5 });
    expect(badged('artifacts')).toBe(true);

    emit({ type: 'panel_focus', panel: 'artifacts' });

    expect(badged('artifacts')).toBe(false);
    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('1000.5');
  });

  test('no ack ⇒ an unseen ring event restores the badge on SSE open', async () => {
    const { open } = await bootAck({
      events: [{ type: 'agent_activity', tool: 'search_logbook', target: { kind: 'panel', panel: 'ariel' }, ts: 1000 }],
    });
    expect(badged('ariel')).toBe(false);

    open();

    await vi.waitFor(() => expect(badged('ariel')).toBe(true));
  });

  test('a ring event at or before the ack does NOT resurrect the badge', async () => {
    localStorage.setItem(`${ACK}ariel`, '1000');
    const { open, settle } = await bootAck({
      events: [
        { type: 'agent_activity', tool: 'search_logbook', target: { kind: 'panel', panel: 'ariel' }, ts: 1000 },
        { type: 'agent_activity', tool: 'search_logbook', target: { kind: 'panel', panel: 'ariel' }, ts: 999 },
      ],
    });

    open();
    await settle();

    expect(badged('ariel')).toBe(false);
  });

  test('a ring event after the ack restores the badge', async () => {
    localStorage.setItem(`${ACK}ariel`, '1000');
    const { open } = await bootAck({
      events: [{ type: 'agent_activity', tool: 'search_logbook', target: { kind: 'panel', panel: 'ariel' }, ts: 1000.5 }],
    });

    open();

    await vi.waitFor(() => expect(badged('ariel')).toBe(true));
  });

  test('non-panel-kind ring rows never badge the rail', async () => {
    const { open, settle } = await bootAck({
      events: [
        { type: 'agent_activity', tool: 'read_channel', target: { kind: 'channel', detail: 'SR01C:BPM1:X' }, ts: 2000 },
        { type: 'agent_activity', tool: 'run_scan', target: { kind: 'run', detail: 'orm-3' }, ts: 1999 },
        // A panel-kind row with no rail entry has nothing to badge either.
        { type: 'agent_activity', tool: 'open_panel', target: { kind: 'panel', panel: 'no-such-panel' }, ts: 1998 },
      ],
    });

    open();
    await settle();

    expect(document.querySelector('.agent-attention')).toBeNull();
  });

  test('the newest of several rows for one panel becomes the ack', async () => {
    const { open, emit } = await bootAck({
      events: [  // newest first, as the endpoint serves them
        { type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'artifacts' }, ts: 30 },
        { type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'artifacts' }, ts: 10 },
      ],
    });

    open();
    await vi.waitFor(() => expect(badged('artifacts')).toBe(true));
    emit({ type: 'panel_focus', panel: 'artifacts' });

    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('30');
  });

  test('a clear with no badge ts on record leaves the stored ack untouched', async () => {
    localStorage.setItem(`${ACK}artifacts`, '1000');
    const { emit } = await bootAck();  // boot surfaces artifacts — an unbadged clear

    emit({ type: 'panel_focus', panel: 'artifacts' });

    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('1000');
  });

  test('the ack is the server ts even when the browser clock is far ahead', async () => {
    // A client clock written as an ack would be ~2e12 here and would swallow
    // every future server ts — badges would never come back after one clear.
    vi.spyOn(Date, 'now').mockReturnValue(2_000_000_000_000);
    const { emit, ring, open } = await bootAck();

    emit({ type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'artifacts' }, ts: 5 });
    emit({ type: 'panel_focus', panel: 'artifacts' });
    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('5');

    // Reconnect with a newer server event: still strictly greater than the ack.
    ring.events = [{ type: 'agent_activity', tool: 'read_file', target: { kind: 'panel', panel: 'artifacts' }, ts: 6 }];
    open();

    await vi.waitFor(() => expect(badged('artifacts')).toBe(true));
  });
});
