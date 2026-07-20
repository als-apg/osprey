/**
 * Contract tests for the dock↔server state-sync bridge (dock-sync.js):
 *   npx vitest run tests/interfaces/web_terminal/dock-sync.test.mjs
 *
 * dock-sync carries the REVERSE half of the panel-state loop: it turns a human's
 * dockview gestures (focusing a tab, closing a tab) back into the same server
 * POSTs an agent MCP call would make, WITHOUT letting the server's own SSE echo
 * bounce back out again. The forward half (server → dock application, and the
 * simple-mode layout synthesis) lives in dock-workspace.js / panel-manager.js and
 * is pinned by the reconcile tests (dock-reconcile.test.mjs) plus the browser
 * suite; this file does not duplicate them. Here we pin dock-sync's own contract:
 *
 *   - the ECHO GUARD — a genuine human focus POSTs exactly once, a server-applied
 *     (echo) focus POSTs never;
 *   - the human tab-CLOSE click POSTs a visibility=false;
 *   - dockPanelBesideActive appends a placeholder beside the active group under
 *     the same guard;
 *   - the listeners wire idempotently and tolerate late DockviewApi arrival.
 *
 * dockview and the two collaborators are stubbed at the module boundary: getDockApi
 * (dock-workspace.js) yields a hand-built fake api, and setPanelFocus /
 * setPanelVisibility (panel-commands.js) are spies. Each test re-imports dock-sync
 * fresh (vi.resetModules) so its module-scoped guard/wire state never leaks across
 * tests.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

const SYNC = '../../../src/osprey/interfaces/web_terminal/static/js/dock-sync.js';

// Boundary stubs. Hoisted so the spies exist before dock-sync's static imports
// resolve; the same spy instances persist across resetModules (they are closed
// over here), while dock-sync's own state is rebuilt on each fresh import.
const { getDockApi, setPanelFocus, setPanelVisibility, state } = vi.hoisted(() => ({
  state: { api: /** @type {any} */ (null) },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
  setPanelFocus: vi.fn(),
  setPanelVisibility: vi.fn(),
}));
// getDockApi reads the mutable holder so a test can swap the live api in place.
getDockApi.mockImplementation(() => state.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  // dock-iframe.js imports these from dock-workspace; the mock must supply
  // them or the transitive import chain resolves them to undefined.
  // (PLACEHOLDER_PREFIX comes from dock-reconcile.js, a pure module that
  // loads for real — no mock entry needed.)
  defaultServiceWidth: () => 600,
  setServiceRedock: () => {},
  onDragGesture: () => [],
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-commands.js', () => ({
  setPanelFocus,
  setPanelVisibility,
}));

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.api = null;
  document.body.innerHTML = '';
});

/**
 * A hand-built stand-in for dockview's DockviewApi exposing only the surface
 * dock-sync touches. `fireActive()` invokes the onDidActivePanelChange listener
 * dock-sync registered (dockview fires it synchronously for both a human tab
 * click and a programmatic setActive — the ambiguity the echo guard resolves).
 * @returns {any}
 */
function makeApi() {
  /** @type {any} */
  const api = {
    activePanel: null,
    activeGroup: null,
    groups: [],
    _panels: /** @type {Record<string, any>} */ ({}),
    _added: /** @type {any[]} */ ([]),
    /** Panel dockview auto-activates when the active one is removed (null = none). */
    _activateOnRemove: /** @type {any} */ (null),
    /** @type {null | (() => void)} */
    _activeCb: null,
    fireActive() {
      if (api._activeCb) api._activeCb();
    },
  };
  api.onDidActivePanelChange = vi.fn((/** @type {() => void} */ cb) => {
    api._activeCb = cb;
    return { dispose() {} };
  });
  api.getPanel = vi.fn((/** @type {string} */ id) => api._panels[id] ?? null);
  api.addPanel = vi.fn((/** @type {any} */ opts) => {
    api._added.push(opts);
  });
  // dockview removes the panel; if a stacked neighbor exists it becomes active and
  // onDidActivePanelChange fires SYNCHRONOUSLY — the echo the guard must cover.
  api.removePanel = vi.fn(() => {
    if (api._activateOnRemove) {
      api.activePanel = api._activateOnRemove;
      api.fireActive();
    }
  });
  return api;
}

/**
 * Put a #dock-root in the DOM, publish `api` as the live DockviewApi, import a
 * fresh dock-sync, and wire it. Returns the module + root for follow-on asserts.
 * @param {any} api
 */
async function wire(api) {
  state.api = api;
  const root = document.createElement('div');
  root.id = 'dock-root';
  document.body.appendChild(root);
  const mod = await import(SYNC);
  mod.initDockSync();
  return { mod, root };
}

/**
 * Build a dockview-shaped group DOM (one `.dv-groupview` holding a tab strip of
 * `.dv-tab` elements, each with the `.dv-default-tab-action` close control) under
 * `root`, and register a matching fake group on `api` whose ordered `panels` line
 * up by index with the rendered tabs — the mapping panelIdForTab relies on.
 * @param {HTMLElement} root
 * @param {any} api
 * @param {string[]} panelIds
 */
function buildGroup(root, api, panelIds) {
  const groupview = document.createElement('div');
  groupview.className = 'dv-groupview';
  const strip = document.createElement('div');
  strip.className = 'dv-tabs-container';
  /** @type {{tab: HTMLElement, action: HTMLElement}[]} */
  const tabs = [];
  for (let i = 0; i < panelIds.length; i++) {
    const tab = document.createElement('div');
    tab.className = 'dv-tab';
    const action = document.createElement('div');
    action.className = 'dv-default-tab-action';
    tab.appendChild(action);
    strip.appendChild(tab);
    tabs.push({ tab, action });
  }
  groupview.appendChild(strip);
  root.appendChild(groupview);
  api.groups = [{ element: groupview, panels: panelIds.map((id) => ({ id })) }];
  return { groupview, tabs };
}

describe('serviceIdOf — service-placeholder id extraction', () => {
  test('strips the iframe: prefix from a service placeholder id', async () => {
    const { serviceIdOf } = await import(SYNC);
    expect(serviceIdOf('iframe:ariel')).toBe('ariel');
    expect(serviceIdOf('iframe:bluesky-scan')).toBe('bluesky-scan');
  });

  test('returns null for the native (non-placeholder) panels and nullish ids', async () => {
    const { serviceIdOf } = await import(SYNC);
    expect(serviceIdOf('terminal')).toBeNull();
    expect(serviceIdOf('workspace')).toBeNull();
    expect(serviceIdOf('')).toBeNull();
    expect(serviceIdOf(null)).toBeNull();
    expect(serviceIdOf(undefined)).toBeNull();
    expect(serviceIdOf(42)).toBeNull();
  });
});

describe('withEchoSuppressed — the guard primitive', () => {
  test('returns the callback result and runs it exactly once', async () => {
    const { withEchoSuppressed } = await import(SYNC);
    const fn = vi.fn(() => 42);
    expect(withEchoSuppressed(fn)).toBe(42);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('restores the guard even when the callback throws (finally)', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    expect(() => mod.withEchoSuppressed(() => {
      throw new Error('boom');
    })).toThrow('boom');

    // Guard is back to 0, so a subsequent genuine focus still POSTs.
    api.activePanel = { id: 'iframe:ariel' };
    api.fireActive();
    expect(setPanelFocus).toHaveBeenCalledExactlyOnceWith('ariel');
  });
});

describe('echo guard — active-panel change routing (core invariant)', () => {
  test('a human focus of a service panel POSTs setPanelFocus exactly once', async () => {
    const api = makeApi();
    await wire(api);

    api.activePanel = { id: 'iframe:ariel' };
    api.fireActive();

    expect(setPanelFocus).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(setPanelVisibility).not.toHaveBeenCalled();
  });

  test('each distinct human focus POSTs once — no coalescing, no duplication', async () => {
    const api = makeApi();
    await wire(api);

    api.activePanel = { id: 'iframe:ariel' };
    api.fireActive();
    api.activePanel = { id: 'iframe:bluesky' };
    api.fireActive();

    expect(setPanelFocus).toHaveBeenCalledTimes(2);
    expect(setPanelFocus).toHaveBeenNthCalledWith(1, 'ariel');
    expect(setPanelFocus).toHaveBeenNthCalledWith(2, 'bluesky');
  });

  test('focusing a native (terminal/workspace) panel never POSTs', async () => {
    const api = makeApi();
    await wire(api);

    api.activePanel = { id: 'terminal' };
    api.fireActive();
    api.activePanel = { id: 'workspace' };
    api.fireActive();

    expect(setPanelFocus).not.toHaveBeenCalled();
  });

  test('a nullish active panel never POSTs', async () => {
    const api = makeApi();
    await wire(api);

    api.activePanel = null;
    api.fireActive();

    expect(setPanelFocus).not.toHaveBeenCalled();
  });

  test('a server-APPLIED focus (fired inside withEchoSuppressed) never POSTs', async () => {
    const api = makeApi();
    const { mod } = await wire(api);

    // This is exactly how panel-manager applies a server-driven focus: it sets the
    // active panel programmatically inside the guard, and dockview echoes an active
    // change synchronously within that window.
    mod.withEchoSuppressed(() => {
      api.activePanel = { id: 'iframe:ariel' };
      api.fireActive();
    });

    expect(setPanelFocus).not.toHaveBeenCalled();
  });

  test('the guard is depth-counted: nested suppression still suppresses, and lifts fully afterward', async () => {
    const api = makeApi();
    const { mod } = await wire(api);

    mod.withEchoSuppressed(() => {
      mod.withEchoSuppressed(() => {
        api.activePanel = { id: 'iframe:ariel' };
        api.fireActive();
      });
      // Still inside the outer guard — the inner exit must not have re-opened POSTs.
      api.activePanel = { id: 'iframe:bluesky' };
      api.fireActive();
    });
    expect(setPanelFocus).not.toHaveBeenCalled();

    // Both windows closed — a real human focus POSTs again.
    api.activePanel = { id: 'iframe:okf' };
    api.fireActive();
    expect(setPanelFocus).toHaveBeenCalledExactlyOnceWith('okf');
  });

  test('no-op in fallback mode: an active change with no live api does not throw or POST', async () => {
    const api = makeApi();
    const { mod } = await wire(api);

    state.api = null; // dock shell torn down after wiring
    expect(() => api.fireActive()).not.toThrow();
    expect(setPanelFocus).not.toHaveBeenCalled();
    expect(mod).toBeTruthy();
  });
});

describe('server-driven hide of the ACTIVE panel (regression)', () => {
  // A server SSE panel_visibility(false) for the currently-active panel drives
  // hidePanel → api.removePanel → dockview auto-activates a stacked neighbor →
  // onDidActivePanelChange. panel-manager applies that hide INSIDE the echo guard
  // (withEchoSuppressed(() => hidePanel(panel))), so the neighbor's activation must
  // not be mistaken for a human focus and POSTed back. This pins the dock-sync
  // guarantee panel-manager relies on; the stub now models removePanel firing the
  // active-change echo (the mock gap that previously let the back-POST slip).
  test('the neighbor auto-activated by a guarded removePanel does not POST setPanelFocus', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    api.activePanel = { id: 'iframe:ariel' };
    api._activateOnRemove = { id: 'iframe:bluesky' }; // the stacked neighbor dockview reveals

    mod.withEchoSuppressed(() => api.removePanel({ id: 'iframe:ariel' }));

    expect(api.removePanel).toHaveBeenCalledTimes(1);
    expect(setPanelFocus).not.toHaveBeenCalled();
    expect(setPanelVisibility).not.toHaveBeenCalled();
  });

  test('the guard lifts after the hide chain — a later genuine human focus still POSTs', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    api.activePanel = { id: 'iframe:ariel' };
    api._activateOnRemove = { id: 'iframe:bluesky' };

    mod.withEchoSuppressed(() => api.removePanel({ id: 'iframe:ariel' }));
    expect(setPanelFocus).not.toHaveBeenCalled();

    // Operator now clicks a different service tab — outside any guard.
    api.activePanel = { id: 'iframe:okf' };
    api.fireActive();
    expect(setPanelFocus).toHaveBeenCalledExactlyOnceWith('okf');
  });
});

describe('human tab close → visibility POST', () => {
  test('clicking a service tab’s close control POSTs setPanelVisibility(id, false) once', async () => {
    const api = makeApi();
    const { mod, root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel', 'terminal']);

    tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(setPanelVisibility).toHaveBeenCalledExactlyOnceWith('ariel', false);
    expect(setPanelFocus).not.toHaveBeenCalled();
    expect(mod).toBeTruthy();
  });

  test('the close click maps by tab position — the second tab resolves the second panel', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel', 'iframe:bluesky']);

    tabs[1].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(setPanelVisibility).toHaveBeenCalledExactlyOnceWith('bluesky', false);
  });

  test('closing a native panel’s tab never POSTs (no server-side visibility)', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['terminal', 'iframe:ariel']);

    tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(setPanelVisibility).not.toHaveBeenCalled();
  });

  test('a click elsewhere on a tab (not the close control) does not POST', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);

    // The tab body, not its `.dv-default-tab-action` close control.
    tabs[0].tab.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(setPanelVisibility).not.toHaveBeenCalled();
  });

  test('a close click on an unresolvable tab (no matching group) is a silent no-op', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);
    api.groups = []; // group vanished between render and click

    expect(() =>
      tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true })),
    ).not.toThrow();
    expect(setPanelVisibility).not.toHaveBeenCalled();
  });
});

describe('dockPanelBesideActive — placeholder append (register/open path)', () => {
  test('adds a placeholder by the adapter id + component, positioned right of the active group', async () => {
    const api = makeApi();
    const activeGroup = { id: 'group-1' };
    api.activeGroup = activeGroup;
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel', 'ARIEL');

    expect(api.addPanel).toHaveBeenCalledTimes(1);
    expect(api._added[0]).toEqual({
      id: 'iframe:ariel',
      component: 'dock-iframe-placeholder',
      title: 'ARIEL',
      position: { referenceGroup: activeGroup, direction: 'right' },
    });
  });

  test('defaults the tab title to the service id when none is given', async () => {
    const api = makeApi();
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('okf');

    expect(api._added[0].title).toBe('okf');
    expect(api._added[0].id).toBe('iframe:okf');
  });

  test('omits the position when there is no active group', async () => {
    const api = makeApi();
    api.activeGroup = null;
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel');

    expect(api._added[0].position).toBeUndefined();
  });

  test('is a no-op when the panel is already docked', async () => {
    const api = makeApi();
    api._panels['iframe:ariel'] = { id: 'iframe:ariel' };
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel');

    expect(api.addPanel).not.toHaveBeenCalled();
  });

  test('the add is wrapped in the echo guard — its own active change does not POST a focus', async () => {
    const api = makeApi();
    api.activeGroup = { id: 'group-1' };
    // Model dockview activating the freshly-added panel synchronously on add.
    api.addPanel = vi.fn((/** @type {any} */ opts) => {
      api._added.push(opts);
      api.activePanel = { id: opts.id };
      api.fireActive();
    });
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel');

    expect(api.addPanel).toHaveBeenCalledTimes(1);
    expect(setPanelFocus).not.toHaveBeenCalled(); // the add-menu's show path owns the focus POST
  });

  test('a failing addPanel is caught and does not propagate', async () => {
    const api = makeApi();
    api.addPanel = vi.fn(() => {
      throw new Error('dockview rejected');
    });
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const { mod } = await wire(api);

    expect(() => mod.dockPanelBesideActive('ariel')).not.toThrow();
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  test('no-op in fallback mode (no live api)', async () => {
    const mod = await import(SYNC);
    state.api = null;

    expect(() => mod.dockPanelBesideActive('ariel')).not.toThrow();
    expect(setPanelFocus).not.toHaveBeenCalled();
  });
});

describe('initDockSync — wiring, idempotency, late arrival', () => {
  test('wires the active-panel listener and the capture-phase close click when api + root exist', async () => {
    const api = makeApi();
    const { root } = await wire(api);

    expect(api.onDidActivePanelChange).toHaveBeenCalledTimes(1);

    // The click listener is live: a service close routes a visibility POST.
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);
    tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(setPanelVisibility).toHaveBeenCalledExactlyOnceWith('ariel', false);
  });

  test('is idempotent — a second initDockSync does not double-wire', async () => {
    const api = makeApi();
    const { mod } = await wire(api);

    mod.initDockSync();
    mod.initDockSync();

    expect(api.onDidActivePanelChange).toHaveBeenCalledTimes(1);
  });

  test('retries and wires once a late DockviewApi arrives', async () => {
    vi.useFakeTimers();
    try {
      // No api and no #dock-root yet: init must defer, not wire.
      const mod = await import(SYNC);
      mod.initDockSync();

      const api = makeApi();
      state.api = api;
      const root = document.createElement('div');
      root.id = 'dock-root';
      document.body.appendChild(root);
      expect(api.onDidActivePanelChange).not.toHaveBeenCalled();

      // The bounded retry fires and now finds both the api and the root.
      vi.advanceTimersByTime(150);
      expect(api.onDidActivePanelChange).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  test('gives up after the bounded retry budget when the shell never appears', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import(SYNC);
      mod.initDockSync();
      // 31 ticks exhausts the wireAttempts budget without a root/api present.
      vi.advanceTimersByTime(150 * 33);
      // Shell finally arrives, but the retry loop has already stopped.
      const api = makeApi();
      state.api = api;
      const root = document.createElement('div');
      root.id = 'dock-root';
      document.body.appendChild(root);
      vi.advanceTimersByTime(150 * 5);
      expect(api.onDidActivePanelChange).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('module isolation (vi.resetModules per test)', () => {
  test('exposes exactly its public surface as functions', async () => {
    const mod = await import(SYNC);
    for (const name of ['withEchoSuppressed', 'dockPanelBesideActive', 'serviceIdOf', 'initDockSync']) {
      expect(typeof mod[name]).toBe('function');
    }
  });

  test('module-scoped guard/wire state does not leak across imports', async () => {
    // First instance: raise the guard, wire it.
    const first = makeApi();
    const { mod: modA } = await wire(first);
    modA.withEchoSuppressed(() => {}); // touches suppressDepth
    expect(first.onDidActivePanelChange).toHaveBeenCalledTimes(1);

    // A fresh import (resetModules ran in beforeEach for the NEXT test, but here we
    // force a new instance) starts with wired=false and suppressDepth=0: it wires
    // again, and a plain human focus POSTs — proving no stale suppression carried.
    vi.resetModules();
    const second = makeApi();
    state.api = second;
    document.body.innerHTML = '';
    const root = document.createElement('div');
    root.id = 'dock-root';
    document.body.appendChild(root);
    const modB = await import(SYNC);
    modB.initDockSync();
    expect(second.onDidActivePanelChange).toHaveBeenCalledTimes(1);

    second.activePanel = { id: 'iframe:ariel' };
    second.fireActive();
    expect(setPanelFocus).toHaveBeenCalledExactlyOnceWith('ariel');
  });
});
