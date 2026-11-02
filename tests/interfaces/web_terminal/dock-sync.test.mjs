/**
 * Contract tests for the dock↔server state-sync bridge (dock-sync.js):
 *   npx vitest run tests/interfaces/web_terminal/dock-sync.test.mjs
 *
 * dock-sync carries the REVERSE half of the panel-state loop: it turns a human's
 * dockview FOCUS gestures back into the same server POST an agent MCP call would
 * make, WITHOUT letting the server's own SSE echo bounce back out again. A human
 * tab CLOSE is deliberately NOT a server change (tile occupancy is per-client
 * layout state under the launcher-rail model) — it routes to a locally-registered
 * vacate handler instead. The forward half (server → dock application, and the
 * chat layout synthesis) lives in dock-workspace.js / panel-manager.js and
 * is pinned by the reconcile tests (dock-reconcile.test.mjs) plus the browser
 * suite; this file does not duplicate them. Here we pin dock-sync's own contract:
 *
 *   - the ECHO GUARD — a genuine human focus POSTs exactly once, a server-applied
 *     (echo) focus POSTs never;
 *   - the human tab-CLOSE click fires the registered tile-close handler, never a
 *     POST;
 *   - the LAYOUT REPORT reads the layout holder (never dockview's toJSON), sends
 *     the page's client_id and the pop-outs beside the layout, and hands the
 *     ack's rev back to the holder;
 *   - dockPanelBesideActive and dockPanelAt run one engine op on the layout
 *     holder, which applies it under the same guard;
 *   - the listeners wire idempotently and tolerate late DockviewApi arrival.
 *
 * dockview and the collaborators are stubbed at the module boundary: getDockApi
 * (dock-workspace.js) yields a hand-built fake api, getLayoutHolder (dock-apply.js)
 * a fake holder (or a real one over the fake api, for the placement verbs), and
 * setPanelFocus / setPanelVisibility (panel-commands.js) are spies, and fetch
 * is stubbed so `serverReport` plays the layout-report route. Each test
 * re-imports dock-sync fresh (vi.resetModules) so its module-scoped guard/wire
 * state never leaks across tests.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SYNC = '../../../src/osprey/interfaces/web_terminal/static/js/dock-sync.js';
const API = '../../../src/osprey/interfaces/web_terminal/static/js/api.js';

// Boundary stubs. Hoisted so the spies exist before dock-sync's static imports
// resolve; the same spy instances persist across resetModules (they are closed
// over here), while dock-sync's own state is rebuilt on each fresh import.
const {
  getDockApi,
  setPanelFocus,
  setPanelVisibility,
  serverReport,
  setLayoutRestoredHook,
  state,
} = vi.hoisted(() => ({
  state: {
    api: /** @type {any} */ (null),
    restoreHook: /** @type {any} */ (null),
    holder: /** @type {any} */ (null),
  },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
  setLayoutRestoredHook: vi.fn(),
  setPanelFocus: vi.fn(),
  setPanelVisibility: vi.fn(),
  // The layout-report route: (layout, popouts, dock) -> the server's
  // acknowledgement, or null for a report that did not land.
  serverReport: vi.fn(),
}));
// getDockApi reads the mutable holder so a test can swap the live api in place.
getDockApi.mockImplementation(() => state.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  // The boot-layout-settled seam. dock-workspace calls the registered hook at
  // every exit of initPersistence; tests capture it and fire it by hand to model
  // a restore landing (or deliberately never landing).
  setLayoutRestoredHook,
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
// The layout holder is the one thing the reporter and the placement verbs
// read; the page's holder is swapped per test (a fake with just a layout for
// the reporter, a real one over the fake api for the verbs).
const DOCK_APPLY = '../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js';
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js',
  async (importOriginal) => ({
    .../** @type {object} */ (await importOriginal()),
    getLayoutHolder: () => state.holder,
  })
);
// The one question dock-sync asks the pop-out seam: whether a panel lives in
// its own window. The seam's own behaviour is popout.test.mjs's contract.
const { isPoppedOut, popoutListeners } = vi.hoisted(() => ({
  isPoppedOut: vi.fn(/** @type {(id: string) => boolean} */ (() => false)),
  /** The pop-out change listeners dock-sync registered. @type {Function[]} */
  popoutListeners: /** @type {Function[]} */ ([]),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({
  isPoppedOut,
  onPopoutChange: (/** @type {Function} */ fn) => {
    popoutListeners.push(fn);
  },
}));

/** Every body POSTed to /api/panel-layout, in order. @type {any[]} */
let fetchBodies = [];
/** Flip `ok` to false to model the route refusing a report. */
const fetchStatus = { ok: true };

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.restoreHook = null;
  setLayoutRestoredHook.mockImplementation((/** @type {any} */ fn) => {
    state.restoreHook = fn;
  });
  // Default: the server acknowledges every report as a genuine update and
  // mints the next rev, the way the real route does.
  let rev = 0;
  serverReport.mockImplementation(
    (/** @type {any} */ layout, /** @type {string[]} */ popouts, /** @type {boolean} */ dock) =>
      Promise.resolve({ status: 'ok', layout, popouts, dock, rev: ++rev, updated: true })
  );
  fetchBodies = [];
  fetchStatus.ok = true;
  popoutListeners.length = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ _url, /** @type {any} */ opts) => {
      const body = JSON.parse(opts.body);
      fetchBodies.push(body);
      const ack = await serverReport(body.layout, body.popouts, body.dock);
      if (!fetchStatus.ok || ack === null) return { ok: false, json: async () => null };
      return { ok: true, json: async () => ack };
    })
  );
  state.api = null;
  state.holder = null;
  document.body.innerHTML = '';
});

/**
 * A hand-built stand-in for dockview's DockviewApi exposing only the surface
 * dock-sync touches. `fireActive()` invokes the onDidActivePanelChange listener
 * dock-sync registered (dockview fires it synchronously for both a human tab
 * click and a programmatic setActive — the ambiguity the echo guard resolves);
 * `fireLayout()` invokes the onDidLayoutChange listener the occupancy reporter
 * registered. A layout holder applies a whole layout at once through
 * `fromJSON` (and reads the stored panel entries through `toJSON`).
 *
 * The layout report reads the holder, never this api; `state.holder` is
 * null by default, which keeps the settle timer wiring arms at boot inert in
 * every test that is not about the reporter.
 * @returns {any}
 */
function makeApi() {
  /** @type {any} */
  const api = {
    activePanel: null,
    groups: [],
    width: 1600,
    height: 900,
    /** @type {null | (() => void)} */
    _activeCb: null,
    /** @type {null | (() => void)} */
    _layoutCb: null,
    fireActive() {
      if (api._activeCb) api._activeCb();
    },
    /** dockview's settled-layout-change event (panel add/move/close, sash end). */
    fireLayout() {
      if (api._layoutCb) api._layoutCb();
    },
  };
  api.onDidActivePanelChange = vi.fn((/** @type {() => void} */ cb) => {
    api._activeCb = cb;
    return { dispose() {} };
  });
  api.onDidLayoutChange = vi.fn((/** @type {() => void} */ cb) => {
    api._layoutCb = cb;
    return { dispose() {} };
  });
  api.getPanel = vi.fn(() => null);
  api.toJSON = vi.fn(() => ({ panels: {} }));
  api.fromJSON = vi.fn();
  return api;
}

/**
 * A dock layout row of tiles, one per tab list (tile ids '1', '2', ...; each
 * tile shows its first tab); a single list is a root tile.
 * @param {...string[]} tabLists
 * @returns {any}
 */
function row(...tabLists) {
  const tiles = tabLists.map((tabs, i) => ({ tile: String(i + 1), tabs, showing: tabs[0] }));
  return tiles.length === 1 ? tiles[0] : { split: 'row', items: tiles };
}

/** The tile-close vacate handler panel-manager registers in production —
 *  wire() registers this spy so close-click tests can assert the routing. */
const tileClose = vi.fn();

/**
 * Put a #dock-root in the DOM, publish `api` as the live DockviewApi, import a
 * fresh dock-sync, wire it, and register the tile-close spy. Returns the
 * module + root for follow-on asserts.
 * @param {any} api
 */
async function wire(api) {
  state.api = api;
  const root = document.createElement('div');
  root.id = 'dock-root';
  document.body.appendChild(root);
  const mod = await import(SYNC);
  mod.initDockSync();
  mod.setTileCloseHandler(tileClose);
  return { mod, root };
}

/**
 * Build a dockview-shaped group DOM (one `.dv-groupview` holding a tab strip of
 * `.dv-tab` elements, each wrapping a `.tile-tab` root that carries the
 * `data-panel-id` stamp dock-tab.js writes, with the `.tile-tab-close` close
 * control inside) under `root` — the stamp is what panelIdForTab resolves.
 * @param {HTMLElement} root
 * @param {any} api
 * @param {string[]} panelIds
 */
function buildGroup(root, api, panelIds) {
  const groupview = document.createElement('div');
  groupview.className = 'dv-groupview';
  const strip = document.createElement('div');
  strip.className = 'dv-tabs-container';
  /** @type {{tab: HTMLElement, tile: HTMLElement, action: HTMLElement}[]} */
  const tabs = [];
  for (const id of panelIds) {
    const tab = document.createElement('div');
    tab.className = 'dv-tab';
    const tile = document.createElement('div');
    tile.className = 'tile-tab';
    tile.dataset.panelId = id;
    const action = document.createElement('button');
    action.className = 'tile-tab-close';
    tile.appendChild(action);
    tab.appendChild(tile);
    strip.appendChild(tab);
    tabs.push({ tab, tile, action });
  }
  groupview.appendChild(strip);
  root.appendChild(groupview);
  api.groups = [{ element: groupview, panels: panelIds.map((id) => ({ id })) }];
  return { groupview, tabs };
}

/**
 * A fake layout holder with the surface the reporter reads: `layout`,
 * `popouts`, `dirty`, `ackedRev` and `ack(rev)` (dock-apply.js's
 * createLayoutHolder).
 * @param {any} layout @param {string[]} [popouts]
 */
function makeHolder(layout, popouts = []) {
  /** @type {any} */
  const holder = { layout, popouts, dirty: true, ackedRev: null };
  holder.ack = vi.fn((/** @type {number} */ rev) => {
    holder.ackedRev = rev;
    holder.dirty = false;
  });
  return holder;
}

describe('layout reporter — from the holder, settle debounce, content dedupe, rev', () => {
  const SETTLE = 250;

  /**
   * Wire dock-sync with a holder holding `layout`, then fire the boot-layout-
   * settled hook — the normal boot. Reporting is inert until that fires, so
   * tests about the boot window itself withhold it (`settled: false`) and fire
   * state.restoreHook themselves.
   * @param {any} layout
   * @param {{settled?: boolean}} [opts]
   */
  async function wireReporting(layout, { settled = true } = {}) {
    state.holder = makeHolder(layout);
    const api = makeApi();
    const { mod } = await wire(api);
    if (settled) state.restoreHook?.();
    return { api, mod, holder: state.holder };
  }

  /**
   * The operator's dock moved on: the holder now holds `layout` (dirty), and
   * dockview fires the layout change the holder's apply causes.
   * @param {any} api
   * @param {any} layout
   */
  function change(api, layout) {
    state.holder.layout = layout;
    state.holder.dirty = true;
    api.fireLayout();
  }

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('reports the holder layout — tabs, showing tab and terminal — with the client id', async () => {
    const layout = row(['terminal'], ['ariel', 'okf'], ['bluesky']);
    await wireReporting(layout);
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(serverReport).toHaveBeenCalledExactlyOnceWith(layout, [], true);
    const body = fetchBodies[0];
    expect(Object.keys(body).sort()).toEqual(['client_id', 'dock', 'layout', 'popouts']);
    expect(body.layout.items[1]).toEqual({
      tile: '2',
      tabs: ['ariel', 'okf'],
      showing: 'ariel',
    });
    expect(typeof body.client_id).toBe('string');
    expect(body.client_id.length).toBeGreaterThan(0);
    const { getPageClientId } = await import(API);
    expect(body.client_id).toBe(getPageClientId());
  });

  test('never reads dockview: an api whose toJSON throws changes nothing', async () => {
    const { api } = await wireReporting(row(['terminal']));
    api.toJSON = vi.fn(() => {
      throw new Error('toJSON must not be called');
    });
    change(api, row(['terminal'], ['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(api.toJSON).not.toHaveBeenCalled();
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(row(['terminal'], ['ariel']), [], true);
  });

  test('the client id is the same on every report of a page', async () => {
    const { api } = await wireReporting(row(['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    change(api, row(['terminal'], ['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(fetchBodies).toHaveLength(2);
    expect(fetchBodies[0].client_id).toBe(fetchBodies[1].client_id);
  });

  test('the ack rev goes to the holder: ackedRev kept, dirty cleared', async () => {
    serverReport.mockImplementation((/** @type {any} */ layout) =>
      Promise.resolve({ status: 'ok', layout, rev: 7, updated: true })
    );
    const { holder } = await wireReporting(row(['terminal'], ['ariel']));
    expect(holder.dirty).toBe(true);
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(holder.ack).toHaveBeenCalledExactlyOnceWith(7);
    expect(holder.ackedRev).toBe(7);
    expect(holder.dirty).toBe(false);
  });

  test('a layout that moved on while the report was in flight stays dirty and goes next', async () => {
    /** @type {(() => void)[]} */
    const pending = [];
    let rev = 0;
    serverReport.mockImplementation(
      (/** @type {any} */ layout) =>
        new Promise((resolve) =>
          pending.push(() => resolve({ status: 'ok', layout, rev: ++rev, updated: true }))
        )
    );
    const { api, holder } = await wireReporting(row(['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);

    change(api, row(['terminal'], ['ariel']));
    pending[0]();
    await vi.advanceTimersByTimeAsync(0);
    expect(holder.ack).not.toHaveBeenCalled();
    expect(holder.dirty).toBe(true);

    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(2);
    expect(serverReport).toHaveBeenLastCalledWith(row(['terminal'], ['ariel']), [], true);
    pending[1]();
    await vi.advanceTimersByTimeAsync(0);
    expect(holder.ack).toHaveBeenCalledExactlyOnceWith(2);
    expect(holder.dirty).toBe(false);
  });

  test('the unchanged-since-ack check ignores rev', async () => {
    // A later writer (the op-result route) may move the holder's ackedRev; the
    // layout is what decides whether there is news, never the rev.
    const { api, holder } = await wireReporting(row(['terminal'], ['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    holder.ack(99);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(serverReport).toHaveBeenCalledTimes(1);
  });

  test('a dirty holder whose layout equals the acked one settles without a POST', async () => {
    const { api, holder } = await wireReporting(row(['terminal'], ['ariel', 'okf']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    const ackedRev = holder.ackedRev;
    change(api, row(['terminal'], ['ariel', 'okf']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);
    expect(holder.dirty).toBe(false);
    expect(holder.ackedRev).toBe(ackedRev);
  });

  test('a SLOW restore never publishes the pre-restore layout as fact', async () => {
    const { api } = await wireReporting(row(['terminal']), { settled: false });
    await vi.advanceTimersByTimeAsync(SETTLE * 20);
    expect(serverReport).not.toHaveBeenCalled();

    change(api, row(['ariel'], ['okf'], ['terminal']));
    state.restoreHook();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(
      row(['ariel'], ['okf'], ['terminal']),
      [],
      true
    );
  });

  test('layout churn before the restore is ignored, not merely debounced', async () => {
    const { api } = await wireReporting(row(['terminal']), { settled: false });
    change(api, row(['ariel'], ['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    change(api, row(['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(serverReport).not.toHaveBeenCalled();
  });

  test('the boot deadline is a backstop when the restore hook never fires', async () => {
    const { api } = await wireReporting(row(['ariel']), { settled: false });
    await vi.advanceTimersByTimeAsync(SETTLE * 8);
    expect(serverReport).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(10000);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(row(['ariel']), [], true);
    expect(api.onDidLayoutChange).toHaveBeenCalledTimes(1);
  });

  test('a burst of layout changes reports ONCE, from the layout it settles on', async () => {
    const { api } = await wireReporting(row(['terminal']));
    change(api, row(['ariel'], ['terminal']));
    await vi.advanceTimersByTimeAsync(100);
    change(api, row(['ariel'], ['bluesky'], ['terminal']));
    await vi.advanceTimersByTimeAsync(100);
    change(api, row(['bluesky'], ['ariel'], ['terminal']));
    expect(serverReport).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(
      row(['bluesky'], ['ariel'], ['terminal']),
      [],
      true
    );
  });

  test('reads at FLUSH time — an intermediate layout is never reported', async () => {
    const { api } = await wireReporting(row(['ariel']));
    change(api, row(['terminal']));
    change(api, row(['okf']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(row(['okf']), [], true);
  });

  test('switching the showing tab is a change, not a duplicate', async () => {
    const { api } = await wireReporting(row(['terminal'], ['ariel', 'okf']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    const shown = row(['terminal'], ['ariel', 'okf']);
    shown.items[1].showing = 'okf';
    change(api, shown);
    api.fireActive();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(2);
    expect(serverReport).toHaveBeenLastCalledWith(shown, [], true);
  });

  test('closing the last service tab reports the terminal-only layout', async () => {
    const { api } = await wireReporting(row(['ariel'], ['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    change(api, row(['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenLastCalledWith(row(['terminal']), [], true);
  });

  test('reports during an echo-guarded apply — the guard must NOT gate it', async () => {
    const { api, mod } = await wireReporting(row(['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    serverReport.mockClear();
    mod.withEchoSuppressed(() => change(api, row(['ariel'], ['terminal'])));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(row(['ariel'], ['terminal']), [], true);
  });

  test('an applied layout settles in ONE report, then goes quiet (no loop)', async () => {
    const { api } = await wireReporting(row(['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    serverReport.mockClear();
    change(api, row(['ariel'], ['okf'], ['terminal']));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(serverReport).toHaveBeenCalledTimes(1);
  });

  test('no holder, or a holder with no layout yet, reports nothing', async () => {
    const { api } = await wireReporting(null);
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).not.toHaveBeenCalled();

    state.holder = null;
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).not.toHaveBeenCalled();

    state.holder = makeHolder(row(['ariel']));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(row(['ariel']), [], true);
  });

  test('fallback mode sends ONE dock:false capability report with no layout', async () => {
    const mod = await import(SYNC);
    mod.initDockSync();
    await vi.advanceTimersByTimeAsync(150 * 33);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(null, [], false);
    const { getPageClientId } = await import(API);
    expect(fetchBodies[0].client_id).toBe(getPageClientId());

    mod.initDockSync();
    mod.initDockSync();
    await vi.advanceTimersByTimeAsync(150 * 10);
    expect(serverReport).toHaveBeenCalledTimes(1);
  });

  test('a dock:false claim is RETRACTED when a slow shell finally arrives', async () => {
    const mod = await import(SYNC);
    mod.initDockSync();
    await vi.advanceTimersByTimeAsync(150 * 33);
    expect(serverReport).toHaveBeenCalledExactlyOnceWith(null, [], false);

    state.holder = makeHolder(row(['ariel'], ['terminal']));
    const api = makeApi();
    state.api = api;
    const root = document.createElement('div');
    root.id = 'dock-root';
    document.body.appendChild(root);
    await vi.advanceTimersByTimeAsync(2000);
    expect(api.onDidActivePanelChange).toHaveBeenCalledTimes(1);

    state.restoreHook();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenLastCalledWith(row(['ariel'], ['terminal']), [], true);
  });

  test('a deduped no-op ack (updated:false) still sets the dedupe baseline', async () => {
    serverReport.mockImplementation((/** @type {any} */ layout) =>
      Promise.resolve({ status: 'ok', layout, rev: 3, updated: false })
    );
    const { api, holder } = await wireReporting(row(['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(holder.ackedRev).toBe(3);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(serverReport).toHaveBeenCalledTimes(1);
  });

  test('a person undoing an agent open is reported: noteReported moves the baseline', async () => {
    const before = row(['ariel'], ['terminal']);
    const { api, mod } = await wireReporting(before);
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);
    // An agent opens bluesky: the holder moves and the op's answer (not this
    // module's report) is what the server acknowledged.
    const opened = row(['ariel', 'bluesky'], ['terminal']);
    state.holder.layout = opened;
    mod.noteReported({ layout: opened, popouts: [] });
    // The person closes it again: back to the layout this module last sent,
    // which the server no longer holds. It must go out, not dedupe away.
    change(api, before);
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(2);
    expect(serverReport).toHaveBeenLastCalledWith(before, [], true);
  });

  test('a layout equal to one acknowledged through noteReported is not re-reported', async () => {
    const { api, mod } = await wireReporting(row(['ariel'], ['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    const opened = row(['ariel', 'bluesky'], ['terminal']);
    mod.noteReported({ layout: opened, popouts: [] });
    change(api, opened);
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);
  });

  test('a read-back differing only by size rounding from the acknowledged layout is not re-reported', async () => {
    const { api, mod } = await wireReporting(row(['ariel'], ['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    const planned = row(['ariel', 'bluesky'], ['terminal']);
    mod.noteReported({ layout: planned, popouts: [] });
    // dockview reads the applied tree back a pixel's fraction off.
    const measured = structuredClone(planned);
    measured.items[0].size = planned.items[0].size + 0.0002;
    measured.items[1].size = planned.items[1].size - 0.0002;
    change(api, measured);
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);
  });

  test('a panel leaving for its own window is reported, with the pop-outs beside the layout', async () => {
    const { api, holder } = await wireReporting(row(['ariel'], ['terminal']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);
    // popout.js vacates the tile (the holder's layout moves) and tells the
    // holder; dock-sync hears the change and reports both.
    holder.layout = row(['terminal']);
    holder.popouts = ['ariel'];
    holder.dirty = true;
    for (const fn of popoutListeners) fn('ariel', true);
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenLastCalledWith(row(['terminal']), ['ariel'], true);
    expect(fetchBodies.at(-1).popouts).toEqual(['ariel']);
    // Its return, with the layout as before the pop-out, is news too.
    holder.layout = row(['ariel'], ['terminal']);
    holder.popouts = [];
    holder.dirty = true;
    for (const fn of popoutListeners) fn('ariel', false);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(3);
    expect(serverReport).toHaveBeenLastCalledWith(row(['ariel'], ['terminal']), [], true);
  });

  test('a report that never LANDED is retried, not remembered as sent', async () => {
    serverReport.mockImplementation(() => Promise.resolve(null));
    const { api, holder } = await wireReporting(row(['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);
    expect(holder.ack).not.toHaveBeenCalled();
    expect(holder.dirty).toBe(true);

    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(2);

    serverReport.mockImplementation((/** @type {any} */ layout) =>
      Promise.resolve({ status: 'ok', layout, rev: 1, updated: true })
    );
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(3);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(3);
  });

  test('a refused report (HTTP error) is not acked and is retried', async () => {
    fetchStatus.ok = false;
    const { api, holder } = await wireReporting(row(['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(holder.ack).not.toHaveBeenCalled();
    fetchStatus.ok = true;
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(2);
    expect(holder.ack).toHaveBeenCalledTimes(1);
  });

  test('a fetch that throws does not strand the in-flight flag and kill reporting', async () => {
    serverReport.mockImplementation(() => Promise.reject(new Error('network down')));
    const { api } = await wireReporting(row(['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);

    serverReport.mockImplementation((/** @type {any} */ layout) =>
      Promise.resolve({ status: 'ok', layout, rev: 1, updated: true })
    );
    change(api, row(['ariel'], ['okf']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(2);
    expect(serverReport).toHaveBeenLastCalledWith(row(['ariel'], ['okf']), [], true);
  });

  test('a second report never overlaps one in flight — it waits for the ack', async () => {
    /** @type {(() => void)[]} */
    const pending = [];
    serverReport.mockImplementation(
      (/** @type {any} */ layout) =>
        new Promise((resolve) =>
          pending.push(() => resolve({ status: 'ok', layout, rev: pending.length, updated: true }))
        )
    );
    const { api } = await wireReporting(row(['ariel']));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(1);

    change(api, row(['ariel'], ['okf']));
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(serverReport).toHaveBeenCalledTimes(1);

    pending[0]();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(serverReport).toHaveBeenCalledTimes(2);
    expect(serverReport).toHaveBeenLastCalledWith(row(['ariel'], ['okf']), [], true);

    pending[1]();
    await vi.advanceTimersByTimeAsync(SETTLE);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(serverReport).toHaveBeenCalledTimes(2);
  });
});

describe('withEchoSuppressed — the guard primitive', () => {
  test('restores the guard even when the callback throws (finally)', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    expect(() =>
      mod.withEchoSuppressed(() => {
        throw new Error('boom');
      })
    ).toThrow('boom');

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

  test('focusing a native (terminal/workspace) panel, or none at all, never POSTs', async () => {
    const api = makeApi();
    await wire(api);

    api.activePanel = { id: 'terminal' };
    api.fireActive();
    api.activePanel = { id: 'workspace' };
    api.fireActive();
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
    await wire(api);

    state.api = null; // dock shell torn down after wiring
    expect(() => api.fireActive()).not.toThrow();
    expect(setPanelFocus).not.toHaveBeenCalled();
  });
});

describe('human tab close → local vacate handler (never a POST)', () => {
  test('clicking a service tab’s close control fires the registered handler once', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel', 'terminal']);

    tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(tileClose).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(setPanelVisibility).not.toHaveBeenCalled();
    expect(setPanelFocus).not.toHaveBeenCalled();
  });

  test('the close click resolves by the data-panel-id stamp — the second tab resolves the second panel', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel', 'iframe:bluesky']);

    tabs[1].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(tileClose).toHaveBeenCalledExactlyOnceWith('bluesky');
    expect(setPanelVisibility).not.toHaveBeenCalled();
  });

  test('closing a native panel’s tab fires nothing (no service id)', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['terminal', 'iframe:ariel']);

    tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(tileClose).not.toHaveBeenCalled();
    expect(setPanelVisibility).not.toHaveBeenCalled();
  });

  test('a click elsewhere on a tab (not the close control) fires nothing', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);

    // The tab body, not its `.tile-tab-close` control.
    tabs[0].tab.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(tileClose).not.toHaveBeenCalled();
  });

  test('a close click on an unstamped tab (no data-panel-id) is a silent no-op', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);
    tabs[0].tile.removeAttribute('data-panel-id'); // stamp lost — nothing to resolve

    expect(() =>
      tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    ).not.toThrow();
    expect(tileClose).not.toHaveBeenCalled();
  });

  test('a close click with NO registered handler is a silent no-op', async () => {
    const api = makeApi();
    const { mod, root } = await wire(api);
    mod.setTileCloseHandler(null);
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);

    expect(() =>
      tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    ).not.toThrow();
    expect(setPanelVisibility).not.toHaveBeenCalled();
  });
});

/**
 * A REAL layout holder (dock-apply.js's createLayoutHolder, loaded past the
 * module mock) over `api`, published as the page's holder and seeded with
 * `layout`; `focus` (a tile id) is the tile a person last focused. Its applies
 * call `api.fromJSON` under the guard the page passes in — dock-sync's own echo
 * guard here, as in production. Call after `wire()` so the holder shares the
 * freshly imported dock-sync.
 * @param {any} api
 * @param {any} layout
 * @param {string} [focus]
 * @returns {Promise<any>}
 */
async function seedHolder(api, layout, focus) {
  const actual = /** @type {any} */ (await vi.importActual(DOCK_APPLY));
  const sync = await import(SYNC);
  const holder = actual.createLayoutHolder({
    api,
    guard: sync.withEchoSuppressed,
    dockPx: { width: 1600, height: 900 },
  });
  expect(holder.restore(layout)).toBe(true);
  if (focus) {
    const tile = tilesIn(holder.layout).find((t) => t.tile === focus);
    expect(holder.show(focus, tile.showing)).toBe(true);
  }
  api.fromJSON.mockClear();
  state.holder = holder;
  return holder;
}

/**
 * Every tile of a dock layout in reading order.
 * @param {any} layout
 * @returns {any[]}
 */
function tilesIn(layout) {
  if (!layout) return [];
  return 'tile' in layout ? [layout] : layout.items.flatMap(tilesIn);
}

/**
 * A dock layout reduced to its shape: a tile is its tab list, a split is
 * `{split, items}` (tile ids and sizes dropped).
 * @param {any} layout
 * @returns {any}
 */
function shapeOf(layout) {
  if ('tile' in layout) return layout.tabs;
  return { split: layout.split, items: layout.items.map(shapeOf) };
}

describe('dockPanelBesideActive — a new tile right of the focused tile (register/open path)', () => {
  test('adds a placeholder by the adapter id + component, in a new tile right of the focused tile', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal'], ['bluesky']), '2');

    mod.dockPanelBesideActive('ariel', 'ARIEL');

    expect(shapeOf(holder.layout)).toEqual({
      split: 'row',
      items: [['terminal'], ['bluesky'], ['ariel']],
    });
    // One whole-layout apply, carrying the placeholder under the adapter id.
    expect(api.fromJSON).toHaveBeenCalledTimes(1);
    const serialized = api.fromJSON.mock.calls[0][0];
    expect(serialized.panels['iframe:ariel']).toMatchObject({
      id: 'iframe:ariel',
      contentComponent: 'dock-iframe-placeholder',
      title: 'ARIEL',
    });
  });

  test('defaults the tab title to the service id when none is given', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    await seedHolder(api, row(['terminal'], ['bluesky']), '2');

    mod.dockPanelBesideActive('okf');

    const serialized = api.fromJSON.mock.calls[0][0];
    expect(serialized.panels['iframe:okf'].title).toBe('okf');
    expect(serialized.panels['iframe:okf'].id).toBe('iframe:okf');
  });

  test('with no focused tile the panel opens as the first service tile, left of the terminal', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal']));

    mod.dockPanelBesideActive('ariel');

    expect(shapeOf(holder.layout)).toEqual({ split: 'row', items: [['ariel'], ['terminal']] });
  });

  test('already docked with NO focused tile: a no-op (no meaningful target)', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal'], ['ariel']));
    const before = holder.layout;

    mod.dockPanelBesideActive('ariel');

    expect(api.fromJSON).not.toHaveBeenCalled();
    expect(holder.layout).toBe(before);
  });

  test("a move is echo-guarded — the rebuild's active change does not POST a focus", async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    await seedHolder(api, row(['terminal'], ['ariel'], ['bluesky']), '3');
    // Model dockview activating a panel synchronously on apply.
    api.fromJSON = vi.fn(() => {
      api.activePanel = { id: 'iframe:bluesky' };
      api.fireActive();
    });

    mod.dockPanelBesideActive('ariel');

    expect(api.fromJSON).toHaveBeenCalledTimes(1);
    expect(setPanelFocus).not.toHaveBeenCalled();
  });

  test('the add is wrapped in the echo guard — its own active change does not POST a focus', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    await seedHolder(api, row(['terminal'], ['bluesky']), '2');
    // Model dockview activating the freshly-added panel synchronously on apply.
    api.fromJSON = vi.fn(() => {
      api.activePanel = { id: 'iframe:ariel' };
      api.fireActive();
    });

    mod.dockPanelBesideActive('ariel');

    expect(api.fromJSON).toHaveBeenCalledTimes(1);
    expect(setPanelFocus).not.toHaveBeenCalled(); // the add-menu's show path owns the focus POST
  });

  test('a failing apply is caught and does not propagate', async () => {
    const api = makeApi();
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const { mod } = await wire(api);
    await seedHolder(api, row(['terminal'], ['bluesky']), '2');
    api.fromJSON = vi.fn(() => {
      throw new Error('dockview rejected');
    });

    expect(() => mod.dockPanelBesideActive('ariel')).not.toThrow();
    expect(api.fromJSON).toHaveBeenCalledTimes(1);
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  test('no-op in fallback mode (no live api)', async () => {
    const mod = await import(SYNC);
    state.api = null;

    expect(() => mod.dockPanelBesideActive('ariel')).not.toThrow();
  });
});

describe('dockPanelAt — placement at an explicit dock position (rail drag / drop routing)', () => {
  test('adds a fresh placeholder split off the given tile on the given side', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal'], ['bluesky']));

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: { id: '2' }, direction: 'bottom' });

    expect(shapeOf(holder.layout)).toEqual({
      split: 'row',
      items: [['terminal'], { split: 'column', items: [['bluesky'], ['okf']] }],
    });
    expect(api.fromJSON.mock.calls[0][0].panels['iframe:okf'].title).toBe('KNOWLEDGE');
  });

  test("dockview's own words for the vertical sides are taken (the rail's drop routing)", async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal'], ['bluesky']));

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: { id: '1' }, direction: 'below' });
    mod.dockPanelAt('ariel', 'ARIEL', { referenceGroup: { id: '2' }, direction: 'above' });

    expect(shapeOf(holder.layout)).toEqual({
      split: 'row',
      items: [
        { split: 'column', items: [['terminal'], ['okf']] },
        { split: 'column', items: [['ariel'], ['bluesky']] },
      ],
    });
  });

  test('a tile may be named by id as well as by its dockview group', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal'], ['bluesky']));

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceTile: '1', direction: 'left' });

    expect(shapeOf(holder.layout)).toEqual({
      split: 'row',
      items: [['okf'], ['terminal'], ['bluesky']],
    });
  });

  test('no tile named: split against the grid edge', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal'], ['bluesky']));

    mod.dockPanelAt('okf', 'KNOWLEDGE', { direction: 'right' });

    expect(shapeOf(holder.layout)).toEqual({
      split: 'row',
      items: [['terminal'], ['bluesky'], ['okf']],
    });
  });

  test('moves an existing placeholder to the given position, never duplicating it', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['okf'], ['terminal'], ['bluesky']));

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: { id: '3' }, direction: 'right' });

    expect(shapeOf(holder.layout)).toEqual({
      split: 'row',
      items: [['terminal'], ['bluesky'], ['okf']],
    });
  });

  test('dropping a panel onto its OWN tile is a no-op', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['okf'], ['terminal']));
    const before = holder.layout;

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: { id: '1' }, direction: 'right' });

    expect(api.fromJSON).not.toHaveBeenCalled();
    expect(holder.layout).toBe(before);
  });

  test('a panel living in its own pop-out window never gets a placeholder, by any placement verb', async () => {
    const api = makeApi();
    const { mod } = await wire(api);
    const holder = await seedHolder(api, row(['terminal'], ['bluesky']), '2');
    isPoppedOut.mockImplementation((/** @type {string} */ id) => id === 'okf');

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: { id: '2' }, direction: 'right' });
    mod.dockPanelBesideActive('okf', 'KNOWLEDGE');
    mod.dockPanelBesideActive('ariel', 'ARIEL');

    expect(shapeOf(holder.layout)).toEqual({
      split: 'row',
      items: [['terminal'], ['bluesky'], ['ariel']],
    });
    isPoppedOut.mockImplementation(() => false);
  });
});

describe('initDockSync — wiring, idempotency, late arrival', () => {
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

  test('gives up once BOTH retry budgets are spent and the shell never appears', async () => {
    vi.useFakeTimers();
    try {
      const mod = await import(SYNC);
      mod.initDockSync();
      // The fast budget only concludes "no dock seen yet" — a slow 1s poll keeps
      // watching for about a minute so a late shell can still wire (see the
      // reporter suite's retraction test). Both must be spent to stop for good.
      vi.advanceTimersByTime(150 * 33 + 1000 * 70);
      // Shell finally arrives, but the retry loop has already stopped.
      const api = makeApi();
      state.api = api;
      const root = document.createElement('div');
      root.id = 'dock-root';
      document.body.appendChild(root);
      vi.advanceTimersByTime(1000 * 10);
      expect(api.onDidActivePanelChange).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});
