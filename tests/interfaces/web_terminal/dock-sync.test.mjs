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
 * simple-mode layout synthesis) lives in dock-workspace.js / panel-manager.js and
 * is pinned by the reconcile tests (dock-reconcile.test.mjs) plus the browser
 * suite; this file does not duplicate them. Here we pin dock-sync's own contract:
 *
 *   - the ECHO GUARD — a genuine human focus POSTs exactly once, a server-applied
 *     (echo) focus POSTs never;
 *   - the human tab-CLOSE click fires the registered tile-close handler, never a
 *     POST;
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

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SYNC = '../../../src/osprey/interfaces/web_terminal/static/js/dock-sync.js';

// Boundary stubs. Hoisted so the spies exist before dock-sync's static imports
// resolve; the same spy instances persist across resetModules (they are closed
// over here), while dock-sync's own state is rebuilt on each fresh import.
const {
  getDockApi, setPanelFocus, setPanelVisibility, reportPanelLayout, setLayoutRestoredHook, state,
} = vi.hoisted(
  () => ({
    state: { api: /** @type {any} */ (null), restoreHook: /** @type {any} */ (null) },
    getDockApi: vi.fn(() => /** @type {any} */ (null)),
    setLayoutRestoredHook: vi.fn(),
    setPanelFocus: vi.fn(),
    setPanelVisibility: vi.fn(),
    // Resolves with the server's acknowledgement; the reporter sets its dedupe
    // baseline from that, so tests can model a lost report by resolving null.
    reportPanelLayout: vi.fn(),
  }),
);
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
  reportPanelLayout,
}));

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.restoreHook = null;
  setLayoutRestoredHook.mockImplementation((/** @type {any} */ fn) => {
    state.restoreHook = fn;
  });
  // Default: the server acknowledges every report as a genuine update, echoing
  // the tiles back the way the real route does.
  reportPanelLayout.mockImplementation((/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
    Promise.resolve({ status: 'ok', tiles, dock, updated: true }),
  );
  state.api = null;
  document.body.innerHTML = '';
});

/**
 * A hand-built stand-in for dockview's DockviewApi exposing only the surface
 * dock-sync touches. `fireActive()` invokes the onDidActivePanelChange listener
 * dock-sync registered (dockview fires it synchronously for both a human tab
 * click and a programmatic setActive — the ambiguity the echo guard resolves);
 * `fireLayout()` invokes the onDidLayoutChange listener the occupancy reporter
 * registered.
 *
 * Deliberately has NO toJSON: an api that cannot serialize its grid reports
 * nothing, which keeps the settle timer wiring arms at boot inert in every test
 * that is not about the reporter. Reporter tests opt in via {@link withGrid}.
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

/**
 * A serialized grid LEAF (one tile) in the shape `api.toJSON()` emits — `views`
 * is the tile's tab order, one entry under the one-panel-per-tile invariant.
 * @param {...string} views
 */
function leaf(...views) {
  return {
    type: 'leaf',
    data: { views, activeView: views[0], id: `group-${views.join('+') || 'empty'}` },
    size: 500,
  };
}

/**
 * A serialized grid BRANCH: children in spatial order along the branch's own
 * axis — left-to-right for a horizontal branch, top-to-bottom for a vertical
 * one (the root's axis is `grid.orientation`, nested branches alternate).
 * @param {...any} children
 */
function branch(...children) {
  return { type: 'branch', data: children, size: 1000 };
}

/**
 * A serialized dockview layout carrying `root` as its grid tree.
 * @param {any} root  the grid tree, or undefined for a layout with no grid
 */
function gridPayload(root) {
  return {
    grid:
      root === undefined ? undefined : { root, width: 1600, height: 900, orientation: 'HORIZONTAL' },
    panels: {},
    activeGroup: 'group-1',
  };
}

/**
 * A minimal api stub exposing only toJSON — serializeOpenTiles takes the api as
 * an argument, so these tests need no wiring, no DOM, and no dockview.
 * @param {any} root
 */
function apiWithGrid(root) {
  return { toJSON: () => gridPayload(root) };
}

/**
 * Teach a {@link makeApi} stub to serialize `root`, opting that api into the
 * occupancy reporter. Returns the api for chaining; call again to model the
 * operator rearranging tiles before firing the next layout change.
 * @param {any} api
 * @param {any} root
 */
function withGrid(api, root) {
  api.toJSON = () => gridPayload(root);
  return api;
}

describe('serializeOpenTiles — occupancy in spatial reading order', () => {
  test('returns null with no api at all (fallback mode — never an empty report)', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    expect(serializeOpenTiles(null)).toBeNull();
    expect(serializeOpenTiles(undefined)).toBeNull();
  });

  test('returns null for an api that cannot serialize (no toJSON)', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    expect(serializeOpenTiles({})).toBeNull();
  });

  test('returns null when toJSON throws — the layout is unknown, not empty', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const api = {
      toJSON: () => {
        throw new Error('dockview mid-rebuild');
      },
    };

    expect(serializeOpenTiles(api)).toBeNull();
    expect(errSpy).toHaveBeenCalled();
    errSpy.mockRestore();
  });

  test('returns null when the payload carries no grid tree', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    expect(serializeOpenTiles(apiWithGrid(undefined))).toBeNull();
    expect(serializeOpenTiles(apiWithGrid(null))).toBeNull();
  });

  test('an EMPTY dock reports an empty list (distinct from the unknown null)', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    expect(serializeOpenTiles(apiWithGrid(branch()))).toEqual([]);
  });

  test('a dock holding only the terminal reports no tiles', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    expect(serializeOpenTiles(apiWithGrid(branch(leaf('terminal'))))).toEqual([]);
  });

  test('maps placeholder ids back to service ids and drops the native tiles', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(leaf('iframe:ariel'), leaf('terminal'));

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['ariel']);
  });

  test('a horizontal split reports left-to-right, terminal excluded from the order', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(leaf('iframe:ariel'), leaf('iframe:bluesky'), leaf('terminal'));

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['ariel', 'bluesky']);
  });

  test('a single group holding several placeholders reports them in tab order', async () => {
    // The one-panel-per-tile invariant makes this a pre-strict-model layout, but
    // the serializer must still describe it rather than drop the stack.
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(leaf('iframe:ariel', 'iframe:bluesky', 'iframe:okf'));

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['ariel', 'bluesky', 'okf']);
  });

  test('a NESTED branch (what a rail-drag vertical split produces) flattens in child order', async () => {
    // Named for what this can actually observe: a serialized branch carries no
    // orientation, so the axis itself is not visible here — the browser suite
    // pins that a vertical split reads top-to-bottom.
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(branch(leaf('iframe:ariel'), leaf('iframe:bluesky')));

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['ariel', 'bluesky']);
  });

  test('a 2D arrangement flattens column-first: a stacked pair before its neighbor', async () => {
    // ariel over bluesky on the left, okf full-height on the right. Flattened,
    // not raster-scanned — pixel geometry is out of scope for the report.
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(
      branch(leaf('iframe:ariel'), leaf('iframe:bluesky')),
      leaf('iframe:okf'),
      leaf('terminal'),
    );

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['ariel', 'bluesky', 'okf']);
  });

  test('nests to arbitrary depth', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(
      leaf('iframe:a'),
      branch(leaf('iframe:b'), branch(leaf('iframe:c'), leaf('terminal'))),
      leaf('iframe:d'),
    );

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['a', 'b', 'c', 'd']);
  });

  test('a leaf with a bare (non-branch) root is walked too', async () => {
    // dockview requires a branch root on fromJSON, but toJSON is read here
    // without that assumption.
    const { serializeOpenTiles } = await import(SYNC);
    expect(serializeOpenTiles(apiWithGrid(leaf('iframe:ariel')))).toEqual(['ariel']);
  });

  test('an UNRECOGNIZED ROOT is unreadable (null), never an affirmative empty workspace', async () => {
    // The whole tree is unreadable, so there is nothing to assert about
    // occupancy — distinct from an inner node, whose siblings still describe
    // real tiles and which is therefore skipped rather than fatal.
    const { serializeOpenTiles } = await import(SYNC);

    expect(serializeOpenTiles(apiWithGrid({ type: 'mystery', data: [] }))).toBeNull();
    expect(serializeOpenTiles(apiWithGrid({ data: [] }))).toBeNull();
    expect(serializeOpenTiles(apiWithGrid({ type: 'leaf', data: { views: [] } }))).toEqual([]);
  });

  test('malformed nodes are skipped, not thrown on', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(
      null,
      { type: 'leaf' },
      { type: 'leaf', data: { views: 'not-an-array' } },
      { type: 'mystery', data: [leaf('iframe:ignored')] },
      leaf('iframe:ariel'),
    );

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['ariel']);
  });

  test('a panel referenced twice by a malformed tree is reported once', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(leaf('iframe:ariel'), leaf('iframe:bluesky'), leaf('iframe:ariel'));

    expect(serializeOpenTiles(apiWithGrid(root))).toEqual(['ariel', 'bluesky']);
  });

  test('does not mutate the serialized layout it reads', async () => {
    const { serializeOpenTiles } = await import(SYNC);
    const root = branch(branch(leaf('iframe:ariel'), leaf('terminal')), leaf('iframe:okf'));
    const before = JSON.stringify(root);

    serializeOpenTiles(apiWithGrid(root));

    expect(JSON.stringify(root)).toBe(before);
  });
});

describe('occupancy reporter — settle debounce, content dedupe, capability', () => {
  const SETTLE = 250;

  /**
   * Wire dock-sync against an api that serializes `root`, then fire the
   * boot-layout-settled hook — the normal boot, where dock-workspace finishes
   * deciding the starting layout. Reporting is inert until that fires, so tests
   * about steady-state behavior go through here; tests about the boot window
   * itself withhold it (pass `settled: false`) and fire state.restoreHook
   * themselves.
   * @param {any} root
   * @param {{settled?: boolean}} [opts]
   */
  async function wireReporting(root, { settled = true } = {}) {
    const api = withGrid(makeApi(), root);
    const { mod } = await wire(api);
    if (settled) state.restoreHook?.();
    return { api, mod };
  }

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('a SLOW restore never publishes the pre-restore layout as fact', async () => {
    // The critical boot case. dock-workspace restores only after an /api/panels
    // round trip; if reporting ran on a deadline, a fetch slower than that
    // deadline would publish "no tiles open" as a confident fact, and a hook
    // reading /api/panels in that window would believe it.
    const { api } = await wireReporting(branch(leaf('terminal')), { settled: false });

    // Far past any settle window, with the restore still in flight.
    await vi.advanceTimersByTimeAsync(SETTLE * 20);
    expect(reportPanelLayout).not.toHaveBeenCalled();

    // The restore finally lands, and only now is there a layout worth reporting.
    withGrid(api, branch(leaf('iframe:ariel'), leaf('iframe:okf'), leaf('terminal')));
    state.restoreHook();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel', 'okf'], true);
    expect(reportPanelLayout).not.toHaveBeenCalledWith([], true);
  });

  test('layout churn before the restore is ignored, not merely debounced', async () => {
    const { api } = await wireReporting(branch(leaf('terminal')), { settled: false });

    // The adapter docking its first placeholders, pre-restore.
    withGrid(api, branch(leaf('iframe:ariel'), leaf('terminal')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    withGrid(api, branch(leaf('terminal')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);

    expect(reportPanelLayout).not.toHaveBeenCalled();
  });

  test('the boot deadline is a backstop when the restore hook never fires', async () => {
    // A dock-workspace that never reached initPersistence, or a fetch that
    // neither resolves nor rejects: reporting must not be disabled forever.
    const { api } = await wireReporting(branch(leaf('iframe:ariel')), { settled: false });

    await vi.advanceTimersByTimeAsync(SETTLE * 8);
    expect(reportPanelLayout).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(10000);
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);
    expect(api.onDidLayoutChange).toHaveBeenCalledTimes(1);
  });

  test('a restore arriving after the deadline is an ordinary layout change', async () => {
    const { api } = await wireReporting(branch(leaf('terminal')), { settled: false });
    await vi.advanceTimersByTimeAsync(10000 + SETTLE);
    reportPanelLayout.mockClear();

    withGrid(api, branch(leaf('iframe:ariel'), leaf('terminal')));
    state.restoreHook();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);
  });

  test('reports the settled layout once the window expires', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel'), leaf('terminal')));

    expect(api.onDidLayoutChange).toHaveBeenCalledTimes(1);
    expect(reportPanelLayout).not.toHaveBeenCalled(); // still settling

    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);
  });

  test('a burst of layout changes reports ONCE, from the arrangement it settles on', async () => {
    // The shape of a mode flip / reset / fromJSON restore: panels land one at a
    // time, and only the final arrangement may reach the server.
    const { api } = await wireReporting(branch(leaf('terminal')));

    withGrid(api, branch(leaf('iframe:ariel'), leaf('terminal')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(100);
    withGrid(api, branch(leaf('iframe:ariel'), leaf('iframe:bluesky'), leaf('terminal')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(100);
    withGrid(api, branch(leaf('iframe:bluesky'), leaf('iframe:ariel'), leaf('terminal')));
    api.fireLayout();

    expect(reportPanelLayout).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['bluesky', 'ariel'], true);
  });

  test('serializes at FLUSH time — an intermediate arrangement is never reported', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));

    // Mid-rebuild the dock is momentarily empty; it settles holding okf.
    withGrid(api, branch());
    api.fireLayout();
    withGrid(api, branch(leaf('iframe:okf')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['okf'], true);
  });

  test('an api.clear()/fromJSON rebuild never records its transient EMPTY window', async () => {
    // The correctness case behind subscribe-after-apply + settle debounce: a
    // rebuild legitimately serializes to [] between clear() and fromJSON, and []
    // is a VALID report ("operator closed everything"), so nothing downstream
    // could tell that transient from real occupancy. Only the debounce can — the
    // window must never expire mid-rebuild.
    const { api } = await wireReporting(branch(leaf('iframe:ariel'), leaf('terminal')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    reportPanelLayout.mockClear();

    withGrid(api, branch()); // api.clear()
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE - 50);
    withGrid(api, branch(leaf('terminal'))); // fromJSON, panel by panel
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE - 50);
    withGrid(api, branch(leaf('iframe:okf'), leaf('terminal')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['okf'], true);
    expect(reportPanelLayout).not.toHaveBeenCalledWith([], true);
  });

  test('an unchanged layout is not re-POSTed (client half of the dedupe contract)', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);

    // A sash drag or a focus change settles the layout without moving a tile.
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledTimes(1);
  });

  test('a genuine change after a deduped repeat still reports', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);

    api.fireLayout(); // same list — deduped
    await vi.advanceTimersByTimeAsync(SETTLE);
    withGrid(api, branch(leaf('iframe:ariel'), leaf('iframe:okf')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['ariel', 'okf'], true);
  });

  test('REORDERING the same panels is a change, not a duplicate', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel'), leaf('iframe:okf')));
    await vi.advanceTimersByTimeAsync(SETTLE);

    withGrid(api, branch(leaf('iframe:okf'), leaf('iframe:ariel')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['okf', 'ariel'], true);
  });

  test('closing the last service tile reports an empty list (a real report)', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel'), leaf('terminal')));
    await vi.advanceTimersByTimeAsync(SETTLE);

    withGrid(api, branch(leaf('terminal')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenLastCalledWith([], true);
  });

  test('reports during an echo-guarded apply — the guard must NOT gate it', async () => {
    // dockview defers onDidLayoutChange to a microtask, so it lands after the
    // synchronous suppress window closes; gating here would suppress nothing
    // while looking correct. Dedupe is what stops the loop instead.
    const { api, mod } = await wireReporting(branch(leaf('terminal')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    reportPanelLayout.mockClear();

    mod.withEchoSuppressed(() => {
      withGrid(api, branch(leaf('iframe:ariel'), leaf('terminal')));
      api.fireLayout();
    });
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);
  });

  test('an applied arrangement settles in ONE report, then goes quiet (no loop)', async () => {
    // The convergence criterion: a server-applied arrange fires layout changes,
    // which report once; the induced report matches, so nothing follows.
    const { api } = await wireReporting(branch(leaf('terminal')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    reportPanelLayout.mockClear();

    withGrid(api, branch(leaf('iframe:ariel'), leaf('iframe:okf'), leaf('terminal')));
    api.fireLayout();
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);

    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);
  });

  test('an api that cannot serialize reports nothing and leaves the baseline intact', async () => {
    const api = makeApi(); // no toJSON — occupancy is unknown, not empty
    await wire(api);
    state.restoreHook();

    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).not.toHaveBeenCalled();

    // Once it can serialize, the first real report still goes out.
    withGrid(api, branch(leaf('iframe:ariel')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);
  });

  test('a dock torn down inside the settle window reports nothing', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    api.fireLayout();
    state.api = null; // shell gone before the window expires

    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).not.toHaveBeenCalled();
  });

  test('fallback mode sends ONE dock:false capability report and no tile reports', async () => {
    const mod = await import(SYNC);
    mod.initDockSync(); // no api, no #dock-root — the shell never arrives

    await vi.advanceTimersByTimeAsync(150 * 33); // exhaust the bounded retry budget
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith([], false);

    // Re-entering after the budget is spent must not re-report.
    mod.initDockSync();
    mod.initDockSync();
    await vi.advanceTimersByTimeAsync(150 * 10);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);
  });

  test('a dock:false claim is RETRACTED when a slow shell finally arrives', async () => {
    // The fast budget expiring means "no dock seen yet", not "no dock exists".
    // A shell that boots slowly must be able to correct the record rather than
    // leaving the server believing this client is dockless.
    const mod = await import(SYNC);
    mod.initDockSync();
    await vi.advanceTimersByTimeAsync(150 * 33);
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith([], false);

    // The shell shows up late; the slow poll is still watching.
    const api = withGrid(makeApi(), branch(leaf('iframe:ariel'), leaf('terminal')));
    state.api = api;
    const root = document.createElement('div');
    root.id = 'dock-root';
    document.body.appendChild(root);
    await vi.advanceTimersByTimeAsync(2000);
    expect(api.onDidActivePanelChange).toHaveBeenCalledTimes(1);

    state.restoreHook();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['ariel'], true);
  });

  test('the slow poll is finite — a page with no dock stops watching', async () => {
    const mod = await import(SYNC);
    mod.initDockSync();
    await vi.advanceTimersByTimeAsync(150 * 33 + 1000 * 70); // past both budgets

    const api = makeApi();
    state.api = api;
    const root = document.createElement('div');
    root.id = 'dock-root';
    document.body.appendChild(root);
    await vi.advanceTimersByTimeAsync(1000 * 10);

    expect(api.onDidActivePanelChange).not.toHaveBeenCalled();
  });

  test('a deduped no-op ack (updated:false) still sets the dedupe baseline', async () => {
    // The server already held this list, so it is acknowledged even though
    // nothing changed — the client must treat it as reported, not retry it.
    reportPanelLayout.mockImplementation((/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
      Promise.resolve({ status: 'ok', tiles, dock, updated: false }),
    );
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);

    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);
  });

  test('a report that never LANDED is retried, not remembered as sent', async () => {
    // Optimistically marking a failed POST as reported would dedupe that layout
    // away forever, leaving the server permanently wrong about occupancy.
    reportPanelLayout.mockImplementation(() => Promise.resolve(null));
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);

    api.fireLayout(); // same layout — but the server never got it
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['ariel'], true);

    // Once a report lands, the baseline takes hold and dedupe resumes.
    reportPanelLayout.mockImplementation((/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
      Promise.resolve({ status: 'ok', tiles, dock, updated: true }),
    );
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(3);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(3);
  });

  test('a REJECTED report does not strand the in-flight flag and kill reporting', async () => {
    // reportPanelLayout is documented never to reject, but that contract lives in
    // another module. Were it ever broken, an unhandled rejection would leave the
    // serialization flag raised and silently suppress every later report — the one
    // failure mode the whole design errs away from. Pinned so it stays recoverable.
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    reportPanelLayout.mockImplementation(() => Promise.reject(new Error('contract broken')));
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1);
    expect(errSpy).toHaveBeenCalled();

    // The reporter is still alive: the next layout change reports normally.
    reportPanelLayout.mockImplementation((/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
      Promise.resolve({ status: 'ok', tiles, dock, updated: true }),
    );
    withGrid(api, branch(leaf('iframe:ariel'), leaf('iframe:okf')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['ariel', 'okf'], true);
    errSpy.mockRestore();
  });

  test('the baseline follows the ACKNOWLEDGED tiles, not the ones sent', async () => {
    // A server that normalized the list would otherwise leave the client
    // re-POSTing forever; the baseline is whatever the server says it holds.
    reportPanelLayout.mockImplementation((/** @type {string[]} */ _tiles, /** @type {boolean} */ dock) =>
      Promise.resolve({ status: 'ok', tiles: ['ariel'], dock, updated: true }),
    );
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);

    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);

    expect(reportPanelLayout).toHaveBeenCalledTimes(1);
  });

  test('a second report never overlaps one in flight — it waits for the ack', async () => {
    // Two concurrent reports could be applied by the server in one order and
    // acknowledged in the other, and no client-side rule could tell which write
    // stuck. Overlap is therefore prevented rather than arbitrated.
    /** Resolvers for the reports still in flight; calling one delivers its ack.
     *  @type {(() => void)[]} */
    const pending = [];
    reportPanelLayout.mockImplementation((/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
      new Promise((resolve) => pending.push(() => resolve({
        status: 'ok', tiles, dock, updated: true,
      }))),
    );

    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1); // first report, unacked

    // The operator keeps working while that report is still outstanding.
    withGrid(api, branch(leaf('iframe:ariel'), leaf('iframe:okf')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(reportPanelLayout).toHaveBeenCalledTimes(1); // still only the first

    // Once it acks, the deferred report goes out — with the CURRENT layout.
    pending[0]();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['ariel', 'okf'], true);
  });

  test('a late ack cannot rewrite a baseline set by a newer report', async () => {
    // The out-of-order-acks scenario, made unreachable by serialization: because
    // the second report is not sent until the first is acknowledged, their acks
    // cannot race. Pinned so a future change that reintroduces overlap fails here.
    /** @type {(() => void)[]} */
    const pending = [];
    reportPanelLayout.mockImplementation((/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
      new Promise((resolve) => pending.push(() => resolve({
        status: 'ok', tiles, dock, updated: true,
      }))),
    );

    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    withGrid(api, branch(leaf('iframe:okf')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 2);

    expect(pending).toHaveLength(1); // never two in flight, so no ack race exists
    pending[0]();
    await vi.advanceTimersByTimeAsync(SETTLE);
    pending[1]();
    await vi.advanceTimersByTimeAsync(SETTLE);

    // Baseline ended on the newer list: a repeat of it is deduped away.
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['okf'], true);
  });

  test('a STALE ack causes a redundant report, never suppression', async () => {
    // The invariant behind the module header's "no sequence guard" rule. Feed an
    // ack that names an older list than the one just sent — what a reordered or
    // superseded response would carry — and the client must end up re-reporting
    // the real layout. Erring toward a redundant POST is recoverable (the server
    // dedupes it); erring toward suppression is not, because a suppressed report
    // leaves the server wrong with nothing scheduled to correct it.
    reportPanelLayout.mockImplementation((/** @type {string[]} */ _tiles, /** @type {boolean} */ dock) =>
      Promise.resolve({ status: 'ok', tiles: ['stale-older-list'], dock, updated: true }),
    );
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);

    // Layout unchanged, but the baseline disagrees with it, so it reports again
    // rather than concluding the server already knows.
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
    expect(reportPanelLayout).toHaveBeenLastCalledWith(['ariel'], true);
  });

  test('a client WITH a dock never sends the dock:false capability report', async () => {
    await wireReporting(branch(leaf('iframe:ariel')));
    await vi.advanceTimersByTimeAsync(SETTLE * 4);

    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);
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

describe('human tab close → local vacate handler (never a POST)', () => {
  test('clicking a service tab’s close control fires the registered handler once', async () => {
    const api = makeApi();
    const { mod, root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel', 'terminal']);

    tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(tileClose).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(setPanelVisibility).not.toHaveBeenCalled();
    expect(setPanelFocus).not.toHaveBeenCalled();
    expect(mod).toBeTruthy();
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

    // The tab body, not its `.dv-default-tab-action` close control.
    tabs[0].tab.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(tileClose).not.toHaveBeenCalled();
  });

  test('a close click on an unstamped tab (no data-panel-id) is a silent no-op', async () => {
    const api = makeApi();
    const { root } = await wire(api);
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);
    tabs[0].tile.removeAttribute('data-panel-id'); // stamp lost — nothing to resolve

    expect(() =>
      tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true })),
    ).not.toThrow();
    expect(tileClose).not.toHaveBeenCalled();
  });

  test('a close click with NO registered handler is a silent no-op', async () => {
    const api = makeApi();
    const { mod, root } = await wire(api);
    mod.setTileCloseHandler(null);
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);

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

  test('is a no-op in simple mode — the locked layout has exactly one service tile', async () => {
    // The add-menu pick then falls through to the plain show path, which takes
    // the single tile over like a rail click (one panel per tile).
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    try {
      const api = makeApi();
      api.activeGroup = { id: 'group-1' };
      const { mod } = await wire(api);

      mod.dockPanelBesideActive('ariel');

      expect(api.addPanel).not.toHaveBeenCalled();
    } finally {
      document.documentElement.removeAttribute('data-ui-mode');
    }
  });

  test('MOVES an already-docked placeholder beside the active group (remove + re-add)', async () => {
    const api = makeApi();
    const activeGroup = { id: 'group-1' };
    api.activeGroup = activeGroup;
    const existing = { id: 'iframe:ariel', group: { id: 'group-0' } };
    api._panels['iframe:ariel'] = existing;
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel', 'ARIEL');

    expect(api.removePanel).toHaveBeenCalledExactlyOnceWith(existing);
    expect(api._added[0]).toEqual({
      id: 'iframe:ariel',
      component: 'dock-iframe-placeholder',
      title: 'ARIEL',
      position: { referenceGroup: activeGroup, direction: 'right' },
    });
  });

  test('a move is echo-guarded — the removal\'s auto-activation does not POST a focus', async () => {
    const api = makeApi();
    api.activeGroup = { id: 'group-1' };
    api._panels['iframe:ariel'] = { id: 'iframe:ariel', group: { id: 'group-0' } };
    api._activateOnRemove = { id: 'iframe:bluesky' };
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel');

    expect(setPanelFocus).not.toHaveBeenCalled();
  });

  test('already docked IN the active group: a no-op (never remove a tile to re-create it in place)', async () => {
    const api = makeApi();
    const activeGroup = { id: 'group-1' };
    api.activeGroup = activeGroup;
    api._panels['iframe:ariel'] = { id: 'iframe:ariel', group: activeGroup };
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel');

    expect(api.removePanel).not.toHaveBeenCalled();
    expect(api.addPanel).not.toHaveBeenCalled();
  });

  test('already docked with NO active group: a no-op (no meaningful target)', async () => {
    const api = makeApi();
    api.activeGroup = null;
    api._panels['iframe:ariel'] = { id: 'iframe:ariel', group: { id: 'group-0' } };
    const { mod } = await wire(api);

    mod.dockPanelBesideActive('ariel');

    expect(api.removePanel).not.toHaveBeenCalled();
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

describe('dockPanelAt — placement at an explicit dock position (rail drag / drop routing)', () => {
  test('adds a fresh placeholder at the given group + direction', async () => {
    const api = makeApi();
    const target = { id: 'group-2' };
    const { mod } = await wire(api);

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: target, direction: 'bottom' });

    expect(api._added[0]).toEqual({
      id: 'iframe:okf',
      component: 'dock-iframe-placeholder',
      title: 'KNOWLEDGE',
      position: { referenceGroup: target, direction: 'bottom' },
    });
  });

  test('moves an existing placeholder to the given position', async () => {
    const api = makeApi();
    const target = { id: 'group-2' };
    const existing = { id: 'iframe:okf', group: { id: 'group-0' } };
    api._panels['iframe:okf'] = existing;
    const { mod } = await wire(api);

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: target, direction: 'left' });

    expect(api.removePanel).toHaveBeenCalledExactlyOnceWith(existing);
    expect(api._added[0].position).toEqual({ referenceGroup: target, direction: 'left' });
  });

  test('dropping a panel onto its OWN group is a no-op (would destroy the tile mid-move)', async () => {
    const api = makeApi();
    const home = { id: 'group-0' };
    api._panels['iframe:okf'] = { id: 'iframe:okf', group: home };
    const { mod } = await wire(api);

    mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: home, direction: 'right' });

    expect(api.removePanel).not.toHaveBeenCalled();
    expect(api.addPanel).not.toHaveBeenCalled();
  });

  test('is a no-op in simple mode (locked layout)', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    try {
      const api = makeApi();
      const { mod } = await wire(api);
      mod.dockPanelAt('okf', 'KNOWLEDGE', { referenceGroup: { id: 'g' }, direction: 'right' });
      expect(api.addPanel).not.toHaveBeenCalled();
    } finally {
      document.documentElement.removeAttribute('data-ui-mode');
    }
  });
});

describe('initDockSync — wiring, idempotency, late arrival', () => {
  test('wires the active-panel listener and the capture-phase close click when api + root exist', async () => {
    const api = makeApi();
    const { root } = await wire(api);

    expect(api.onDidActivePanelChange).toHaveBeenCalledTimes(1);

    // The click listener is live: a service close routes to the vacate handler.
    const { tabs } = buildGroup(root, api, ['iframe:ariel']);
    tabs[0].action.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(tileClose).toHaveBeenCalledExactlyOnceWith('ariel');
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

describe('module isolation (vi.resetModules per test)', () => {
  test('exposes exactly its public surface as functions', async () => {
    const mod = await import(SYNC);
    for (const name of ['withEchoSuppressed', 'dockPanelBesideActive', 'dockPanelAt', 'serviceIdOf', 'initDockSync', 'setTileCloseHandler']) {
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
