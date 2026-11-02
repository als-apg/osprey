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
 *   - the OCCUPANCY REPORT reads the layout holder (never dockview's toJSON);
 *   - dockPanelBesideActive and dockPanelAt run one engine op on the layout
 *     holder, which applies it under the same guard;
 *   - the listeners wire idempotently and tolerate late DockviewApi arrival.
 *
 * dockview and the collaborators are stubbed at the module boundary: getDockApi
 * (dock-workspace.js) yields a hand-built fake api, getLayoutHolder (dock-apply.js)
 * a fake holder (or a real one over the fake api, for the placement verbs), and
 * setPanelFocus / setPanelVisibility (panel-commands.js) are spies. Each test
 * re-imports dock-sync fresh (vi.resetModules) so its module-scoped guard/wire
 * state never leaks across tests.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SYNC = '../../../src/osprey/interfaces/web_terminal/static/js/dock-sync.js';

// Boundary stubs. Hoisted so the spies exist before dock-sync's static imports
// resolve; the same spy instances persist across resetModules (they are closed
// over here), while dock-sync's own state is rebuilt on each fresh import.
const {
  getDockApi,
  setPanelFocus,
  setPanelVisibility,
  reportPanelLayout,
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
  // Resolves with the server's acknowledgement; the reporter sets its dedupe
  // baseline from that, so tests can model a lost report by resolving null.
  reportPanelLayout: vi.fn(),
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
  reportPanelLayout,
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
const { isPoppedOut } = vi.hoisted(() => ({
  isPoppedOut: vi.fn(/** @type {(id: string) => boolean} */ (() => false)),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({ isPoppedOut }));

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
  reportPanelLayout.mockImplementation(
    (/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
      Promise.resolve({ status: 'ok', tiles, dock, updated: true })
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
 * The occupancy report reads the holder, never this api; `state.holder` is
 * null by default, which keeps the settle timer wiring arms at boot inert in
 * every test that is not about the reporter. Reporter tests opt in via
 * {@link withGrid}.
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
 * A dock layout TILE holding `views` as its tabs, named by dockview id
 * (`terminal`, or an `iframe:` placeholder) the way the old grid helpers
 * were; the first view shows.
 * @param {...string} views
 */
function leaf(...views) {
  const tabs = views.map((v) => (v.startsWith('iframe:') ? v.slice('iframe:'.length) : v));
  return { tile: `group-${views.join('+')}`, tabs, showing: tabs[0] };
}

/**
 * A dock layout SPLIT: a row of its children in spatial order (a column at the
 * next depth, as the engine alternates). One child is that child; none is the
 * empty dock (null).
 * @param {...any} children
 */
function branch(...children) {
  if (children.length === 0) return null;
  if (children.length === 1) return children[0];
  return { split: 'row', items: children.map((item) => ({ ...item, size: 1 / children.length })) };
}

/**
 * A fake layout holder with the surface the reporter reads: `layout`,
 * `dirty`, `ackedRev` and `ack(rev)` (dock-apply.js's createLayoutHolder).
 * @param {any} layout
 */
function makeHolder(layout) {
  /** @type {any} */
  const holder = { layout, dirty: true, ackedRev: null };
  holder.ack = vi.fn((/** @type {number} */ rev) => {
    holder.ackedRev = rev;
    holder.dirty = false;
  });
  return holder;
}

/**
 * Give the page a holder holding `root`, opting the reporter in. Returns the
 * api for chaining; call again to model the operator rearranging tiles before
 * firing the next layout change.
 * @param {any} api
 * @param {any} root
 */
function withGrid(api, root) {
  state.holder = makeHolder(root);
  return api;
}

describe('openServiceTiles — occupancy in reading order, from the holder', () => {
  test('returns null with no holder (fallback mode — never an empty report)', async () => {
    const { openServiceTiles } = await import(SYNC);
    expect(openServiceTiles()).toBeNull();
  });

  test('an EMPTY dock reports an empty list (distinct from the unknown null)', async () => {
    const { openServiceTiles } = await import(SYNC);
    withGrid(makeApi(), branch());
    expect(openServiceTiles()).toEqual([]);
  });

  test('a dock holding only the terminal reports no tiles', async () => {
    const { openServiceTiles } = await import(SYNC);
    withGrid(makeApi(), branch(leaf('terminal')));
    expect(openServiceTiles()).toEqual([]);
  });

  test('a row reports left-to-right, terminal excluded from the order', async () => {
    const { openServiceTiles } = await import(SYNC);
    withGrid(makeApi(), branch(leaf('iframe:ariel'), leaf('terminal'), leaf('iframe:bluesky')));
    expect(openServiceTiles()).toEqual(['ariel', 'bluesky']);
  });

  test('a tile holding several tabs reports them in tab order', async () => {
    const { openServiceTiles } = await import(SYNC);
    withGrid(makeApi(), branch(leaf('iframe:ariel', 'iframe:bluesky'), leaf('iframe:okf')));
    expect(openServiceTiles()).toEqual(['ariel', 'bluesky', 'okf']);
  });

  test('a 2D arrangement flattens column-first: a stacked pair before its neighbor', async () => {
    const { openServiceTiles } = await import(SYNC);
    const column = {
      split: 'column',
      items: [
        { ...leaf('iframe:ariel'), size: 0.5 },
        { ...leaf('iframe:bluesky'), size: 0.5 },
      ],
    };
    withGrid(makeApi(), branch(column, leaf('iframe:okf')));
    expect(openServiceTiles()).toEqual(['ariel', 'bluesky', 'okf']);
  });

  test('a root tile is read too', async () => {
    const { openServiceTiles } = await import(SYNC);
    withGrid(makeApi(), leaf('iframe:ariel'));
    expect(openServiceTiles()).toEqual(['ariel']);
  });

  test('does not mutate the layout it reads', async () => {
    const { openServiceTiles } = await import(SYNC);
    const root = branch(leaf('iframe:ariel'), leaf('terminal'));
    const before = JSON.stringify(root);
    withGrid(makeApi(), root);
    openServiceTiles();
    expect(JSON.stringify(state.holder.layout)).toBe(before);
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
    // The shape of a reset / fromJSON restore: panels land one at a
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

  test('no holder yet reports nothing and leaves the baseline intact', async () => {
    const api = makeApi(); // no holder — occupancy is unknown, not empty
    await wire(api);
    state.restoreHook();

    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).not.toHaveBeenCalled();

    // Once the holder is there, the first real report still goes out.
    withGrid(api, branch(leaf('iframe:ariel')));
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE);
    expect(reportPanelLayout).toHaveBeenCalledExactlyOnceWith(['ariel'], true);
  });

  test('a dock torn down inside the settle window reports nothing', async () => {
    const { api } = await wireReporting(branch(leaf('iframe:ariel')));
    api.fireLayout();
    state.api = null; // shell gone before the window expires
    state.holder = null;

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

  test('a deduped no-op ack (updated:false) still sets the dedupe baseline', async () => {
    // The server already held this list, so it is acknowledged even though
    // nothing changed — the client must treat it as reported, not retry it.
    reportPanelLayout.mockImplementation(
      (/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
        Promise.resolve({ status: 'ok', tiles, dock, updated: false })
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
    reportPanelLayout.mockImplementation(
      (/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
        Promise.resolve({ status: 'ok', tiles, dock, updated: true })
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
    reportPanelLayout.mockImplementation(
      (/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
        Promise.resolve({ status: 'ok', tiles, dock, updated: true })
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
    reportPanelLayout.mockImplementation(
      (/** @type {string[]} */ _tiles, /** @type {boolean} */ dock) =>
        Promise.resolve({ status: 'ok', tiles: ['ariel'], dock, updated: true })
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
    reportPanelLayout.mockImplementation(
      (/** @type {string[]} */ tiles, /** @type {boolean} */ dock) =>
        new Promise((resolve) =>
          pending.push(() =>
            resolve({
              status: 'ok',
              tiles,
              dock,
              updated: true,
            })
          )
        )
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

    // Its ack sets the baseline to the newer list: a repeat is deduped away.
    pending[1]();
    await vi.advanceTimersByTimeAsync(SETTLE);
    api.fireLayout();
    await vi.advanceTimersByTimeAsync(SETTLE * 4);
    expect(reportPanelLayout).toHaveBeenCalledTimes(2);
  });

  test('a STALE ack causes a redundant report, never suppression', async () => {
    // The invariant behind the module header's "no sequence guard" rule. Feed an
    // ack that names an older list than the one just sent — what a reordered or
    // superseded response would carry — and the client must end up re-reporting
    // the real layout. Erring toward a redundant POST is recoverable (the server
    // dedupes it); erring toward suppression is not, because a suppressed report
    // leaves the server wrong with nothing scheduled to correct it.
    reportPanelLayout.mockImplementation(
      (/** @type {string[]} */ _tiles, /** @type {boolean} */ dock) =>
        Promise.resolve({ status: 'ok', tiles: ['stale-older-list'], dock, updated: true })
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
