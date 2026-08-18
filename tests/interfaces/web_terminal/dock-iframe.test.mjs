// @ts-check
/**
 * Unit tests for the dock iframe adapter's PLACEMENT engine (dock-iframe.js):
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-iframe.test.mjs
 *
 * One panel per tile is a workspace invariant — the rail is the tab system.
 * These tests pin the placement rules that enforce it:
 *
 *   - the first service placeholder docks LEFT of the terminal at the 60/40
 *     split;
 *   - an ACTIVATION ('replace' intent) takes over the target service tile:
 *     the previous occupant's placeholder is evicted and reported through the
 *     replaced-panel handler (panel-manager closes it server-side);
 *   - the terminal is never evicted;
 *   - focusing a panel that already has a tile jumps to it — no new tile, no
 *     eviction;
 *   - a REDOCK after a layout rebuild ('beside' intent) re-materializes each
 *     visible panel as its own tile — they must not evict one another;
 *   - hidePanel drops a placeholder even with no managed entry (a tile
 *     restored from storage for a panel closed elsewhere);
 *   - pruning removes placeholders of server-closed panels once the visible
 *     set is seeded, and leaves everything alone before that.
 *
 * dockview and dock-workspace are stubbed at the module boundary: getDockApi
 * yields a hand-built fake api whose addPanel/removePanel model dockview's
 * group bookkeeping (a 'within' add joins the reference group; anything else
 * opens a new group; removal collapses an emptied group). Each test re-imports
 * the adapter fresh (vi.resetModules) so its module state never leaks.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const ADAPTER = '../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js';

const { getDockApi, state, redock } = vi.hoisted(() => ({
  state: { api: /** @type {any} */ (null) },
  redock: { fn: /** @type {null | (() => void)} */ (null) },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
}));
getDockApi.mockImplementation(() => state.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  defaultServiceWidth: () => 600,
  setServiceRedock: (/** @type {() => void} */ fn) => { redock.fn = fn; },
  onDragGesture: () => [],
}));

/** The adapter's live-follow observer; geometry itself is browser-suite turf. */
class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.api = null;
  redock.fn = null;
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  document.body.innerHTML = '<div class="main-container"></div>';
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/**
 * A hand-built DockviewApi stand-in modeling exactly the group bookkeeping the
 * placement engine relies on. addPanel with `direction: 'within'` joins the
 * reference group (the stacking dockview would do); any other position opens a
 * fresh group. The added panel becomes active (dockview's default), firing the
 * active-panel listeners synchronously.
 * @returns {any}
 */
function makeApi() {
  let groupSeq = 0;
  /** @type {any} */
  const api = {
    activePanel: null,
    groups: /** @type {any[]} */ ([]),
    panels: /** @type {any[]} */ ([]),
    _added: /** @type {any[]} */ ([]),
    _activeCbs: /** @type {(() => void)[]} */ ([]),
    onDidLayoutChange: vi.fn(() => ({ dispose() {} })),
    onDidActivePanelChange: vi.fn((/** @type {() => void} */ cb) => {
      api._activeCbs.push(cb);
      return { dispose() {} };
    }),
    getPanel: (/** @type {string} */ id) => api.panels.find((/** @type {any} */ p) => p.id === id) ?? null,
    addPanel: (/** @type {any} */ opts) => {
      api._added.push(opts);
      const group = opts.position?.referenceGroup && opts.position.direction === 'within'
        ? opts.position.referenceGroup
        : makeGroup();
      const panel = { id: opts.id, title: opts.title, group, api: { setActive: vi.fn() } };
      group.panels.push(panel);
      group.activePanel = panel;
      api.panels.push(panel);
      api.activePanel = panel;
      for (const cb of api._activeCbs) cb();
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
      if (api.activePanel === panel) api.activePanel = group.panels[0] ?? api.panels[0] ?? null;
    },
  };
  function makeGroup() {
    const element = document.createElement('div');
    const content = document.createElement('div');
    content.className = 'dv-content-container';
    element.appendChild(content);
    const group = { id: `group-${++groupSeq}`, panels: [], activePanel: null, element };
    api.groups.push(group);
    return group;
  }
  api._makeGroup = makeGroup;
  return api;
}

/** Seed the fake api with the native terminal card in its own group. @param {any} api */
function addTerminal(api) {
  const group = api._makeGroup();
  const terminal = { id: 'terminal', group, api: { setActive: vi.fn() } };
  group.panels.push(terminal);
  group.activePanel = terminal;
  api.panels.push(terminal);
  api.activePanel = terminal;
  return terminal;
}

function makeIframe() {
  return document.createElement('iframe');
}

/** @param {any} api */
async function freshAdapter(api) {
  state.api = api;
  const mod = await import(ADAPTER);
  mod.initDockIframeAdapter({ fallbackHost: null });
  return mod;
}

describe('first placement — the classic 60/40 split', () => {
  test('the first service placeholder docks LEFT of the terminal at the default width', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);

    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });

    expect(api._added).toHaveLength(1);
    expect(api._added[0]).toMatchObject({
      id: 'iframe:artifacts',
      position: { referencePanel: 'terminal', direction: 'left' },
      initialWidth: 600,
    });
  });
});

describe("activation ('replace') — the new panel takes over the tile", () => {
  test('the previous occupant is evicted and reported; the tile holds exactly the new panel', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    const replaced = vi.fn();
    mod.setReplacedPanelHandler(replaced);
    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });

    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });

    const arielAdd = api._added.find((/** @type {any} */ o) => o.id === 'iframe:ariel');
    expect(arielAdd.position.direction).toBe('within'); // joins the tile, then evicts
    expect(api.getPanel('iframe:artifacts')).toBeNull();
    expect(replaced).toHaveBeenCalledExactlyOnceWith('artifacts');
    const tile = api.getPanel('iframe:ariel').group;
    expect(tile.panels.map((/** @type {any} */ p) => p.id)).toEqual(['iframe:ariel']);
  });

  test('the evicted iframe is concealed but kept cached for a later re-activation', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    const replaced = vi.fn();
    mod.setReplacedPanelHandler(replaced);
    const artifactsFrame = makeIframe();
    mod.adoptIframe('artifacts', artifactsFrame, { title: 'WORKSPACE' });

    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    expect(artifactsFrame.style.display).toBe('none');
    expect(artifactsFrame.isConnected).toBe(true); // cached in the overlay, state intact

    // Re-activating the evicted panel takes the tile back the same way.
    mod.focusPanel('artifacts');
    expect(api.getPanel('iframe:artifacts')).not.toBeNull();
    expect(api.getPanel('iframe:ariel')).toBeNull();
    expect(replaced).toHaveBeenLastCalledWith('ariel');
  });

  test('with the terminal focused, the take-over targets the last-focused service tile — never the terminal', async () => {
    const api = makeApi();
    const terminal = addTerminal(api);
    const mod = await freshAdapter(api);
    const replaced = vi.fn();
    mod.setReplacedPanelHandler(replaced);
    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });

    api.activePanel = terminal; // operator clicked back into the chat
    mod.adoptIframe('okf', makeIframe(), { title: 'KNOWLEDGE' });

    expect(api.getPanel('terminal')).not.toBeNull();
    expect(api.getPanel('iframe:artifacts')).toBeNull(); // the service tile was taken over
    expect(replaced).toHaveBeenCalledExactlyOnceWith('artifacts');
  });
});

describe('focusing an existing tile — jump, not move', () => {
  test('a panel that already has a tile is activated in place; other tiles are untouched', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });
    // A second tile beside the first (the "+" add-menu path pre-creates it).
    api.addPanel({ id: 'iframe:ariel', component: 'dock-iframe-placeholder', title: 'ARIEL' });
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });
    const addsBefore = api._added.length;

    mod.focusPanel('artifacts');

    expect(api._added).toHaveLength(addsBefore); // no new tile
    expect(api.getPanel('iframe:artifacts').api.setActive).toHaveBeenCalled();
    expect(api.getPanel('iframe:ariel')).not.toBeNull(); // no eviction
  });
});

describe("redock after a layout rebuild ('beside') — tiles do not evict one another", () => {
  test('each visible panel re-materializes as its own tile', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    const replaced = vi.fn();
    mod.setReplacedPanelHandler(replaced);
    mod.adoptIframe('artifacts', makeIframe(), { title: 'WORKSPACE' });
    api.addPanel({ id: 'iframe:ariel', component: 'dock-iframe-placeholder', title: 'ARIEL' });
    mod.adoptIframe('ariel', makeIframe(), { title: 'ARIEL' });

    // A layout rebuild wiped the grid; only the terminal came back.
    api.panels = [];
    api.groups = [];
    addTerminal(api);
    api._added.length = 0;
    if (!redock.fn) throw new Error('adapter never registered its redock hook');
    redock.fn();

    expect(api.getPanel('iframe:artifacts')).not.toBeNull();
    expect(api.getPanel('iframe:ariel')).not.toBeNull();
    expect(replaced).not.toHaveBeenCalled();
    // First tile re-opens at the terminal split; the second splits beside it.
    expect(api._added[0].position).toMatchObject({ referencePanel: 'terminal', direction: 'left' });
    expect(api._added[1].position.direction).toBe('right');
    expect(api._added[1].position.referenceGroup).toBe(api.getPanel('iframe:artifacts').group);
  });
});

describe('hidePanel — tiles never outlive their panel', () => {
  test('drops a placeholder even when the adapter never adopted the panel', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    api.addPanel({ id: 'iframe:ghost', component: 'dock-iframe-placeholder', title: 'GHOST' });

    mod.hidePanel('ghost');

    expect(api.getPanel('iframe:ghost')).toBeNull();
  });
});

describe('pruning — placeholders of server-closed panels', () => {
  test('a known but server-closed panel loses its restored tile', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    api.addPanel({ id: 'iframe:lattice', component: 'dock-iframe-placeholder', title: 'LATTICE' });

    mod.setServerVisiblePanels(new Set(['artifacts']));
    mod.setKnownServicePanels(['artifacts', 'lattice']);

    expect(api.getPanel('iframe:lattice')).toBeNull();
  });

  test('a server-visible panel keeps its restored tile', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    api.addPanel({ id: 'iframe:lattice', component: 'dock-iframe-placeholder', title: 'LATTICE' });

    mod.setServerVisiblePanels(new Set(['lattice']));
    mod.setKnownServicePanels(['lattice']);

    expect(api.getPanel('iframe:lattice')).not.toBeNull();
  });

  test('before the visible set is seeded, known placeholders are left alone', async () => {
    // An early layout restore must not drop every placeholder just because
    // panel-manager has not seeded the server-visible set yet.
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    api.addPanel({ id: 'iframe:lattice', component: 'dock-iframe-placeholder', title: 'LATTICE' });

    mod.setKnownServicePanels(['lattice']);

    expect(api.getPanel('iframe:lattice')).not.toBeNull();
  });
});

describe('sash drags — dockview owns the iframe shield (#638)', () => {
  /**
   * dockview-core 7.x shields iframes during sash drags itself
   * (disableIframePointEvents in its splitview): on sash pointerdown it
   * SNAPSHOTS each iframe's inline pointer-events into a WeakMap and sets
   * 'none'; its document-level, bubble-phase pointerup handler restores the
   * snapshot. The adapter must not touch pointer-events around that gesture:
   * an adapter shield running in the capture phase sets 'none' BEFORE dockview
   * snapshots, so dockview records 'none' as the original value and its
   * restore re-freezes every overlay iframe after the drag — permanently,
   * since each further drag re-snapshots the poisoned value (the #638 freeze).
   * This test emulates dockview's exact contract and pins that a full sash
   * gesture ends with the managed iframe interactive.
   */
  test('a full sash gesture leaves managed overlay iframes interactive', async () => {
    const api = makeApi();
    addTerminal(api);
    const mod = await freshAdapter(api);
    const iframe = makeIframe();
    mod.adoptIframe('artifacts', iframe, { title: 'WORKSPACE' });
    expect(iframe.style.pointerEvents).toBe('auto');

    const sash = document.createElement('div');
    sash.className = 'dv-sash';
    document.body.appendChild(sash);

    // dockview's splitview sash handling, verbatim contract: target-phase
    // pointerdown snapshot, bubble-phase document pointerup restore.
    const snapshot = new WeakMap();
    sash.addEventListener('pointerdown', () => {
      for (const f of document.querySelectorAll('iframe')) {
        snapshot.set(f, f.style.pointerEvents);
        f.style.pointerEvents = 'none';
      }
      const release = () => {
        document.removeEventListener('pointerup', release);
        for (const f of document.querySelectorAll('iframe')) {
          f.style.pointerEvents = snapshot.get(f) ?? 'auto';
        }
      };
      document.addEventListener('pointerup', release);
    });

    sash.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    document.dispatchEvent(new Event('pointerup', { bubbles: true }));

    expect(iframe.style.pointerEvents).toBe('auto');

    // A second gesture must also end interactive (the old shield could never
    // self-heal: every drag re-snapshotted the poisoned 'none').
    sash.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    document.dispatchEvent(new Event('pointerup', { bubbles: true }));

    expect(iframe.style.pointerEvents).toBe('auto');
  });
});
