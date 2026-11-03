// @ts-check
// @vitest-environment jsdom
/**
 * Unit tests for the dock iframe adapter's PLACEMENT seam (dock-iframe.js):
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-iframe.test.mjs
 *
 * Every open, redock, prune and hide is one operation on the page's layout
 * holder (dock-apply.js), which applies it to dockview. These tests run
 * against the pinned real dockview-core in jsdom and a real holder, and pin:
 *
 *   - the first service placeholder docks LEFT of the terminal at the 60/40
 *     split;
 *   - an OPEN lands as a new tab in front of the service tile the operator
 *     used last: the tab it covers stays, its iframe hidden behind, nothing
 *     resizes and nothing is evicted;
 *   - the terminal's tile never takes a service tab;
 *   - focusing a panel that already has a tab brings it forward in place;
 *   - a REDOCK after a layout rebuild brings every visible panel the layout
 *     lacks back by the same rule — as tabs, so none evicts another;
 *   - hidePanel drops a placeholder even with no managed entry (a tile
 *     restored from storage for a panel closed elsewhere);
 *   - pruning removes placeholders of server-closed panels once the visible
 *     set is seeded, and leaves everything alone before that.
 *
 * dock-workspace is stubbed at the module boundary: getDockApi yields the real
 * dockview mounted for the test. Each test re-imports the adapter and the
 * holder fresh (vi.resetModules) so module state never leaks.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { DEFAULT_PX, mountDock, unmountDocks } from './_dock-real.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const ADAPTER = `${JS}/dock-iframe.js`;
const DOCK_APPLY = `${JS}/dock-apply.js`;

const { getDockApi, state, redock } = vi.hoisted(() => ({
  state: { api: /** @type {any} */ (null) },
  redock: { fn: /** @type {null | (() => void)} */ (null) },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
}));
getDockApi.mockImplementation(() => state.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  setServiceRedock: (/** @type {() => void} */ fn) => {
    redock.fn = fn;
  },
  onDragGesture: () => [],
}));
// The one question the adapter asks the pop-out seam: whether a panel lives
// in its own window. The seam itself is popout.test.mjs's contract.
const { isPoppedOut } = vi.hoisted(() => ({
  isPoppedOut: vi.fn(/** @type {(id: string) => boolean} */ (() => false)),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({ isPoppedOut }));

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.api = null;
  redock.fn = null;
  document.body.innerHTML = '<div class="main-container"></div>';
});

afterEach(() => {
  unmountDocks();
  state.api = null;
  document.body.innerHTML = '';
});

/** @param {string} id @param {string[]} tabs @param {string} [showing] */
function tile(id, tabs, showing = tabs[0]) {
  return { tile: id, tabs, showing };
}

/** @param {any[]} items */
function row(...items) {
  const size = 1 / items.length;
  return { split: 'row', items: items.map((item) => ({ ...item, size })) };
}

const TERMINAL = tile('tt', ['terminal']);

/** Every tile of a layout, in tree order. @param {any} node @returns {any[]} */
function tiles(node) {
  if (node === null) return [];
  return 'tabs' in node ? [node] : node.items.flatMap(tiles);
}

/** The tile holding `key`. @param {any} layout @param {string} key */
function tileWith(layout, key) {
  return tiles(layout).find((t) => t.tabs.includes(key)) ?? null;
}

function makeIframe() {
  return document.createElement('iframe');
}

/**
 * A real dock and a real holder showing `layout` (the terminal alone by
 * default; null for an empty dock), then a fresh adapter over both.
 * @param {any} [layout]
 */
async function freshAdapter(layout = TERMINAL) {
  const { api } = mountDock();
  state.api = api;
  const dockApply = await import(DOCK_APPLY);
  const holder = dockApply.initLayoutHolder({
    api,
    guard: (/** @type {() => any} */ fn) => fn(),
    dockPx: DEFAULT_PX,
  });
  if (layout !== null) expect(holder.restore(layout)).toBe(true);
  const mod = await import(ADAPTER);
  mod.initDockIframeAdapter({ fallbackHost: null });
  return { mod, api, holder };
}

/**
 * Adopt a panel's iframe and show it, as panel-manager's activation does.
 * @param {any} mod @param {string} id
 */
function open(mod, id) {
  const iframe = makeIframe();
  mod.adoptIframe(id, iframe, { title: id.toUpperCase() });
  mod.focusPanel(id);
  return iframe;
}

/** Show `id` in its tile, as a tab click does. @param {any} api @param {string} id */
function switchTab(api, id) {
  api.getPanel(id).api.setActive();
}

describe('first placement — the classic 60/40 split', () => {
  test('the first service placeholder docks LEFT of the terminal at the service share', async () => {
    const { mod, api, holder } = await freshAdapter();

    open(mod, 'artifacts');

    const [left, right] = tiles(holder.layout);
    expect(left.tabs).toEqual(['artifacts']);
    expect(right.tabs).toEqual(['terminal']);
    expect(holder.layout.split).toBe('row');
    expect(left.size).toBeCloseTo(0.6, 2);
    expect(api.getPanel('iframe:artifacts')).toBeTruthy();
    expect(api.getPanel('iframe:artifacts').title).toBe('ARTIFACTS');
  });

  test("the overlay iframe takes the card's surface and its bottom corners", async () => {
    const { mod } = await freshAdapter();
    const iframe = makeIframe();

    mod.adoptIframe('artifacts', iframe, { title: 'Artifacts' });

    expect(iframe.style.borderRadius).toContain('var(--radius-xl)');
    expect(iframe.style.background).toContain('var(--surface-card)');
  });

  test('with no tile at all, the panel gets a tile of its own', async () => {
    const { mod, api, holder } = await freshAdapter(null);

    open(mod, 'artifacts');

    expect(tiles(holder.layout).map((t) => t.tabs)).toEqual([['artifacts']]);
    expect(api.getPanel('iframe:artifacts')).toBeTruthy();
  });
});

describe('an open — a new tab in front of the service tile, nothing evicted', () => {
  test('the previous occupant stays; the tile holds both, the new one in front', async () => {
    const { mod, api, holder } = await freshAdapter();
    open(mod, 'artifacts');
    const serviceTile = tileWith(holder.layout, 'artifacts').tile;

    open(mod, 'ariel');

    expect(tileWith(holder.layout, 'ariel')).toMatchObject({
      tile: serviceTile,
      tabs: ['artifacts', 'ariel'],
      showing: 'ariel',
    });
    expect(api.getPanel('iframe:artifacts')).toBeTruthy();
    const group = api.getPanel('iframe:ariel').group;
    expect(group.panels.map((/** @type {any} */ p) => p.id)).toEqual([
      'iframe:artifacts',
      'iframe:ariel',
    ]);
    expect(group.activePanel?.id).toBe('iframe:ariel');
  });

  test('the covered iframe is hidden but kept; re-activating it brings it forward in place', async () => {
    const { mod, api, holder } = await freshAdapter();
    const artifactsFrame = open(mod, 'artifacts');

    const arielFrame = open(mod, 'ariel');
    expect(artifactsFrame.style.display).toBe('none');
    expect(artifactsFrame.isConnected).toBe(true); // cached in the overlay, state intact

    mod.focusPanel('artifacts');
    expect(tileWith(holder.layout, 'artifacts')?.tabs).toEqual(['artifacts', 'ariel']);
    expect(api.getPanel('iframe:artifacts').group.activePanel?.id).toBe('iframe:artifacts');
    expect(arielFrame.style.display).toBe('none');
    expect(artifactsFrame.style.display).not.toBe('none');
  });

  test('with the terminal focused, the tab goes to the last-focused service tile — never the terminal', async () => {
    const { mod, api, holder } = await freshAdapter();
    open(mod, 'artifacts');

    switchTab(api, 'terminal'); // operator clicked back into the chat
    holder.show('tt', 'terminal');
    open(mod, 'okf');

    expect(tileWith(holder.layout, 'terminal')?.tabs).toEqual(['terminal']);
    expect(tileWith(holder.layout, 'okf')?.tabs).toEqual(['artifacts', 'okf']);
  });
});

describe('focusing an existing tile — jump, not move', () => {
  test('a panel that already has a tile is activated in place; other tiles are untouched', async () => {
    const { mod, api, holder } = await freshAdapter(
      row(tile('t1', ['artifacts']), tile('t2', ['ariel']), TERMINAL)
    );
    open(mod, 'artifacts');
    open(mod, 'ariel');
    const before = JSON.stringify(tiles(holder.layout).map((t) => [t.tile, t.tabs]));

    mod.focusPanel('artifacts');

    expect(JSON.stringify(tiles(holder.layout).map((t) => [t.tile, t.tabs]))).toBe(before);
    expect(api.activePanel?.id).toBe('iframe:artifacts');
    expect(api.getPanel('iframe:ariel')).toBeTruthy(); // no eviction
  });
});

describe('redock after a layout rebuild — visible panels come back as tabs', () => {
  test('each visible panel the layout lacks re-materializes by the open rule', async () => {
    const { mod, api, holder } = await freshAdapter(
      row(tile('t1', ['artifacts']), tile('t2', ['ariel']), TERMINAL)
    );
    open(mod, 'artifacts');
    open(mod, 'ariel');

    // A layout rebuild wiped the grid; only the terminal came back.
    expect(holder.restore(TERMINAL)).toBe(true);
    if (!redock.fn) throw new Error('adapter never registered its redock hook');
    redock.fn();

    const artifacts = tileWith(holder.layout, 'artifacts');
    const ariel = tileWith(holder.layout, 'ariel');
    expect(artifacts).not.toBeNull();
    expect(ariel).not.toBeNull();
    // The first re-opens left of the terminal at the classic split; the next
    // joins it as a tab. Neither evicted the other, and the terminal is alone.
    expect(ariel.tile).toBe(artifacts.tile);
    expect(artifacts.tabs).toEqual(['artifacts', 'ariel']);
    expect(api.getPanel('iframe:ariel').group).toBe(api.getPanel('iframe:artifacts').group);
    expect(tiles(holder.layout).map((t) => t.tabs[0])).toEqual(['artifacts', 'terminal']);
    expect(tileWith(holder.layout, 'terminal')?.tabs).toEqual(['terminal']);
  });
});

describe('hidePanel — tiles never outlive their panel', () => {
  test('drops a placeholder even when the adapter never adopted the panel', async () => {
    const { mod, api } = await freshAdapter(row(tile('tg', ['ghost']), TERMINAL));
    expect(api.getPanel('iframe:ghost')).toBeTruthy();

    mod.hidePanel('ghost');

    expect(api.getPanel('iframe:ghost')).toBeFalsy();
  });

  test('a hidden panel keeps its iframe cached and comes back on the next activation', async () => {
    const { mod, api } = await freshAdapter();
    const iframe = open(mod, 'artifacts');

    mod.hidePanel('artifacts');
    expect(api.getPanel('iframe:artifacts')).toBeFalsy();
    expect(iframe.style.display).toBe('none');
    expect(iframe.isConnected).toBe(true);

    mod.focusPanel('artifacts');
    expect(api.getPanel('iframe:artifacts')).toBeTruthy();
  });
});

describe('pruning — placeholders of server-closed panels', () => {
  const LATTICE = row(tile('tl', ['lattice']), TERMINAL);

  test('a known but server-closed panel loses its restored tile', async () => {
    const { mod, api } = await freshAdapter(LATTICE);

    mod.setServerVisiblePanels(new Set(['artifacts']));
    mod.setKnownServicePanels(['artifacts', 'lattice']);

    expect(api.getPanel('iframe:lattice')).toBeFalsy();
  });

  test('a server-visible panel keeps its restored tile', async () => {
    const { mod, api } = await freshAdapter(LATTICE);

    mod.setServerVisiblePanels(new Set(['lattice']));
    mod.setKnownServicePanels(['lattice']);

    expect(api.getPanel('iframe:lattice')).toBeTruthy();
  });

  test('before the visible set is seeded, known placeholders are left alone', async () => {
    // An early layout restore must not drop every placeholder just because
    // panel-manager has not seeded the server-visible set yet.
    const { mod, api } = await freshAdapter(LATTICE);

    mod.setKnownServicePanels(['lattice']);

    expect(api.getPanel('iframe:lattice')).toBeTruthy();
  });

  test('a panel living in its own pop-out window loses its restored tile, visible or not', async () => {
    const { mod, api } = await freshAdapter(LATTICE);
    isPoppedOut.mockImplementation((/** @type {string} */ id) => id === 'lattice');

    mod.setServerVisiblePanels(new Set(['lattice']));
    mod.setKnownServicePanels(['lattice']);

    expect(api.getPanel('iframe:lattice')).toBeFalsy();
    isPoppedOut.mockImplementation(() => false);
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
    const { mod } = await freshAdapter();
    const iframe = open(mod, 'artifacts');
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

describe('panelFocusAnchor — where a panel sits in the host tab order', () => {
  test("dock mode: the placeholder group's content container, with the overlay to skip", async () => {
    const { mod, api } = await freshAdapter(row(tile('t1', ['ariel']), TERMINAL));
    const iframe = open(mod, 'ariel');
    const found = mod.panelFocusAnchor(iframe.contentWindow);
    const group = api.getPanel('iframe:ariel').group;
    expect(found?.anchor).toBe(group.element.querySelector('.dv-content-container'));
    expect(found?.skip(iframe)).toBe(true);
    expect(found?.skip(group.element)).toBe(false);
  });

  test('a window that is no managed frame anchors nowhere', async () => {
    const { mod } = await freshAdapter(row(tile('t1', ['ariel']), TERMINAL));
    open(mod, 'ariel');
    expect(mod.panelFocusAnchor(/** @type {any} */ ({}))).toBeNull();
    expect(mod.panelFocusAnchor(null)).toBeNull();
  });

  test('fallback mount: the iframe itself', async () => {
    const mod = await import(ADAPTER);
    const host = document.createElement('div');
    document.body.appendChild(host);
    mod.initDockIframeAdapter({ fallbackHost: host });
    const iframe = makeIframe();
    mod.adoptIframe('ariel', iframe, { title: 'ARIEL' });
    const found = mod.panelFocusAnchor(iframe.contentWindow);
    expect(found?.anchor).toBe(iframe);
    expect(found?.skip(iframe)).toBe(false);
  });
});
