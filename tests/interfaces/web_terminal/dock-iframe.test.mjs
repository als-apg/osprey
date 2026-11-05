// @ts-check
// @vitest-environment jsdom
/**
 * Unit tests for the dock iframe adapter's PLACEMENT seam (dock-iframe.js):
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-iframe.test.mjs
 *
 * Every open, prune and hide is one operation on the page's layout holder
 * (dock-apply.js), which applies it to dockview. These tests run against the
 * pinned real dockview-core in jsdom and a real holder, and pin:
 *
 *   - the first service placeholder docks LEFT of the terminal at the 60/40
 *     split;
 *   - a tile holds ONE panel: an OPEN takes the service tile the operator used
 *     last, replacing the panel it showed — nothing resizes, the replaced
 *     panel's iframe is concealed and kept, and its next open replaces back;
 *   - the terminal's tile is never the target;
 *   - focusing a panel that already has a tile activates it in place;
 *   - after a WHOLE-LAYOUT apply, a visible panel the layout lacks is
 *     concealed, never re-docked;
 *   - hidePanel drops a placeholder even with no managed entry (a tile
 *     restored from storage for a panel closed elsewhere), and leaves the
 *     other tiles and their iframes as they were;
 *   - pruning removes placeholders of server-closed panels once the visible
 *     set is seeded, and leaves everything alone before that.
 *
 * dock-workspace is stubbed at the module boundary: getDockApi yields the real
 * dockview mounted for the test. Each test re-imports the adapter and the
 * holder fresh (vi.resetModules) so module state never leaks.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { MIN_PX, geometry, keysOf } from '/static/vendor/dock-engine.mjs';
import { DEFAULT_PX, mountDock, settle, unmountDocks } from './_dock-real.mjs';

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
  vi.restoreAllMocks();
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

/**
 * A person clicks in tile `tile`, which shows `key`: dockview activates its
 * panel and the bar tells the holder (dock-workspace's wiring, stubbed here).
 * @param {any} api @param {any} holder @param {string} tile @param {string} key
 */
function clickTile(api, holder, tile, key) {
  api.getPanel(key === 'terminal' ? key : `iframe:${key}`).api.setActive();
  holder.show(tile, key);
}

/**
 * Lay the dock out as a browser would, which jsdom does not: each group's
 * content container takes its tile's rect in the engine's geometry of the
 * held layout. Everything else keeps jsdom's zero rect, so the overlay is the
 * origin.
 * @param {any} api @param {any} holder
 */
function layOut(api, holder) {
  const zero = Element.prototype.getBoundingClientRect;
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(
    /** @this {Element} */
    function () {
      if (!this.classList.contains('dv-content-container')) return zero.call(this);
      const group = api.groups.find((/** @type {any} */ g) => g.element.contains(this));
      const box = { x: 0, y: 0, ...DEFAULT_PX };
      const rect = group ? geometry(holder.layout, box, MIN_PX).tiles[group.id] : undefined;
      return rect ? new DOMRect(rect.x, rect.y, rect.width, rect.height) : zero.call(this);
    }
  );
}

/** Where an overlay iframe is drawn. @param {HTMLIFrameElement} iframe */
function drawn(iframe) {
  const { display, left, top, width, height } = iframe.style;
  return { display, left, top, width, height };
}

/**
 * Drawn over tile `id`'s content, in the held layout.
 * @param {any} holder @param {string} id
 */
function over(holder, id) {
  const r = geometry(holder.layout, { x: 0, y: 0, ...DEFAULT_PX }, MIN_PX).tiles[id];
  const px = (/** @type {number} */ n) => `${Math.round(n)}px`;
  return { display: '', left: px(r.x), top: px(r.y), width: px(r.width), height: px(r.height) };
}

/** Each dockview group's views, by tile id. @param {any} api */
function groups(api) {
  return Object.fromEntries(
    api.groups.map((/** @type {any} */ g) => [g.id, g.panels.map((/** @type {any} */ p) => p.id)])
  );
}

/** @param {HTMLIFrameElement} iframe */
function inOverlay(iframe) {
  return iframe.parentElement?.classList.contains('dock-iframe-overlay') ?? false;
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

/**
 * Kirk saw both bugs in a browser with several panels in one tile: an open
 * stacked a panel into the service tile used last, and removing a panel from
 * the sidebar (the panel_visibility hide) had its fallback stack another one
 * into a tile that still showed a panel. A crowded tile's header collapsed.
 * Here no group ever holds more than one panel, and the tiles an action does
 * not touch keep their panel and their iframe where they were.
 */
describe('one panel per tile — the regressions Kirk saw', () => {
  const THREE = row(
    tile('t1', ['artifacts']),
    tile('t2', ['ariel']),
    tile('t3', ['okf']),
    TERMINAL
  );

  /** @param {any} mod */
  function openThree(mod) {
    return { artifacts: open(mod, 'artifacts'), ariel: open(mod, 'ariel'), okf: open(mod, 'okf') };
  }

  test('hiding one of three panels leaves the other tiles and their shown iframes as they were', async () => {
    const { mod, api, holder } = await freshAdapter(THREE);
    layOut(api, holder);
    const frames = openThree(mod);
    expect(groups(api)).toEqual({
      t1: ['iframe:artifacts'],
      t2: ['iframe:ariel'],
      t3: ['iframe:okf'],
      tt: ['terminal'],
    });

    mod.hidePanel('ariel');
    await settle();

    expect(groups(api)).toEqual({ t1: ['iframe:artifacts'], t3: ['iframe:okf'], tt: ['terminal'] });
    expect(drawn(frames.artifacts)).toEqual(over(holder, 't1'));
    expect(drawn(frames.okf)).toEqual(over(holder, 't3'));
    expect(frames.ariel.style.display).toBe('none');
    expect(inOverlay(frames.ariel)).toBe(true);
  });

  test('an open replaces the panel of the service tile used last; the panel it replaced comes back by its own open', async () => {
    const { mod, api, holder } = await freshAdapter(THREE);
    layOut(api, holder);
    const frames = openThree(mod);
    clickTile(api, holder, 't3', 'okf');

    const lattice = open(mod, 'lattice');
    await settle();

    expect(groups(api)).toEqual({
      t1: ['iframe:artifacts'],
      t2: ['iframe:ariel'],
      t3: ['iframe:lattice'],
      tt: ['terminal'],
    });
    expect(drawn(lattice)).toEqual(over(holder, 't3'));
    // okf is concealed, not closed: its iframe stays cached in the overlay.
    expect(frames.okf.style.display).toBe('none');
    expect(inOverlay(frames.okf)).toBe(true);
    // Nothing resized, and the other tiles kept their panel and their iframe.
    expect(tiles(holder.layout).map((t) => t.size)).toEqual(tiles(THREE).map((t) => t.size));
    expect(drawn(frames.artifacts)).toEqual(over(holder, 't1'));
    expect(drawn(frames.ariel)).toEqual(over(holder, 't2'));

    // Its row opens it again: it replaces lattice in the same tile, and the
    // iframe drawn there is the one okf had.
    mod.focusPanel('okf');
    await settle();

    expect(groups(api)).toEqual({
      t1: ['iframe:artifacts'],
      t2: ['iframe:ariel'],
      t3: ['iframe:okf'],
      tt: ['terminal'],
    });
    expect(drawn(frames.okf)).toEqual(over(holder, 't3'));
    expect(lattice.style.display).toBe('none');
    expect(inOverlay(lattice)).toBe(true);
  });

  test('with the Agent pane focused, an open still lands in the service tile used last', async () => {
    const { mod, api, holder } = await freshAdapter(
      row(tile('t1', ['artifacts']), tile('t2', ['ariel']), TERMINAL)
    );
    open(mod, 'artifacts');
    open(mod, 'ariel');
    clickTile(api, holder, 't2', 'ariel');
    clickTile(api, holder, 'tt', 'terminal'); // the operator went back to the chat

    open(mod, 'okf');

    expect(groups(api)).toEqual({ t1: ['iframe:artifacts'], t2: ['iframe:okf'], tt: ['terminal'] });
  });

  test('a whole-layout apply conceals a panel it lacks, and a later layout that docks it shows it', async () => {
    const { mod, api, holder } = await freshAdapter(
      row(tile('t1', ['artifacts']), tile('t2', ['ariel']), TERMINAL)
    );
    layOut(api, holder);
    const applied = vi.fn();
    mod.setPlaceholdersAppliedHandler(applied);
    const artifacts = open(mod, 'artifacts');
    const ariel = open(mod, 'ariel');
    if (!redock.fn) throw new Error('the adapter never registered its whole-layout hook');

    // A stored layout without ariel is applied whole (a reset, a restore).
    expect(holder.restore(row(tile('t1', ['artifacts']), TERMINAL))).toBe(true);
    redock.fn();
    await settle();

    expect(applied).toHaveBeenCalledTimes(1);
    expect(keysOf(holder.layout)).toEqual(['artifacts', 'terminal']);
    expect(groups(api)).toEqual({ t1: ['iframe:artifacts'], tt: ['terminal'] });
    expect(drawn(artifacts)).toEqual(over(holder, 't1'));
    expect(ariel.style.display).toBe('none');
    expect(inOverlay(ariel)).toBe(true);

    // Concealed while it has no tile, and shown as soon as a layout gives it
    // one again: here a layout the holder takes directly, as an agent
    // arrangement or a preset does, with no whole-layout hook after it. A tile
    // the layout holds never stays a blank card.
    expect(holder.restore(row(tile('t1', ['artifacts']), tile('t2', ['ariel']), TERMINAL))).toBe(
      true
    );
    await settle();
    expect(drawn(ariel)).toEqual(over(holder, 't2'));
    expect(drawn(artifacts)).toEqual(over(holder, 't1'));
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

describe('setPanelBodyState — a centred state over a tile body', () => {
  /** @param {HTMLElement} host */
  const paint = (host) => {
    host.textContent = 'Starting…';
  };
  const stateEl = () =>
    /** @type {HTMLElement | null} */ (document.querySelector('.tile-body-state'));

  test("covers the panel's tile body, hides its frame, and gives it back when cleared", async () => {
    const { mod, api, holder } = await freshAdapter(row(tile('tj', ['jupyter']), TERMINAL));
    const iframe = open(mod, 'jupyter');
    layOut(api, holder);

    mod.setPanelBodyState('jupyter', paint);

    const el = /** @type {HTMLElement} */ (stateEl());
    expect(el.textContent).toBe('Starting…');
    expect(el.parentElement?.classList.contains('dock-iframe-overlay')).toBe(true);
    const { display, left, top, width, height } = el.style;
    expect({ display, left, top, width, height }).toEqual(over(holder, 'tj'));
    expect(el.style.pointerEvents).toBe('auto');
    expect(iframe.style.display).toBe('none');

    mod.setPanelBodyState('jupyter', null);

    expect(stateEl()).toBeNull();
    expect(drawn(iframe)).toEqual(over(holder, 'tj'));
  });

  test('shows over a restored tile whose panel has no frame yet', async () => {
    const { mod, api, holder } = await freshAdapter(row(tile('tj', ['jupyter']), TERMINAL));
    layOut(api, holder);

    mod.setPanelBodyState('jupyter', paint);

    const { display, left, top, width, height } = /** @type {HTMLElement} */ (stateEl()).style;
    expect({ display, left, top, width, height }).toEqual(over(holder, 'tj'));
  });

  test('is hidden while the panel has no tile', async () => {
    const { mod } = await freshAdapter();

    mod.setPanelBodyState('jupyter', paint);

    const el = stateEl();
    expect(el === null || el.style.display === 'none').toBe(true);
  });

  test('the drag shield covers it as it covers the frames', async () => {
    const { mod, api, holder } = await freshAdapter(row(tile('tj', ['jupyter']), TERMINAL));
    layOut(api, holder);
    mod.setPanelBodyState('jupyter', paint);

    mod.setIframePointerShield(true);
    expect(/** @type {HTMLElement} */ (stateEl()).style.pointerEvents).toBe('none');
    mod.setIframePointerShield(false);
    expect(/** @type {HTMLElement} */ (stateEl()).style.pointerEvents).toBe('auto');
  });

  test('fallback mount: it covers the host while the host shows that panel', async () => {
    const mod = await import(ADAPTER);
    const host = document.createElement('div');
    document.body.appendChild(host);
    mod.initDockIframeAdapter({ fallbackHost: host });
    mod.adoptIframe('jupyter', makeIframe(), { title: 'Jupyter' });
    mod.adoptIframe('okf', makeIframe(), { title: 'OKF' });
    mod.focusPanel('jupyter');

    mod.setPanelBodyState('jupyter', paint);

    const el = /** @type {HTMLElement} */ (stateEl());
    expect(el.parentElement).toBe(host);
    expect(el.style.display).toBe('');
    mod.focusPanel('okf');
    expect(el.style.display).toBe('none');
  });
});
