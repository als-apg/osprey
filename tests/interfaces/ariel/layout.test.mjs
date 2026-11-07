// @ts-check
/**
 * Unit tests for the ARIEL screen layout module (layout.js): rail splitter
 * with snap and `railCollapsed` both ways, storage keys, the list splitter,
 * `stacked` from a ResizeObserver, Back, and both drawers mapped both ways onto
 * `drawer`.
 *
 *   npx vitest run tests/interfaces/ariel/layout.test.mjs
 *
 * Pointer drags ride pointer capture (browser-level), so a drag release is
 * reproduced through the keyboard nudge path, which commits the same way.
 * Drawers are stand-ins with the <osprey-drawer> open()/close() contract and
 * its drawer:open / drawer:close events, so the veto path can be driven.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} StoreMod */
/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/layout.js')} LayoutMod */

/** @type {StoreMod} */ let store;
/** @type {LayoutMod} */ let layout;
/** @type {{destroy: () => void} | null} */ let api = null;

/** @type {Array<(entries: Array<{contentRect: {width: number}}>) => void>} */
let observers = [];

class FakeResizeObserver {
  /** @param {(entries: Array<{contentRect: {width: number}}>) => void} cb */
  constructor(cb) {
    observers.push(cb);
  }
  observe() {}
  unobserve() {}
  disconnect() {}
}

/** @param {number} width */
function resizePanel(width) {
  for (const cb of observers) cb([{ contentRect: { width } }]);
}

/** Fixture mirroring index.html's pane roots and drawers. */
function mountFixture({ settings = true } = {}) {
  document.body.innerHTML = `
    <div class="app-container" id="ariel-app">
      <div class="ariel-shell">
        <nav class="ariel-pane ariel-rail" id="ariel-rail"></nav>
        <div class="osprey-splitter" id="ariel-rail-splitter" role="separator"
             aria-orientation="vertical" aria-label="Resize mailbox rail" tabindex="0"></div>
        <div class="ariel-list-col" id="ariel-list-col">
          <section id="ariel-list"></section>
          <section id="ariel-compose" hidden></section>
        </div>
        <div class="osprey-splitter" id="ariel-list-splitter" role="separator"
             aria-orientation="vertical" aria-label="Resize entry list" tabindex="0"></div>
        <section id="ariel-reader">
          <div class="osprey-pane-head"><button type="button" id="ariel-reader-back"></button></div>
          <div id="ariel-reader-body"></div>
        </section>
      </div>
    </div>
    ${settings ? '<div id="settings-drawer"></div>' : ''}
    <div id="status-drawer"><button id="refresh-status"></button><div id="status-content"></div></div>
  `;
  for (const id of ['settings-drawer', 'status-drawer']) {
    const el = document.getElementById(id);
    if (el) fakeDrawer(el);
  }
}

/**
 * Give `el` the <osprey-drawer> open/close contract: idempotent, single
 * active, vetoable close, drawer:open/drawer:close on every change.
 * @param {HTMLElement} el
 */
function fakeDrawer(el) {
  const d = /** @type {any} */ (el);
  d.veto = false;
  const set = (/** @type {boolean} */ on) => {
    el.toggleAttribute('open', on);
    el.dispatchEvent(new CustomEvent(on ? 'drawer:open' : 'drawer:close', { bubbles: true }));
  };
  d.close = () => {
    if (!el.hasAttribute('open')) return true;
    if (d.veto) return false;
    set(false);
    return true;
  };
  d.open = () => {
    if (el.hasAttribute('open')) return true;
    for (const other of document.querySelectorAll('[open]')) {
      if (other !== el && !(/** @type {any} */ (other).close())) return false;
    }
    set(true);
    return true;
  };
}

/** @param {string} id @returns {HTMLElement} */
function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

/** @param {HTMLElement} el @param {string} key */
function press(el, key) {
  el.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

/** @param {string} url */
async function boot(url = '/', fixture = {}) {
  history.replaceState(null, '', url);
  mountFixture(fixture);
  vi.resetModules();
  store = await import('../../../src/osprey/interfaces/ariel/static/js/store.js');
  layout = await import('../../../src/osprey/interfaces/ariel/static/js/layout.js');
  api = layout.initLayout();
}

/** @param {string} url @param {string} key @param {unknown} value */
function seed(url, key, value) {
  history.replaceState(null, '', url);
  localStorage.setItem(key, JSON.stringify(value));
}

const app = () => byId('ariel-app');
const rail = () => byId('ariel-rail');
const railHandle = () => byId('ariel-rail-splitter');

beforeEach(() => {
  localStorage.clear();
  observers = [];
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
});

afterEach(() => {
  api?.destroy();
  api = null;
  vi.unstubAllGlobals();
  history.replaceState(null, '', '/');
});

describe('rail: default state and storage', () => {
  test('starts collapsed to the icon strip', async () => {
    await boot('/?embedded=true');
    expect(store.getState().railCollapsed).toBe(true);
    expect(app().hasAttribute('data-rail-collapsed')).toBe(true);
    expect(rail().style.flexBasis).toBe(`${layout.RAIL_STRIP}px`);
  });

  test('storage keys keep their suffix, so a stored size survives', async () => {
    await boot('/?embedded=true');
    expect(layout.railStorageKey()).toBe('ariel-rail-embedded');
    expect(layout.listStorageKey()).toBe('ariel-list-width-embedded');
  });

  test('a stored width is restored', async () => {
    seed('/', 'ariel-rail-embedded', { size: 240, collapsed: false });
    await boot('/');
    expect(rail().style.flexBasis).toBe('240px');
  });

  test('storage scope prefixes the key', async () => {
    document.documentElement.dataset.ospreyStorageScope = 'alice';
    try {
      await boot('/');
      expect(layout.railStorageKey()).toBe('ariel-rail-embedded--alice');
    } finally {
      delete document.documentElement.dataset.ospreyStorageScope;
    }
  });
});

describe('rail: DOM -> store', () => {
  test('double-click on the handle toggles railCollapsed', async () => {
    await boot('/');
    railHandle().dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(store.getState().railCollapsed).toBe(false);
    expect(app().hasAttribute('data-rail-collapsed')).toBe(false);
    expect(rail().style.flexBasis).toBe(`${layout.RAIL_DEFAULT}px`);
    railHandle().dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(store.getState().railCollapsed).toBe(true);
    expect(app().hasAttribute('data-rail-collapsed')).toBe(true);
  });

  test('Enter on the handle toggles too', async () => {
    await boot('/');
    press(railHandle(), 'Enter');
    expect(store.getState().railCollapsed).toBe(false);
  });

  test('an expanded release below the snap threshold collapses to the strip and expands to a usable width', async () => {
    seed('/', 'ariel-rail-embedded', { size: 120, collapsed: false });
    await boot('/');
    expect(rail().style.flexBasis).toBe('120px');
    press(railHandle(), 'ArrowLeft'); // 120 - 16 = 104 < snap
    expect(store.getState().railCollapsed).toBe(true);
    expect(rail().style.flexBasis).toBe(`${layout.RAIL_STRIP}px`);
    const stored = JSON.parse(String(localStorage.getItem('ariel-rail-embedded')));
    expect(stored.collapsed).toBe(true);
    expect(stored.size).toBeGreaterThanOrEqual(layout.RAIL_SNAP);

    store.setState({ railCollapsed: false });
    const px = parseInt(rail().style.flexBasis, 10);
    expect(px).toBeGreaterThanOrEqual(layout.RAIL_SNAP);
  });

  test('a release at or above the threshold stays expanded and becomes the width expand returns to', async () => {
    seed('/', 'ariel-rail-embedded', { size: 240, collapsed: false });
    await boot('/');
    press(railHandle(), 'ArrowRight'); // 256
    expect(store.getState().railCollapsed).toBe(false);
    store.setState({ railCollapsed: true });
    store.setState({ railCollapsed: false });
    expect(rail().style.flexBasis).toBe('256px');
  });

  test('the snap writes railCollapsed exactly once', async () => {
    seed('/', 'ariel-rail-embedded', { size: 120, collapsed: false });
    await boot('/');
    const seen = vi.fn();
    store.subscribe((_s, _p, changed) => {
      if (changed.has('railCollapsed')) seen(store.getState().railCollapsed);
    });
    press(railHandle(), 'ArrowLeft');
    expect(seen.mock.calls).toEqual([[true]]);
  });
});

describe('rail: store -> DOM', () => {
  test('railCollapsed from elsewhere is applied to the splitter and persisted', async () => {
    await boot('/');
    store.setState({ railCollapsed: false });
    expect(rail().style.flexBasis).toBe(`${layout.RAIL_DEFAULT}px`);
    expect(app().hasAttribute('data-rail-collapsed')).toBe(false);
    store.setState({ railCollapsed: true });
    expect(rail().style.flexBasis).toBe(`${layout.RAIL_STRIP}px`);
    expect(JSON.parse(String(localStorage.getItem('ariel-rail-embedded'))).collapsed).toBe(true);
  });
});

describe('list splitter', () => {
  test('restores the stored list width', async () => {
    seed('/', 'ariel-list-width-embedded', { size: 420, collapsed: false });
    await boot('/');
    expect(byId('ariel-list-col').style.flexBasis).toBe('420px');
  });

  test('double-click never leaves the list collapsed', async () => {
    await boot('/');
    byId('ariel-list-splitter').dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(byId('ariel-list-col').style.flexBasis).not.toBe('0px');
  });

  test('is inert while stacked', async () => {
    seed('/', 'ariel-list-width-embedded', { size: 420, collapsed: false });
    await boot('/');
    resizePanel(500);
    press(byId('ariel-list-splitter'), 'ArrowRight');
    expect(byId('ariel-list-col').style.flexBasis).toBe('420px');
  });
});

describe('stacked and Back', () => {
  test('the ResizeObserver writes stacked across the threshold, only on change', async () => {
    await boot('/');
    const seen = vi.fn();
    store.subscribe((_s, _p, changed) => {
      if (changed.has('stacked')) seen(store.getState().stacked);
    });
    resizePanel(500);
    resizePanel(520);
    resizePanel(900);
    expect(seen.mock.calls).toEqual([[true], [false]]);
  });

  test('stacked and narrowView are stamped on the root', async () => {
    await boot('/');
    resizePanel(500);
    expect(app().hasAttribute('data-stacked')).toBe(true);
    expect(app().dataset.narrowView).toBe('list');
    store.setState({ narrowView: 'reader' });
    expect(app().dataset.narrowView).toBe('reader');
  });

  test('Back sets narrowView to list', async () => {
    await boot('/');
    store.setState({ narrowView: 'reader' });
    byId('ariel-reader-back').click();
    expect(store.getState().narrowView).toBe('list');
    expect(byId('ariel-reader-back').querySelector('svg')).not.toBeNull();
  });
});

describe('drawers', () => {
  test('store -> DOM opens and closes each drawer', async () => {
    await boot('/');
    store.setState({ drawer: 'status' });
    expect(byId('status-drawer').hasAttribute('open')).toBe(true);
    store.setState({ drawer: 'settings' });
    expect(byId('settings-drawer').hasAttribute('open')).toBe(true);
    expect(byId('status-drawer').hasAttribute('open')).toBe(false);
    store.setState({ drawer: null });
    expect(byId('settings-drawer').hasAttribute('open')).toBe(false);
  });

  test('a drawer opening or closing itself is written back', async () => {
    await boot('/');
    /** @type {any} */ (byId('settings-drawer')).open(); // e.g. the display menu's Settings
    expect(store.getState().drawer).toBe('settings');
    /** @type {any} */ (byId('settings-drawer')).close(); // e.g. Esc or the backdrop
    expect(store.getState().drawer).toBe(null);
  });

  test('a vetoed close leaves the store on the drawer actually open', async () => {
    await boot('/');
    store.setState({ drawer: 'settings' });
    /** @type {any} */ (byId('settings-drawer')).veto = true;
    store.setState({ drawer: 'status' });
    expect(byId('settings-drawer').hasAttribute('open')).toBe(true);
    expect(byId('status-drawer').hasAttribute('open')).toBe(false);
    expect(store.getState().drawer).toBe('settings');
    store.setState({ drawer: null });
    expect(store.getState().drawer).toBe('settings');
  });

  test("drawer:'settings' is a no-op when the settings drawer was removed", async () => {
    await boot('/', { settings: false });
    store.setState({ drawer: 'settings' });
    expect(byId('status-drawer').hasAttribute('open')).toBe(false);
  });
});
