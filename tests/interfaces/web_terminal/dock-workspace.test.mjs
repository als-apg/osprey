// @ts-check
// @vitest-environment jsdom
/**
 * Unit tests for dock-workspace.js's persistence, defaults and the terminal's
 * own close/reopen, driven through the real module on the pinned dockview
 * build (see _dock-real.mjs).
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-workspace.test.mjs
 *
 * What is pinned here: the dock persists the layout holder as
 * `{version: 2, layout, panels}` under the user-scoped key; a reload restores
 * the same tiles, tabs, showing tab and sizes through the holder; a layout the
 * dockview-era shell stored (dockview's own serialization) is read into the
 * dock form; any other version or an invalid value silently opens the default;
 * reset goes through the holder; the terminal closes as its tile and reopens
 * at the dock's right edge; and a change dockview makes on its own (its
 * drag-and-drop) is read back into the holder, never re-applied.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import { loadDockview, settle } from './_dock-real.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

// The refit the workspace runs on every settled change needs a live xterm; the
// rest of terminal.js stays real for the modules that read it on the way in.
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/terminal.js',
  async (importOriginal) => ({
    .../** @type {object} */ (await importOriginal()),
    fitTerminal: vi.fn(),
  })
);

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */

const KEY = 'osprey-dock-layout-p1';
const SCOPE_ATTR = 'data-osprey-storage-scope';
const PX = { width: 1000, height: 600 };

/** @param {unknown} body */
const jsonOk = (body) => ({ ok: true, status: 200, statusText: 'OK', json: async () => body });

/** A service left of the terminal at 0.7 / 0.3. */
/** @type {Layout} */
const SERVICE_AND_TERMINAL = {
  split: 'row',
  items: [
    { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.7 },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.3 },
  ],
};

/** The default: the terminal alone. */
const TERMINAL_ONLY = { tile: 't1', tabs: ['terminal'], showing: 'terminal' };

/**
 * A stored v2 value.
 * @param {unknown} layout
 * @param {Record<string, any>} [panels]
 */
const v2 = (layout, panels = {}) => JSON.stringify({ version: 2, layout, panels });

/**
 * The value the dockview-era shell stored: dockview's own serialization of a
 * service tile left of the terminal.
 */
const dockviewStore = () =>
  JSON.stringify({
    grid: {
      root: {
        type: 'branch',
        data: [
          {
            type: 'leaf',
            data: { views: ['iframe:ariel'], activeView: 'iframe:ariel', id: 'g1' },
            size: 700,
          },
          {
            type: 'leaf',
            data: { views: ['terminal'], activeView: 'terminal', id: 'g2' },
            size: 300,
          },
        ],
        size: 600,
      },
      width: 1000,
      height: 600,
      orientation: 'HORIZONTAL',
    },
    panels: {
      'iframe:ariel': {
        id: 'iframe:ariel',
        contentComponent: 'dock-iframe-placeholder',
        title: 'ARIEL',
      },
      terminal: { id: 'terminal', contentComponent: 'terminal', title: 'SESSION' },
    },
    activeGroup: 'g1',
  });

/**
 * Boot a fresh page for one user (null: an unscoped page) on a real dockview,
 * and wait until the workspace has settled its boot layout.
 * @param {string | null} scope
 */
async function boot(scope) {
  vi.resetModules();
  if (typeof scope === 'string') document.documentElement.setAttribute(SCOPE_ATTR, scope);
  else document.documentElement.removeAttribute(SCOPE_ATTR);
  document.body.innerHTML =
    '<div class="main-container"><div class="terminal-panel"><div class="terminal-header"></div></div><div id="right-panel"></div><div id="dock-root"></div></div>';
  window.__OSPREY_PREFIX__ = '';
  const info = {
    project_key: 'p1',
    enabled: ['artifacts', 'ariel'],
    custom: [{ id: 'logbook', url: '/panel/logbook' }],
  };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') return jsonOk(info);
      throw new Error(`unexpected ${url}`);
    })
  );
  loadDockview();
  const workspace = await import(`${JS}/dock-workspace.js`);
  const apply = await import(`${JS}/dock-apply.js`);
  const open = await import(`${JS}/dock-open.js`);
  workspace.initDockWorkspace();
  const api = workspace.getDockApi();
  api.layout(PX.width, PX.height);
  await vi.waitFor(() => expect(workspace.bootLayoutSettled()).toBe(true));
  await settle();
  const holder = apply.getLayoutHolder();
  if (!holder) throw new Error('no holder');
  return { api, holder, workspace, open, apply };
}

/**
 * Open a service the way an activation does: a tab in front of the service
 * tile used last (dock-open's openTab through the holder).
 * @param {Awaited<ReturnType<typeof boot>>} page
 * @param {string} key
 */
async function openService(page, key) {
  page.holder.run((/** @type {any} */ layout, /** @type {any} */ ctx) =>
    page.open.openTab(layout, key, ctx)
  );
  await settle();
}

/** Wait for the debounced persist, then read one stored value. @param {string} key */
async function storedAt(key) {
  await vi.waitFor(() => expect(localStorage.getItem(key)).not.toBeNull());
  return JSON.parse(localStorage.getItem(key) ?? '');
}

/**
 * A layout with every size rounded to 4 places (the engine's fractions carry
 * float noise), for exact comparison.
 * @param {any} layout
 * @returns {any}
 */
const rounded = (layout) =>
  JSON.parse(
    JSON.stringify(layout, (key, value) =>
      key === 'size' && typeof value === 'number' ? Math.round(value * 1e4) / 1e4 : value
    )
  );

/** Console calls that speak of the stored layout (other modules' notices are not this test's). */
const layoutNotices = (/** @type {any} */ spy) =>
  spy.mock.calls.filter((/** @type {any[]} */ args) =>
    /layout|stored|corrupt|persist/i.test(args.map(String).join(' '))
  );

/** The dockview groups as {tile, views, active}. @param {any} api */
const groups = (api) =>
  api.groups.map((/** @type {any} */ g) => ({
    tile: g.id,
    views: g.panels.map((/** @type {any} */ p) => p.id),
    active: g.activePanel?.id ?? null,
  }));

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.documentElement.removeAttribute(SCOPE_ATTR);
  document.documentElement.removeAttribute('data-agent');
  localStorage.clear();
});

describe('dock layout persistence under the user-scoped key', () => {
  test('the holder is stored as {version: 2, layout, panels} and restored for that user', async () => {
    const page = await boot('carol');
    page.holder.setPanelRecord('ariel', { title: 'ARIEL' });
    await openService(page, 'ariel');

    const stored = await storedAt(`${KEY}--carol`);
    expect(stored.version).toBe(2);
    expect(stored.layout).toEqual(page.holder.layout);
    expect(stored.layout.items.map((/** @type {any} */ i) => i.tabs)).toEqual([
      ['ariel'],
      ['terminal'],
    ]);
    expect(stored.panels).toEqual({ ariel: { title: 'ARIEL' } });
    expect(localStorage.getItem(KEY)).toBeNull();

    const next = await boot('carol');
    expect(next.holder.layout).toEqual(stored.layout);
    expect(next.api.getPanel('iframe:ariel')).toBeTruthy();
    expect(next.api.getPanel('iframe:ariel').title).toBe('ARIEL');
    expect(next.api.getPanel('terminal')).toBeTruthy();
  });

  test("another user's stored layout is never read", async () => {
    localStorage.setItem(`${KEY}--carol`, v2(SERVICE_AND_TERMINAL));

    const { holder, api } = await boot('bob');

    expect(holder.layout).toEqual(TERMINAL_ONLY);
    expect(groups(api)).toEqual([{ tile: 't1', views: ['terminal'], active: 'terminal' }]);
  });

  test('an unscoped page stores under the bare key', async () => {
    const page = await boot(null);
    await openService(page, 'ariel');

    await storedAt(KEY);
    expect(localStorage.getItem(`${KEY}--carol`)).toBeNull();
  });
});

describe('a change made just before the page goes away is stored', () => {
  // A write waits 150 ms after the first change of a burst. A reload or a
  // closed tab inside that window must still find the change stored.
  /** @type {[string, () => void][]} */
  const leaving = [
    ['pagehide', () => window.dispatchEvent(new Event('pagehide'))],
    [
      'the page going out of sight',
      () => {
        Object.defineProperty(document, 'visibilityState', {
          configurable: true,
          value: 'hidden',
        });
        document.dispatchEvent(new Event('visibilitychange'));
      },
    ],
  ];

  test.each(leaving)('%s writes the change still waiting to be stored', async (_name, leave) => {
    const page = await boot(null);
    localStorage.removeItem(KEY);
    await openService(page, 'ariel');
    expect(localStorage.getItem(KEY)).toBeNull();

    try {
      leave();
    } finally {
      Reflect.deleteProperty(document, 'visibilityState');
    }

    const stored = localStorage.getItem(KEY);
    expect(stored).not.toBeNull();
    expect(JSON.parse(stored ?? '').layout).toEqual(page.holder.layout);
  });
});

describe('a reload restores tiles, the showing tab and sizes', () => {
  test('the stored tiles come back as stored, with their records', async () => {
    localStorage.setItem(KEY, v2(SERVICE_AND_TERMINAL, { ariel: { title: 'ARIEL' } }));

    const { holder, api } = await boot(null);

    expect(holder.layout).toEqual(SERVICE_AND_TERMINAL);
    expect(groups(api)).toEqual([
      { tile: 't1', views: ['iframe:ariel'], active: 'iframe:ariel' },
      { tile: 't2', views: ['terminal'], active: 'terminal' },
    ]);
    expect(api.getPanel('iframe:ariel').title).toBe('ARIEL');
    expect(api.getPanel('terminal').title).toBe('Agent');
  });

  test('the restored panel records are persisted again on the next write', async () => {
    localStorage.setItem(KEY, v2(SERVICE_AND_TERMINAL, { ariel: { title: 'ARIEL' } }));
    const page = await boot(null);
    localStorage.removeItem(KEY);

    await openService(page, 'artifacts');
    const stored = await storedAt(KEY);

    // The restored record rides along with the new tab's.
    expect(stored.panels).toEqual({ ariel: { title: 'ARIEL' }, artifacts: { title: 'artifacts' } });
    expect(stored.layout.items[0].tabs).toEqual(['ariel', 'artifacts']);
  });

  test('a key the dock can no longer build is closed; the rest is kept', async () => {
    localStorage.setItem(
      KEY,
      v2({
        split: 'row',
        items: [
          { tile: 't1', tabs: ['gone'], showing: 'gone', size: 0.3 },
          { tile: 't3', tabs: ['ariel'], showing: 'ariel', size: 0.3 },
          { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
        ],
      })
    );

    const { holder } = await boot(null);

    expect(rounded(holder.layout)).toEqual({
      split: 'row',
      items: [
        { tile: 't3', tabs: ['ariel'], showing: 'ariel', size: 0.6 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
  });

  test('a custom panel the server lists is buildable', async () => {
    const layout = {
      split: 'row',
      items: [
        { tile: 't1', tabs: ['logbook'], showing: 'logbook', size: 0.6 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    };
    localStorage.setItem(KEY, v2(layout));

    const { holder } = await boot(null);

    expect(holder.layout).toEqual(layout);
  });

  test('a stored layout without the terminal gets it back at the right edge', async () => {
    localStorage.setItem(KEY, v2({ tile: 't1', tabs: ['ariel'], showing: 'ariel' }));

    const { holder } = await boot(null);

    expect(rounded(holder.layout)).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.6 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
  });

  test('a layout the dockview-era shell stored comes back as tiles, with its records', async () => {
    localStorage.setItem(KEY, dockviewStore());

    const page = await boot(null);
    const { holder, api } = page;

    expect(rounded(holder.layout)).toEqual({
      split: 'row',
      items: [
        { tile: 'g1', tabs: ['ariel'], showing: 'ariel', size: 0.7 },
        { tile: 'g2', tabs: ['terminal'], showing: 'terminal', size: 0.3 },
      ],
    });
    expect(api.getPanel('iframe:ariel').title).toBe('ARIEL');
    // The old title is healed; the next write is in this format.
    expect(api.getPanel('terminal').title).toBe('Agent');
    localStorage.removeItem(KEY);
    await openService(page, 'artifacts');
    const stored = await storedAt(KEY);
    expect(stored.version).toBe(2);
    expect(stored.layout.items[0].tabs).toEqual(['ariel', 'artifacts']);
  });
});

describe('any other version, or an invalid value, silently opens the default', () => {
  /** @type {[string, string][]} */
  const cases = [
    [
      'a dockview serialization with no tile left',
      JSON.stringify({
        grid: {
          root: { type: 'branch', data: [] },
          width: 0,
          height: 0,
          orientation: 'HORIZONTAL',
        },
        panels: { terminal: { id: 'terminal', contentComponent: 'terminal', title: '' } },
      }),
    ],
    ['version 1', JSON.stringify({ version: 1, layout: SERVICE_AND_TERMINAL, panels: {} })],
    ['version 3', JSON.stringify({ version: 3, layout: SERVICE_AND_TERMINAL, panels: {} })],
    ['an invalid layout', v2({ tile: 't1', tabs: [], showing: 'x' })],
    [
      'a layout with a key twice',
      v2({ split: 'row', items: [TERMINAL_ONLY, { ...TERMINAL_ONLY, tile: 't2' }] }),
    ],
    ['unparseable JSON', '{not json'],
    ['a bare string', JSON.stringify('layout')],
    ['version 2 without a layout', JSON.stringify({ version: 2, panels: {} })],
    ['every key unbuildable', v2({ tile: 't1', tabs: ['gone'], showing: 'gone' })],
  ];

  test.each(cases)('%s', async (_name, raw) => {
    localStorage.setItem(KEY, raw);
    const errors = vi.spyOn(console, 'error');
    const warns = vi.spyOn(console, 'warn');

    const { holder, api } = await boot(null);

    expect(holder.layout).toEqual(TERMINAL_ONLY);
    expect(groups(api)).toEqual([{ tile: 't1', views: ['terminal'], active: 'terminal' }]);
    expect(layoutNotices(errors)).toEqual([]);
    expect(layoutNotices(warns)).toEqual([]);
  });
});

describe('reset and the terminal go through the holder', () => {
  test('reset applies the terminal-only default with one fromJSON and forgets the store', async () => {
    localStorage.setItem(KEY, v2(SERVICE_AND_TERMINAL));
    const { api, holder, workspace } = await boot(null);
    const fromJSON = vi.spyOn(api, 'fromJSON');

    workspace.resetDockLayout();

    expect(fromJSON).toHaveBeenCalledTimes(1);
    expect(holder.layout).toEqual(TERMINAL_ONLY);
    expect(groups(api)).toEqual([{ tile: 't1', views: ['terminal'], active: 'terminal' }]);
  });

  test('the terminal closes as its tile and reopens at the right edge at 0.4', async () => {
    localStorage.setItem(KEY, v2(SERVICE_AND_TERMINAL));
    const { api, holder, workspace } = await boot(null);

    expect(workspace.closeTerminalPanel()).toBe(true);
    expect(holder.layout).toEqual({ tile: 't1', tabs: ['ariel'], showing: 'ariel' });
    expect(api.getPanel('terminal')).toBeFalsy();
    expect(workspace.isTerminalPanelOpen()).toBe(false);
    expect(workspace.closeTerminalPanel()).toBe(false);

    workspace.openTerminalPanel();
    expect(rounded(holder.layout)).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.6 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
    expect(api.getPanel('terminal')).toBeTruthy();
    expect(workspace.isTerminalPanelOpen()).toBe(true);
  });

  test('a service tile closes through the holder too', async () => {
    localStorage.setItem(KEY, v2(SERVICE_AND_TERMINAL));
    const { api, holder, workspace } = await boot(null);

    expect(workspace.closeDockPanel('ariel')).toBe(true);
    expect(holder.layout).toEqual({ tile: 't2', tabs: ['terminal'], showing: 'terminal' });
    expect(api.getPanel('iframe:ariel')).toBeFalsy();
    expect(workspace.closeDockPanel('ariel')).toBe(false);
  });
});

describe('what dockview does on its own is read back into the holder (sizes only)', () => {
  test('a tab click records the focused tile', async () => {
    localStorage.setItem(KEY, v2(SERVICE_AND_TERMINAL));
    const { api, holder } = await boot(null);

    api.getPanel('terminal').api.setActive();
    expect(holder.focusedTile).toBe('t2');
    api.getPanel('iframe:ariel').api.setActive();
    expect(holder.focusedTile).toBe('t1');
    expect(holder.lastServiceTile).toBe('t1');
  });

  test("a sash drag's sizes reach the holder", async () => {
    localStorage.setItem(KEY, v2(SERVICE_AND_TERMINAL));
    const { api, holder } = await boot(null);
    const fromJSON = vi.spyOn(api, 'fromJSON');

    api.getPanel('iframe:ariel').api.setSize({ width: 500 });
    await settle();

    expect(fromJSON).not.toHaveBeenCalled();
    const items = holder.layout && 'items' in holder.layout ? holder.layout.items : [];
    expect(items.map((/** @type {any} */ i) => i.tile)).toEqual(['t1', 't2']);
    expect(items[0].size).toBeCloseTo(0.5, 2);
  });
});
