// @ts-check
/**
 * Unit tests for the ARIEL embedded tile-bar header (header.js): the
 * `[search, compose, ⋯]` contribution, whole re-sends on state change,
 * re-keying the search item on a panel-set query, the omitted search item on
 * invalid configuration, the header-action round trip into the store, and the
 * standalone no-op.
 *
 *   npx vitest run tests/interfaces/ariel/header.test.mjs
 *
 * `window.parent` is replaced by a stand-in whose `postMessage` records each
 * contribution; hub actions are dispatched as same-origin `message` events.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/rail.js', () => ({
  promptSaveSearch: vi.fn(),
}));

/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} StoreMod */
/** @typedef {typeof import('../../../src/osprey/interfaces/ariel/static/js/header.js')} HeaderMod */

/** @type {StoreMod} */ let store;
/** @type {HeaderMod} */ let header;
/** @type {{destroy: () => void} | null} */ let api = null;
/** @type {ReturnType<typeof vi.fn>} */ let postMessage;
/** @type {ReturnType<typeof vi.fn>} */ let promptSaveSearch;

/** @returns {any[]} every contribution's items, oldest first. */
function contributions() {
  return postMessage.mock.calls
    .map((c) => c[0])
    .filter((m) => m && m.type === 'osprey-header-contribution')
    .map((m) => m.items);
}

/** @returns {any[]} the latest contribution's items. */
function latest() {
  const all = contributions();
  if (all.length === 0) throw new Error('nothing contributed');
  return all[all.length - 1];
}

/** @param {any[]} items @param {string} kind */
function itemOf(items, kind) {
  return items.find((i) => i.kind === kind);
}

/** @param {string} id @param {string} [value] */
function hubAction(id, value) {
  window.dispatchEvent(
    new MessageEvent('message', {
      data: { type: 'osprey-header-action', id, value },
      origin: window.location.origin,
    })
  );
}

/**
 * @param {{embedded?: boolean, capabilities?: any}} [opts]
 */
async function boot({ embedded = true, capabilities = { status: 'ok' } } = {}) {
  history.replaceState(null, '', embedded ? '/?embedded=true' : '/');
  document.body.className = embedded ? 'embedded' : '';
  vi.resetModules();
  store = await import('../../../src/osprey/interfaces/ariel/static/js/store.js');
  header = await import('../../../src/osprey/interfaces/ariel/static/js/header.js');
  const rail = await import('../../../src/osprey/interfaces/ariel/static/js/rail.js');
  promptSaveSearch = /** @type {any} */ (rail.promptSaveSearch);
  promptSaveSearch.mockClear();
  api = header.initHeader({ capabilities });
}

beforeEach(() => {
  postMessage = vi.fn();
  vi.stubGlobal('parent', { postMessage });
});

afterEach(() => {
  api?.destroy();
  api = null;
  vi.unstubAllGlobals();
  document.body.className = '';
  history.replaceState(null, '', '/');
});

describe('contribution', () => {
  test('embedded init contributes search, compose and the ⋯ menu in that order', async () => {
    await boot();
    expect(window.parent).not.toBe(window);
    const items = latest();
    expect(items.map((/** @type {any} */ i) => i.kind)).toEqual(['search', 'button', 'menu']);
    const menu = itemOf(items, 'menu');
    expect(menu.items.map((/** @type {any} */ e) => e.id)).toEqual(['toggle-rail', 'save-search']);
    expect(postMessage.mock.calls[0][1]).toBe(window.location.origin);
  });

  test('the search item is seeded with the current query', async () => {
    await boot();
    store.setState({ query: 'kicker', rerank: 'now' });
    expect(itemOf(latest(), 'search').value).toBe('kicker');
  });

  test('invalid configuration omits the search item but keeps compose and menu', async () => {
    await boot({ capabilities: { status: 'configuration_invalid' } });
    expect(latest().map((/** @type {any} */ i) => i.kind)).toEqual(['button', 'menu']);
    store.setState({ query: 'x', rerank: 'now' });
    expect(itemOf(latest(), 'search')).toBeUndefined();
  });

  test('missing capabilities still contribute the search item', async () => {
    await boot({ capabilities: null });
    expect(itemOf(latest(), 'search')).toBeDefined();
  });

  test('a railCollapsed change re-sends the whole contribution with the new menu label', async () => {
    await boot();
    const before = itemOf(latest(), 'menu').items[0].label;
    const n = contributions().length;
    store.setState({ railCollapsed: !store.getState().railCollapsed });
    expect(contributions().length).toBe(n + 1);
    const items = latest();
    expect(items).toHaveLength(3);
    expect(itemOf(items, 'menu').items[0].label).not.toBe(before);
  });

  test('an unrelated state change sends nothing', async () => {
    await boot();
    const n = contributions().length;
    store.setState({ sel: 'e1' });
    store.setState({ stacked: true });
    expect(contributions().length).toBe(n);
  });
});

describe('re-keying the search item', () => {
  test('a panel-set query (rerank now) re-keys the search item', async () => {
    await boot();
    const firstId = itemOf(latest(), 'search').id;
    store.setState({ query: 'saved one', rerank: 'now' });
    const secondId = itemOf(latest(), 'search').id;
    expect(secondId).not.toBe(firstId);
    store.setState({ query: 'from hash', rerank: 'now' });
    expect(itemOf(latest(), 'search').id).not.toBe(secondId);
  });

  test('a tile-bar report (rerank when-stable) keeps the id', async () => {
    await boot();
    const id = itemOf(latest(), 'search').id;
    hubAction(id, 'typed');
    expect(store.getState().query).toBe('typed');
    expect(itemOf(latest(), 'search').id).toBe(id);
    hubAction(id, 'typed more');
    expect(itemOf(latest(), 'search').id).toBe(id);
  });

  test('a panel-set query after a report re-keys again', async () => {
    await boot();
    const id = itemOf(latest(), 'search').id;
    hubAction(id, 'typed');
    store.setState({ query: 'alias', rerank: 'now' });
    expect(itemOf(latest(), 'search').id).not.toBe(id);
    expect(itemOf(latest(), 'search').value).toBe('alias');
  });

  test('a forced write of the same query (rerank now) re-keys', async () => {
    await boot();
    store.setState({ query: 'q', rerank: 'now' });
    const id = itemOf(latest(), 'search').id;
    store.setState({ query: 'q', rerank: 'now' }, { force: true });
    expect(itemOf(latest(), 'search').id).not.toBe(id);
  });

  test('a re-keyed id still routes back as a search report', async () => {
    await boot();
    store.setState({ query: 'from hash', rerank: 'now' });
    hubAction(itemOf(latest(), 'search').id, 'next');
    expect(store.getState()).toMatchObject({ query: 'next', rerank: 'when-stable' });
  });
});

describe('header actions', () => {
  test('search value writes {query, rerank: when-stable}', async () => {
    await boot();
    hubAction(itemOf(latest(), 'search').id, 'beam dump');
    expect(store.getState()).toMatchObject({ query: 'beam dump', rerank: 'when-stable' });
  });

  test('an empty search report clears the query', async () => {
    await boot();
    store.setState({ query: 'old', rerank: 'now' });
    hubAction(itemOf(latest(), 'search').id, '');
    expect(store.getState()).toMatchObject({ query: '', rerank: 'when-stable' });
  });

  test('compose opens an empty compose', async () => {
    await boot();
    hubAction(itemOf(latest(), 'button').id);
    expect(store.getState().compose).toEqual({ draft: null });
  });

  test('⋯ toggle-rail flips railCollapsed both ways', async () => {
    await boot();
    const menuId = itemOf(latest(), 'menu').id;
    const start = store.getState().railCollapsed;
    hubAction(menuId, 'toggle-rail');
    expect(store.getState().railCollapsed).toBe(!start);
    hubAction(menuId, 'toggle-rail');
    expect(store.getState().railCollapsed).toBe(start);
  });

  test('⋯ save-search calls promptSaveSearch', async () => {
    await boot();
    hubAction(itemOf(latest(), 'menu').id, 'save-search');
    expect(promptSaveSearch).toHaveBeenCalledTimes(1);
  });

  test('unknown ids and menu values are ignored', async () => {
    await boot();
    const before = store.getState();
    hubAction('nope', 'x');
    hubAction(itemOf(latest(), 'menu').id, 'nope');
    expect(store.getState()).toBe(before);
    expect(promptSaveSearch).not.toHaveBeenCalled();
  });

  test('a cross-origin action is ignored', async () => {
    await boot();
    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'osprey-header-action', id: 'ariel-compose' },
        origin: 'https://evil.example',
      })
    );
    expect(store.getState().compose).toBeNull();
  });

  test('after destroy, actions and state changes do nothing', async () => {
    await boot();
    const menuId = itemOf(latest(), 'menu').id;
    api?.destroy();
    api = null;
    const n = contributions().length;
    hubAction(menuId, 'toggle-rail');
    hubAction('ariel-compose');
    store.setState({ query: 'z', rerank: 'now' });
    expect(store.getState().compose).toBeNull();
    expect(contributions().length).toBe(n);
  });
});

describe('standalone', () => {
  test('contributes nothing and ignores actions', async () => {
    await boot({ embedded: false });
    store.setState({ query: 'q', rerank: 'now' });
    store.setState({ railCollapsed: true });
    hubAction('ariel-compose');
    header.handleHeaderAction('ariel-compose', undefined);
    expect(postMessage).not.toHaveBeenCalled();
    expect(store.getState().compose).toBeNull();
  });
});
