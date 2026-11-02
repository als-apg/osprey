// @ts-check
/**
 * Hash contract: router.js maps the location hash onto the store and writes
 * the canonical hash back.
 *
 * Run: npx vitest run tests/interfaces/ariel/router.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const STORE = '../../../src/osprey/interfaces/ariel/static/js/store.js';
const ROUTER = '../../../src/osprey/interfaces/ariel/static/js/router.js';

/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} */
let store;
/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/router.js')} */
let router;

/** @param {string} url */
function go(url) {
  history.replaceState(null, '', url);
}

/** Fire the event the browser fires when only the fragment changes. */
function hashchange() {
  window.dispatchEvent(new Event('hashchange'));
}

/** Listeners each test's router adds to the shared window; removed after the test. */
/** @type {Array<[string, EventListenerOrEventListenerObject]>} */
let added = [];
const realAdd = window.addEventListener.bind(window);

beforeEach(async () => {
  go('/');
  added = [];
  vi.spyOn(window, 'addEventListener').mockImplementation((type, fn, opts) => {
    added.push([type, fn]);
    realAdd(type, fn, opts);
  });
  vi.resetModules();
  store = await import(STORE);
  router = await import(ROUTER);
});

afterEach(() => {
  vi.restoreAllMocks();
  for (const [type, fn] of added) window.removeEventListener(type, fn);
});

const LIST = {
  query: '',
  mailbox: 'all',
  sel: null,
  att: null,
  compose: null,
  drawer: null,
  narrowView: 'list',
  rerank: 'now',
};

describe('serializeState', () => {
  test('writes every canonical key, URL-encoded, in contract order', () => {
    const s = {
      ...store.getState(),
      query: 'beam dump & rf',
      mode: 'keyword',
      mailbox: /** @type {const} */ ({ saved: 'my search' }),
      sel: 'e/1',
      att: 'a 2',
    };
    expect(router.serializeState(s)).toBe(
      '#q=beam%20dump%20%26%20rf&mode=keyword&box=saved%3Amy%20search&sel=e%2F1&att=a%202'
    );
  });

  test('omits empty keys', () => {
    expect(router.serializeState(store.getState())).toBe('#box=all');
    expect(router.serializeState({ ...store.getState(), mailbox: 'drafts', sel: 'x' })).toBe(
      '#box=drafts&sel=x'
    );
  });
});

describe('parseHash: canonical', () => {
  test('canonical round-trip reproduces the canonical keys', () => {
    const s = {
      ...store.getState(),
      query: 'vacuum + leak?',
      mode: 'semantic',
      mailbox: /** @type {const} */ ({ saved: 'a:b' }),
      sel: 'id=7',
      att: 'att&1',
    };
    const hash = router.serializeState(s);
    const patch = router.parseHash(hash);
    expect(patch).toMatchObject({
      query: s.query,
      mode: s.mode,
      mailbox: s.mailbox,
      sel: s.sel,
      att: s.att,
    });
    expect(router.serializeState({ ...store.getState(), ...patch })).toBe(hash);
  });

  test('every mailbox value round-trips', () => {
    for (const mailbox of ['all', 'drafts', 'mine', { saved: 'x y' }]) {
      const hash = router.serializeState({
        ...store.getState(),
        mailbox: /** @type {any} */ (mailbox),
      });
      expect(router.parseHash(hash).mailbox).toEqual(mailbox);
    }
  });

  test('applying any canonical hash writes rerank now; a sel also opens the reader', () => {
    expect(router.parseHash('#q=x')).toMatchObject({ query: 'x', rerank: 'now' });
    expect(router.parseHash('#q=x').narrowView).toBeUndefined();
    expect(router.parseHash('#sel=E1')).toMatchObject({
      sel: 'E1',
      narrowView: 'reader',
      rerank: 'now',
    });
  });

  test('omitted canonical keys fall back to empty', () => {
    expect(router.parseHash('#q=x')).toEqual({
      query: 'x',
      mode: null,
      mailbox: 'all',
      sel: null,
      att: null,
      rerank: 'now',
    });
  });

  test('an empty hash is no patch', () => {
    expect(router.parseHash('')).toEqual({});
    expect(router.parseHash('#')).toEqual({});
  });
});

describe('parseHash: aliases', () => {
  test('#entry?id&attachment selects and opens the reader, decoding ids', () => {
    expect(router.parseHash('#entry?id=a%2Fb&attachment=att%201')).toEqual({
      sel: 'a/b',
      att: 'att 1',
      narrowView: 'reader',
      rerank: 'now',
    });
    expect(router.parseHash('#entry?id=E1')).toEqual({
      sel: 'E1',
      att: null,
      narrowView: 'reader',
      rerank: 'now',
    });
  });

  test('#create and #create?draft open compose', () => {
    expect(router.parseHash('#create')).toEqual({
      compose: { draft: null },
      narrowView: 'list',
      rerank: 'now',
    });
    expect(router.parseHash('#create?draft=D%201')).toEqual({
      compose: { draft: 'D 1' },
      narrowView: 'list',
      rerank: 'now',
    });
  });

  test('#search and #browse show the list with an empty query', () => {
    expect(router.parseHash('#search')).toEqual(LIST);
    expect(router.parseHash('#browse')).toEqual(LIST);
  });

  test('#status opens the Status drawer standalone', () => {
    expect(router.parseHash('#status')).toEqual({ drawer: 'status', rerank: 'now' });
  });

  test('#status shows the list when embedded', () => {
    go('/?embedded=true');
    expect(router.parseHash('#status')).toEqual(LIST);
  });

  test('an unknown alias shows the list', () => {
    expect(router.parseHash('#nowhere')).toEqual(LIST);
  });
});

/**
 * One row per alias: the hash, and the state it must leave in each frame.
 * @type {Array<[string, Record<string, unknown>, Record<string, unknown>]>}
 */
const ALIASES = [
  [
    '#entry?id=E1&attachment=A1',
    { sel: 'E1', att: 'A1', narrowView: 'reader' },
    { sel: 'E1', att: 'A1', narrowView: 'reader' },
  ],
  [
    '#entry?id=E1',
    { sel: 'E1', att: null, narrowView: 'reader' },
    { sel: 'E1', att: null, narrowView: 'reader' },
  ],
  ['#create', { compose: { draft: null } }, { compose: { draft: null } }],
  ['#create?draft=D1', { compose: { draft: 'D1' } }, { compose: { draft: 'D1' } }],
  [
    '#search',
    { query: '', sel: null, compose: null, narrowView: 'list' },
    { query: '', sel: null, compose: null, narrowView: 'list' },
  ],
  [
    '#browse',
    { query: '', sel: null, compose: null, narrowView: 'list' },
    { query: '', sel: null, compose: null, narrowView: 'list' },
  ],
  ['#status', { drawer: 'status' }, { drawer: null, query: '', sel: null, narrowView: 'list' }],
];

for (const [frame, prefix] of /** @type {const} */ ([
  ['standalone', '/'],
  ['embedded', '/?embedded=true'],
])) {
  describe(`initRouter (${frame})`, () => {
    for (const [hash, standalone, embedded] of ALIASES) {
      const expected = frame === 'standalone' ? standalone : embedded;

      test(`${hash} applied at init`, () => {
        store.setState({ query: 'old', sel: 'S0', compose: null });
        go(prefix + hash);
        router.initRouter();
        expect(store.getState()).toMatchObject({ ...expected, rerank: 'now' });
      });

      test(`${hash} applied on hashchange, with force, and re-applied when repeated`, () => {
        router.initRouter();
        const changed = vi.fn();
        store.subscribe((_s, _p, c) => changed([...c]));

        go(prefix + hash);
        hashchange();
        expect(store.getState()).toMatchObject(expected);
        expect(changed).toHaveBeenCalledTimes(1);
        // Forced: every patch key reported, even ones already at that value.
        const keys = Object.keys(router.parseHash(hash));
        expect(new Set(changed.mock.calls[0][0])).toEqual(new Set(keys));

        // The alias was replaced by the canonical hash ...
        expect(window.location.hash).not.toBe(hash);
        expect(window.location.hash).toBe(router.serializeState(store.getState()));

        // ... so following the same link again changes the fragment and re-applies.
        go(prefix + hash);
        hashchange();
        expect(changed).toHaveBeenCalledTimes(2);
        expect(store.getState()).toMatchObject(expected);
      });
    }
  });
}

describe('initRouter: canonical hash and write-back', () => {
  test('a canonical hash at init is applied', () => {
    go('/#q=rf%20trip&mode=keyword&box=mine&sel=E9');
    router.initRouter();
    expect(store.getState()).toMatchObject({
      query: 'rf trip',
      mode: 'keyword',
      mailbox: 'mine',
      sel: 'E9',
      narrowView: 'reader',
      rerank: 'now',
    });
  });

  test('a canonical hashchange is applied without force', () => {
    router.initRouter();
    const changed = vi.fn();
    store.subscribe((_s, _p, c) => changed([...c]));
    go(router.serializeState(store.getState()));
    hashchange();
    expect(changed).not.toHaveBeenCalled();
    go('/#q=new&box=drafts');
    hashchange();
    expect(changed).toHaveBeenCalledTimes(1);
    expect(store.getState()).toMatchObject({ query: 'new', mailbox: 'drafts' });
  });

  test('store changes write the canonical hash with replaceState', () => {
    go('/?embedded=true');
    router.initRouter();
    const replace = vi.spyOn(history, 'replaceState');
    const assignHash = vi.spyOn(window, 'dispatchEvent');
    store.setState({ query: 'beam', mailbox: { saved: 'mine2' }, rerank: 'now' });
    expect(replace).toHaveBeenCalledTimes(1);
    expect(window.location.hash).toBe('#q=beam&box=saved%3Amine2');
    expect(window.location.search).toBe('?embedded=true');
    expect(assignHash).not.toHaveBeenCalled();
  });

  test('a store change that leaves the canonical hash unchanged writes nothing', () => {
    router.initRouter();
    const replace = vi.spyOn(history, 'replaceState');
    store.setState({ railCollapsed: true, drawer: 'settings' });
    expect(replace).not.toHaveBeenCalled();
  });

  test('applying a hash writes the canonical hash back exactly once', () => {
    router.initRouter();
    // A subscriber reacting to the hash write must not cause a second write-back.
    store.subscribe((_s, _p, changed) => {
      if (changed.has('sel')) store.setState({ narrowView: 'reader', mode: 'keyword' });
    });
    go('/#entry?id=E5');
    const replace = vi.spyOn(history, 'replaceState');
    hashchange();
    expect(replace).toHaveBeenCalledTimes(1);
    expect(window.location.hash).toBe('#mode=keyword&box=all&sel=E5');
  });

  test('initRouter twice does not double-apply hashchange', () => {
    router.initRouter();
    router.initRouter();
    const changed = vi.fn();
    store.subscribe(changed);
    go('/#create');
    hashchange();
    expect(changed).toHaveBeenCalledTimes(1);
  });
});
