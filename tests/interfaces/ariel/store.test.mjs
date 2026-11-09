// @ts-check
/**
 * Store contract: the single state object behind the ARIEL panel.
 *
 * Run: npx vitest run tests/interfaces/ariel/store.test.mjs
 */
import { test, expect, describe, beforeEach, vi } from 'vitest';

const STORE = '../../../src/osprey/interfaces/ariel/static/js/store.js';

/** @type {typeof import('../../../src/osprey/interfaces/ariel/static/js/store.js')} */
let store;

beforeEach(async () => {
  vi.resetModules();
  store = await import(STORE);
});

describe('initial state', () => {
  test('matches the contract', () => {
    expect(store.getState()).toEqual({
      query: '',
      mode: null,
      filters: {},
      rerank: 'now',
      mailbox: 'all',
      sel: null,
      att: null,
      compose: null,
      draftsRev: 0,
      drawer: null,
      railCollapsed: false,
      stacked: false,
      narrowView: 'list',
      mineAuthor: null,
      filter: {},
      notice: null,
    });
  });

  test('the filter and the knobs are separate keys', () => {
    const seen = /** @type {Set<string>[]} */ ([]);
    store.subscribe((_s, _p, changed) => seen.push(changed));
    store.setState({ filter: { author: ['alice'] } });
    expect(store.getState().filters).toEqual({});
    expect(store.getState().filter).toEqual({ author: ['alice'] });
    store.setState({ filter: { author: ['alice'] } });
    store.setState({ notice: 'Dropped: shift' });
    expect(seen.map((c) => [...c])).toEqual([['filter'], ['notice']]);
  });
});

describe('setState', () => {
  test('shallow-merges the patch and leaves other keys alone', () => {
    store.setState({ query: 'rf trip', sel: '42' });
    const s = store.getState();
    expect(s.query).toBe('rf trip');
    expect(s.sel).toBe('42');
    expect(s.mailbox).toBe('all');
    expect(s.narrowView).toBe('list');
  });

  test('replaces object-valued keys wholesale rather than deep-merging', () => {
    store.setState({ filters: { author: 'a', rerank: true } });
    store.setState({ filters: { author: 'b' } });
    expect(store.getState().filters).toEqual({ author: 'b' });
  });

  test('a change produces a new state object; the previous one is not mutated', () => {
    const before = store.getState();
    store.setState({ query: 'x' });
    expect(store.getState()).not.toBe(before);
    expect(before.query).toBe('');
  });
});

describe('subscribe', () => {
  test('receives the new state, the previous state and the changed keys', () => {
    const fn = vi.fn();
    store.subscribe(fn);
    const prev = store.getState();
    store.setState({ query: 'beam', mailbox: 'mine' });
    expect(fn).toHaveBeenCalledTimes(1);
    const [state, previous, changed] = fn.mock.calls[0];
    expect(state).toBe(store.getState());
    expect(previous).toBe(prev);
    expect(previous.query).toBe('');
    expect(changed).toBeInstanceOf(Set);
    expect([...changed].sort()).toEqual(['mailbox', 'query']);
  });

  test('a patch that changes nothing does not notify', () => {
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ query: '', mailbox: 'all', sel: null });
    store.setState({});
    expect(fn).not.toHaveBeenCalled();
  });

  test('only keys whose value differs are reported changed', () => {
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ query: '', sel: '7' });
    expect([...fn.mock.calls[0][2]]).toEqual(['sel']);
  });

  test('object-valued keys compare by content, not reference', () => {
    store.setState({ filters: { author: 'a', limit: 5 }, mailbox: { saved: 'rf' } });
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({
      filters: { limit: 5, author: 'a' },
      mailbox: { saved: 'rf' },
      compose: null,
    });
    expect(fn).not.toHaveBeenCalled();

    store.setState({ mailbox: { saved: 'vacuum' } });
    expect(fn).toHaveBeenCalledTimes(1);
    expect([...fn.mock.calls[0][2]]).toEqual(['mailbox']);
  });

  test('nested and array values compare structurally', () => {
    store.setState({ filters: { tags: ['a', 'b'], range: { from: 1 } } });
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ filters: { tags: ['a', 'b'], range: { from: 1 } } });
    expect(fn).not.toHaveBeenCalled();
    store.setState({ filters: { tags: ['b', 'a'], range: { from: 1 } } });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('null versus an object counts as a change', () => {
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ compose: { draft: null } });
    store.setState({ compose: null });
    expect(fn).toHaveBeenCalledTimes(2);
    expect([...fn.mock.calls[0][2]]).toEqual(['compose']);
    expect([...fn.mock.calls[1][2]]).toEqual(['compose']);
  });

  test('a key gaining an extra entry counts as changed', () => {
    store.setState({ filters: { a: 1 } });
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ filters: { a: 1, b: undefined } });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('unsubscribe stops further notifications', () => {
    const fn = vi.fn();
    const off = store.subscribe(fn);
    store.setState({ sel: '1' });
    off();
    store.setState({ sel: '2' });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('every subscriber is notified, in subscription order', () => {
    /** @type {string[]} */
    const order = [];
    store.subscribe(() => order.push('a'));
    store.subscribe(() => order.push('b'));
    store.setState({ drawer: 'status' });
    expect(order).toEqual(['a', 'b']);
  });

  test('a subscriber unsubscribing during notification does not skip the others', () => {
    const b = vi.fn();
    /** @type {() => void} */
    let offA = () => {};
    offA = store.subscribe(() => offA());
    store.subscribe(b);
    store.setState({ sel: '1' });
    expect(b).toHaveBeenCalledTimes(1);
  });

  test('a write from inside a subscriber is seen by later notifications', () => {
    /** @type {Array<string|null>} */
    const seen = [];
    store.subscribe((state, _prev, changed) => {
      if (changed.has('sel') && state.sel === '1') store.setState({ narrowView: 'reader' });
    });
    store.subscribe((state) => seen.push(state.narrowView));
    store.setState({ sel: '1' });
    expect(store.getState().narrowView).toBe('reader');
    expect(seen).toContain('reader');
  });
});

describe('force', () => {
  test('notifies even when nothing changed, marking every patch key changed', () => {
    store.setState({ query: 'rf', rerank: 'now' });
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ query: 'rf', rerank: 'now' }, { force: true });
    expect(fn).toHaveBeenCalledTimes(1);
    const [state, prev, changed] = fn.mock.calls[0];
    expect([...changed].sort()).toEqual(['query', 'rerank']);
    expect(state.query).toBe('rf');
    expect(prev.query).toBe('rf');
  });

  test('marks only the patch keys, not untouched ones', () => {
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ sel: 'X', att: 'Y', narrowView: 'reader' }, { force: true });
    expect([...fn.mock.calls[0][2]].sort()).toEqual(['att', 'narrowView', 'sel']);
  });

  test('repeating a forced write notifies each time', () => {
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ compose: { draft: null } }, { force: true });
    store.setState({ compose: { draft: null } }, { force: true });
    expect(fn).toHaveBeenCalledTimes(2);
  });

  test('force: false behaves like an ordinary write', () => {
    const fn = vi.fn();
    store.subscribe(fn);
    store.setState({ query: '' }, { force: false });
    expect(fn).not.toHaveBeenCalled();
  });
});

describe('compose and draftsRev (C-COMPOSE)', () => {
  test('a follow-up is a distinct compose value from a new entry', () => {
    const seen = /** @type {string[][]} */ ([]);
    store.subscribe((_s, _p, changed) => seen.push([...changed]));
    store.setState({ compose: { draft: null } });
    store.setState({ compose: { draft: null, followsUp: 'E1' } });
    store.setState({ compose: { draft: null, followsUp: 'E1' } });
    store.setState({ compose: { draft: 'D1' } });
    expect(seen).toEqual([['compose'], ['compose'], ['compose']]);
    expect(store.getState().compose).toEqual({ draft: 'D1' });
  });

  test('opening and closing the sheet leaves narrowView alone', () => {
    store.setState({ narrowView: 'reader' });
    store.setState({ compose: { draft: null } });
    store.setState({ compose: null });
    expect(store.getState().narrowView).toBe('reader');
  });

  test('bumping draftsRev notifies with draftsRev changed', () => {
    const seen = /** @type {string[][]} */ ([]);
    store.subscribe((_s, _p, changed) => seen.push([...changed]));
    store.setState({ draftsRev: store.getState().draftsRev + 1 });
    store.setState({ draftsRev: store.getState().draftsRev + 1 });
    expect(store.getState().draftsRev).toBe(2);
    expect(seen).toEqual([['draftsRev'], ['draftsRev']]);
  });
});
