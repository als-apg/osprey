// @ts-check
/**
 * Paging for the session tree's scroll boxes (sidebar-artifact-paging.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifact-paging.test.mjs
 *
 * A scroll near the end of a box (within two rows) asks for its next page; a
 * box with room for everything never scrolls, so `fill` asks again after a
 * render and `watch` after each resize, for boxes that have layout. Layout
 * does not exist in the test DOM, so box metrics are stubbed.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const { createBoxPager } = await import(`${JS}/sidebar-artifact-paging.js`);

/**
 * @param {string} id
 * @param {{ scrollHeight: number, scrollTop: number, clientHeight: number }} m
 */
function box(id, m) {
  const el = document.createElement('div');
  el.dataset.sessionBox = id;
  for (const [k, v] of Object.entries(m)) {
    Object.defineProperty(el, k, { value: v, configurable: true });
  }
  document.body.append(el);
  return el;
}

/** @param {{ served?: boolean, boxes?: Element[] }} [o] */
function pager({ served = true, boxes = [] } = {}) {
  const loadMore = vi.fn();
  const p = createBoxPager({ isServed: () => served, loadMore, boxes: () => boxes });
  return { p, loadMore };
}

beforeEach(() => {
  document.body.innerHTML = '';
  vi.useFakeTimers();
  vi.stubGlobal('requestAnimationFrame', (/** @type {() => void} */ cb) => setTimeout(cb, 0));
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('onScroll', () => {
  test('loads the next page when the box end is within 52 px', () => {
    const b = box('t1', { scrollHeight: 1000, scrollTop: 450, clientHeight: 500 });
    const { p, loadMore } = pager();
    p.onScroll(/** @type {any} */ ({ target: b }));
    expect(loadMore).toHaveBeenCalledWith('t1');
  });

  test('does nothing while farther than 52 px from the end', () => {
    const b = box('t1', { scrollHeight: 1000, scrollTop: 447, clientHeight: 500 });
    const { p, loadMore } = pager();
    p.onScroll(/** @type {any} */ ({ target: b }));
    expect(loadMore).not.toHaveBeenCalled();
  });

  test('finds the box from a scroll on a child inside it', () => {
    const b = box('t2', { scrollHeight: 100, scrollTop: 0, clientHeight: 100 });
    const child = document.createElement('span');
    b.append(child);
    const { p, loadMore } = pager();
    p.onScroll(/** @type {any} */ ({ target: child }));
    expect(loadMore).toHaveBeenCalledWith('t2');
  });

  test('ignores a target outside any box, a non-element and a null target', () => {
    const outside = document.createElement('div');
    document.body.append(outside);
    const { p, loadMore } = pager();
    p.onScroll(/** @type {any} */ ({ target: outside }));
    p.onScroll(/** @type {any} */ ({ target: document }));
    p.onScroll(/** @type {any} */ ({ target: null }));
    expect(loadMore).not.toHaveBeenCalled();
  });

  test('does nothing while the tree is not served', () => {
    const b = box('t1', { scrollHeight: 100, scrollTop: 0, clientHeight: 100 });
    const { p, loadMore } = pager({ served: false });
    p.onScroll(/** @type {any} */ ({ target: b }));
    expect(loadMore).not.toHaveBeenCalled();
  });
});

describe('fill', () => {
  test('pages every box with layout whose content fits, on the next frame', () => {
    const fits = box('fits', { scrollHeight: 300, scrollTop: 0, clientHeight: 300 });
    const long = box('long', { scrollHeight: 3000, scrollTop: 0, clientHeight: 300 });
    const hidden = box('hidden', { scrollHeight: 0, scrollTop: 0, clientHeight: 0 });
    const { p, loadMore } = pager({ boxes: [fits, long, hidden] });
    p.fill();
    expect(loadMore).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(loadMore).toHaveBeenCalledTimes(1);
    expect(loadMore).toHaveBeenCalledWith('fits');
  });

  test('coalesces calls made before the frame, and works again afterwards', () => {
    const fits = box('fits', { scrollHeight: 300, scrollTop: 0, clientHeight: 300 });
    const { p, loadMore } = pager({ boxes: [fits] });
    p.fill();
    p.fill();
    p.fill();
    vi.runAllTimers();
    expect(loadMore).toHaveBeenCalledTimes(1);
    p.fill();
    vi.runAllTimers();
    expect(loadMore).toHaveBeenCalledTimes(2);
  });
});

describe('watch', () => {
  /** @type {Array<{ cb: () => void, observed: Element[], disconnect: () => void }>} */
  let observers;
  beforeEach(() => {
    observers = [];
    vi.stubGlobal(
      'ResizeObserver',
      class {
        /** @param {() => void} cb */
        constructor(cb) {
          /** @type {{ cb: () => void, observed: Element[], disconnect: () => void }} */
          this.entry = { cb, observed: [], disconnect: vi.fn() };
          observers.push(this.entry);
        }
        /** @param {Element} el */
        observe(el) {
          this.entry.observed.push(el);
        }
        disconnect() {
          this.entry.disconnect();
        }
      }
    );
  });

  test('observes each element, fills on a resize, and disconnects on abort', () => {
    const fits = box('fits', { scrollHeight: 300, scrollTop: 0, clientHeight: 300 });
    const other = box('other', { scrollHeight: 3000, scrollTop: 0, clientHeight: 300 });
    const { p, loadMore } = pager({ boxes: [fits, other] });
    const ctl = new AbortController();
    p.watch([fits, other], ctl.signal);
    expect(observers).toHaveLength(1);
    expect(observers[0].observed).toEqual([fits, other]);
    observers[0].cb();
    vi.runAllTimers();
    expect(loadMore).toHaveBeenCalledWith('fits');
    expect(observers[0].disconnect).not.toHaveBeenCalled();
    ctl.abort();
    expect(observers[0].disconnect).toHaveBeenCalledTimes(1);
  });

  test('is a no-op without ResizeObserver', () => {
    vi.stubGlobal('ResizeObserver', undefined);
    const { p } = pager();
    expect(() => p.watch([], new AbortController().signal)).not.toThrow();
    expect(observers).toHaveLength(0);
  });
});
