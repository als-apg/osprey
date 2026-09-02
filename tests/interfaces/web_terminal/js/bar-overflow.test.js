/**
 * The bar crowding ladder — happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/bar-overflow.test.js
 *
 * happy-dom has no layout engine: every element is 0 px wide, so a real
 * measurement would report a bar that is never crowded and none of the rungs
 * would ever fire. The ladder therefore takes its width from ONE injectable
 * probe, and these tests drive it through `mockCrowding()` — which is also the
 * seam the Playwright suite replaces with a real browser to prove the CSS half
 * (spacing shrink and text ellipsis) actually prevents overflow.
 *
 * What is pinned here is the POLICY, which is the part a browser cannot
 * assert cheaply: fold order follows the catalog's priority, only the six
 * foldable types ever fold, the header chrome and search never do, spacing is
 * untouched by any rung, an unfolded item is the SAME node object it was
 * before (a rebuild would silently drop the adopted chrome the host exists to
 * protect), and a folded item is reachable by name from the overflow menu.
 */

import { test, expect, describe, beforeEach, afterEach } from 'vitest';

import {
  hostElement,
  hydrate,
  poolElement,
  reconcile,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';
import {
  applyOverflow,
  mockCrowding,
  resetOverflow,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-overflow.js';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayout} BarLayout */

/** Undo the probe installed by the test currently running. @type {(() => void)[]} */
let restores = [];

/**
 * A layout document naming `header` and `status` items in order.
 * @param {string[]} header
 * @param {string[]} [status]
 * @returns {BarLayout}
 */
function layoutOf(header, status = []) {
  /** @param {string} type */
  const item = (type) => ({ type, options: {} });
  return {
    version: 1,
    rev: 0,
    header: header.map(item),
    status: status.map(item),
    header_visible: true,
    status_visible: true,
  };
}

/** The SSR DOM: both hosts and the hidden pool. */
function seedDom() {
  document.body.innerHTML = `
    <header class="header">
      <div class="header-actions" data-bar-host="header"></div>
    </header>
    <footer class="status-bar" data-bar-host="status"></footer>
    <div id="bar-item-pool" hidden></div>
  `;
  hydrate(document);
}

/**
 * Install a probe and remember how to uninstall it.
 * @param {(container: HTMLElement, host: 'header' | 'status') =>
 *          {overflow: number, width: number}} fake
 */
function useCrowding(fake) {
  restores.push(mockCrowding(fake));
}

/**
 * A probe for a host that fits exactly `capacity` items: every item beyond
 * that is 100 px of overflow. Collapsed search counts as no item at all, which
 * is what makes rung 2 buy room the same way a fold does.
 * @param {number} capacity
 * @param {number} [width]
 */
function crowdingAtCapacity(capacity, width = 800) {
  useCrowding((container) => {
    const shells = Array.from(container.querySelectorAll('.bar-item')).filter(
      (shell) => /** @type {HTMLElement} */ (shell).dataset.barCollapsed !== 'true'
    );
    return { overflow: (shells.length - capacity) * 100, width };
  });
}

/** @returns {string[]} the `data-bar-item` types a host shows, in DOM order */
function typesIn(/** @type {'header' | 'status'} */ host) {
  const container = hostElement(host, document);
  if (!container) return [];
  return Array.from(container.querySelectorAll('.bar-item')).map(
    (el) => /** @type {HTMLElement} */ (el).dataset.barItem ?? ''
  );
}

/** @returns {string[]} the `data-bar-item` types currently parked in the pool */
function typesInPool() {
  const pool = poolElement(document);
  if (!pool) return [];
  return Array.from(pool.querySelectorAll('.bar-item')).map(
    (el) => /** @type {HTMLElement} */ (el).dataset.barItem ?? ''
  );
}

/** The header's overflow trigger, if the ladder is showing one. */
function trigger() {
  return /** @type {HTMLButtonElement | null} */ (
    document.querySelector('.header-actions .bar-overflow-trigger')
  );
}

/** The labels of the rows in the open overflow popover, in order. */
function menuLabels() {
  return Array.from(document.querySelectorAll('.contrib-menu-popover .contrib-menu-item')).map(
    (row) => row.textContent ?? ''
  );
}

/** The shell for a type in whichever container it currently sits in. */
function shellOf(/** @type {string} */ type) {
  return /** @type {HTMLElement} */ (document.querySelector(`.bar-item[data-bar-item="${type}"]`));
}

beforeEach(() => {
  seedDom();
});

afterEach(() => {
  resetOverflow();
  for (const restore of restores) restore();
  restores = [];
  document.body.innerHTML = '';
});

describe('rung 3 — foldable items fold, lowest priority first', () => {
  test('folds in ascending catalog priority and stops as soon as the bar fits', () => {
    reconcile(layoutOf(['logo', 'docs', 'clock', 'stopwatch']));
    // stopwatch 10 < clock 40 < docs 50: two rungs, and the third never fires.
    crowdingAtCapacity(2);
    applyOverflow(document);

    expect(typesIn('header')).toEqual(['logo', 'docs']);
    // The pool is in arrival order, so it reads as the order they folded in.
    expect(typesInPool()).toEqual(['stopwatch', 'clock']);
  });

  test('only the six overflowLabel types are candidates', () => {
    // control-target and search both declare a null overflowLabel:
    // a bar of nothing but those has no rung 3 at all, however crowded it is.
    reconcile(layoutOf(['control-target', 'search']));
    crowdingAtCapacity(1);
    applyOverflow(document);

    expect(typesIn('header')).toEqual(['control-target', 'search']);
    expect(typesInPool()).toEqual([]);
    expect(trigger()).toBeNull();
  });

  test('the header chrome never folds, however narrow the bar gets', () => {
    reconcile(layoutOf(['logo', 'identity', 'control-target', 'display', 'clock']));
    crowdingAtCapacity(0);
    applyOverflow(document);

    expect(typesIn('header')).toEqual(['logo', 'identity', 'control-target', 'display']);
    expect(typesInPool()).toEqual(['clock']);
  });

  test('folding parks the shell rather than hiding it', () => {
    // `hidden` on a shell is the mirror's output channel in bar-host.js; a
    // ladder that wrote it would be overwritten on the next placement pass.
    reconcile(layoutOf(['logo', 'clock']));
    const clock = shellOf('clock');
    crowdingAtCapacity(1);
    applyOverflow(document);

    expect(poolElement(document)?.contains(clock)).toBe(true);
    expect(clock.hasAttribute('hidden')).toBe(false);
    expect(clock.dataset.barDensity).toBeUndefined();
  });

  test('each host climbs its own ladder', () => {
    reconcile(layoutOf(['logo', 'clock'], ['docs', 'stopwatch']));
    useCrowding((container, host) => ({
      overflow: host === 'status' ? (container.querySelectorAll('.bar-item').length - 1) * 100 : -1,
      width: 800,
    }));
    applyOverflow(document);

    expect(typesIn('header')).toEqual(['logo', 'clock']);
    expect(typesIn('status')).toEqual(['docs']);
  });
});

describe('rung 2 — search collapses but never folds', () => {
  test('search collapses before any item folds', () => {
    reconcile(layoutOf(['logo', 'search', 'clock']));
    crowdingAtCapacity(2);
    applyOverflow(document);

    expect(shellOf('search').dataset.barCollapsed).toBe('true');
    expect(typesIn('header')).toEqual(['logo', 'search', 'clock']);
    expect(typesInPool()).toEqual([]);
  });

  test('search stays in the bar even when everything else has folded', () => {
    reconcile(layoutOf(['search', 'docs', 'clock', 'stopwatch']));
    crowdingAtCapacity(0);
    applyOverflow(document);

    expect(typesIn('header')).toEqual(['search']);
    expect(shellOf('search').dataset.barCollapsed).toBe('true');
    expect(typesInPool()).toEqual(['stopwatch', 'clock', 'docs']);
  });

  test('a bar that can never fit still keeps search, and stops climbing', () => {
    // The pathological case: a probe that reports crowding no matter what.
    // The ladder must run out of rungs rather than fold the last controls
    // away, and it must terminate rather than spin on its step budget.
    reconcile(layoutOf(['logo', 'search', 'docs', 'clock']));
    useCrowding(() => ({ overflow: 500, width: 320 }));
    applyOverflow(document);

    expect(typesIn('header')).toEqual(['logo', 'search']);
    expect(shellOf('search').dataset.barCollapsed).toBe('true');
    expect(typesInPool()).toEqual(['clock', 'docs']);
  });

  test('search un-collapses last, after every folded item is back', () => {
    reconcile(layoutOf(['search', 'clock']));
    crowdingAtCapacity(0, 400);
    applyOverflow(document);
    expect(typesInPool()).toEqual(['clock']);

    // Room for one item again: the clock comes back, search stays collapsed.
    restores.pop()?.();
    crowdingAtCapacity(1, 600);
    applyOverflow(document);
    expect(typesIn('header')).toEqual(['search', 'clock']);
    expect(shellOf('search').dataset.barCollapsed).toBe('true');

    // Room for both: the last rung is undone too.
    restores.pop()?.();
    crowdingAtCapacity(2, 800);
    applyOverflow(document);
    expect(shellOf('search').dataset.barCollapsed).toBeUndefined();
  });
});

describe('spacing is never touched by the ladder', () => {
  test('spaces and separators stay put and keep their declared flex hints', () => {
    // Spacing yields continuously through CSS flex-shrink, ahead of every JS
    // rung. A ladder that folded a space would be racing CSS for those pixels.
    reconcile(layoutOf(['logo', 'space', 'separator', 'clock']));
    const space = shellOf('space');
    const before = space.style.getPropertyValue('flex-basis');
    crowdingAtCapacity(0);
    applyOverflow(document);

    expect(typesIn('header')).toEqual(['logo', 'space', 'separator']);
    expect(typesInPool()).toEqual(['clock']);
    expect(space.style.getPropertyValue('flex-basis')).toBe(before);
  });
});

describe('unfolding restores the item that folded', () => {
  test('the same node object comes back, in its old place', () => {
    reconcile(layoutOf(['logo', 'clock', 'docs']));
    const clock = shellOf('clock');
    /** @type {any} */ (clock).__ospreyMarker = Symbol('clock');

    crowdingAtCapacity(2, 400);
    applyOverflow(document);
    expect(typesIn('header')).toEqual(['logo', 'docs']);

    restores.pop()?.();
    crowdingAtCapacity(3, 800);
    applyOverflow(document);

    const back = shellOf('clock');
    expect(back).toBe(clock);
    expect(/** @type {any} */ (back).__ospreyMarker).toBe(
      /** @type {any} */ (clock).__ospreyMarker
    );
    expect(typesIn('header')).toEqual(['logo', 'clock', 'docs']);
    expect(back.dataset.barDensity).toBe('comfortable');
  });

  test('a wider bar alone unfolds nothing — the width must exceed the fold width', () => {
    reconcile(layoutOf(['logo', 'clock']));
    crowdingAtCapacity(1, 500);
    applyOverflow(document);
    expect(typesInPool()).toEqual(['clock']);

    // Same width, no crowding reported: the rung stays taken rather than
    // oscillating between folded and unfolded at one width.
    restores.pop()?.();
    useCrowding(() => ({ overflow: 0, width: 500 }));
    applyOverflow(document);
    expect(typesInPool()).toEqual(['clock']);
  });

  test('items come back highest priority first', () => {
    reconcile(layoutOf(['docs', 'clock', 'stopwatch']));
    crowdingAtCapacity(0, 300);
    applyOverflow(document);
    expect(typesIn('header')).toEqual([]);

    restores.pop()?.();
    crowdingAtCapacity(1, 500);
    applyOverflow(document);
    // stopwatch folded first, docs last, so docs is the first back.
    expect(typesIn('header')).toEqual(['docs']);

    restores.pop()?.();
    crowdingAtCapacity(2, 700);
    applyOverflow(document);
    expect(typesIn('header')).toEqual(['docs', 'clock']);
  });
});

describe('the overflow menu names what folded', () => {
  test('a trigger appears only while something is folded', () => {
    reconcile(layoutOf(['logo', 'clock']));
    crowdingAtCapacity(2);
    applyOverflow(document);
    expect(trigger()).toBeNull();

    restores.pop()?.();
    crowdingAtCapacity(1, 400);
    applyOverflow(document);
    const btn = trigger();
    expect(btn).toBeTruthy();
    expect(btn?.getAttribute('aria-label')).toBe('More items');
    // Last in the bar, so it reads as the tail of the item list.
    expect(hostElement('header', document)?.lastElementChild).toBe(btn);

    restores.pop()?.();
    crowdingAtCapacity(2, 800);
    applyOverflow(document);
    expect(trigger()).toBeNull();
  });

  test("each folded item is one row, under its catalog overflowLabel", () => {
    reconcile(layoutOf(['logo', 'docs', 'clock', 'stopwatch']));
    crowdingAtCapacity(1);
    applyOverflow(document);
    expect(typesInPool()).toEqual(['stopwatch', 'clock', 'docs']);

    trigger()?.click();
    // Most recently folded first — docs folded last, so it heads the list.
    expect(menuLabels()).toEqual(['Documentation', 'Clock', 'Stopwatch']);
  });

  test('picking a row brings that item back and folds a different one', () => {
    reconcile(layoutOf(['logo', 'docs', 'clock', 'stopwatch']));
    const stopwatch = shellOf('stopwatch');
    crowdingAtCapacity(3);
    applyOverflow(document);
    expect(typesIn('header')).toEqual(['logo', 'docs', 'clock']);

    trigger()?.click();
    expect(menuLabels()).toEqual(['Stopwatch']);
    /** @type {HTMLElement} */ (
      document.querySelector('.contrib-menu-popover .contrib-menu-item')
    ).click();

    // The promoted item is back and the next-lowest priority folded in its
    // place — picking a row must not simply undo itself.
    expect(shellOf('stopwatch')).toBe(stopwatch);
    expect(typesIn('header')).toEqual(['logo', 'docs', 'stopwatch']);
    expect(typesInPool()).toEqual(['clock']);
  });
});

describe('data-follows tracks what the bar actually shows', () => {
  test('folding an item moves the adjacency claim onto its neighbour', () => {
    reconcile(layoutOf(['logo', 'clock', 'identity']));
    expect(shellOf('identity').dataset.follows).toBe('clock');

    crowdingAtCapacity(2);
    applyOverflow(document);
    expect(shellOf('identity').dataset.follows).toBe('logo');
  });
});
