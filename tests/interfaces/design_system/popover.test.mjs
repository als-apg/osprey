// @ts-check
/**
 * Unit tests for the shared popover (design_system/static/js/popover.js):
 *   npx vitest run tests/interfaces/design_system/popover.test.mjs
 *
 * happy-dom does no layout, so each test stubs getBoundingClientRect on the
 * anchor (and the edge) and asserts the fixed-position style the module
 * computes from it.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import {
  closeAllPopovers,
  closePopover,
  openPopover,
  togglePopover,
} from '/design-system/js/popover.js';

/**
 * @param {HTMLElement} el
 * @param {{top: number, bottom: number, left: number, right: number}} rect
 */
function stubRect(el, rect) {
  el.getBoundingClientRect = () =>
    /** @type {DOMRect} */ ({
      ...rect,
      width: rect.right - rect.left,
      height: rect.bottom - rect.top,
      x: rect.left,
      y: rect.top,
      toJSON() {},
    });
}

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** @param {EventTarget} target */
function pointerDown(target) {
  target.dispatchEvent(new Event('pointerdown', { bubbles: true }));
}

beforeEach(() => {
  closeAllPopovers();
  document.body.innerHTML = `
    <nav id="sidebar">
      <button id="opener" type="button" aria-controls="kids" aria-expanded="false">Artifacts</button>
      <button id="other-opener" type="button">More</button>
      <div id="kids"></div>
    </nav>
    <main id="outside">outside</main>
    <div id="menu" class="osprey-popover" role="menu" hidden>
      <button id="menu-first" class="osprey-popover-item" type="button">Refresh</button>
      <button id="nested-opener" class="osprey-mini" type="button">⋯</button>
    </div>
    <div id="child" class="osprey-popover" role="menu" hidden>
      <button id="child-first" class="osprey-popover-item" type="button">All sessions</button>
    </div>
    <div id="more" class="osprey-popover" role="menu" hidden>
      <button class="osprey-popover-item" type="button">Settings</button>
    </div>
  `;
  stubRect($('opener'), { top: 100, bottom: 128, left: 8, right: 200 });
  stubRect($('other-opener'), { top: 600, bottom: 628, left: 8, right: 200 });
  stubRect($('sidebar'), { top: 0, bottom: 700, left: 0, right: 232 });
  stubRect($('nested-opener'), { top: 110, bottom: 130, left: 400, right: 420 });
});

describe('openPopover', () => {
  test('shows the popover, marks the opener and focuses the first item', () => {
    openPopover($('menu'), $('opener'));

    expect($('menu').hidden).toBe(false);
    expect($('opener').getAttribute('aria-expanded')).toBe('true');
    expect(document.activeElement).toBe($('menu-first'));
  });

  test('keeps the opener aria-controls in step with the popover it opened', () => {
    expect($('opener').getAttribute('aria-controls')).toBe('kids');

    openPopover($('menu'), $('opener'));

    expect($('opener').getAttribute('aria-controls')).toBe('menu');
  });

  test("'below-end' right-aligns 4 px under the anchor", () => {
    openPopover($('menu'), $('opener'), { place: 'below-end' });

    expect($('menu').style.top).toBe('132px');
    expect($('menu').style.right).toBe(`${window.innerWidth - 200}px`);
    expect($('menu').style.left).toBe('');
  });

  test("'beside-right' starts 8 px right of the edge, level with the anchor", () => {
    openPopover($('menu'), $('opener'), { place: 'beside-right', edge: $('sidebar') });

    expect($('menu').style.left).toBe('240px');
    expect($('menu').style.top).toBe('100px');
  });

  test("'beside-right' without an edge uses the anchor's right edge", () => {
    openPopover($('menu'), $('opener'), { place: 'beside-right' });

    expect($('menu').style.left).toBe('208px');
  });

  test("'beside-right-bottom' aligns the bottom edges", () => {
    openPopover($('more'), $('other-opener'), { place: 'beside-right-bottom', edge: $('sidebar') });

    expect($('more').style.left).toBe('240px');
    expect($('more').style.bottom).toBe(`${window.innerHeight - 628}px`);
    expect($('more').style.top).toBe('');
  });

  test('opening one popover closes another', () => {
    openPopover($('menu'), $('opener'));
    openPopover($('more'), $('other-opener'));

    expect($('menu').hidden).toBe(true);
    expect($('opener').getAttribute('aria-expanded')).toBe('false');
    expect($('more').hidden).toBe(false);
  });
});

describe('nesting', () => {
  test('a popover opened from inside another keeps its parent open', () => {
    openPopover($('menu'), $('opener'));
    openPopover($('child'), $('nested-opener'));

    expect($('menu').hidden).toBe(false);
    expect($('child').hidden).toBe(false);
    expect($('nested-opener').getAttribute('aria-expanded')).toBe('true');
  });

  test('a press inside the parent keeps it open and closes the child', () => {
    openPopover($('menu'), $('opener'));
    openPopover($('child'), $('nested-opener'));

    pointerDown($('menu-first'));

    expect($('menu').hidden).toBe(false);
    expect($('child').hidden).toBe(true);
  });

  test('a press inside the child keeps both open', () => {
    openPopover($('menu'), $('opener'));
    openPopover($('child'), $('nested-opener'));

    pointerDown($('child-first'));

    expect($('menu').hidden).toBe(false);
    expect($('child').hidden).toBe(false);
  });

  test('closing the parent closes the child too', () => {
    openPopover($('menu'), $('opener'));
    openPopover($('child'), $('nested-opener'));

    closePopover($('menu'));

    expect($('child').hidden).toBe(true);
    expect($('nested-opener').getAttribute('aria-expanded')).toBe('false');
  });
});

describe('dismissal', () => {
  test('a press outside closes every popover and resets the opener', () => {
    openPopover($('menu'), $('opener'));

    pointerDown($('outside'));

    expect($('menu').hidden).toBe(true);
    expect($('opener').getAttribute('aria-expanded')).toBe('false');
  });

  test('a press on the opener leaves the toggle to its click', () => {
    openPopover($('menu'), $('opener'));

    pointerDown($('opener'));

    expect($('menu').hidden).toBe(false);
  });

  test('Escape closes and returns focus to the opener', () => {
    openPopover($('menu'), $('opener'));
    expect(document.activeElement).toBe($('menu-first'));

    document.activeElement?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect($('menu').hidden).toBe(true);
    expect(document.activeElement).toBe($('opener'));
  });

  test('Escape from a nested popover closes both and focuses the outer opener', () => {
    openPopover($('menu'), $('opener'));
    openPopover($('child'), $('nested-opener'));

    $('child-first').dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect($('menu').hidden).toBe(true);
    expect($('child').hidden).toBe(true);
    expect(document.activeElement).toBe($('opener'));
  });

  test('a window resize closes', () => {
    openPopover($('menu'), $('opener'));

    window.dispatchEvent(new Event('resize'));

    expect($('menu').hidden).toBe(true);
  });
});

describe('togglePopover and closeAllPopovers', () => {
  test('toggle opens a closed popover and closes an open one', () => {
    togglePopover($('opener'), $('menu'));
    expect($('menu').hidden).toBe(false);

    togglePopover($('opener'), $('menu'));
    expect($('menu').hidden).toBe(true);
    expect($('opener').getAttribute('aria-expanded')).toBe('false');
  });

  test('closeAllPopovers spares the popovers in except', () => {
    openPopover($('menu'), $('opener'));
    openPopover($('child'), $('nested-opener'));

    closeAllPopovers([$('menu')]);

    expect($('menu').hidden).toBe(false);
    expect($('child').hidden).toBe(true);
  });

  test('a popover this module did not open is left alone', () => {
    $('more').hidden = false;
    openPopover($('menu'), $('opener'));

    closeAllPopovers();

    expect($('menu').hidden).toBe(true);
    expect($('more').hidden).toBe(false);
  });
});
