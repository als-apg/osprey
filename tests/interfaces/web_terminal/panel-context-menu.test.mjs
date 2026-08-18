/**
 * Unit tests for the rail/tile context-menu popover (panel-context-menu.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-context-menu.test.mjs
 *
 * The module is purely presentational, so everything it owns is testable here:
 * the DOM contract, the dismissal paths (pointerdown, Escape, Tab, run, window
 * blur/resize, anchor-scoped scroll), and the focus hand-back rules. happy-dom
 * implements focus/activeElement well enough for the focus assertions; layout
 * is faked where clamping is under test, since happy-dom reports zero rects.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  openContextMenu,
  closeContextMenu,
  isContextMenuOpen,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-context-menu.js';

/** The open menu element, or null. @returns {HTMLElement | null} */
function menu() {
  return /** @type {HTMLElement | null} */ (document.querySelector('.rail-context-menu'));
}

/** The menu's rows, dividers excluded. @returns {HTMLElement[]} */
function rows() {
  return /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.rail-context-item')]);
}

/** The rendered label text of every row, in order. @returns {string[]} */
function rowLabels() {
  return rows().map((r) => /** @type {string} */ (r.querySelector('.rail-context-label')?.textContent));
}

/**
 * A rail entry inside a scrollable rail, plus an unrelated scroller standing in
 * for the terminal's xterm viewport.
 * @returns {{entry: HTMLButtonElement, rail: HTMLElement, viewport: HTMLElement}}
 */
function mountSurfaces() {
  document.body.innerHTML = `
    <div id="rail"><button id="entry" type="button">ARIEL</button></div>
    <div id="xterm-viewport"></div>`;
  return {
    entry: /** @type {HTMLButtonElement} */ (document.getElementById('entry')),
    rail: /** @type {HTMLElement} */ (document.getElementById('rail')),
    viewport: /** @type {HTMLElement} */ (document.getElementById('xterm-viewport')),
  };
}

/** Open a three-row service menu at the cursor. @param {object} [over] */
function openServiceMenu(over = {}) {
  const items = [
    { label: 'Focus ARIEL' },
    { label: 'Open in a new tile', glyph: '⊞' },
    { divider: true },
    { label: 'Remove from rail', glyph: '×', danger: true },
  ].map((i) => ({ run: () => {}, ...i }));
  openContextMenu({ x: 10, y: 20, ariaLabel: 'ARIEL actions', items, ...over });
}

/** @param {string} key */
function pressKey(key) {
  document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

beforeEach(() => {
  document.body.innerHTML = '';
});

afterEach(() => {
  closeContextMenu();
  vi.restoreAllMocks();
});

describe('openContextMenu — DOM contract', () => {
  test('renders rows, glyphs, dividers and danger styling', () => {
    openServiceMenu();

    const el = /** @type {HTMLElement} */ (menu());
    expect(el).not.toBeNull();
    expect(el.getAttribute('role')).toBe('menu');
    expect(el.getAttribute('aria-label')).toBe('ARIEL actions');
    expect(rowLabels()).toEqual(['Focus ARIEL', 'Open in a new tile', 'Remove from rail']);
    expect(rows().every((r) => r.getAttribute('role') === 'menuitem')).toBe(true);
    expect(rows()[2].classList.contains('danger')).toBe(true);

    const glyph = rows()[1].querySelector('.rail-context-glyph');
    expect(glyph?.textContent).toBe('⊞');
    expect(glyph?.getAttribute('aria-hidden')).toBe('true');

    const divider = el.querySelector('.rail-context-divider');
    expect(divider?.getAttribute('role')).toBe('separator');
  });

  test('labels are text, never markup (panel labels are server-supplied)', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'actions',
      items: [{ label: 'Focus <img src=x onerror=boom>', run: () => {} }],
    });

    expect(rows()[0].querySelector('img')).toBeNull();
    expect(rowLabels()[0]).toBe('Focus <img src=x onerror=boom>');
  });

  test('a disabled row is inert: aria-disabled, no run, menu stays open', () => {
    const run = vi.fn();
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'ARIEL actions',
      items: [{ label: 'Open in a new window', glyph: '↗', disabled: true, run }],
    });

    expect(rows()[0].getAttribute('aria-disabled')).toBe('true');
    rows()[0].click();
    expect(run).not.toHaveBeenCalled();
    expect(isContextMenuOpen()).toBe(true);
  });

  test('right-clicking the popover never stacks the native menu', () => {
    openServiceMenu();

    const e = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    rows()[0].dispatchEvent(e);

    expect(e.defaultPrevented).toBe(true);
  });

  test('only one menu is open at a time — a second replaces the first', () => {
    openServiceMenu();
    openContextMenu({ x: 0, y: 0, ariaLabel: 'Lattice actions', items: [{ label: 'Focus Lattice' }] });

    expect(document.querySelectorAll('.rail-context-menu')).toHaveLength(1);
    expect(menu()?.getAttribute('aria-label')).toBe('Lattice actions');
  });

  test('clamps inward rather than overflowing the viewport', () => {
    const width = 200;
    const height = 150;
    const x = window.innerWidth - 10;
    const y = window.innerHeight - 10;
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({ width, height, right: x + width, bottom: y + height })
    );

    openServiceMenu({ x, y });

    expect(menu()?.style.left).toBe(`${x - width}px`);
    expect(menu()?.style.top).toBe(`${y - height}px`);
  });

  test('clamping never pushes the menu off the top-left corner', () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({ width: 200, height: 150, right: 1e6, bottom: 1e6 })
    );

    openServiceMenu({ x: 5, y: 5 });

    expect(menu()?.style.left).toBe('0px');
    expect(menu()?.style.top).toBe('0px');
  });
});

describe('keyboard behavior', () => {
  test('the first enabled row takes focus on open', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'ARIEL actions',
      items: [
        { label: 'Open in a new window', disabled: true },
        { label: 'Focus ARIEL', run: () => {} },
      ],
    });

    expect(document.activeElement).toBe(rows()[1]);
  });

  test('arrows walk the enabled rows and wrap, skipping disabled ones', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'ARIEL actions',
      items: [
        { label: 'Focus ARIEL', run: () => {} },
        { label: 'Open in a new window', disabled: true },
        { label: 'Remove from rail', danger: true, run: () => {} },
      ],
    });

    pressKey('ArrowDown');
    expect(document.activeElement).toBe(rows()[2]);
    pressKey('ArrowDown');
    expect(document.activeElement).toBe(rows()[0]);
    pressKey('ArrowUp');
    expect(document.activeElement).toBe(rows()[2]);
  });
});

describe('dismissal', () => {
  test('pointerdown outside closes; inside does not', () => {
    const { entry } = mountSurfaces();
    openServiceMenu({ anchorEl: entry });

    rows()[0].dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    expect(isContextMenuOpen()).toBe(true);

    entry.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    expect(isContextMenuOpen()).toBe(false);
  });

  test('Escape and Tab both close the menu', () => {
    openServiceMenu();
    pressKey('Escape');
    expect(isContextMenuOpen()).toBe(false);

    openServiceMenu();
    pressKey('Tab');
    expect(isContextMenuOpen()).toBe(false);
  });

  test('Tab does not also move focus onward — its default is cancelled', () => {
    openServiceMenu();

    const e = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    document.dispatchEvent(e);

    expect(e.defaultPrevented).toBe(true);
  });

  test('window blur closes the menu (a click landing in a panel iframe)', () => {
    openServiceMenu();
    window.dispatchEvent(new Event('blur'));
    expect(isContextMenuOpen()).toBe(false);
  });

  test('window resize closes the menu', () => {
    openServiceMenu();
    window.dispatchEvent(new Event('resize'));
    expect(isContextMenuOpen()).toBe(false);
  });

  test('a document scroll closes the menu', () => {
    const { entry } = mountSurfaces();
    openServiceMenu({ anchorEl: entry });

    document.dispatchEvent(new Event('scroll'));

    expect(isContextMenuOpen()).toBe(false);
  });

  test('a scroll of a container holding the anchor closes the menu', () => {
    const { entry, rail } = mountSurfaces();
    openServiceMenu({ anchorEl: entry });

    rail.dispatchEvent(new Event('scroll'));

    expect(isContextMenuOpen()).toBe(false);
  });

  test('an unrelated scroller — xterm streaming output — never dismisses', () => {
    const { entry, viewport } = mountSurfaces();
    openServiceMenu({ anchorEl: entry });

    viewport.dispatchEvent(new Event('scroll'));

    expect(isContextMenuOpen()).toBe(true);
  });

  test('without an anchor only document scrolls dismiss', () => {
    const { viewport } = mountSurfaces();
    openServiceMenu();

    viewport.dispatchEvent(new Event('scroll'));
    expect(isContextMenuOpen()).toBe(true);

    document.dispatchEvent(new Event('scroll'));
    expect(isContextMenuOpen()).toBe(false);
  });

  test('closing detaches every listener', () => {
    const { entry } = mountSurfaces();
    entry.focus();
    openServiceMenu({ anchorEl: entry });
    pressKey('Escape');

    // Nothing left listening: these would each close or restore focus if the
    // teardown had missed a handler.
    const other = document.createElement('button');
    document.body.appendChild(other);
    other.focus();
    pressKey('Escape');
    window.dispatchEvent(new Event('blur'));
    window.dispatchEvent(new Event('resize'));
    document.dispatchEvent(new Event('scroll'));

    expect(document.activeElement).toBe(other);
  });

  test('closeContextMenu is safe with no menu open', () => {
    expect(() => closeContextMenu()).not.toThrow();
    expect(isContextMenuOpen()).toBe(false);
  });
});

describe('focus hand-back', () => {
  test('Escape returns focus to whatever held it at open', () => {
    const { entry } = mountSurfaces();
    entry.focus();
    openServiceMenu({ anchorEl: entry });
    expect(document.activeElement).toBe(rows()[0]);

    pressKey('Escape');

    expect(document.activeElement).toBe(entry);
  });

  test('Tab returns focus to whatever held it at open', () => {
    const { entry } = mountSurfaces();
    entry.focus();
    openServiceMenu({ anchorEl: entry });

    pressKey('Tab');

    expect(document.activeElement).toBe(entry);
  });

  test('replacing a menu keeps the ORIGINAL previous focus', () => {
    const { entry } = mountSurfaces();
    entry.focus();
    openServiceMenu({ anchorEl: entry });
    // Second menu opens while a row of the first holds focus — the row is not
    // somewhere focus can be handed back to.
    openServiceMenu({ anchorEl: entry });

    pressKey('Escape');

    expect(document.activeElement).toBe(entry);
  });

  test('pointer, blur, scroll and resize dismissals leave focus alone', () => {
    const { entry, rail } = mountSurfaces();

    for (const dismissBy of [
      () => entry.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true })),
      () => window.dispatchEvent(new Event('blur')),
      () => window.dispatchEvent(new Event('resize')),
      () => rail.dispatchEvent(new Event('scroll')),
    ]) {
      entry.focus();
      openServiceMenu({ anchorEl: entry });
      const rowEl = rows()[0];
      expect(document.activeElement).toBe(rowEl);

      dismissBy();

      expect(isContextMenuOpen()).toBe(false);
      // Focus stayed with the removed row rather than being yanked back — in a
      // real browser it has already moved on (the iframe, the clicked target).
      expect(document.activeElement).not.toBe(entry);
    }
  });

  test('closeContextMenu() does not move focus', () => {
    const { entry } = mountSurfaces();
    entry.focus();
    openServiceMenu({ anchorEl: entry });

    closeContextMenu();

    expect(document.activeElement).not.toBe(entry);
  });
});

describe('running an item', () => {
  test('the menu is gone and focus is back before the action runs', () => {
    const { entry } = mountSurfaces();
    entry.focus();
    const seen = /** @type {{open: boolean, focus: Element | null}[]} */ ([]);
    openContextMenu({
      x: 0,
      y: 0,
      anchorEl: entry,
      ariaLabel: 'ARIEL actions',
      items: [{ label: 'Focus ARIEL', run: () => seen.push({ open: isContextMenuOpen(), focus: document.activeElement }) }],
    });

    rows()[0].click();

    expect(seen).toEqual([{ open: false, focus: entry }]);
  });

  test('an action that moves focus itself wins', () => {
    const { entry } = mountSurfaces();
    const target = document.createElement('button');
    document.body.appendChild(target);
    entry.focus();
    openContextMenu({
      x: 0,
      y: 0,
      anchorEl: entry,
      ariaLabel: 'ARIEL actions',
      items: [{ label: 'Open in a new tile', run: () => target.focus() }],
    });

    rows()[0].click();

    expect(document.activeElement).toBe(target);
  });

  test('a row with no run closes the menu without throwing', () => {
    openContextMenu({ x: 0, y: 0, ariaLabel: 'ARIEL actions', items: [{ label: 'Focus ARIEL' }] });

    expect(() => rows()[0].click()).not.toThrow();
    expect(isContextMenuOpen()).toBe(false);
  });
});
