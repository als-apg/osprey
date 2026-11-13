/**
 * Unit tests for the shared context menu (design_system/static/js/context-menu.js).
 *
 *   npx vitest run tests/interfaces/design_system/context-menu.test.mjs
 *
 * The dismissal and focus contract is exercised through the web terminal's
 * wrapper in tests/interfaces/web_terminal/panel-context-menu.test.mjs, which
 * runs the same code under the hub's class names. This file covers what the
 * shared module adds: its own default class names, heading and toggle rows,
 * the monospace value segment, the keyboard opener, the key predicate and the
 * browser-passthrough predicate.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  openContextMenu,
  openContextMenuAt,
  isContextMenuKey,
  declinesToBrowser,
  closeContextMenu,
} from '/design-system/js/context-menu.js';

/** @returns {HTMLElement | null} */
function menu() {
  return /** @type {HTMLElement | null} */ (document.querySelector('.osprey-context-menu'));
}

/** @returns {HTMLElement[]} */
function rows() {
  return /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.osprey-context-item')]);
}

beforeEach(() => {
  document.body.innerHTML = '';
});

afterEach(() => {
  document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
  vi.restoreAllMocks();
});

describe('default class names', () => {
  test('renders under the osprey-context stem', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'Row actions',
      items: [{ label: 'Copy', glyph: '⎘' }, { divider: true }],
    });
    const el = menu();
    expect(el?.getAttribute('role')).toBe('menu');
    expect(el?.getAttribute('aria-label')).toBe('Row actions');
    expect(rows()[0].querySelector('.osprey-context-label')?.textContent).toBe('Copy');
    expect(rows()[0].querySelector('.osprey-context-glyph')?.textContent).toBe('⎘');
    expect(el?.querySelector('.osprey-context-divider')?.getAttribute('role')).toBe('separator');
  });

  test('a caller-supplied prefix renames every part', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      classPrefix: 'rail-context',
      items: [{ label: 'Focus' }],
    });
    expect(menu()).toBeNull();
    expect(
      document.querySelector('.rail-context-menu .rail-context-item .rail-context-label')
    ).not.toBeNull();
  });

  test('the next open without a prefix goes back to the default', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      classPrefix: 'rail-context',
      items: [{ label: 'A' }],
    });
    openContextMenu({ x: 0, y: 0, ariaLabel: 'b', items: [{ label: 'B' }] });
    expect(document.querySelector('.rail-context-menu')).toBeNull();
    expect(menu()).not.toBeNull();
  });
});

describe('row kinds', () => {
  test('a heading is an inert caption, not a focusable row', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'Columns',
      items: [{ heading: 'Columns' }, { label: 'Device', checked: true }],
    });
    const heading = menu()?.querySelector('.osprey-context-heading');
    expect(heading?.textContent).toBe('Columns');
    expect(heading?.tagName).toBe('DIV');
    expect(rows()).toHaveLength(1);
    expect(document.activeElement).toBe(rows()[0]);
  });

  test('a toggle row is a menuitemcheckbox carrying its state and a tick', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'Columns',
      items: [
        { label: 'Device', checked: true },
        { label: 'System', checked: false },
        { label: 'Reset' },
      ],
    });
    const [on, off, plain] = rows();
    expect(on.getAttribute('role')).toBe('menuitemcheckbox');
    expect(on.getAttribute('aria-checked')).toBe('true');
    expect(on.classList.contains('checked')).toBe(true);
    expect(on.querySelector('.osprey-context-glyph')?.textContent).toBe('✓');
    expect(off.getAttribute('aria-checked')).toBe('false');
    expect(off.querySelector('.osprey-context-glyph')).toBeNull();
    expect(plain.getAttribute('role')).toBe('menuitem');
    expect(plain.hasAttribute('aria-checked')).toBe(false);
  });

  test('the value segment is a <code> element holding text, never markup', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      items: [{ label: 'Only section', code: '<b>SR01C</b>' }],
    });
    const label = rows()[0].querySelector('.osprey-context-label');
    expect(label?.querySelector('b')).toBeNull();
    expect(label?.querySelector('code')?.textContent).toBe('<b>SR01C</b>');
    expect(label?.textContent).toBe('Only section <b>SR01C</b>');
  });

  test('running a row closes the menu first, then runs it', () => {
    let openWhenRun = true;
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      items: [
        {
          label: 'Go',
          run: () => {
            openWhenRun = menu() !== null;
          },
        },
      ],
    });
    rows()[0].click();
    expect(openWhenRun).toBe(false);
  });
});

describe('row columns', () => {
  /** @param {HTMLElement} row @returns {string[]} the row's child classes, in order */
  const columns = (row) => [...row.children].map((c) => c.className);

  test('the glyph leads the label; nothing trails it', () => {
    openContextMenu({ x: 0, y: 0, ariaLabel: 'a', items: [{ label: 'Copy', glyph: '⎘' }] });
    expect(columns(rows()[0])).toEqual(['osprey-context-glyph', 'osprey-context-label']);
    expect(rows()[0].firstElementChild?.textContent).toBe('⎘');
  });

  test('a row without a glyph keeps an empty leading slot, so labels align', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      items: [{ label: 'Copy', glyph: '⎘' }, { label: 'Reset' }],
    });
    const plain = rows()[1];
    expect(columns(plain)).toEqual(['osprey-context-glyph-blank', 'osprey-context-label']);
    expect(plain.firstElementChild?.textContent).toBe('');
    expect(plain.firstElementChild?.getAttribute('aria-hidden')).toBe('true');
    expect(plain.querySelector('.osprey-context-glyph')).toBeNull();
  });

  test('a toggle draws its tick in the leading slot', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      items: [
        { label: 'Device', checked: true },
        { label: 'System', checked: false },
      ],
    });
    const [on, off] = rows();
    expect(columns(on)[0]).toBe('osprey-context-glyph');
    expect(on.firstElementChild?.textContent).toBe('✓');
    expect(columns(off)[0]).toBe('osprey-context-glyph-blank');
  });

  test('a hint is a third column after the label, outside the label text', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      items: [{ label: 'Close', glyph: '×', hint: '⌘W' }],
    });
    const row = rows()[0];
    expect(columns(row)).toEqual([
      'osprey-context-glyph',
      'osprey-context-label',
      'osprey-context-hint',
    ]);
    expect(row.querySelector('.osprey-context-label')?.textContent).toBe('Close');
    expect(row.querySelector('.osprey-context-hint')?.textContent).toBe('⌘W');
  });

  test('the hint is text, never markup', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      items: [{ label: 'Go', hint: '<b>x</b>' }],
    });
    const hint = rows()[0].querySelector('.osprey-context-hint');
    expect(hint?.querySelector('b')).toBeNull();
    expect(hint?.textContent).toBe('<b>x</b>');
  });

  test('a row without a hint has no hint column', () => {
    openContextMenu({ x: 0, y: 0, ariaLabel: 'a', items: [{ label: 'Go', glyph: '↗' }] });
    expect(rows()[0].querySelector('.osprey-context-hint')).toBeNull();
  });

  test('the rail prefix gets the same columns', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      classPrefix: 'rail-context',
      items: [{ label: 'Focus' }, { label: 'Pop out', glyph: '↗', hint: 'tab' }],
    });
    const items = [...document.querySelectorAll('.rail-context-item')];
    expect(columns(/** @type {HTMLElement} */ (items[0]))).toEqual([
      'rail-context-glyph-blank',
      'rail-context-label',
    ]);
    expect(columns(/** @type {HTMLElement} */ (items[1]))).toEqual([
      'rail-context-glyph',
      'rail-context-label',
      'rail-context-hint',
    ]);
  });
});

describe('keyboard opener', () => {
  test('opens at the element’s lower-left corner and anchors to it', () => {
    document.body.innerHTML = '<div id="row" tabindex="0">row</div>';
    const row = /** @type {HTMLElement} */ (document.getElementById('row'));
    vi.spyOn(row, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({
        left: 100,
        top: 40,
        right: 300,
        bottom: 60,
        width: 200,
        height: 20,
        x: 100,
        y: 40,
      })
    );
    openContextMenuAt(row, { ariaLabel: 'Row actions', items: [{ label: 'Copy' }] });
    expect(menu()?.style.left).toBe('108px');
    expect(menu()?.style.top).toBe('60px');
  });

  test('Shift+F10 and the ContextMenu key are requests; plain F10 is not', () => {
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'ContextMenu' }))).toBe(true);
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'F10', shiftKey: true }))).toBe(
      true
    );
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'F10' }))).toBe(false);
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'Enter' }))).toBe(false);
  });
});

describe('trigger button (a ⋯)', () => {
  /** A button with a known rect, and a menu with a known size. */
  function setup() {
    document.body.innerHTML = '<button id="more">⋯</button>';
    const btn = /** @type {HTMLElement} */ (document.getElementById('more'));
    vi.spyOn(btn, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({ left: 500, right: 528, top: 10, bottom: 38, width: 28, height: 28 })
    );
    const menuRect = /** @type {DOMRect} */ ({ width: 200, height: 150, right: 200, bottom: 150 });
    const proto = vi.spyOn(HTMLDivElement.prototype, 'getBoundingClientRect');
    proto.mockReturnValue(menuRect);
    return { btn, menuRect };
  }

  /** @param {HTMLElement} btn */
  const open = (btn) =>
    openContextMenu({
      x: 0,
      y: 0,
      triggerEl: btn,
      ariaLabel: 'Panel menu',
      items: [{ label: 'Go' }],
    });

  test('hangs under the button, right edges aligned, and marks it expanded', () => {
    const { btn } = setup();
    open(btn);
    expect(menu()?.style.left).toBe('328px'); // 528 - 200
    expect(menu()?.style.top).toBe('42px'); // 38 + 4
    expect(btn.getAttribute('aria-expanded')).toBe('true');
  });

  test('flips above the button when the viewport is short, never off the top', () => {
    const { btn, menuRect } = setup();
    vi.spyOn(window, 'innerHeight', 'get').mockReturnValue(100);
    open(btn);
    expect(menu()?.style.top).toBe('0px'); // 10 - 150 - 4 clamps at 0
    menuRect.height = 5;
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    open(btn);
    expect(menu()?.style.top).toBe('42px'); // 42 + 5 fits in 100
  });

  test("side: 'right' opens beside the button, and at the right edge on its left", () => {
    const { btn } = setup();
    vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(1000);
    openContextMenu({ x: 0, y: 0, triggerEl: btn, side: 'right', ariaLabel: 'View', items: [] });
    expect(menu()?.style.left).toBe('526px'); // 528 - the 2 px submenu overlap
    expect(menu()?.style.top).toBe('6px'); // 10 - the 4 px submenu rise
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(600);
    openContextMenu({ x: 0, y: 0, triggerEl: btn, side: 'right', ariaLabel: 'View', items: [] });
    expect(menu()?.style.left).toBe('302px'); // 500 - 200 + 2
  });

  test('a press on the button is not an outside press; the next open from it toggles the menu closed', () => {
    const { btn } = setup();
    open(btn);
    btn.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    expect(menu()).not.toBeNull();

    open(btn);
    expect(menu()).toBeNull();
    expect(btn.getAttribute('aria-expanded')).toBe('false');

    // A third open is an open again.
    open(btn);
    expect(menu()).not.toBeNull();
  });

  test('every other dismissal clears aria-expanded; another trigger replaces the menu', () => {
    const { btn } = setup();
    open(btn);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(menu()).toBeNull();
    expect(btn.getAttribute('aria-expanded')).toBe('false');

    open(btn);
    const other = document.createElement('button');
    document.body.appendChild(other);
    openContextMenu({ x: 0, y: 0, triggerEl: other, ariaLabel: 'b', items: [{ label: 'B' }] });
    expect(document.querySelectorAll('.osprey-context-menu').length).toBe(1);
    expect(btn.getAttribute('aria-expanded')).toBe('false');
    expect(other.getAttribute('aria-expanded')).toBe('true');
  });

  test('without a trigger the cursor placement and clamping are as before', () => {
    setup();
    openContextMenu({ x: 30, y: 40, ariaLabel: 'a', items: [{ label: 'A' }] });
    expect(menu()?.style.left).toBe('30px');
    expect(menu()?.style.top).toBe('40px');
  });
});

describe('submenus', () => {
  /** @returns {HTMLElement[]} every open level, root first */
  const levels = () =>
    /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.osprey-context-menu')]);

  /** @param {string} label */
  const row = (label) =>
    /** @type {HTMLElement} */ (
      rows().find((r) => r.querySelector('.osprey-context-label')?.textContent === label)
    );

  /** @param {string} key */
  const press = (key) =>
    document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));

  /** @param {HTMLElement} el */
  const hover = (el) => el.dispatchEvent(new MouseEvent('pointerenter'));

  /** @param {{ run?: () => void }} [opts] */
  function open(opts = {}) {
    openContextMenu({
      x: 10,
      y: 10,
      ariaLabel: 'Tile',
      items: [
        { label: 'Rename' },
        {
          label: 'Open panel',
          hint: '⌘O',
          submenu: [
            { label: 'Logs', run: opts.run },
            { label: 'Disabled', disabled: true },
            { label: 'More', submenu: [{ label: 'Deep', run: opts.run }] },
          ],
        },
        { label: 'Locked', disabled: true, submenu: [{ label: 'X' }] },
      ],
    });
  }

  test('a submenu row shows ▸ after its hint, carries the menu ARIA, and runs nothing', () => {
    open();
    const parent = row('Open panel');
    const classes = [...parent.children].map((c) => c.className);
    expect(classes.slice(-2)).toEqual(['osprey-context-hint', 'osprey-context-submenu']);
    const arrow = /** @type {HTMLElement} */ (parent.querySelector('.osprey-context-submenu'));
    expect(arrow.textContent).toBe('▸');
    expect(arrow.getAttribute('aria-hidden')).toBe('true');
    expect(parent.getAttribute('aria-haspopup')).toBe('menu');
    expect(parent.getAttribute('aria-expanded')).toBe('false');
    expect(row('Rename').hasAttribute('aria-haspopup')).toBe(false);
    expect(row('Rename').querySelector('.osprey-context-submenu')).toBeNull();

    parent.click();
    expect(levels().length).toBe(2);
    expect(parent.getAttribute('aria-expanded')).toBe('true');
    const child = levels()[1];
    expect(child.getAttribute('role')).toBe('menu');
    expect(child.getAttribute('aria-label')).toBe('Open panel');
    expect(child.dataset.depth).toBe('1');
    // Clicking again does not toggle it closed.
    parent.click();
    expect(levels().length).toBe(2);
  });

  test('hover opens the submenu after a short rest, not before; leaving cancels it', () => {
    vi.useFakeTimers();
    try {
      open();
      const parent = row('Open panel');
      hover(parent);
      vi.advanceTimersByTime(50);
      expect(levels().length).toBe(1);
      vi.advanceTimersByTime(200);
      expect(levels().length).toBe(2);
      expect(parent.getAttribute('aria-expanded')).toBe('true');

      // Resting on a sibling row closes it again.
      hover(row('Rename'));
      expect(levels().length).toBe(1);
      expect(parent.getAttribute('aria-expanded')).toBe('false');

      hover(parent);
      parent.dispatchEvent(new MouseEvent('pointerleave'));
      vi.advanceTimersByTime(500);
      expect(levels().length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  test.each(['Enter', ' ', 'ArrowRight'])(
    '%j on a submenu row opens it and focuses its first enabled row',
    (key) => {
      open();
      row('Open panel').focus();
      press(key);
      expect(levels().length).toBe(2);
      expect(document.activeElement).toBe(row('Logs'));
    }
  );

  test('ArrowRight on a plain row opens nothing; a disabled submenu row never opens', () => {
    open();
    press('ArrowRight');
    expect(levels().length).toBe(1);
    const locked = row('Locked');
    expect(locked.getAttribute('aria-haspopup')).toBe('menu');
    locked.click();
    locked.focus();
    press('Enter');
    expect(levels().length).toBe(1);
  });

  test('arrows move only among the focused level’s enabled rows', () => {
    open();
    row('Open panel').focus();
    press('ArrowRight');
    press('ArrowDown');
    expect(document.activeElement).toBe(row('More'));
    press('ArrowDown');
    expect(document.activeElement).toBe(row('Logs'));
    press('ArrowUp');
    expect(document.activeElement).toBe(row('More'));
  });

  test('ArrowLeft and Escape close one level and focus the parent row', () => {
    open();
    row('Open panel').focus();
    press('ArrowRight');
    row('More').focus();
    press('ArrowRight');
    expect(levels().length).toBe(3);
    expect(levels()[2].dataset.depth).toBe('2');
    expect(document.activeElement).toBe(row('Deep'));

    press('ArrowLeft');
    expect(levels().length).toBe(2);
    expect(document.activeElement).toBe(row('More'));
    expect(row('More').getAttribute('aria-expanded')).toBe('false');

    press('Escape');
    expect(levels().length).toBe(1);
    expect(document.activeElement).toBe(row('Open panel'));

    // ArrowLeft on the root closes nothing; Escape there closes the menu.
    press('ArrowLeft');
    expect(levels().length).toBe(1);
    press('Escape');
    expect(levels().length).toBe(0);
  });

  test('a press on a child row is inside: the row runs and the whole chain closes', () => {
    const run = vi.fn();
    open({ run });
    row('Open panel').click();
    row('More').click();
    const deep = row('Deep');
    deep.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    expect(levels().length).toBe(3);
    deep.click();
    expect(run).toHaveBeenCalledTimes(1);
    expect(levels().length).toBe(0);
  });

  test('an outside press, a blur, a document scroll and closeContextMenu close every level', () => {
    /** @type {Array<() => void>} */
    const closers = [
      () => document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true })),
      () => window.dispatchEvent(new Event('blur')),
      () => document.dispatchEvent(new Event('scroll')),
      () => closeContextMenu(),
    ];
    for (const close of closers) {
      open();
      row('Open panel').click();
      row('More').click();
      expect(levels().length).toBe(3);
      close();
      expect(levels().length).toBe(0);
    }
  });

  test('a scroll inside a submenu does not close anything', () => {
    open();
    row('Open panel').click();
    levels()[1].dispatchEvent(new Event('scroll'));
    expect(levels().length).toBe(2);
  });

  test('opening another submenu replaces a sibling’s, and its children', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      items: [
        { label: 'A', submenu: [{ label: 'A1', submenu: [{ label: 'A2' }] }] },
        { label: 'B', submenu: [{ label: 'B1' }] },
      ],
    });
    row('A').click();
    row('A1').click();
    expect(levels().length).toBe(3);
    row('B').click();
    expect(levels().length).toBe(2);
    expect(row('A').getAttribute('aria-expanded')).toBe('false');
    expect(row('B1')).toBeTruthy();
    expect(row('A1')).toBeUndefined();
  });

  test('a caller prefix carries to every level', () => {
    openContextMenu({
      x: 0,
      y: 0,
      ariaLabel: 'a',
      classPrefix: 'rail-context',
      items: [{ label: 'Move to', submenu: [{ label: 'Left' }] }],
    });
    const parent = /** @type {HTMLElement} */ (document.querySelector('.rail-context-item'));
    expect(parent.querySelector('.rail-context-submenu')?.textContent).toBe('▸');
    parent.click();
    expect(document.querySelectorAll('.rail-context-menu').length).toBe(2);
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    expect(document.querySelectorAll('.rail-context-menu').length).toBe(0);
  });

  describe('placement', () => {
    /**
     * Rows report `rowRect`, menus a 200×150 box.
     * @param {Partial<DOMRect>} rowRect
     */
    function geometry(rowRect) {
      vi.spyOn(HTMLButtonElement.prototype, 'getBoundingClientRect').mockReturnValue(
        /** @type {DOMRect} */ ({ width: 180, height: 24, ...rowRect })
      );
      vi.spyOn(HTMLDivElement.prototype, 'getBoundingClientRect').mockReturnValue(
        /** @type {DOMRect} */ ({
          width: 200,
          height: 150,
          left: 0,
          top: 0,
          right: 200,
          bottom: 150,
        })
      );
      vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(1000);
      vi.spyOn(window, 'innerHeight', 'get').mockReturnValue(800);
    }

    const child = () => levels()[levels().length - 1];

    test('beside its row, on the right, level with it', () => {
      geometry({ left: 100, right: 280, top: 50, bottom: 74 });
      open();
      row('Open panel').click();
      expect(child().style.left).toBe('278px'); // 280 - 2 overlap
      expect(child().style.top).toBe('46px'); // 50 - 4
    });

    test('flips to the row’s left side at the right edge', () => {
      geometry({ left: 700, right: 880, top: 50, bottom: 74 });
      open();
      row('Open panel').click();
      expect(child().style.left).toBe('502px'); // 700 - 200 + 2
    });

    test('a cascade that went left keeps going left, and flips back right at the left edge', () => {
      geometry({ left: 700, right: 880, top: 50, bottom: 74 });
      open();
      row('Open panel').click();
      expect(child().style.left).toBe('502px');
      // The next row sits near the left edge: left has no room, so it flips right.
      vi.spyOn(HTMLButtonElement.prototype, 'getBoundingClientRect').mockReturnValue(
        /** @type {DOMRect} */ ({ left: 100, right: 280, top: 50, bottom: 74, width: 180 })
      );
      row('More').click();
      expect(child().style.left).toBe('278px');
    });

    test('a level that went left keeps going left while there is room', () => {
      geometry({ left: 700, right: 880, top: 50, bottom: 74 });
      open();
      row('Open panel').click();
      vi.spyOn(HTMLButtonElement.prototype, 'getBoundingClientRect').mockReturnValue(
        /** @type {DOMRect} */ ({ left: 400, right: 580, top: 50, bottom: 74, width: 180 })
      );
      row('More').click();
      expect(child().style.left).toBe('202px'); // 400 - 200 + 2, though right would fit too
    });

    test('lifts above the bottom edge, never off the top', () => {
      geometry({ left: 100, right: 280, top: 750, bottom: 774 });
      open();
      row('Open panel').click();
      expect(child().style.top).toBe('650px'); // 800 - 150
    });

    test('mounts in the root menu’s document', () => {
      geometry({ left: 100, right: 280, top: 50, bottom: 74 });
      open();
      row('Open panel').click();
      expect(child().parentElement).toBe(document.body);
      expect(child().ownerDocument).toBe(levels()[0].ownerDocument);
    });
  });
});

describe('Home, End and typeahead', () => {
  /** @returns {HTMLElement[]} every open level, root first */
  const levels = () =>
    /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.osprey-context-menu')]);

  /** @param {string} label */
  const row = (label) =>
    /** @type {HTMLElement} */ (
      rows().find((r) => r.querySelector('.osprey-context-label')?.textContent === label)
    );

  /**
   * @param {string} key
   * @param {KeyboardEventInit} [init]
   * @returns {KeyboardEvent} the dispatched event
   */
  const press = (key, init = {}) => {
    const e = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    (document.activeElement ?? document).dispatchEvent(e);
    return e;
  };

  const focused = () => document.activeElement?.querySelector('.osprey-context-label')?.textContent;

  function open() {
    openContextMenu({
      x: 10,
      y: 10,
      ariaLabel: 'Tile',
      items: [
        { heading: 'Actions' },
        { label: 'Apply', disabled: true },
        { label: 'Rename' },
        { label: 'Reload' },
        { divider: true },
        {
          label: 'Open panel',
          submenu: [
            { label: 'Alpha', disabled: true },
            { label: 'Logs' },
            { label: 'Lattice' },
            { label: 'Zeta', disabled: true },
          ],
        },
        { label: 'remove' },
        { label: 'Zoom', disabled: true },
      ],
    });
  }

  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  test('Home and End focus the first and last enabled rows of the root', () => {
    open();
    row('Reload').focus();
    press('End');
    expect(focused()).toBe('remove');
    press('Home');
    expect(focused()).toBe('Rename');
  });

  test('Home and End act on the focused submenu level only', () => {
    open();
    row('Open panel').focus();
    press('ArrowRight');
    expect(levels().length).toBe(2);
    press('End');
    expect(focused()).toBe('Lattice');
    press('Home');
    expect(focused()).toBe('Logs');
    expect(levels().length).toBe(2);
  });

  test('Home and End are consumed', () => {
    open();
    const seen = vi.fn();
    window.addEventListener('keydown', seen);
    const e = press('End');
    window.removeEventListener('keydown', seen);
    expect(e.defaultPrevented).toBe(true);
    expect(seen).not.toHaveBeenCalled();
  });

  test('a letter moves to the next enabled row starting with it, case-insensitive, wrapping', () => {
    open();
    row('Rename').focus();
    press('r');
    expect(focused()).toBe('Reload');
    vi.advanceTimersByTime(600);
    press('R');
    expect(focused()).toBe('remove');
    vi.advanceTimersByTime(600);
    press('r');
    expect(focused()).toBe('Rename'); // wrapped; disabled Apply never matched
  });

  test('keys typed in quick succession build a prefix', () => {
    open();
    row('Reload').focus();
    press('r');
    press('e');
    press('m');
    expect(focused()).toBe('remove');
    press('o');
    expect(focused()).toBe('remove');
  });

  test('the prefix resets after the pause', () => {
    open();
    row('Reload').focus();
    press('r');
    expect(focused()).toBe('remove');
    vi.advanceTimersByTime(600);
    press('o');
    expect(focused()).toBe('Open panel'); // a fresh "o", not "ro"
  });

  test('disabled rows, headings and dividers are never matched', () => {
    open();
    row('Rename').focus();
    press('a'); // "Apply" is disabled, "Actions" a heading
    expect(focused()).toBe('Rename');
    vi.advanceTimersByTime(600);
    press('z'); // "Zoom" is disabled
    expect(focused()).toBe('Rename');
  });

  test('typeahead works in a submenu and stays in that level', () => {
    open();
    row('Open panel').focus();
    press('ArrowRight');
    expect(focused()).toBe('Logs');
    press('l');
    expect(focused()).toBe('Lattice');
    vi.advanceTimersByTime(600);
    press('z'); // disabled Zeta; root rows are not this level's
    expect(focused()).toBe('Lattice');
    vi.advanceTimersByTime(600);
    press('r');
    expect(focused()).toBe('Lattice');
    expect(levels().length).toBe(2);
  });

  test('a typeahead key is consumed even without a match, so page shortcuts stay silent', () => {
    open();
    row('Rename').focus();
    const seen = vi.fn();
    window.addEventListener('keydown', seen);
    const hit = press('o');
    const miss = press('q');
    window.removeEventListener('keydown', seen);
    expect(hit.defaultPrevented).toBe(true);
    expect(miss.defaultPrevented).toBe(true);
    expect(seen).not.toHaveBeenCalled();
  });

  test('modified keys are not typeahead', () => {
    open();
    row('Rename').focus();
    const e = press('r', { metaKey: true });
    expect(e.defaultPrevented).toBe(false);
    expect(focused()).toBe('Rename');
    const c = press('r', { ctrlKey: true });
    expect(c.defaultPrevented).toBe(false);
    expect(focused()).toBe('Rename');
  });

  test('a reopened menu starts with an empty prefix', () => {
    open();
    row('Rename').focus();
    press('r');
    expect(focused()).toBe('Reload');
    closeContextMenu();
    open();
    row('Rename').focus();
    press('o');
    expect(focused()).toBe('Open panel'); // not "ro"
  });
});

describe('browser passthrough', () => {
  // Each test's fake clock starts a minute past the previous one's, so a
  // menu-key window an earlier test opened has always expired.
  let clock = Date.now();

  beforeEach(() => {
    clock += 60_000;
    vi.useFakeTimers({ now: clock });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  /**
   * Dispatch a right-click `contextmenu` on `target` and return the event.
   * @param {EventTarget | null} target
   * @param {MouseEventInit} [init]
   */
  function rightClick(target, init = {}) {
    if (!target) throw new Error('rightClick: no target element');
    const e = new MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      button: 2,
      ...init,
    });
    target.dispatchEvent(e);
    return e;
  }

  /** @param {KeyboardEventInit} init */
  function keydown(init) {
    document.dispatchEvent(
      new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init })
    );
  }

  test('a plain right-click on an ordinary element is not declined', () => {
    document.body.innerHTML = '<div id="d">x</div>';
    expect(declinesToBrowser(rightClick(document.getElementById('d')))).toBe(false);
  });

  test('Shift+right-click declines', () => {
    document.body.innerHTML = '<div id="d">x</div>';
    expect(declinesToBrowser(rightClick(document.getElementById('d'), { shiftKey: true }))).toBe(
      true
    );
  });

  test('a Shift contextmenu carrying a pointer type declines', () => {
    document.body.innerHTML = '<div id="d">x</div>';
    const e = new PointerEvent('contextmenu', {
      bubbles: true,
      shiftKey: true,
      button: 0,
      pointerType: 'mouse',
    });
    document.getElementById('d')?.dispatchEvent(e);
    expect(declinesToBrowser(e)).toBe(true);
  });

  test('Shift+F10 (contextmenu with Shift, right after the keydown) is not declined', () => {
    document.body.innerHTML = '<div id="d" tabindex="0">x</div>';
    keydown({ key: 'F10', shiftKey: true });
    // Windows/Linux: the keydown fires a contextmenu with Shift still held.
    // Even one that looks like a right-click counts as keyboard here.
    const e = rightClick(document.getElementById('d'), { shiftKey: true });
    expect(declinesToBrowser(e)).toBe(false);
  });

  test('the ContextMenu key with Shift held is not declined', () => {
    document.body.innerHTML = '<div id="d" tabindex="0">x</div>';
    keydown({ key: 'ContextMenu', shiftKey: true });
    const e = rightClick(document.getElementById('d'), { shiftKey: true, button: 0 });
    expect(declinesToBrowser(e)).toBe(false);
  });

  test('a menu key long ago does not shield a later Shift+right-click', () => {
    document.body.innerHTML = '<div id="d">x</div>';
    keydown({ key: 'ContextMenu' });
    vi.advanceTimersByTime(2000);
    expect(declinesToBrowser(rightClick(document.getElementById('d'), { shiftKey: true }))).toBe(
      true
    );
  });

  test('a keyboard-shaped contextmenu (button 0, no pointer type) never declines for Shift', () => {
    document.body.innerHTML = '<div id="d">x</div>';
    const e = rightClick(document.getElementById('d'), { shiftKey: true, button: 0 });
    expect(declinesToBrowser(e)).toBe(false);
  });

  test('a synthetic contextmenu built by a keyboard route is not declined', () => {
    document.body.innerHTML = '<div id="d">x</div>';
    const e = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    Object.defineProperty(e, 'target', { value: document.getElementById('d') });
    expect(declinesToBrowser(e)).toBe(false);
  });

  test('Shift on a non-contextmenu event does not decline', () => {
    document.body.innerHTML = '<div id="d">x</div>';
    const e = new KeyboardEvent('keydown', { key: 'F10', shiftKey: true, bubbles: true });
    document.getElementById('d')?.dispatchEvent(e);
    expect(declinesToBrowser(e)).toBe(false);
  });

  test.each([
    ['<input id="t">', 'untyped input'],
    ['<input id="t" type="text">', 'text input'],
    ['<input id="t" type="search">', 'search input'],
    ['<input id="t" type="email">', 'email input'],
    ['<input id="t" type="password">', 'password input'],
    ['<input id="t" type="number">', 'number input'],
    ['<textarea id="t"></textarea>', 'textarea'],
    ['<div id="t" contenteditable="true">x</div>', 'contenteditable element'],
    ['<div contenteditable="true"><span id="t">x</span></div>', 'contenteditable ancestor'],
  ])('right-click in a text field declines: %s (%s)', (html) => {
    document.body.innerHTML = html;
    expect(
      declinesToBrowser(rightClick(/** @type {Element} */ (document.getElementById('t'))))
    ).toBe(true);
  });

  test('a text field declines for keyboard-originated requests too', () => {
    document.body.innerHTML = '<input id="t" type="text">';
    keydown({ key: 'F10', shiftKey: true });
    const e = rightClick(/** @type {Element} */ (document.getElementById('t')), {
      shiftKey: true,
      button: 0,
    });
    expect(declinesToBrowser(e)).toBe(true);
  });

  test.each([
    ['<input id="t" type="checkbox">', 'checkbox'],
    ['<input id="t" type="button" value="b">', 'button input'],
    ['<input id="t" type="range">', 'range'],
    ['<select id="t"><option>a</option></select>', 'select'],
    ['<button id="t">b</button>', 'button'],
    [
      '<div contenteditable="true"><span contenteditable="false" id="t">x</span></div>',
      'non-editable island',
    ],
  ])('right-click on a non-text control is not declined: %s (%s)', (html) => {
    document.body.innerHTML = html;
    expect(
      declinesToBrowser(rightClick(/** @type {Element} */ (document.getElementById('t'))))
    ).toBe(false);
  });

  test('xterm’s helper textarea is not a text field', () => {
    document.body.innerHTML =
      '<div class="xterm"><textarea id="t" class="xterm-helper-textarea"></textarea></div>';
    expect(
      declinesToBrowser(rightClick(/** @type {Element} */ (document.getElementById('t'))))
    ).toBe(false);
  });

  test('Shift+right-click on xterm’s helper textarea still declines', () => {
    document.body.innerHTML = '<textarea id="t" class="xterm-helper-textarea"></textarea>';
    const e = rightClick(/** @type {Element} */ (document.getElementById('t')), { shiftKey: true });
    expect(declinesToBrowser(e)).toBe(true);
  });

  test('a text node target resolves to its element', () => {
    document.body.innerHTML = '<div id="d" contenteditable="true">hello</div>';
    const text = /** @type {Node} */ (document.getElementById('d')?.firstChild);
    const e = new MouseEvent('contextmenu', { button: 2 });
    Object.defineProperty(e, 'target', { value: text });
    expect(declinesToBrowser(e)).toBe(true);
  });

  test('a declined event is left for the browser: the predicate never cancels it', () => {
    document.body.innerHTML = '<textarea id="t"></textarea>';
    const e = rightClick(/** @type {Element} */ (document.getElementById('t')), { shiftKey: true });
    declinesToBrowser(e);
    expect(e.defaultPrevented).toBe(false);
  });
});
