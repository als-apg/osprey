/**
 * Unit tests for the shared context menu (design_system/static/js/context-menu.js).
 *
 *   npx vitest run tests/interfaces/design_system/context-menu.test.mjs
 *
 * The dismissal and focus contract is exercised through the web terminal's
 * wrapper in tests/interfaces/web_terminal/panel-context-menu.test.mjs, which
 * runs the same code under the hub's class names. This file covers what the
 * shared module adds: its own default class names, heading and toggle rows,
 * the monospace value segment, the keyboard opener and the key predicate.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  openContextMenu,
  openContextMenuAt,
  isContextMenuKey,
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
    openContextMenu({ x: 0, y: 0, ariaLabel: 'Row actions', items: [{ label: 'Copy', glyph: '⎘' }, { divider: true }] });
    const el = menu();
    expect(el?.getAttribute('role')).toBe('menu');
    expect(el?.getAttribute('aria-label')).toBe('Row actions');
    expect(rows()[0].querySelector('.osprey-context-label')?.textContent).toBe('Copy');
    expect(rows()[0].querySelector('.osprey-context-glyph')?.textContent).toBe('⎘');
    expect(el?.querySelector('.osprey-context-divider')?.getAttribute('role')).toBe('separator');
  });

  test('a caller-supplied prefix renames every part', () => {
    openContextMenu({ x: 0, y: 0, ariaLabel: 'a', classPrefix: 'rail-context', items: [{ label: 'Focus' }] });
    expect(menu()).toBeNull();
    expect(document.querySelector('.rail-context-menu .rail-context-item .rail-context-label')).not.toBeNull();
  });

  test('the next open without a prefix goes back to the default', () => {
    openContextMenu({ x: 0, y: 0, ariaLabel: 'a', classPrefix: 'rail-context', items: [{ label: 'A' }] });
    openContextMenu({ x: 0, y: 0, ariaLabel: 'b', items: [{ label: 'B' }] });
    expect(document.querySelector('.rail-context-menu')).toBeNull();
    expect(menu()).not.toBeNull();
  });
});

describe('row kinds', () => {
  test('a heading is an inert caption, not a focusable row', () => {
    openContextMenu({ x: 0, y: 0, ariaLabel: 'Columns', items: [{ heading: 'Columns' }, { label: 'Device', checked: true }] });
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
      items: [{ label: 'Device', checked: true }, { label: 'System', checked: false }, { label: 'Reset' }],
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
    openContextMenu({ x: 0, y: 0, ariaLabel: 'a', items: [{ label: 'Only section', code: '<b>SR01C</b>' }] });
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
      items: [{ label: 'Go', run: () => { openWhenRun = menu() !== null; } }],
    });
    rows()[0].click();
    expect(openWhenRun).toBe(false);
  });
});

describe('keyboard opener', () => {
  test('opens at the element’s lower-left corner and anchors to it', () => {
    document.body.innerHTML = '<div id="row" tabindex="0">row</div>';
    const row = /** @type {HTMLElement} */ (document.getElementById('row'));
    vi.spyOn(row, 'getBoundingClientRect').mockReturnValue(
      /** @type {DOMRect} */ ({ left: 100, top: 40, right: 300, bottom: 60, width: 200, height: 20, x: 100, y: 40 }),
    );
    openContextMenuAt(row, { ariaLabel: 'Row actions', items: [{ label: 'Copy' }] });
    expect(menu()?.style.left).toBe('108px');
    expect(menu()?.style.top).toBe('60px');
  });

  test('Shift+F10 and the ContextMenu key are requests; plain F10 is not', () => {
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'ContextMenu' }))).toBe(true);
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'F10', shiftKey: true }))).toBe(true);
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'F10' }))).toBe(false);
    expect(isContextMenuKey(new KeyboardEvent('keydown', { key: 'Enter' }))).toBe(false);
  });
});
