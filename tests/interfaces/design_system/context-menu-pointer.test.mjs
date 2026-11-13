/**
 * Unit tests for the pointer half of the context menu
 * (design_system/static/js/context-menu-pointer.js).
 *
 *   npx vitest run tests/interfaces/design_system/context-menu-pointer.test.mjs
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  LEAVE_DELAY,
  followPointer,
} from '../../../src/osprey/interfaces/design_system/static/js/context-menu-pointer.js';

/** @param {string} type @param {string} [pointerType] */
function pointer(type, pointerType) {
  const event = new MouseEvent(type);
  if (pointerType) Object.defineProperty(event, 'pointerType', { value: pointerType });
  return event;
}

/**
 * A level of two enabled rows and one disabled row, following the pointer.
 * @returns {{ menu: HTMLElement, rows: HTMLElement[], close: import('vitest').Mock }}
 */
function level() {
  const menu = document.createElement('div');
  const rows = ['One', 'Two', 'Off'].map((label) => {
    const row = document.createElement('button');
    row.appendChild(document.createElement('span')).textContent = label;
    menu.appendChild(row);
    return row;
  });
  rows[2].setAttribute('aria-disabled', 'true');
  document.body.appendChild(menu);
  const close = vi.fn();
  followPointer(menu, close);
  return { menu, rows, close };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  document.body.replaceChildren();
});

describe('focus follows the pointer', () => {
  test('the row under the pointer takes focus from the row that had it', () => {
    const { rows } = level();
    rows[0].focus();
    rows[1].dispatchEvent(pointer('pointerenter'));
    expect(document.activeElement).toBe(rows[1]);
  });

  test('a disabled row does not take focus', () => {
    const { rows } = level();
    rows[0].focus();
    rows[2].dispatchEvent(pointer('pointerenter'));
    expect(document.activeElement).toBe(rows[0]);
  });

  test('a row the pointer leaves gives its focus to the level', () => {
    const { menu, rows } = level();
    rows[1].dispatchEvent(pointer('pointerenter'));
    rows[1].dispatchEvent(pointer('pointerleave'));
    expect(document.activeElement).toBe(menu);
  });

  test('the pointer moving between the parts of one row keeps the row focused', () => {
    const { rows } = level();
    rows[1].dispatchEvent(pointer('pointerenter'));
    /** @type {HTMLElement} */ (rows[1].firstElementChild).dispatchEvent(pointer('pointerleave'));
    expect(document.activeElement).toBe(rows[1]);
  });

  test('a row the pointer leaves does not take focus away from a different row', () => {
    const { rows } = level();
    rows[0].focus();
    rows[1].dispatchEvent(pointer('pointerleave'));
    expect(document.activeElement).toBe(rows[0]);
  });
});

describe('submenus close when the pointer leaves', () => {
  test('they close after the delay, not before', () => {
    const { menu, close } = level();
    menu.dispatchEvent(pointer('pointerleave', 'mouse'));
    vi.advanceTimersByTime(LEAVE_DELAY - 1);
    expect(close).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(close).toHaveBeenCalledTimes(1);
  });

  test('leaving a row inside the level closes nothing', () => {
    const { rows, close } = level();
    rows[0].dispatchEvent(pointer('pointerleave', 'mouse'));
    vi.advanceTimersByTime(LEAVE_DELAY * 3);
    expect(close).not.toHaveBeenCalled();
  });

  test('entering a second level in time keeps them open', () => {
    const parent = level();
    const child = level();
    parent.menu.dispatchEvent(pointer('pointerleave', 'mouse'));
    child.menu.dispatchEvent(pointer('pointerenter', 'mouse'));
    vi.advanceTimersByTime(LEAVE_DELAY * 3);
    expect(parent.close).not.toHaveBeenCalled();
  });

  test('a touch closes nothing', () => {
    const { menu, close } = level();
    menu.dispatchEvent(pointer('pointerleave', 'touch'));
    vi.advanceTimersByTime(LEAVE_DELAY * 3);
    expect(close).not.toHaveBeenCalled();
  });

  test('a level that was removed in the meantime closes nothing', () => {
    const { menu, close } = level();
    menu.dispatchEvent(pointer('pointerleave', 'mouse'));
    menu.remove();
    vi.advanceTimersByTime(LEAVE_DELAY * 3);
    expect(close).not.toHaveBeenCalled();
  });
});
