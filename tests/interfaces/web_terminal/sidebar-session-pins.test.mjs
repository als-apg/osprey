// @ts-check
/**
 * The sidebar's pinned-session group (sidebar-session-pins.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-session-pins.test.mjs
 *
 * Pins the group's place (first in the rail, hidden with nothing pinned), its
 * heading (click, Enter and Space fold it; right-click and Shift+F10 open
 * Expand/Collapse · Rename · Hide pinned sessions), the inline rename, the
 * view state in per-user browser storage, the first-pin rule, a hidden group
 * staying hidden through later pins and reloads, and Show pinned sessions on a
 * right-click of empty sidebar space.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

const model = vi.hoisted(() => ({ setPinned: vi.fn(async () => true) }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', () => model);

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-session-pins.js')} */
let pins;

beforeEach(async () => {
  vi.resetModules();
  model.setPinned.mockClear();
  localStorage.clear();
  document.body.innerHTML = `
    <nav id="sidebar">
      <div id="panel-rail">
        <div class="sidebar-group" data-group="workspace">
          <button class="osprey-nav-row" id="sidebar-agent-row">Sessions</button>
        </div>
      </div>
    </nav>`;
  pins = await import(`${JS}/sidebar-session-pins.js`);
  pins.initPinnedMenus();
});

afterEach(() => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  pins.resetSessionPins();
  document.body.replaceChildren();
});

const group = () => /** @type {HTMLElement} */ (document.querySelector('.sidebar-pinned'));
const head = () => /** @type {HTMLElement} */ (group().querySelector('.sidebar-pinned-head'));
const list = () => /** @type {HTMLElement} */ (group().querySelector('.sidebar-pinned-list'));
const stored = () => JSON.parse(localStorage.getItem(pins.PINS_STORAGE_KEY) ?? 'null');
const menuLabels = () =>
  [...document.querySelectorAll('.rail-context-item .rail-context-label')].map(
    (n) => n.textContent
  );
/** @param {string} label */
const menuRow = (label) =>
  /** @type {HTMLElement} */ (
    [...document.querySelectorAll('.rail-context-item')].find(
      (r) => r.querySelector('.rail-context-label')?.textContent === label
    )
  );
/** @param {Element} el */
const rightClick = (el) => {
  const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
  el.dispatchEvent(ev);
  return ev;
};

test('the group leads the rail, hidden while nothing is pinned', () => {
  pins.paintPinnedGroup(0);
  expect(document.getElementById('panel-rail')?.firstElementChild).toBe(group());
  expect(group().hidden).toBe(true);
  pins.paintPinnedGroup(2);
  expect(group().hidden).toBe(false);
  expect(head().textContent).toContain('Pinned');
  expect(group().querySelector('.sidebar-pinned-count')?.textContent).toBe('2');
  expect(list().matches('[data-session-tree][data-tree-where="pinned"]')).toBe(true);
});

test('a click, Enter or Space folds and unfolds it; the fold is stored', () => {
  pins.paintPinnedGroup(1);
  const redraw = vi.fn(() => pins.paintPinnedGroup(1));
  pins.onPinsChange(redraw);
  expect(head().getAttribute('aria-expanded')).toBe('false');
  head().click();
  expect(head().getAttribute('aria-expanded')).toBe('true');
  expect(list().hidden).toBe(false);
  expect(stored()).toMatchObject({ folded: false });
  head().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  expect(head().getAttribute('aria-expanded')).toBe('false');
  head().dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
  expect(head().getAttribute('aria-expanded')).toBe('true');
  expect(redraw).toHaveBeenCalledTimes(3);
});

test('the heading menu: Expand / Collapse, Rename, Hide pinned sessions; Shift+F10 opens it', () => {
  pins.paintPinnedGroup(1);
  pins.onPinsChange(() => pins.paintPinnedGroup(1));
  expect(rightClick(head()).defaultPrevented).toBe(true);
  expect(menuLabels()).toEqual(['Expand', 'Rename', 'Hide pinned sessions']);
  menuRow('Hide pinned sessions').click();
  expect(group().hidden).toBe(true);
  expect(stored()).toMatchObject({ hidden: true });
  pins.setPinnedHidden(false);
  const ev = new KeyboardEvent('keydown', {
    key: 'F10',
    shiftKey: true,
    bubbles: true,
    cancelable: true,
  });
  head().dispatchEvent(ev);
  expect(ev.defaultPrevented).toBe(true);
  expect(menuLabels()[0]).toBe('Expand');
});

test('Rename edits in place: Enter keeps it (stored), Escape and an empty name keep the old', () => {
  pins.paintPinnedGroup(1);
  pins.onPinsChange(() => pins.paintPinnedGroup(1));
  const name = () => head().querySelector('.sidebar-group-name')?.textContent;
  /** @param {string} value @param {string} key */
  const rename = (value, key) => {
    rightClick(head());
    menuRow('Rename').click();
    const field = /** @type {HTMLInputElement} */ (head().querySelector('.sidebar-group-rename'));
    expect(document.activeElement).toBe(field);
    field.value = value;
    field.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
    expect(head().querySelector('.sidebar-group-rename')).toBeNull();
  };
  rename('Shift notes', 'Enter');
  expect(name()).toBe('Shift notes');
  expect(stored()).toMatchObject({ name: 'Shift notes' });
  rename('Other', 'Escape');
  expect(name()).toBe('Shift notes');
  rename('   ', 'Enter');
  expect(name()).toBe('Shift notes');
  // A rename click on the heading does not fold it.
  expect(head().getAttribute('aria-expanded')).toBe('false');
});

test('the Sessions divider re-measures when the group shows, hides, folds or unfolds', async () => {
  const { FOLD_EVENT } = await import(`${JS}/sidebar-fold.js`);
  const heard = vi.fn();
  document.addEventListener(FOLD_EVENT, heard);
  try {
    pins.paintPinnedGroup(0);
    expect(heard).not.toHaveBeenCalled();
    pins.paintPinnedGroup(1);
    expect(heard).toHaveBeenCalledTimes(1);
    pins.paintPinnedGroup(2);
    expect(heard).toHaveBeenCalledTimes(1);
    pins.setPinnedFolded(false);
    pins.setPinnedHidden(true);
    expect(heard).toHaveBeenCalledTimes(3);
  } finally {
    document.removeEventListener(FOLD_EVENT, heard);
  }
});

test('the view state is read back from per-user storage', async () => {
  localStorage.setItem(
    pins.PINS_STORAGE_KEY,
    JSON.stringify({ hidden: false, folded: false, name: 'Beam work' })
  );
  pins.resetSessionPins();
  pins.paintPinnedGroup(1);
  expect(head().querySelector('.sidebar-group-name')?.textContent).toBe('Beam work');
  expect(head().getAttribute('aria-expanded')).toBe('true');
});

test('the first pin folds the group and shows it; the model sets the pin', async () => {
  pins.paintPinnedGroup(0);
  localStorage.clear();
  pins.setPinnedFolded(false);
  await pins.setSessionPinned('t1', true);
  expect(model.setPinned).toHaveBeenCalledWith('t1', true);
  expect(stored()).toMatchObject({ folded: true, hidden: false });
  pins.paintPinnedGroup(1);
  expect(group().hidden).toBe(false);

  await pins.setSessionPinned('t1', false);
  expect(model.setPinned).toHaveBeenLastCalledWith('t1', false);
});

test('a pin while hidden keeps the group hidden, and keeps its fold', async () => {
  pins.paintPinnedGroup(1);
  pins.onPinsChange(() => pins.paintPinnedGroup(1));
  pins.setPinnedFolded(false);
  rightClick(head());
  menuRow('Hide pinned sessions').click();
  expect(group().hidden).toBe(true);
  await pins.setSessionPinned('t2', true);
  pins.paintPinnedGroup(2);
  expect(group().hidden).toBe(true);
  expect(stored()).toMatchObject({ hidden: true, folded: false });
});

test('a group hidden with nothing pinned stays hidden on the next pin', async () => {
  pins.paintPinnedGroup(1);
  pins.setPinnedHidden(true);
  pins.paintPinnedGroup(0);
  await pins.setSessionPinned('t3', true);
  pins.paintPinnedGroup(1);
  expect(group().hidden).toBe(true);
  expect(stored()).toMatchObject({ hidden: true });
});

test('the hidden group stays hidden across a reload, through further pins', async () => {
  pins.paintPinnedGroup(1);
  pins.setPinnedHidden(true);
  pins.resetSessionPins();
  pins.paintPinnedGroup(1);
  expect(group().hidden).toBe(true);
  await pins.setSessionPinned('t2', true);
  pins.paintPinnedGroup(2);
  expect(group().hidden).toBe(true);
});

test('Show pinned sessions brings the hidden group back with every pin', async () => {
  pins.paintPinnedGroup(1);
  pins.setPinnedHidden(true);
  await pins.setSessionPinned('t2', true);
  pins.paintPinnedGroup(2);
  pins.onPinsChange(() => pins.paintPinnedGroup(2));
  rightClick(/** @type {Element} */ (document.getElementById('panel-rail')));
  expect(menuRow('Show pinned sessions').getAttribute('aria-checked')).toBe('false');
  menuRow('Show pinned sessions').click();
  expect(group().hidden).toBe(false);
  expect(group().querySelector('.sidebar-pinned-count')?.textContent).toBe('2');
  expect(stored()).toMatchObject({ hidden: false });
});

test('a right-click on empty sidebar space offers Show pinned sessions; rows keep theirs', () => {
  pins.paintPinnedGroup(0);
  const rail = /** @type {HTMLElement} */ (document.getElementById('panel-rail'));
  expect(rightClick(rail).defaultPrevented).toBe(true);
  expect(menuLabels()).toEqual(['Show pinned sessions']);
  expect(menuRow('Show pinned sessions').getAttribute('aria-disabled')).toBe('true');
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

  pins.paintPinnedGroup(1);
  pins.onPinsChange(() => pins.paintPinnedGroup(1));
  rightClick(/** @type {Element} */ (document.getElementById('sidebar')));
  expect(menuRow('Show pinned sessions').getAttribute('aria-checked')).toBe('true');
  menuRow('Show pinned sessions').click();
  expect(group().hidden).toBe(true);

  const row = /** @type {Element} */ (document.getElementById('sidebar-agent-row'));
  expect(rightClick(row).defaultPrevented).toBe(false);
});

test('the collapsed sidebar opens no empty-space menu', () => {
  document.documentElement.setAttribute('data-sidebar', 'collapsed');
  try {
    expect(
      rightClick(/** @type {Element} */ (document.getElementById('panel-rail'))).defaultPrevented
    ).toBe(false);
  } finally {
    document.documentElement.removeAttribute('data-sidebar');
  }
});
