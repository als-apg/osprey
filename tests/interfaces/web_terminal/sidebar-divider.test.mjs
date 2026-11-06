// @ts-check
/**
 * The Workspace divider (sidebar-divider.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-divider.test.mjs
 *
 * The divider sets the Agent group's height by drag, steps it by key, resets
 * it by double-click, clamps it, stores it under
 * `osprey-sidebar-workspace-height`, and hides when the group is folded or the
 * sidebar is collapsed. Two cases drive the divider together with
 * sidebar-fold.js. Layout does not exist in the test DOM, so the rail's and
 * group's heights are stubbed.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const HEIGHT = 'osprey-sidebar-workspace-height';

const HTML = `
  <div id="panel-rail">
    <div class="sidebar-group" data-group="workspace">
      <div class="osprey-section-label sidebar-group-head">
        <span class="sidebar-group-name" id="n1">Workspace</span>
        <button type="button" data-group-menu>m</button>
        <input type="text" id="field">
      </div>
      <div class="sidebar-row"><button id="sidebar-agent-row">Agent</button></div>
    </div>
    <div class="sidebar-group" data-group="tools">
      <div class="osprey-section-label sidebar-group-head">
        <span class="sidebar-group-name" id="n2">Tools</span>
        <button type="button" data-group-menu>m</button>
      </div>
    </div>
  </div>`;

const rail = () => byId('panel-rail');
const group = (/** @type {string} */ id) => qs(rail(), `[data-group="${id}"]`);
const name = (/** @type {string} */ id) => qs(group(id), '.sidebar-group-name');
const divider = () => qs(rail(), '.sidebar-divider');
const stored = (/** @type {string} */ key) => JSON.parse(localStorage.getItem(key) ?? 'null');

/** @param {Element} el */
const dbl = (el) =>
  el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));

/** @param {Element} el @param {string} key @param {boolean} [shift] */
function press(el, key, shift = false) {
  el.dispatchEvent(
    new KeyboardEvent('keydown', { key, shiftKey: shift, bubbles: true, cancelable: true })
  );
}

/** @param {number} y @param {string} type */
const ptr = (type, y) =>
  divider().dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, pointerId: 1, clientY: y })
  );

/**
 * Draw the rail, with a 600 px rail and a 200 px group, and wire the divider;
 * `withFold` also wires the fold module the interaction cases need.
 * @param {{ withFold?: boolean }} [opts]
 */
async function boot({ withFold = false } = {}) {
  document.body.innerHTML = HTML;
  Object.defineProperty(rail(), 'clientHeight', { value: 600, configurable: true });
  Object.defineProperty(group('workspace'), 'offsetHeight', { value: 200, configurable: true });
  vi.resetModules();
  if (withFold) (await import(`${JS}/sidebar-fold.js`)).initSidebarFold(rail());
  (await import(`${JS}/sidebar-divider.js`)).initSidebarDivider(rail());
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-sidebar');
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('the Workspace divider', () => {
  test('is a horizontal separator after the group that holds the Agent row', async () => {
    await boot();
    expect(divider().getAttribute('role')).toBe('separator');
    expect(divider().getAttribute('aria-orientation')).toBe('horizontal');
    expect(divider().previousElementSibling).toBe(group('workspace'));
    expect(group('workspace').hasAttribute('data-sized')).toBe(false);
  });

  test('follows the Agent row when the groups are moved', async () => {
    await boot();
    rail().append(group('workspace'));
    document.dispatchEvent(new CustomEvent('osprey-sidebar-rendered'));
    expect(divider().previousElementSibling).toBe(group('workspace'));
    expect(rail().lastElementChild).toBe(divider());
  });

  test('a drag sets the height and stores it once it ends', async () => {
    await boot();
    ptr('pointerdown', 100);
    ptr('pointermove', 160);
    expect(group('workspace').style.getPropertyValue('--sidebar-group-height')).toBe('260px');
    expect(localStorage.getItem(HEIGHT)).toBeNull();
    ptr('pointerup', 160);
    expect(stored(HEIGHT)).toBe(260);
    expect(group('workspace').hasAttribute('data-sized')).toBe(true);
  });

  test('a drag clamps to the minimum and to what leaves the next group a row', async () => {
    await boot();
    ptr('pointerdown', 100);
    ptr('pointermove', -500);
    expect(group('workspace').style.getPropertyValue('--sidebar-group-height')).toBe('120px');
    ptr('pointermove', 5000);
    ptr('pointerup', 5000);
    expect(stored(HEIGHT)).toBe(544);
  });

  test('the cap does not move when a group folds, only the rail content does', async () => {
    localStorage.setItem(HEIGHT, '428');
    await boot({ withFold: true });
    const side = document.createElement('div');
    side.className = 'sidebar';
    const headEl = document.createElement('div');
    const foot = document.createElement('div');
    rail().before(side);
    side.append(headEl, rail(), foot);
    Object.defineProperty(side, 'clientHeight', { value: 800, configurable: true });
    Object.defineProperty(headEl, 'offsetHeight', { value: 36, configurable: true });
    Object.defineProperty(foot, 'offsetHeight', { value: 125, configurable: true });
    // The rail is content-sized: shorter while Tools is folded.
    Object.defineProperty(rail(), 'clientHeight', {
      configurable: true,
      get: () => (group('tools').hasAttribute('data-folded') ? 444 : 531),
    });
    const h = () => group('workspace').style.getPropertyValue('--sidebar-group-height');
    document.dispatchEvent(new CustomEvent('osprey-collapsible-change'));
    expect(h()).toBe('428px');
    dbl(name('tools'));
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    expect(h()).toBe('428px');
    dbl(name('tools'));
    expect(h()).toBe('428px');
  });

  test('a press without a move stores nothing', async () => {
    await boot();
    ptr('pointerdown', 100);
    ptr('pointerup', 100);
    expect(localStorage.getItem(HEIGHT)).toBeNull();
  });

  test('ArrowUp and ArrowDown step, Shift steps further', async () => {
    await boot();
    press(divider(), 'ArrowDown');
    expect(stored(HEIGHT)).toBe(216);
    press(divider(), 'ArrowDown', true);
    expect(stored(HEIGHT)).toBe(280);
    press(divider(), 'ArrowUp');
    expect(stored(HEIGHT)).toBe(264);
  });

  test('a double-click returns to automatic sizing', async () => {
    localStorage.setItem(HEIGHT, '300');
    await boot();
    expect(group('workspace').style.getPropertyValue('--sidebar-group-height')).toBe('300px');
    dbl(divider());
    expect(group('workspace').hasAttribute('data-sized')).toBe(false);
    expect(group('workspace').style.getPropertyValue('--sidebar-group-height')).toBe('');
    expect(localStorage.getItem(HEIGHT)).toBeNull();
  });

  test('a stored value out of range is ignored', async () => {
    localStorage.setItem(HEIGHT, '12');
    await boot();
    expect(group('workspace').hasAttribute('data-sized')).toBe(false);
  });

  test('is hidden while the group is folded and while the sidebar is collapsed', async () => {
    await boot({ withFold: true });
    expect(divider().hidden).toBe(false);
    dbl(name('workspace'));
    expect(divider().hidden).toBe(true);
    dbl(name('workspace'));
    expect(divider().hidden).toBe(false);
    document.documentElement.setAttribute('data-sidebar', 'collapsed');
    document.dispatchEvent(new CustomEvent('osprey-collapsible-change'));
    expect(divider().hidden).toBe(true);
  });

  test('unavailable storage still sets the height', async () => {
    await boot();
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    press(divider(), 'ArrowDown');
    expect(group('workspace').style.getPropertyValue('--sidebar-group-height')).toBe('216px');
  });
});
