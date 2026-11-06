// @ts-check
/**
 * The Workspace divider (sidebar-divider.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-divider.test.mjs
 *
 * The divider caps the Agent group's automatic height, sets it by drag (with
 * a snap to the automatic height), steps it by key, resets it by double-click,
 * clamps it, stores it under `osprey-sidebar-workspace-height` as a fraction
 * of the rail's room (reading an earlier build's whole pixels once), and hides
 * when the group is folded or the sidebar is collapsed. Two cases drive the
 * divider together with sidebar-fold.js. Layout does not exist in the test
 * DOM, so the rail's and groups' heights are stubbed; the row floors are
 * pinned by test_sidebar_divider_browser.py.
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
const tick = () => qs(divider(), '.sidebar-divider-default');
const height = () => group('workspace').style.getPropertyValue('--sidebar-group-height');
const cap = () => group('workspace').style.getPropertyValue('--sidebar-group-cap');
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
 * `withFold` also wires the fold module the interaction cases need, and
 * `toolsHeight` is the Tools group's height.
 * @param {{ withFold?: boolean, toolsHeight?: number }} [opts]
 */
async function boot({ withFold = false, toolsHeight = 0 } = {}) {
  document.body.innerHTML = HTML;
  Object.defineProperty(rail(), 'clientHeight', { value: 600, configurable: true });
  Object.defineProperty(group('workspace'), 'offsetHeight', { value: 200, configurable: true });
  Object.defineProperty(group('tools'), 'offsetHeight', { value: toolsHeight, configurable: true });
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

  test('automatic sizing caps the group at what leaves the other groups whole', async () => {
    await boot({ toolsHeight: 272 });
    expect(group('workspace').hasAttribute('data-capped')).toBe(true);
    expect(group('workspace').hasAttribute('data-sized')).toBe(false);
    expect(cap()).toBe('328px');
    expect(divider().getAttribute('aria-valuetext')).toBe('Automatic');
  });

  test('the cap stays between 240 and 480 px', async () => {
    await boot({ toolsHeight: 500 });
    expect(cap()).toBe('240px');
    await boot();
    expect(cap()).toBe('480px');
  });

  test('the cap never passes the maximum', async () => {
    await boot();
    Object.defineProperty(rail(), 'clientHeight', { value: 300, configurable: true });
    window.dispatchEvent(new Event('resize'));
    expect(cap()).toBe('244px');
  });

  test('a drag sets the height and stores it as a fraction once it ends', async () => {
    await boot();
    ptr('pointerdown', 100);
    ptr('pointermove', 160);
    expect(height()).toBe('260px');
    expect(localStorage.getItem(HEIGHT)).toBeNull();
    ptr('pointerup', 160);
    expect(localStorage.getItem(HEIGHT)).toBe('0.4333');
    expect(group('workspace').hasAttribute('data-sized')).toBe(true);
    expect(divider().getAttribute('aria-valuenow')).toBe('260');
  });

  test('a stored fraction keeps its share of the rail when the window resizes', async () => {
    localStorage.setItem(HEIGHT, '0.5');
    await boot();
    expect(height()).toBe('300px');
    Object.defineProperty(rail(), 'clientHeight', { value: 800, configurable: true });
    window.dispatchEvent(new Event('resize'));
    expect(height()).toBe('400px');
    expect(localStorage.getItem(HEIGHT)).toBe('0.5');
  });

  test("an earlier build's whole pixels are read once and rewritten as a fraction", async () => {
    localStorage.setItem(HEIGHT, '300');
    await boot();
    expect(height()).toBe('300px');
    expect(localStorage.getItem(HEIGHT)).toBe('0.5000');
    await boot();
    expect(height()).toBe('300px');
    expect(localStorage.getItem(HEIGHT)).toBe('0.5000');
  });

  test('while dragging, a tick marks the automatic height and the line snaps to it', async () => {
    await boot();
    expect(tick().hidden).toBe(true);
    ptr('pointerdown', 100);
    ptr('pointermove', 150);
    expect(height()).toBe('250px');
    expect(tick().hidden).toBe(false);
    expect(tick().style.top).toBe('-50px');
    ptr('pointermove', 108);
    expect(height()).toBe('200px');
    expect(tick().style.top).toBe('0px');
    ptr('pointerup', 108);
    expect(tick().hidden).toBe(true);
  });

  test('a drag that ends on the tick goes back to automatic sizing', async () => {
    localStorage.setItem(HEIGHT, '0.5');
    await boot();
    expect(height()).toBe('300px');
    ptr('pointerdown', 300);
    ptr('pointermove', 206);
    ptr('pointerup', 206);
    expect(group('workspace').hasAttribute('data-sized')).toBe(false);
    expect(height()).toBe('');
    expect(localStorage.getItem(HEIGHT)).toBeNull();
  });

  test('a drag clamps to the minimum and to what leaves the next group a row', async () => {
    await boot();
    ptr('pointerdown', 100);
    ptr('pointermove', -500);
    expect(height()).toBe('120px');
    ptr('pointermove', 5000);
    ptr('pointerup', 5000);
    expect(height()).toBe('544px');
    expect(stored(HEIGHT)).toBe(0.9067);
  });

  test('the cap does not move when a group folds, only the rail content does', async () => {
    // 428 px of the 639 px the sidebar leaves the rail.
    localStorage.setItem(HEIGHT, '0.6698');
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
    expect(height()).toBe('216px');
    expect(stored(HEIGHT)).toBe(0.36);
    press(divider(), 'ArrowDown', true);
    expect(height()).toBe('280px');
    press(divider(), 'ArrowUp');
    expect(height()).toBe('264px');
    expect(stored(HEIGHT)).toBe(0.44);
  });

  test('a double-click returns to automatic sizing', async () => {
    localStorage.setItem(HEIGHT, '0.5');
    await boot();
    expect(height()).toBe('300px');
    dbl(divider());
    expect(group('workspace').hasAttribute('data-sized')).toBe(false);
    expect(group('workspace').style.getPropertyValue('--sidebar-group-height')).toBe('');
    expect(localStorage.getItem(HEIGHT)).toBeNull();
  });

  test('a stored value out of range is ignored', async () => {
    for (const raw of ['12', '1.5', '0', 'tall']) {
      localStorage.setItem(HEIGHT, raw);
      await boot();
      expect(group('workspace').hasAttribute('data-sized')).toBe(false);
    }
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
    expect(height()).toBe('216px');
  });
});
