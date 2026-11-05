/**
 * Unit tests for the Lattice Dashboard toolbar-action controller (header.js):
 * the panel toolbar's Refresh / Baseline / Verify, the two-step Baseline
 * arming, and the tile-bar contribution it publishes when embedded (the
 * model name and the layout menu).
 *
 * happy-dom environment (configured globally). header-contrib.js is mocked
 * via vi.mock — in this environment window.parent === window, so the real
 * contributeHeader() is a no-op by construction and the contribution's shape
 * could not be asserted:
 *   npx vitest run tests/interfaces/lattice_dashboard/header.test.mjs
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';

vi.mock('/design-system/js/header-contrib.js', () => ({
  contributeHeader: vi.fn(),
  onHeaderAction: vi.fn(),
}));

import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { readTitle } from '/design-system/js/tooltip.js';
import {
  createHeader,
  LAYOUT_MENU_ID,
} from '../../../src/osprey/interfaces/lattice_dashboard/static/js/header.js';

/** Minimal DOM fixture matching the panel toolbar in lattice_dashboard's index.html. */
function mountFixture() {
  document.body.innerHTML = `
    <button id="btn-refresh" class="osprey-icon-btn" disabled></button>
    <button id="btn-baseline" class="osprey-btn baseline-btn" disabled>
      <span class="baseline-label">
        <span class="baseline-label-rest">Baseline</span>
        <span class="baseline-label-armed">Confirm?</span>
      </span>
    </button>
    <button id="btn-verify" class="osprey-btn primary" disabled>Verify</button>
    <button id="btn-panel-menu" class="osprey-icon-btn" aria-haspopup="menu" hidden></button>
  `;
}

/** @returns {{onRefresh: any, onVerify: any, onBaseline: any, onLayout: any}} */
function makeCallbacks() {
  return { onRefresh: vi.fn(), onVerify: vi.fn(), onBaseline: vi.fn(), onLayout: vi.fn() };
}

/** The items of the most recent contribution. @returns {any[]} */
function lastContribution() {
  const calls = vi.mocked(contributeHeader).mock.calls;
  return calls[calls.length - 1][0];
}

/** @param {string} id @returns {any} */
function contributedItem(id) {
  return lastContribution().find((/** @type {any} */ item) => item.id === id);
}

/** Fire the hub's action round-trip. @param {string} id @param {string} [value] */
function headerAction(id, value) {
  vi.mocked(onHeaderAction).mock.calls[0][0](id, value);
}

/** @param {string} id @returns {HTMLButtonElement} */
function btn(id) {
  return /** @type {HTMLButtonElement} */ (byId(id));
}

const isArmed = () => btn('btn-baseline').hasAttribute('data-armed');

/** A /api/state body whose periodic SR selection is ready. */
const READY = { selection: { model: 'SR', status: 'ready', capabilities: { verify: true } } };

/** A /api/state body whose single-pass selection is ready. */
const SINGLE_PASS = {
  selection: { model: 'LINE', status: 'ready', capabilities: { verify: false } },
};

beforeEach(() => {
  mountFixture();
  vi.mocked(contributeHeader).mockClear();
  vi.mocked(onHeaderAction).mockClear();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('toolbar actions', () => {
  test('Refresh is an icon button and reaches its handler', () => {
    const cb = makeCallbacks();
    const header = createHeader(cb);
    header.init();
    header.syncState(READY);

    expect(btn('btn-refresh').querySelector('svg')).not.toBeNull();
    btn('btn-refresh').click();
    btn('btn-verify').click();
    expect(cb.onRefresh).toHaveBeenCalledTimes(1);
    expect(cb.onVerify).toHaveBeenCalledTimes(1);
  });

  test('every action is disabled until a lattice loads, and Verify says why', () => {
    createHeader(makeCallbacks()).init();

    expect(btn('btn-refresh').disabled).toBe(true);
    expect(btn('btn-baseline').disabled).toBe(true);
    expect(btn('btn-verify').disabled).toBe(true);
    expect(readTitle(btn('btn-verify'))).toBe('Verify needs a lattice');
  });

  test('a ready periodic model enables all three', () => {
    const header = createHeader(makeCallbacks());
    header.init();
    header.syncState(READY);

    expect(btn('btn-refresh').disabled).toBe(false);
    expect(btn('btn-baseline').disabled).toBe(false);
    expect(btn('btn-verify').disabled).toBe(false);
    expect(readTitle(btn('btn-verify'))).toBe('Run DA + LMA verification');
  });

  test('Verify stays disabled for a single-pass model, with the reason as its tooltip', () => {
    const header = createHeader(makeCallbacks());
    header.init();
    header.syncState(SINGLE_PASS);

    expect(btn('btn-verify').disabled).toBe(true);
    expect(readTitle(btn('btn-verify'))).toMatch(/single-pass model/);
    expect(btn('btn-refresh').disabled).toBe(false);
  });
});

describe('baseline arming', () => {
  test('the first click arms instead of overwriting the baseline', () => {
    const cb = makeCallbacks();
    const header = createHeader(cb);
    header.init();
    header.syncState(READY);

    btn('btn-baseline').click();

    expect(cb.onBaseline).not.toHaveBeenCalled();
    expect(isArmed()).toBe(true);
    expect(readTitle(btn('btn-baseline'))).toBe('Click again to overwrite the baseline');
  });

  test('a second click inside the window runs it and disarms', () => {
    const cb = makeCallbacks();
    const header = createHeader(cb);
    header.init();
    header.syncState(READY);

    btn('btn-baseline').click();
    btn('btn-baseline').click();

    expect(cb.onBaseline).toHaveBeenCalledTimes(1);
    expect(isArmed()).toBe(false);
  });

  test('arming lapses on its own, so the next click re-arms', () => {
    vi.useFakeTimers();
    const cb = makeCallbacks();
    const header = createHeader(cb);
    header.init();
    header.syncState(READY);

    btn('btn-baseline').click();
    vi.advanceTimersByTime(4000);
    expect(isArmed()).toBe(false);

    btn('btn-baseline').click();
    expect(cb.onBaseline).not.toHaveBeenCalled();
    expect(isArmed()).toBe(true);
  });

  test('both labels stay in the button, so arming never changes its width', () => {
    // Verify and the theme switcher sit right of Baseline; a label that grew
    // on arming would shove them on the very click that arms. The two labels
    // share one grid cell (dashboard.css) and arming only flips which shows.
    const header = createHeader(makeCallbacks());
    header.init();
    header.syncState(READY);
    const labels = () =>
      Array.from(btn('btn-baseline').querySelectorAll('.baseline-label > span')).map(
        (span) => span.textContent
      );

    expect(labels()).toEqual(['Baseline', 'Confirm?']);
    btn('btn-baseline').click();
    expect(isArmed()).toBe(true);
    expect(labels()).toEqual(['Baseline', 'Confirm?']);
    expect(qs(btn('btn-baseline'), '.baseline-label').childElementCount).toBe(2);
  });

  test('unloading the lattice cancels a pending confirm', () => {
    const cb = makeCallbacks();
    const header = createHeader(cb);
    header.init();
    header.syncState(READY);

    btn('btn-baseline').click();
    header.syncState({ selection: { model: null, status: 'none', capabilities: {} } });

    expect(isArmed()).toBe(false);
    header.syncState(READY);
    // The next click arms afresh rather than firing the stale confirm
    btn('btn-baseline').click();
    expect(cb.onBaseline).not.toHaveBeenCalled();
  });
});

describe('the standalone ⋯', () => {
  afterEach(() => {
    window.history.replaceState(null, '', '/');
    document.querySelector('.osprey-context-menu')?.remove();
  });

  test('standalone shows it, and an entry runs the same handler as the hub ⋯', () => {
    window.history.replaceState(null, '', '/');
    const cb = makeCallbacks();
    const header = createHeader(cb);
    header.init();
    header.syncLayout('stacked');

    const menuBtn = btn('btn-panel-menu');
    expect(menuBtn.hidden).toBe(false);
    menuBtn.click();

    const rows = Array.from(document.querySelectorAll('.osprey-context-menu [role^="menuitem"]'));
    expect(rows.map((row) => row.textContent?.replace('✓', '').trim())).toEqual([
      'Layout: Stacked',
      'Layout: Grid',
    ]);
    expect(rows.map((row) => row.getAttribute('aria-checked'))).toEqual(['true', 'false']);
    /** @type {HTMLElement} */ (rows[1]).click();
    expect(cb.onLayout).toHaveBeenCalledWith('grid');
  });

  test('embedded it does not exist: the hub ⋯ is the one per tile', () => {
    window.history.replaceState(null, '', '/?embedded=true');
    createHeader(makeCallbacks()).init();

    expect(document.getElementById('btn-panel-menu')).toBeNull();
  });
});

describe('contribution', () => {
  test('the tile bar gets the model name and the layout menu, nothing else', () => {
    const header = createHeader(makeCallbacks());
    header.init();
    header.syncState(READY);

    expect(lastContribution().map((/** @type {any} */ i) => [i.kind, i.id])).toEqual([
      ['text', 'lattice-name'],
      ['menu', LAYOUT_MENU_ID],
    ]);
    expect(contributedItem('lattice-name').text).toBe('SR');
  });

  test('with no model there is no subtitle, only the menu', () => {
    createHeader(makeCallbacks()).init();

    expect(lastContribution().map((/** @type {any} */ i) => i.kind)).toEqual(['menu']);
  });

  test('the layout menu checks the layout shown', () => {
    const header = createHeader(makeCallbacks());
    header.init();

    header.syncLayout('stacked');
    expect(contributedItem(LAYOUT_MENU_ID).items).toEqual([
      { id: 'stacked', label: 'Layout: Stacked', checked: true },
      { id: 'grid', label: 'Layout: Grid', checked: false },
    ]);

    header.syncLayout('grid');
    expect(contributedItem(LAYOUT_MENU_ID).items.map((/** @type {any} */ e) => e.checked)).toEqual([
      false,
      true,
    ]);
  });

  test('a layout pick in ⋯ reaches the layout handler with the entry id', () => {
    const cb = makeCallbacks();
    createHeader(cb).init();

    headerAction(LAYOUT_MENU_ID, 'grid');
    expect(cb.onLayout).toHaveBeenCalledWith('grid');
  });

  test('an unknown action or entry is ignored', () => {
    const cb = makeCallbacks();
    createHeader(cb).init();

    headerAction('refresh');
    headerAction('baseline');
    headerAction(LAYOUT_MENU_ID, 'bogus');

    expect(cb.onRefresh).not.toHaveBeenCalled();
    expect(cb.onBaseline).not.toHaveBeenCalled();
    expect(cb.onLayout).not.toHaveBeenCalled();
  });
});
