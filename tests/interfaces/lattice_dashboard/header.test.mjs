/**
 * Unit tests for the Lattice Dashboard header-action controller (header.js):
 * the two-step Baseline arming state machine and the tile-bar contribution
 * it publishes when embedded.
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

import {
  contributeHeader,
  onHeaderAction,
} from '/design-system/js/header-contrib.js';
import { createHeader } from '../../../src/osprey/interfaces/lattice_dashboard/static/js/header.js';

/** Minimal DOM fixture matching the top bar in lattice_dashboard's index.html. */
function mountFixture() {
  document.body.innerHTML = `
    <button id="btn-refresh"></button>
    <button id="btn-verify"></button>
    <button id="btn-baseline" class="action-btn action-btn--baseline">
      <span class="baseline-label">Baseline</span>
    </button>
  `;
}

/** @returns {{onRefresh: any, onVerify: any, onBaseline: any}} */
function makeCallbacks() {
  return { onRefresh: vi.fn(), onVerify: vi.fn(), onBaseline: vi.fn() };
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

/** Fire the hub's action round-trip for a contributed item. @param {string} id */
function headerAction(id) {
  vi.mocked(onHeaderAction).mock.calls[0][0](id);
}

beforeEach(() => {
  mountFixture();
  vi.mocked(contributeHeader).mockClear();
  vi.mocked(onHeaderAction).mockClear();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('plain actions', () => {
  test('the standalone buttons and the tile bar reach the same handlers', () => {
    const cb = makeCallbacks();
    createHeader(cb).init();

    byId('btn-refresh').click();
    byId('btn-verify').click();
    expect(cb.onRefresh).toHaveBeenCalledTimes(1);
    expect(cb.onVerify).toHaveBeenCalledTimes(1);

    headerAction('refresh');
    headerAction('verify');
    expect(cb.onRefresh).toHaveBeenCalledTimes(2);
    expect(cb.onVerify).toHaveBeenCalledTimes(2);
  });

  test('an unknown action id is ignored', () => {
    const cb = makeCallbacks();
    createHeader(cb).init();

    headerAction('bogus');

    expect(cb.onRefresh).not.toHaveBeenCalled();
    expect(cb.onBaseline).not.toHaveBeenCalled();
  });
});

describe('baseline arming', () => {
  test('the first click arms instead of overwriting the baseline', () => {
    const cb = makeCallbacks();
    createHeader(cb).init();

    byId('btn-baseline').click();

    expect(cb.onBaseline).not.toHaveBeenCalled();
    expect(qs(byId('btn-baseline'), '.baseline-label').textContent).toBe('Confirm?');
    expect(byId('btn-baseline').classList.contains('action-btn--armed')).toBe(true);
  });

  test('a second click inside the window runs it and disarms', () => {
    const cb = makeCallbacks();
    createHeader(cb).init();

    byId('btn-baseline').click();
    byId('btn-baseline').click();

    expect(cb.onBaseline).toHaveBeenCalledTimes(1);
    expect(qs(byId('btn-baseline'), '.baseline-label').textContent).toBe('Baseline');
    expect(byId('btn-baseline').classList.contains('action-btn--armed')).toBe(false);
  });

  test('arming lapses on its own, so the next click re-arms', () => {
    vi.useFakeTimers();
    const cb = makeCallbacks();
    createHeader(cb).init();

    byId('btn-baseline').click();
    vi.advanceTimersByTime(4000);
    expect(qs(byId('btn-baseline'), '.baseline-label').textContent).toBe('Baseline');

    byId('btn-baseline').click();
    expect(cb.onBaseline).not.toHaveBeenCalled();
    expect(qs(byId('btn-baseline'), '.baseline-label').textContent).toBe('Confirm?');
  });

  test('one state machine drives both renderings', () => {
    const cb = makeCallbacks();
    createHeader(cb).init();

    // Armed from the tile bar, confirmed from the standalone button
    headerAction('baseline');
    expect(contributedItem('baseline').label).toBe('Confirm?');
    expect(contributedItem('baseline').tone).toBe('accent');
    expect(qs(byId('btn-baseline'), '.baseline-label').textContent).toBe('Confirm?');

    byId('btn-baseline').click();
    expect(cb.onBaseline).toHaveBeenCalledTimes(1);
    expect(contributedItem('baseline').label).toBe('Baseline');
    expect(contributedItem('baseline').tone).toBe('default');
  });
});

describe('contribution', () => {
  test('publishes the name and the three actions, disabled until a lattice loads', () => {
    createHeader(makeCallbacks()).init();

    expect(lastContribution().map((/** @type {any} */ i) => i.id)).toEqual([
      'lattice-name',
      'refresh',
      'verify',
      'baseline',
    ]);
    expect(contributedItem('lattice-name').text).toBe('No lattice loaded');
    expect(contributedItem('refresh').disabled).toBe(true);
    expect(contributedItem('verify').disabled).toBe(true);
    expect(contributedItem('baseline').disabled).toBe(true);
  });

  test('a loaded lattice enables the actions and names the tile bar', () => {
    const header = createHeader(makeCallbacks());
    header.init();

    header.syncState({ base_lattice: '/lattices/alsu/AR_full.mat' });

    expect(contributedItem('lattice-name').text).toBe('AR_full.mat');
    expect(contributedItem('refresh').disabled).toBe(false);
    expect(contributedItem('baseline').disabled).toBe(false);
  });

  test('the armed label matches the resting one in length, so neighbours hold still', () => {
    // Baseline is not the last item in either rendering: the standalone topbar
    // puts the theme switcher after it, and Refresh/Verify sit before it in
    // both. A label that grows on arming therefore shoves a persistent control
    // sideways on the very click that arms this one — exactly the class of
    // jump the two-step confirm is supposed to protect against, reintroduced
    // by the confirm itself.
    //
    // Equal CHARACTER count is exact in the topbar (.action-btn is monospace)
    // and near-exact in the tile bar (--font-display is proportional), which
    // is the best a contributed item can do: the hub owns that button's box
    // and a panel has no way to ask it for a width.
    const header = createHeader(makeCallbacks());
    header.init();
    header.syncState({ base_lattice: '/fake.mat' });

    const resting = contributedItem('baseline').label;
    byId('btn-baseline').click();
    const armed = contributedItem('baseline').label;

    expect(armed).not.toBe(resting);
    expect(armed.length).toBe(resting.length);
  });

  test('unloading the lattice cancels a pending confirm', () => {
    const cb = makeCallbacks();
    const header = createHeader(cb);
    header.init();
    header.syncState({ base_lattice: '/fake.mat' });

    byId('btn-baseline').click();
    header.syncState({ base_lattice: null });

    expect(contributedItem('baseline').label).toBe('Baseline');
    expect(qs(byId('btn-baseline'), '.baseline-label').textContent).toBe('Baseline');
    // The next click arms afresh rather than firing the stale confirm
    byId('btn-baseline').click();
    expect(cb.onBaseline).not.toHaveBeenCalled();
  });
});
