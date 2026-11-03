// @ts-check
/**
 * The Safety tab's runtime safety-switch rows (safety-switches.js):
 *   npx vitest run tests/interfaces/web_terminal/safety-switches.test.mjs
 *
 * Every operator sees the rows; only `can_change: true` enables the toggle. A
 * feature newly seen off raises a notice naming who turned it off, once.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  _resetSafetySwitchesForTest,
  initSafetySwitches,
  isSafetySwitchFrame,
  refreshSafetySwitches,
} from '../../../src/osprey/interfaces/web_terminal/static/js/safety-switches.js';
import { setChatNoticeSink } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js';
import { dismissConfirm } from '../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js';

const ON = {
  name: 'watchdog',
  label: 'Safety watchdog',
  note: 'Open holds stay open.',
  on: true,
  off_by: null,
  off_at: null,
};
const OFF = { ...ON, on: false, off_by: 'alice', off_at: 1_700_000_000 };

/** @type {{switches: object[], can_change: boolean}} */
let payload;
/** @type {Array<{url: string, init: any}>} */
let posts;

beforeEach(() => {
  document.body.innerHTML = '<div id="tab-safety"><div id="safety-switch-section"></div></div>';
  payload = { switches: [ON], can_change: true };
  posts = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url, /** @type {any} */ init) => {
      if (init?.method === 'POST') posts.push({ url, init });
      return { ok: true, status: 200, statusText: 'OK', json: async () => payload };
    })
  );
  _resetSafetySwitchesForTest();
  initSafetySwitches();
});

afterEach(() => {
  dismissConfirm();
  setChatNoticeSink(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** @returns {HTMLInputElement} */
function toggle() {
  return /** @type {HTMLInputElement} */ (document.querySelector('[data-switch="watchdog"] input'));
}

function notice() {
  return document.querySelector('.undo-notice-text')?.textContent ?? '';
}

/** The in-app confirm the off raises, while it is up. */
function dialog() {
  return document.querySelector('.posture-modal-overlay:not([data-closing]) .posture-modal');
}

/** @param {'confirm' | 'cancel'} which */
function press(which) {
  const button = dialog()?.querySelector(`.posture-modal-${which}`);
  if (!(button instanceof HTMLButtonElement)) throw new Error(`no ${which} button`);
  button.click();
}

function switchOff() {
  toggle().checked = false;
  toggle().dispatchEvent(new Event('change'));
}

test('the frame type is recognised', () => {
  expect(isSafetySwitchFrame({ type: 'safety_switch' })).toBe(true);
  expect(isSafetySwitchFrame({ type: 'watchdog_flag' })).toBe(false);
});

test('an admin tier gets a live toggle that reflects the state', async () => {
  await refreshSafetySwitches();
  expect(toggle().checked).toBe(true);
  expect(toggle().disabled).toBe(false);
});

test('a non-admin tier sees the state but cannot move it', async () => {
  payload = { switches: [OFF], can_change: false };
  await refreshSafetySwitches();
  expect(toggle().checked).toBe(false);
  expect(toggle().disabled).toBe(true);
  expect(document.querySelector('.safety-switch-state')?.textContent).toContain('alice');
  expect(document.querySelector('.safety-switch-lock')?.textContent).toBe(
    '🔒 Set by your deployment admin'
  );
});

test('an admin tier has no lock line beside its live toggle', async () => {
  await refreshSafetySwitches();
  expect(document.querySelector('.safety-switch-lock')?.textContent).toBe('');
});

test('a feature newly seen off is announced once, naming who turned it off', async () => {
  payload = { switches: [OFF], can_change: false };
  await refreshSafetySwitches();
  expect(notice()).toContain('Safety watchdog is off in this terminal by alice');

  document.querySelector('.undo-notice')?.remove();
  await refreshSafetySwitches();
  expect(notice()).toBe('');
});

test('turning it off asks in the app dialog first, then posts the move', async () => {
  await refreshSafetySwitches();
  switchOff();
  expect(dialog()?.textContent).toContain('Turn Safety watchdog off?');
  expect(dialog()?.textContent).toContain(
    'It stays off for this terminal until it restarts. Open holds stay open. ' +
      'The move is recorded in the audit trail with your name.'
  );
  expect(toggle().checked).toBe(true);
  expect(posts).toHaveLength(0);
  press('confirm');
  await vi.waitFor(() => expect(posts).toHaveLength(1));
  expect(posts[0].url).toContain('/api/safety-switches/watchdog');
  expect(JSON.parse(posts[0].init.body)).toEqual({ state: 'off' });
});

test('a cancelled confirm leaves it on and posts nothing', async () => {
  await refreshSafetySwitches();
  switchOff();
  press('cancel');
  expect(dialog()).toBeNull();
  expect(toggle().checked).toBe(true);
  expect(posts).toHaveLength(0);
});

test('Escape closes the confirm and posts nothing', async () => {
  await refreshSafetySwitches();
  switchOff();
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
  expect(dialog()).toBeNull();
  expect(posts).toHaveLength(0);
});

test('a refused off stays on and shows the server detail in the dialog', async () => {
  await refreshSafetySwitches();
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: false,
      status: 403,
      statusText: 'Forbidden',
      json: async () => ({ detail: 'the Config panel is disabled' }),
    }))
  );
  switchOff();
  press('confirm');
  await vi.waitFor(() =>
    expect(dialog()?.querySelector('.posture-modal-error')?.textContent).toContain(
      'the Config panel is disabled'
    )
  );
  expect(toggle().checked).toBe(true);
});

test('a refused on reverts the toggle and shows the server detail', async () => {
  payload = { switches: [OFF], can_change: true };
  await refreshSafetySwitches();
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: false,
      status: 403,
      statusText: 'Forbidden',
      json: async () => ({ detail: 'the Config panel is disabled' }),
    }))
  );
  toggle().checked = true;
  toggle().dispatchEvent(new Event('change'));
  await vi.waitFor(() => expect(notice()).toContain('the Config panel is disabled'));
  expect(toggle().checked).toBe(false);
});

test('with the chat on screen the notice goes in the chat, not a toast', async () => {
  /** @type {string[]} */
  const chat = [];
  setChatNoticeSink((text) => (chat.push(text), true));
  payload = { switches: [OFF], can_change: false };
  await refreshSafetySwitches();
  expect(chat).toEqual([
    expect.stringContaining('Safety watchdog is off in this terminal by alice'),
  ]);
  expect(notice()).toBe('');
});

test('with the chat off screen the notice is a toast', async () => {
  setChatNoticeSink(() => false);
  payload = { switches: [OFF], can_change: false };
  await refreshSafetySwitches();
  expect(notice()).toContain('It returns at the next restart.');
});

test('a feature seen back on is announced', async () => {
  payload = { switches: [OFF], can_change: true };
  await refreshSafetySwitches();
  payload = { switches: [ON], can_change: true };
  await refreshSafetySwitches();
  expect(notice()).toBe('Safety watchdog is back on.');
});

test('no switches hides the section', async () => {
  payload = { switches: [], can_change: true };
  await refreshSafetySwitches();
  expect(/** @type {HTMLElement} */ (document.getElementById('safety-switch-section')).hidden).toBe(
    true
  );
});
