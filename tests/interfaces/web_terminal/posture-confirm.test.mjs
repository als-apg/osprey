// @ts-check
/**
 * Unit tests for the shared confirm dialog (posture-confirm.js):
 *   npx vitest run tests/interfaces/web_terminal/posture-confirm.test.mjs
 *
 * Which button has focus when the dialog opens, that Cancel dismisses it, the
 * machine-kind tint, and the waiver checkbox shown only for a waivable move.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  dismissConfirm,
  isConfirmUp,
  showConfirm,
} from '../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js';

/** @param {object} [extra] */
function spec(extra = {}) {
  return {
    title: 'Turn writes on?',
    body: [['For ', { em: 'every session' }, '.']],
    live: null,
    confirmLabel: 'Turn on',
    onConfirm: vi.fn(),
    ...extra,
  };
}

/** @param {string} selector */
function button(selector) {
  return /** @type {HTMLButtonElement} */ (document.querySelector(selector));
}

beforeEach(() => {
  document.body.innerHTML = '';
});

afterEach(() => {
  dismissConfirm();
  vi.restoreAllMocks();
});

test("focus: 'cancel' puts focus on Cancel", () => {
  showConfirm(spec({ focus: 'cancel' }));

  expect(document.activeElement).toBe(button('.posture-modal-cancel'));
});

test('without a focus option the confirm button has focus', () => {
  showConfirm(spec());

  expect(document.activeElement).toBe(button('.posture-modal-confirm'));
});

test('a held key cannot answer the dialog; a fresh press can', () => {
  showConfirm(spec());
  const confirm = button('.posture-modal-confirm');
  /** @param {string} key @param {boolean} repeat */
  const press = (key, repeat) => {
    const event = new KeyboardEvent('keydown', { key, repeat, bubbles: true, cancelable: true });
    confirm.dispatchEvent(event);
    return event;
  };

  expect(press('Enter', true).defaultPrevented).toBe(true);
  expect(press(' ', true).defaultPrevented).toBe(true);
  expect(press('Enter', false).defaultPrevented).toBe(false);
  expect(press('Tab', true).defaultPrevented).toBe(false);
});

test('Cancel dismisses the dialog and runs onDismiss once', () => {
  const onDismiss = vi.fn();
  showConfirm(spec({ onDismiss }));

  button('.posture-modal-cancel').click();

  expect(isConfirmUp()).toBe(false);
  expect(onDismiss).toHaveBeenCalledTimes(1);
});

test('the dialog carries the machine kind it is about as data-target-kind', () => {
  // The stylesheet tones the banner and the confirm button from this one
  // attribute: amber for the rehearsal, red for the real machine.
  showConfirm(spec({ live: 'Rehearsal — nothing moves', targetKind: 'rehearsal' }));

  const dialog = /** @type {HTMLElement} */ (document.querySelector('.posture-modal'));
  expect(dialog.dataset.targetKind).toBe('rehearsal');
  expect(document.querySelector('.posture-modal-live')?.textContent).toBe(
    'Rehearsal — nothing moves'
  );
});

test('a dialog about no machine has no data-target-kind', () => {
  showConfirm(spec());

  const dialog = /** @type {HTMLElement} */ (document.querySelector('.posture-modal'));
  expect(dialog.hasAttribute('data-target-kind')).toBe(false);
});

test('a real-machine turn-on offers no waiver when the raiser passes none', () => {
  showConfirm(spec({ live: 'Live — writes move hardware', targetKind: 'real', skipKeyBase: null }));

  expect(document.querySelector('.posture-modal-skip')).toBeNull();
  expect(
    /** @type {HTMLElement} */ (document.querySelector('.posture-modal')).dataset.targetKind
  ).toBe('real');
});

test('a waivable move shows the checkbox, unticked, and confirming records the choice', () => {
  localStorage.clear();
  const key = 'osprey-ctc-skip-confirm:writes-asked:va:simulator';
  const onConfirm = vi.fn();
  showConfirm(spec({ targetKind: 'simulator', skipKeyBase: key, onConfirm }));

  const box = /** @type {HTMLInputElement} */ (document.querySelector('.posture-modal-skip input'));
  expect(box.checked).toBe(false);
  box.click();
  button('.posture-modal-confirm').click();

  expect(localStorage.getItem(key)).toBe('1');
  expect(onConfirm).toHaveBeenCalledTimes(1);
});

test('a recorded waiver re-shows ticked, and cancelling leaves it as it was', () => {
  localStorage.clear();
  const key = 'osprey-ctc-skip-confirm:writes-asked:va:simulator';
  localStorage.setItem(key, '1');
  showConfirm(spec({ skipKeyBase: key }));

  const box = /** @type {HTMLInputElement} */ (document.querySelector('.posture-modal-skip input'));
  expect(box.checked).toBe(true);
  box.click();
  button('.posture-modal-cancel').click();

  expect(localStorage.getItem(key)).toBe('1');
});

test.each(['real', 'rehearsal', 'simulator'])('the %s kind tints the dialog', (kind) => {
  showConfirm(spec({ targetKind: kind }));

  expect(
    /** @type {HTMLElement} */ (document.querySelector('.posture-modal')).dataset.targetKind
  ).toBe(kind);
});
