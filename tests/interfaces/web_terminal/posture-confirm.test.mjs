// @ts-check
/**
 * Unit tests for the shared confirm dialog (posture-confirm.js):
 *   npx vitest run tests/interfaces/web_terminal/posture-confirm.test.mjs
 *
 * Which button has focus when the dialog opens, and that Cancel dismisses it.
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

test('Cancel dismisses the dialog and runs onDismiss once', () => {
  const onDismiss = vi.fn();
  showConfirm(spec({ onDismiss }));

  button('.posture-modal-cancel').click();

  expect(isConfirmUp()).toBe(false);
  expect(onDismiss).toHaveBeenCalledTimes(1);
});
