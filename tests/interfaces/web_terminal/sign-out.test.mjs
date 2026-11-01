// @ts-check
/**
 * Unit tests for the sign-out dialog (sign-out.js):
 *   npx vitest run tests/interfaces/web_terminal/sign-out.test.mjs
 *
 * `confirmSignOut` asks with the shared confirm dialog (posture-confirm.js)
 * and, only on Sign out, runs logout.js's flow with the landing URL and the
 * button it was given. logout.js is mocked: its flow is pinned by
 * app-logout.test.mjs.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const logout = vi.hoisted(() => vi.fn(async () => {}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/logout.js', () => ({ logout }));

import { confirmSignOut } from '../../../src/osprey/interfaces/web_terminal/static/js/sign-out.js';
import {
  dismissConfirm,
  isConfirmUp,
} from '../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js';

/** The dialog while it is up — one on its way out carries `data-closing`. */
const dialog = () =>
  /** @type {HTMLElement|null} */ (
    document.querySelector('.posture-modal-overlay:not([data-closing])')
  );

/** @param {string} selector */
const inDialog = (selector) => {
  const node = dialog()?.querySelector(selector);
  if (!(node instanceof HTMLElement)) throw new Error(`no ${selector} in the dialog`);
  return node;
};

beforeEach(() => {
  logout.mockClear();
  document.body.innerHTML = '<button id="logout-btn" type="button">Sign out</button>';
});

afterEach(() => {
  dismissConfirm();
  document.body.innerHTML = '';
});

describe('confirmSignOut', () => {
  test('asks in plain words, with Cancel and Sign out', () => {
    confirmSignOut({ landingUrl: '/landing' });

    expect(inDialog('.posture-modal-title').textContent).toBe('Sign out of OSPREY?');
    expect(inDialog('.posture-modal-body').textContent).toBe(
      'Your terminal and agent sessions end. Anything the agent is doing stops.'
    );
    expect(inDialog('.posture-modal-cancel').textContent).toBe('Cancel');
    expect(inDialog('.posture-modal-confirm').textContent).toBe('Sign out');
    expect(logout).not.toHaveBeenCalled();
  });

  test('Cancel has focus, so a stray Enter signs nobody out', () => {
    confirmSignOut({ landingUrl: '/landing' });

    expect(document.activeElement).toBe(inDialog('.posture-modal-cancel'));
  });

  test('Sign out closes the dialog and runs logout with the landing URL and the button', () => {
    const button = /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn'));
    confirmSignOut({ landingUrl: '/landing', button });

    inDialog('.posture-modal-confirm').click();

    expect(isConfirmUp()).toBe(false);
    expect(logout).toHaveBeenCalledTimes(1);
    expect(logout).toHaveBeenCalledWith('/landing', button);
  });

  test('with no button the flow runs with none', () => {
    confirmSignOut({ landingUrl: '/landing' });

    inDialog('.posture-modal-confirm').click();

    expect(logout).toHaveBeenCalledWith('/landing', null);
  });

  test('Cancel does nothing', () => {
    confirmSignOut({ landingUrl: '/landing' });

    inDialog('.posture-modal-cancel').click();

    expect(isConfirmUp()).toBe(false);
    expect(logout).not.toHaveBeenCalled();
  });

  test('Escape is the same as Cancel, and stops listening once the dialog is gone', () => {
    confirmSignOut({ landingUrl: '/landing' });

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', cancelable: true }));
    expect(isConfirmUp()).toBe(false);
    expect(logout).not.toHaveBeenCalled();

    // A later Escape with no dialog up is left alone.
    const later = new KeyboardEvent('keydown', { key: 'Escape', cancelable: true });
    document.dispatchEvent(later);
    expect(later.defaultPrevented).toBe(false);
  });
});
