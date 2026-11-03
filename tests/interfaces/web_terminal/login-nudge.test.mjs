// @ts-check
/**
 * Unit tests for the personal-login nudge (login-nudge.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/login-nudge.test.mjs
 *
 * The card is due only on the server's two facts -- the deployment allows a
 * personal login and this terminal holds none -- and never after "Got it".
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  NUDGE_ID,
  NUDGE_DELAY_MS,
  applyLoginNudge,
  hideLoginNudge,
  loginNudgeDue,
  setDismissed,
  showLoginNudge,
} from '../../../src/osprey/interfaces/web_terminal/static/js/login-nudge.js';

const allowedNoLogin = { personal_login: { allowed: true, present: false } };

beforeEach(() => {
  document.body.innerHTML = '';
  document.body.className = '';
  localStorage.clear();
  vi.useFakeTimers();
});

afterEach(() => {
  hideLoginNudge();
  vi.useRealTimers();
});

describe('when the card is due', () => {
  test('allowed and no login: due', () => {
    expect(loginNudgeDue(allowedNoLogin)).toBe(true);
  });

  test('a terminal that already holds a login shows nothing', () => {
    expect(loginNudgeDue({ personal_login: { allowed: true, present: true } })).toBe(false);
  });

  test('a deployment that does not allow it shows nothing, login or not', () => {
    expect(loginNudgeDue({ personal_login: { allowed: false, present: false } })).toBe(false);
    expect(loginNudgeDue({ personal_login: { allowed: false, present: true } })).toBe(false);
  });

  test('no facts (failed fetch, older server) shows nothing', () => {
    expect(loginNudgeDue(null)).toBe(false);
    expect(loginNudgeDue({})).toBe(false);
  });

  test('a dismissal is sticky', () => {
    setDismissed(true);
    expect(loginNudgeDue(allowedNoLogin)).toBe(false);
  });

  test('an embedded page never shows it', () => {
    document.body.classList.add('embedded');
    expect(loginNudgeDue(allowedNoLogin)).toBe(false);
  });
});

describe('the card', () => {
  test('is scheduled after the shell settles and names /login', () => {
    expect(applyLoginNudge(allowedNoLogin)).toBe(true);
    expect(document.getElementById(NUDGE_ID)).toBeNull();
    vi.advanceTimersByTime(NUDGE_DELAY_MS);
    const card = document.getElementById(NUDGE_ID);
    expect(card).not.toBeNull();
    expect(card?.textContent).toContain('/login');
    expect(card?.textContent).toContain('bills to your own account');
  });

  test('"Got it" dismisses for good; "Not now" only closes it', () => {
    showLoginNudge();
    /** @type {HTMLButtonElement[]} */
    const buttons = Array.from(document.querySelectorAll(`#${NUDGE_ID} button`));
    const later = buttons.find((b) => b.textContent === 'Not now');
    later?.click();
    expect(document.getElementById(NUDGE_ID)).toBeNull();
    expect(loginNudgeDue(allowedNoLogin)).toBe(true);

    showLoginNudge();
    /** @type {HTMLButtonElement[]} */
    const again = Array.from(document.querySelectorAll(`#${NUDGE_ID} button`));
    const gotIt = again.find((b) => b.textContent === 'Got it');
    gotIt?.click();
    expect(document.getElementById(NUDGE_ID)).toBeNull();
    expect(loginNudgeDue(allowedNoLogin)).toBe(false);
  });

  test('showing twice keeps one card', () => {
    showLoginNudge();
    showLoginNudge();
    expect(document.querySelectorAll(`#${NUDGE_ID}`).length).toBe(1);
  });
});
