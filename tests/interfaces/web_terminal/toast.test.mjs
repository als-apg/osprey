// @ts-check
/**
 * Unit tests for the Web Terminal's shared toast (toast.js):
 *   npx vitest run tests/interfaces/web_terminal/toast.test.mjs
 *
 * `showToast(host, message, opts)` keeps one toast per host, renders the error
 * tone as a class, announces itself as a status, and removes itself after
 * `TOAST_TTL_MS`.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  showToast,
  TOAST_TTL_MS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/toast.js';

/** @type {HTMLElement} */
let host;

beforeEach(() => {
  document.body.innerHTML = '<div id="host"></div>';
  host = /** @type {HTMLElement} */ (document.getElementById('host'));
});

afterEach(() => {
  vi.useRealTimers();
});

describe('showToast', () => {
  test('a second toast on one host replaces the first', () => {
    showToast(host, 'first');
    const second = showToast(host, 'second');

    const toasts = host.querySelectorAll('.osprey-toast');
    expect(toasts).toHaveLength(1);
    expect(toasts[0]).toBe(second);
    expect(toasts[0].textContent).toBe('second');
  });

  test('the default tone carries only the toast class', () => {
    expect(showToast(host, 'saved').className).toBe('osprey-toast');
  });

  test('the error tone adds the error class', () => {
    expect(showToast(host, 'failed', { error: true }).className).toBe('osprey-toast error');
  });

  test('the toast is announced as a status', () => {
    expect(showToast(host, 'saved').getAttribute('role')).toBe('status');
  });

  test('the toast stays until TOAST_TTL_MS and is gone after it', () => {
    vi.useFakeTimers();
    const toast = showToast(host, 'saved');

    vi.advanceTimersByTime(TOAST_TTL_MS - 1);
    expect(toast.isConnected).toBe(true);

    vi.advanceTimersByTime(1);
    expect(toast.isConnected).toBe(false);
    expect(host.querySelector('.osprey-toast')).toBeNull();
  });
});
