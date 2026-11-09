/**
 * Unit tests for the design-system toast primitive (C-TOAST).
 *
 *   npx vitest run tests/interfaces/design_system/js/toast.test.mjs
 *
 * Pins the two live regions (status for info/success/warning, alert for
 * error), the queue (one toast on screen at a time, in call order), the
 * auto-hide and the no-DOM no-op.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  showToast,
  clearToasts,
  TOAST_TIMEOUT_MS,
} from '../../../../src/osprey/interfaces/design_system/static/js/toast.js';

/** @param {string} selector */
const one = (selector) => /** @type {HTMLElement} */ (document.querySelector(selector));
const host = () => one('.osprey-toast-host');
const statusRegion = () => one('.osprey-toast-host [role="status"]');
const alertRegion = () => one('.osprey-toast-host [role="alert"]');
const visible = () =>
  /** @type {HTMLElement[]} */ ([...document.querySelectorAll('.osprey-toast')]);

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  clearToasts();
  // The host outlives clearToasts by design; each test starts on a bare page.
  host()?.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('live regions', () => {
  test('one host holds a status region and an alert region', () => {
    showToast('Saved');
    expect(document.querySelectorAll('.osprey-toast-host')).toHaveLength(1);
    expect(host().parentElement).toBe(document.body);
    expect(statusRegion()).not.toBeNull();
    expect(alertRegion()).not.toBeNull();
    expect(statusRegion().getAttribute('aria-live')).toBe('polite');
    expect(alertRegion().getAttribute('aria-live')).toBe('assertive');
  });

  test.each(/** @type {const} */ (['info', 'success', 'warning']))(
    '%s goes to the status region',
    (kind) => {
      showToast(`a ${kind} message`, { kind });
      const [toast] = visible();
      expect(toast.textContent).toBe(`a ${kind} message`);
      expect(toast.parentElement).toBe(statusRegion());
      expect(toast.dataset.kind).toBe(kind);
      expect(toast.classList.contains(`osprey-toast--${kind}`)).toBe(true);
      expect(alertRegion().children).toHaveLength(0);
    }
  );

  test('error goes to the alert region', () => {
    showToast('Draft is gone', { kind: 'error' });
    const [toast] = visible();
    expect(toast.parentElement).toBe(alertRegion());
    expect(toast.dataset.kind).toBe('error');
    expect(statusRegion().children).toHaveLength(0);
  });

  test('kind defaults to info and an unknown kind falls back to info', () => {
    showToast('plain');
    expect(visible()[0].dataset.kind).toBe('info');
    clearToasts();
    showToast('odd', /** @type {any} */ ({ kind: 'bogus' }));
    expect(visible()[0].dataset.kind).toBe('info');
    expect(visible()[0].parentElement).toBe(statusRegion());
  });

  test('message is text, never markup', () => {
    showToast('<img src=x onerror=alert(1)>');
    expect(visible()[0].querySelector('img')).toBeNull();
    expect(visible()[0].textContent).toBe('<img src=x onerror=alert(1)>');
  });

  test('a second call reuses the same host', () => {
    showToast('one');
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS + 1000);
    showToast('two');
    expect(document.querySelectorAll('.osprey-toast-host')).toHaveLength(1);
  });
});

describe('auto-hide', () => {
  test('default timeout is about 3 s', () => {
    expect(TOAST_TIMEOUT_MS).toBeGreaterThanOrEqual(2500);
    expect(TOAST_TIMEOUT_MS).toBeLessThanOrEqual(4000);
  });

  test('a toast hides after the default timeout', () => {
    showToast('bye');
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS - 1);
    expect(visible()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(visible()).toHaveLength(0);
  });

  test('a custom timeout is honoured', () => {
    showToast('quick', { timeout: 500 });
    vi.advanceTimersByTime(499);
    expect(visible()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(visible()).toHaveLength(0);
  });

  test('a non-positive or non-numeric timeout uses the default', () => {
    showToast('x', { timeout: 0 });
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS - 1);
    expect(visible()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(visible()).toHaveLength(0);
    showToast('y', /** @type {any} */ ({ timeout: 'soon' }));
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS);
    expect(visible()).toHaveLength(0);
  });
});

describe('queue', () => {
  test('toasts never overlap and show in call order', () => {
    showToast('first');
    showToast('second', { kind: 'error' });
    showToast('third', { kind: 'success' });

    expect(visible().map((t) => t.textContent)).toEqual(['first']);
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS);
    expect(visible().map((t) => t.textContent)).toEqual(['second']);
    expect(visible()[0].parentElement).toBe(alertRegion());
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS);
    expect(visible().map((t) => t.textContent)).toEqual(['third']);
    expect(visible()[0].parentElement).toBe(statusRegion());
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS);
    expect(visible()).toHaveLength(0);
  });

  test('each queued toast keeps its own timeout', () => {
    showToast('long', { timeout: 5000 });
    showToast('short', { timeout: 200 });
    vi.advanceTimersByTime(4999);
    expect(visible()[0].textContent).toBe('long');
    vi.advanceTimersByTime(1);
    expect(visible()[0].textContent).toBe('short');
    vi.advanceTimersByTime(200);
    expect(visible()).toHaveLength(0);
  });

  test('clearToasts drops the shown toast and the queue', () => {
    showToast('a');
    showToast('b');
    clearToasts();
    expect(visible()).toHaveLength(0);
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS * 3);
    expect(visible()).toHaveLength(0);
  });

  test('a host removed from the page is rebuilt on the next toast', () => {
    showToast('a');
    host().remove();
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS);
    showToast('b');
    expect(host()).not.toBeNull();
    expect(visible().map((t) => t.textContent)).toEqual(['b']);
  });
});

describe('no DOM', () => {
  test('without a document showToast does nothing and does not throw', () => {
    vi.stubGlobal('document', undefined);
    expect(() => showToast('nowhere', { kind: 'error' })).not.toThrow();
    expect(() => clearToasts()).not.toThrow();
    vi.unstubAllGlobals();
    expect(host()).toBeNull();
    // Nothing was queued while there was no DOM.
    vi.advanceTimersByTime(TOAST_TIMEOUT_MS * 2);
    expect(visible()).toHaveLength(0);
  });
});
