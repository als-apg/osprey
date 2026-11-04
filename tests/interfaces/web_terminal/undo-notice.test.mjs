/**
 * OSPREY Web Terminal — the Undo notice.
 *
 * The callers (bar menus, rail, tile headers) prove what a removal reports;
 * this file pins the notice's own contract: one notice at a time, an Undo
 * button only when there is a way back, the Undo runs once and closes, and the
 * countdown waits while the operator's pointer or focus is on the notice.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  UNDO_NOTICE_MS,
  closeUndoNotice,
  showUndoNotice,
} from '../../../src/osprey/interfaces/web_terminal/static/js/undo-notice.js';

/** Every notice element on the page. */
function notices() {
  return Array.from(document.querySelectorAll('#undo-notice'));
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  closeUndoNotice();
  vi.useRealTimers();
  document.body.replaceChildren();
});

describe('showing a notice', () => {
  test('it is a polite status line carrying the message', () => {
    showUndoNotice({ message: 'Clock removed', undo: () => {} });

    const [node] = notices();
    expect(node.getAttribute('role')).toBe('status');
    expect(node.getAttribute('aria-live')).toBe('polite');
    expect(node.querySelector('.undo-notice-text')?.textContent).toBe('Clock removed');
    expect(node.querySelector('.undo-notice-btn')?.textContent).toBe('Undo');
  });

  test('without a way back it offers no Undo button', () => {
    showUndoNotice({ message: 'Cannot remove the last tile' });

    expect(notices()).toHaveLength(1);
    expect(notices()[0].querySelector('.undo-notice-btn')).toBe(null);
  });

  test('a second notice replaces the first', () => {
    showUndoNotice({ message: 'Clock removed', undo: () => {} });
    showUndoNotice({ message: 'Search removed', undo: () => {} });

    expect(notices()).toHaveLength(1);
    expect(notices()[0].querySelector('.undo-notice-text')?.textContent).toBe('Search removed');
  });
});

describe('Undo', () => {
  test('runs the closure once and closes the notice', () => {
    const undo = vi.fn();
    showUndoNotice({ message: 'Clock removed', undo });

    /** @type {HTMLButtonElement} */ (notices()[0].querySelector('.undo-notice-btn')).click();

    expect(undo).toHaveBeenCalledTimes(1);
    expect(notices()).toHaveLength(0);
  });

  test('the returned close is a no-op once the notice has gone', () => {
    const close = showUndoNotice({ message: 'Clock removed', undo: () => {} });
    close();
    showUndoNotice({ message: 'Search removed', undo: () => {} });

    close();

    expect(notices()).toHaveLength(1);
    expect(notices()[0].querySelector('.undo-notice-text')?.textContent).toBe('Search removed');
  });
});

describe('the countdown', () => {
  test('the notice closes after its duration', () => {
    showUndoNotice({ message: 'Clock removed', undo: () => {} });

    vi.advanceTimersByTime(UNDO_NOTICE_MS - 1);
    expect(notices()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(notices()).toHaveLength(0);
  });

  test('it pauses while the pointer is over the notice and restarts on leave', () => {
    showUndoNotice({ message: 'Clock removed', undo: () => {}, durationMs: 100 });
    const [node] = notices();

    node.dispatchEvent(new Event('pointerenter'));
    vi.advanceTimersByTime(1000);
    expect(notices()).toHaveLength(1);

    node.dispatchEvent(new Event('pointerleave'));
    vi.advanceTimersByTime(99);
    expect(notices()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(notices()).toHaveLength(0);
  });

  test('it pauses while the focus is inside the notice', () => {
    showUndoNotice({ message: 'Clock removed', undo: () => {}, durationMs: 100 });
    const [node] = notices();

    node.dispatchEvent(new Event('focusin'));
    vi.advanceTimersByTime(1000);
    expect(notices()).toHaveLength(1);

    node.dispatchEvent(new Event('focusout'));
    vi.advanceTimersByTime(100);
    expect(notices()).toHaveLength(0);
  });
});
