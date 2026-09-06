// @ts-check
/**
 * Unit tests for the browser's session pointer (session-pointer.js):
 *   npx vitest run tests/interfaces/web_terminal/session-pointer.test.mjs
 *
 * The pointer is the one slot both views read to decide which session they
 * are on, so the contract under test is small and load-bearing: the key is
 * resolved per persona, a write of an unchanged value is not a change, and
 * subscribers hear every value the tab settles on — including the null of a
 * clear, which is how the chat learns the session is gone.
 *
 * Module isolation: the listener list and the write mirror are module-private
 * with no reset API, so each test gets a fresh module instance via
 * vi.resetModules() + dynamic import (same pattern as terminal-resume.test.mjs).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js')} */
let pointer;

const STORAGE_KEY = 'osprey-pty-session';
const SCOPE_ATTR = 'data-osprey-storage-scope';

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  pointer = await import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js');
});

afterEach(() => {
  document.documentElement.removeAttribute(SCOPE_ATTR);
  vi.restoreAllMocks();
});

describe('reading and writing', () => {
  test('an unset pointer reads null', () => {
    expect(pointer.getPointer()).toBeNull();
  });

  test('a set key reads back, and lands in the storage slot both views read', () => {
    pointer.setPointer('session-k');

    expect(pointer.getPointer()).toBe('session-k');
    expect(localStorage.getItem(STORAGE_KEY)).toBe('session-k');
  });

  test('a key written by an earlier page load is what the next one reads', () => {
    localStorage.setItem(STORAGE_KEY, 'session-from-last-load');

    expect(pointer.getPointer()).toBe('session-from-last-load');
  });

  test('clearing empties the slot', () => {
    pointer.setPointer('session-k');

    pointer.clearPointer();

    expect(pointer.getPointer()).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});

describe('per-persona scope', () => {
  // On a multi-user mount every `/u/<user>/` shares one origin, so the bare
  // key is a slot any persona may have written. A session key replayed across
  // personas would attach one persona's surface to another's process.

  test("bob's write goes to bob's key and leaves the shared slot alone", () => {
    localStorage.setItem(STORAGE_KEY, 'someone-elses-session');
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

    pointer.setPointer('bobs-session');

    expect(localStorage.getItem(`${STORAGE_KEY}--bob`)).toBe('bobs-session');
    expect(localStorage.getItem(STORAGE_KEY)).toBe('someone-elses-session');
  });

  test("a bare key left by another persona is not bob's pointer", () => {
    localStorage.setItem(STORAGE_KEY, 'someone-elses-session');
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

    expect(pointer.getPointer()).toBeNull();
  });

  test("clearing removes bob's key only", () => {
    localStorage.setItem(STORAGE_KEY, 'someone-elses-session');
    localStorage.setItem(`${STORAGE_KEY}--bob`, 'bobs-session');
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

    pointer.clearPointer();

    expect(localStorage.getItem(`${STORAGE_KEY}--bob`)).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBe('someone-elses-session');
  });
});

describe('subscribers', () => {
  test('a set notifies with the new key', () => {
    const seen = vi.fn();
    pointer.subscribe(seen);

    pointer.setPointer('session-k');

    expect(seen).toHaveBeenCalledTimes(1);
    expect(seen).toHaveBeenCalledWith('session-k');
  });

  test('a clear notifies with null', () => {
    pointer.setPointer('session-k');
    const seen = vi.fn();
    pointer.subscribe(seen);

    pointer.clearPointer();

    expect(seen).toHaveBeenCalledWith(null);
  });

  test('re-writing the same key is not a change and notifies nobody', () => {
    pointer.setPointer('session-k');
    const seen = vi.fn();
    pointer.subscribe(seen);

    pointer.setPointer('session-k');

    expect(seen).not.toHaveBeenCalled();
  });

  test('clearing an already-empty pointer notifies nobody', () => {
    const seen = vi.fn();
    pointer.subscribe(seen);

    pointer.clearPointer();

    expect(seen).not.toHaveBeenCalled();
  });

  test('every subscriber hears the change, even when one throws', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const first = vi.fn(() => { throw new Error('listener blew up'); });
    const second = vi.fn();
    pointer.subscribe(first);
    pointer.subscribe(second);

    pointer.setPointer('session-k');

    expect(first).toHaveBeenCalled();
    expect(second).toHaveBeenCalledWith('session-k');
  });

  test('unsubscribing stops the notifications', () => {
    const seen = vi.fn();
    const unsubscribe = pointer.subscribe(seen);

    unsubscribe();
    pointer.setPointer('session-k');

    expect(seen).not.toHaveBeenCalled();
  });
});

describe('storage unavailable', () => {
  // Private browsing and disabled storage both throw on access. The pointer
  // is a convenience there, not a hard dependency: nothing may throw out, and
  // the value still has to hold for the life of the page so a flip mid-session
  // does not lose the key.

  /** Make every localStorage access throw, as a locked-down browser does. */
  function lockStorage() {
    for (const method of /** @type {const} */ (['getItem', 'setItem', 'removeItem'])) {
      vi.spyOn(Storage.prototype, method).mockImplementation(() => {
        throw new Error('storage is not available');
      });
    }
  }

  test('a set still holds for this page, and still notifies', () => {
    const seen = vi.fn();
    pointer.subscribe(seen);
    lockStorage();

    pointer.setPointer('session-k');

    expect(pointer.getPointer()).toBe('session-k');
    expect(seen).toHaveBeenCalledWith('session-k');
  });

  test('a clear still empties it, and still notifies', () => {
    pointer.setPointer('session-k');
    const seen = vi.fn();
    pointer.subscribe(seen);
    lockStorage();

    pointer.clearPointer();

    expect(pointer.getPointer()).toBeNull();
    expect(seen).toHaveBeenCalledWith(null);
  });

  test('a repeated set is still not a change', () => {
    lockStorage();
    pointer.setPointer('session-k');
    const seen = vi.fn();
    pointer.subscribe(seen);

    pointer.setPointer('session-k');

    expect(seen).not.toHaveBeenCalled();
  });
});
