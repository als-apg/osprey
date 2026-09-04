/**
 * Unit tests for the shared clipboard helper (design_system/static/js/clipboard.js).
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/design_system/clipboard.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach } from 'vitest';

import { copyText } from '/design-system/js/clipboard.js';

/** @returns {number} number of <textarea> elements currently in the document. */
function textareaCount() {
  return document.querySelectorAll('textarea').length;
}

/**
 * Shadow `navigator.clipboard` as absent for this test.
 *
 * happy-dom exposes `clipboard` as a getter on `Navigator.prototype`, so
 * `delete navigator.clipboard` is a no-op (there is no own property to
 * remove) and the prototype's Clipboard object keeps shining through. This
 * defines an own, configurable `undefined` value instead, which actually
 * shadows the getter; `restoreClipboard()` removes that own property again.
 */
function stubClipboardAbsent() {
  Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
}

describe('copyText', () => {
  /** @type {any} */
  let originalClipboard;

  beforeEach(() => {
    originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
  });

  afterEach(() => {
    if (originalClipboard) {
      Object.defineProperty(navigator, 'clipboard', originalClipboard);
    } else {
      // @ts-expect-error - deleting a test-only stub
      delete navigator.clipboard;
    }
    // @ts-expect-error - clearing a test-only stub
    document.execCommand = undefined;
    for (const el of Array.from(document.querySelectorAll('textarea'))) el.remove();
  });

  test('resolves true via navigator.clipboard.writeText when present', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: async () => {} },
      configurable: true,
    });

    await expect(copyText('hello')).resolves.toBe(true);
    expect(textareaCount()).toBe(0);
  });

  test('falls back to execCommand when navigator.clipboard is undefined', async () => {
    stubClipboardAbsent();
    document.execCommand = () => true;

    await expect(copyText('hello')).resolves.toBe(true);
    expect(textareaCount()).toBe(0);
  });

  test('resolves false when both rungs are absent', async () => {
    stubClipboardAbsent();
    // @ts-expect-error - clearing a test-only stub
    document.execCommand = undefined;

    await expect(copyText('hello')).resolves.toBe(false);
    expect(textareaCount()).toBe(0);
  });

  test('falls back to the textarea rung when writeText rejects, and removes it', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: async () => { throw new Error('denied'); } },
      configurable: true,
    });
    document.execCommand = () => true;

    await expect(copyText('hello')).resolves.toBe(true);
    expect(textareaCount()).toBe(0);
  });

  test('removes the textarea even when execCommand throws', async () => {
    stubClipboardAbsent();
    document.execCommand = () => { throw new Error('blocked'); };

    await expect(copyText('hello')).resolves.toBe(false);
    expect(textareaCount()).toBe(0);
  });
});
