// @ts-check
/**
 * Unit tests for the pointer reports of a framed artifact page
 * (frame-pointer.js), which show the content strip over a sandboxed plot or
 * table frame:
 *   npx vitest run tests/interfaces/artifacts/frame-pointer.test.mjs
 *
 * The module runs inside the sandboxed page (the injected snippets inline
 * it), so these tests hand it a fake window whose `parent` is the gallery.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  POINTER_MESSAGE,
  reportPointer,
} from '../../../src/osprey/interfaces/artifacts/static/js/frame-pointer.js';

/**
 * The page's window in a frame, or alone.
 * @param {{framed?: boolean}} [options]
 */
function pageWindow({ framed = true } = {}) {
  const gallery = { postMessage: vi.fn() };
  const root = document.createElement('html');
  /** @type {any} */
  const win = { document: { documentElement: root } };
  win.parent = framed ? gallery : win;
  return {
    win,
    gallery,
    /** @param {string} type */
    fire: (type) => root.dispatchEvent(new Event(type)),
  };
}

afterEach(() => {
  vi.useRealTimers();
});

describe('reportPointer', () => {
  test('framed, it reports the moving pointer at most twice a second, and its leaving', () => {
    vi.useFakeTimers();
    const page = pageWindow();
    reportPointer(page.win);

    page.fire('pointermove');
    page.fire('pointermove');
    vi.advanceTimersByTime(499);
    page.fire('pointermove');
    vi.advanceTimersByTime(1);
    page.fire('pointermove');
    page.fire('pointerleave');
    page.fire('pointermove');

    expect(page.gallery.postMessage.mock.calls).toEqual([
      [{ type: POINTER_MESSAGE, over: true }, '*'],
      [{ type: POINTER_MESSAGE, over: true }, '*'],
      [{ type: POINTER_MESSAGE, over: false }, '*'],
      [{ type: POINTER_MESSAGE, over: true }, '*'],
    ]);
  });

  test('on its own, the page reports nothing', () => {
    const page = pageWindow({ framed: false });
    reportPointer(page.win);

    page.fire('pointermove');
    page.fire('pointerleave');

    expect(page.gallery.postMessage).not.toHaveBeenCalled();
  });
});
