// @ts-check
/**
 * Unit tests for the shared pane splitter (design_system/static/js/splitter.js):
 *   npx vitest run tests/interfaces/design_system/splitter.test.mjs
 *
 * This module is the single implementation behind the OKF sidebar resizer, the
 * artifact gallery's browse splitter, and the PLAN panel's browser split, so it
 * carries the tests those call sites used to duplicate: clamping, stored-width
 * restore, the ARIA-separator keyboard nudge, and the pointer drag itself.
 *
 * happy-dom reports a 0-width bounding rect, so drag/nudge assertions are
 * written against that baseline rather than a laid-out width.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import { clampWidth, initSplitter } from '/design-system/js/splitter.js';

const KEY = 'osprey-test-split-width';

/** @returns {{handle: HTMLElement, pane: HTMLElement}} */
function render() {
  document.body.innerHTML = `
    <div class="row">
      <aside id="pane"></aside>
      <div id="handle" class="osprey-splitter" role="separator"
           aria-orientation="vertical" tabindex="0"></div>
      <main id="rest"></main>
    </div>
  `;
  return {
    handle: /** @type {HTMLElement} */ (document.getElementById('handle')),
    pane: /** @type {HTMLElement} */ (document.getElementById('pane')),
  };
}

/** Wire a splitter over freshly rendered markup with test-friendly bounds. */
function setup(overrides = {}) {
  const { handle, pane } = render();
  const api = initSplitter({
    handle,
    pane,
    storageKey: KEY,
    min: 180,
    max: 560,
    step: 16,
    ...overrides,
  });
  return { handle, pane, api };
}

/**
 * @param {string} type
 * @param {number} clientX
 * @returns {PointerEvent}
 */
function pointer(type, clientX) {
  return new PointerEvent(type, { clientX, pointerId: 1, bubbles: true });
}

describe('clampWidth', () => {
  test('passes through in-range widths, rounded', () => {
    expect(clampWidth(300, 180, 560)).toBe(300);
    expect(clampWidth(300.6, 180, 560)).toBe(301);
  });

  test('clamps below the minimum and above the maximum', () => {
    expect(clampWidth(10, 180, 560)).toBe(180);
    expect(clampWidth(5000, 180, 560)).toBe(560);
  });
});

describe('initSplitter', () => {
  beforeEach(() => {
    localStorage.clear();
    document.body.innerHTML = '';
  });

  test('no-ops with a safe API when the pane is absent', () => {
    const api = initSplitter({
      handle: null,
      pane: null,
      storageKey: KEY,
      min: 180,
      max: 560,
    });
    localStorage.setItem(KEY, '300');
    expect(() => api.applyWidth(300)).not.toThrow();
    expect(() => api.clearWidth()).not.toThrow();
    expect(api.restoreWidth()).toBeNull();
  });

  test('a pane with no handle still sizes — only the interaction is skipped', () => {
    // The gallery mounts its orientation toggle against a sidebar whose handle
    // may be absent; dropping width restore there would silently lose the
    // operator's persisted split.
    const { pane } = render();
    localStorage.setItem(KEY, '340');
    const api = initSplitter({ handle: null, pane, storageKey: KEY, min: 180, max: 560 });

    expect(api.restoreWidth()).toBe(340);
    expect(pane.style.flexBasis).toBe('340px');
    api.clearWidth();
    expect(pane.style.flexBasis).toBe('');
  });

  test('applyWidth writes a clamped flex-basis and max-width', () => {
    const { pane, api } = setup();
    expect(api.applyWidth(420)).toBe(420);
    expect(pane.style.flexBasis).toBe('420px');
    expect(pane.style.maxWidth).toBe('420px');

    expect(api.applyWidth(9999)).toBe(560);
    expect(pane.style.flexBasis).toBe('560px');
  });

  test('clearWidth drops the inline sizes back to the stylesheet default', () => {
    const { pane, api } = setup();
    api.applyWidth(420);
    api.clearWidth();
    expect(pane.style.flexBasis).toBe('');
    expect(pane.style.maxWidth).toBe('');
  });

  test('restoreWidth applies a stored width, clamped', () => {
    localStorage.setItem(KEY, '420');
    const { pane, api } = setup();
    expect(api.restoreWidth()).toBe(420);
    expect(pane.style.flexBasis).toBe('420px');

    localStorage.setItem(KEY, '5000');
    expect(api.restoreWidth()).toBe(560);
  });

  test('restoreWidth ignores a non-numeric stored value', () => {
    localStorage.setItem(KEY, 'garbage');
    const { pane, api } = setup();
    expect(api.restoreWidth()).toBeNull();
    expect(pane.style.flexBasis).toBe('');
  });

  test('Arrow keys nudge the width by one step and persist it', () => {
    const { handle, pane } = setup();
    // happy-dom reports a 0-width rect, so the first nudge clamps up to min.
    handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(pane.style.flexBasis).toBe('180px');
    expect(localStorage.getItem(KEY)).toBe('180');
  });

  test('ignores keys that are not Arrow Left/Right', () => {
    const { handle, pane } = setup();
    handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(pane.style.flexBasis).toBe('');
    expect(localStorage.getItem(KEY)).toBeNull();
  });

  test('a pointer drag resizes live, flags .dragging, and persists on release', () => {
    const { handle, pane } = setup();

    handle.dispatchEvent(pointer('pointerdown', 0));
    expect(handle.classList.contains('dragging')).toBe(true);

    handle.dispatchEvent(pointer('pointermove', 300));
    expect(pane.style.flexBasis).toBe('300px');
    // Live drag must not thrash storage — only the release persists.
    expect(localStorage.getItem(KEY)).toBeNull();

    handle.dispatchEvent(pointer('pointerup', 300));
    expect(handle.classList.contains('dragging')).toBe(false);
    expect(localStorage.getItem(KEY)).toBe('300');

    // Move handlers are detached on release: a stray move does nothing.
    handle.dispatchEvent(pointer('pointermove', 500));
    expect(pane.style.flexBasis).toBe('300px');
  });

  test('pointercancel ends the drag like a release', () => {
    const { handle } = setup();
    handle.dispatchEvent(pointer('pointerdown', 0));
    handle.dispatchEvent(pointer('pointermove', 240));
    handle.dispatchEvent(pointer('pointercancel', 240));
    expect(handle.classList.contains('dragging')).toBe(false);
    expect(localStorage.getItem(KEY)).toBe('240');
  });

  test('isEnabled gates both the drag and the keyboard nudge', () => {
    const { handle, pane } = setup({ isEnabled: () => false });

    handle.dispatchEvent(pointer('pointerdown', 0));
    expect(handle.classList.contains('dragging')).toBe(false);
    handle.dispatchEvent(pointer('pointermove', 300));
    expect(pane.style.flexBasis).toBe('');

    handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(pane.style.flexBasis).toBe('');
  });

  test('survives blocked storage (sandboxed iframe) without throwing', () => {
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    try {
      const { handle, pane, api } = setup();
      expect(api.restoreWidth()).toBeNull();
      expect(() =>
        handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })),
      ).not.toThrow();
      // The split still holds for this page lifetime.
      expect(pane.style.flexBasis).toBe('180px');
    } finally {
      setItem.mockRestore();
      getItem.mockRestore();
    }
  });
});
