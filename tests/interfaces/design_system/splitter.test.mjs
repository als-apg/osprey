// @ts-check
/**
 * Unit tests for the shared pane splitter (design_system/static/js/splitter.js):
 *   npx vitest run tests/interfaces/design_system/splitter.test.mjs
 *
 * This module is the single implementation behind every resizable pane in the
 * product — the OKF sidebar, the artifact gallery's browse split, the PLAN
 * panel's browser split, the LATTICE sidebar, the settings drawer, and the
 * event dashboard's three zones — so it carries the tests those call sites
 * would otherwise duplicate.
 *
 * Two environment facts shape how these are written:
 *
 * - happy-dom reports a 0-width bounding rect, so drag/nudge assertions are
 *   written against that baseline rather than a laid-out size. This is also why
 *   the module tracks its own last-applied size instead of measuring: a
 *   collapse/restore round-trip that read the rect would always land on `min`.
 * - the drag coalesces writes into `requestAnimationFrame`. Most tests stub it
 *   to run synchronously so live-drag assertions stay readable; the ordering
 *   test below deliberately defers it to prove the release path.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { clampSize, parseStoredState, initSplitter } from '/design-system/js/splitter.js';

const KEY = 'osprey-test-split-size';

/** @returns {{handle: HTMLElement, pane: HTMLElement}} */
function render() {
  document.body.innerHTML = `
    <div class="row">
      <aside id="pane"></aside>
      <div id="handle" class="osprey-splitter" tabindex="0"></div>
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
 * @param {number} coord along the splitter's axis
 * @param {'x'|'y'} [axis]
 * @returns {PointerEvent}
 */
function pointer(type, coord, axis = 'x') {
  const position = axis === 'x' ? { clientX: coord, clientY: 0 } : { clientX: 0, clientY: coord };
  return new PointerEvent(type, { ...position, pointerId: 1, bubbles: true });
}

/** The persisted record, parsed. */
function stored(key = KEY) {
  const raw = localStorage.getItem(key);
  return raw === null ? null : JSON.parse(raw);
}

describe('clampSize', () => {
  test('passes through in-range sizes, rounded', () => {
    expect(clampSize(300, 180, 560)).toBe(300);
    expect(clampSize(300.6, 180, 560)).toBe(301);
  });

  test('clamps below the minimum and above the maximum', () => {
    expect(clampSize(10, 180, 560)).toBe(180);
    expect(clampSize(5000, 180, 560)).toBe(560);
  });
});

describe('parseStoredState', () => {
  test('reads the current record shape', () => {
    expect(parseStoredState('{"size":420,"collapsed":true}', false)).toEqual({
      size: 420,
      collapsed: true,
    });
  });

  test('reads a legacy bare integer as an expanded size', () => {
    // Every splitter host stored a bare width before collapse existed.
    expect(parseStoredState('340', false)).toEqual({ size: 340, collapsed: false });
  });

  test('reads a legacy "true"/"false" flag as a collapsed state with no size', () => {
    // LATTICE's lattice-sidebar-collapsed key holds exactly this.
    expect(parseStoredState('true', false)).toEqual({ size: null, collapsed: true });
    expect(parseStoredState('false', true)).toEqual({ size: null, collapsed: false });
  });

  test('an absent value honours the host default in both directions', () => {
    // LATTICE's old read was `getItem(key) !== 'false'`, so absent means
    // COLLAPSED there. Every other host wants the stylesheet default.
    expect(parseStoredState(null, true)).toEqual({ size: null, collapsed: true });
    expect(parseStoredState(null, false)).toEqual({ size: null, collapsed: false });
  });

  test('falls back to the default on unparseable input', () => {
    expect(parseStoredState('garbage', false)).toEqual({ size: null, collapsed: false });
  });
});

describe('initSplitter', () => {
  beforeEach(() => {
    localStorage.clear();
    document.body.innerHTML = '';
    // Run coalesced writes immediately so live-drag assertions read naturally.
    vi.stubGlobal('requestAnimationFrame', (/** @type {() => void} */ cb) => {
      cb();
      return 1;
    });
    vi.stubGlobal('cancelAnimationFrame', () => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test('no-ops with a safe API when the pane is absent', () => {
    const api = initSplitter({ handle: null, pane: null, storageKey: KEY, min: 180, max: 560 });
    localStorage.setItem(KEY, '300');
    expect(() => api.applySize(300)).not.toThrow();
    expect(() => api.clearSize()).not.toThrow();
    expect(() => api.toggleCollapse()).not.toThrow();
    expect(() => api.destroy()).not.toThrow();
    expect(api.restoreSize()).toBeNull();
  });

  test('a pane with no handle still sizes — only the interaction is skipped', () => {
    // The gallery mounts its orientation toggle against a sidebar whose handle
    // may be absent; dropping size restore there would silently lose the
    // operator's persisted split.
    const { pane } = render();
    localStorage.setItem(KEY, '340');
    const api = initSplitter({ handle: null, pane, storageKey: KEY, min: 180, max: 560 });

    expect(api.restoreSize()).toBe(340);
    expect(pane.style.flexBasis).toBe('340px');
    api.clearSize();
    expect(pane.style.flexBasis).toBe('');
  });

  describe('sizing modes', () => {
    test("'flex' (the default) writes flex-basis and the max cap", () => {
      const { pane, api } = setup();
      expect(api.applySize(420)).toBe(420);
      expect(pane.style.flexBasis).toBe('420px');
      expect(pane.style.maxWidth).toBe('420px');

      expect(api.applySize(9999)).toBe(560);
      expect(pane.style.flexBasis).toBe('560px');
    });

    test("'box' writes width directly and lifts the cap", () => {
      // The settings drawer is position:fixed, so a flex-basis on it is inert;
      // and on a column flex item an inline flex-basis outranks even
      // `height: …px !important`, which would defeat a zone's .collapsed rule.
      const { pane, api } = setup({ sizing: 'box' });
      api.applySize(420);
      expect(pane.style.width).toBe('420px');
      expect(pane.style.maxWidth).toBe('none');
      expect(pane.style.flexBasis).toBe('');
    });

    test("axis 'y' sizes the height, not the width", () => {
      const { pane, api } = setup({ axis: 'y', sizing: 'box' });
      api.applySize(300);
      expect(pane.style.height).toBe('300px');
      expect(pane.style.maxHeight).toBe('none');
      expect(pane.style.width).toBe('');
    });

    test('clearSize drops the inline sizes back to the stylesheet default', () => {
      const { pane, api } = setup();
      api.applySize(420);
      api.clearSize();
      expect(pane.style.flexBasis).toBe('');
      expect(pane.style.maxWidth).toBe('');
    });
  });

  describe('restore', () => {
    test('applies a stored record, clamped', () => {
      localStorage.setItem(KEY, JSON.stringify({ size: 420, collapsed: false }));
      const { pane, api } = setup();
      expect(api.restoreSize()).toBe(420);
      expect(pane.style.flexBasis).toBe('420px');

      localStorage.setItem(KEY, JSON.stringify({ size: 5000, collapsed: false }));
      expect(api.restoreSize()).toBe(560);
    });

    test('a pre-upgrade bare integer still restores', () => {
      localStorage.setItem(KEY, '420');
      const { pane, api } = setup();
      expect(api.restoreSize()).toBe(420);
      expect(pane.style.flexBasis).toBe('420px');
    });

    test('a stored collapsed record restores collapsed, not sized', () => {
      localStorage.setItem(KEY, JSON.stringify({ size: 420, collapsed: true }));
      const { pane, api } = setup({ collapsedSize: 36, sizing: 'box' });
      api.restoreSize();
      expect(pane.style.width).toBe('36px');
      expect(api.isCollapsed()).toBe(true);
    });

    test('an absent key with defaultCollapsed restores collapsed', () => {
      // Without this, every LATTICE operator who never touched the chevron
      // silently flips from a collapsed sidebar to an expanded one.
      const { pane, api } = setup({ defaultCollapsed: true, collapsedSize: 36, sizing: 'box' });
      api.restoreSize();
      expect(api.isCollapsed()).toBe(true);
      expect(pane.style.width).toBe('36px');
    });

    test('ignores a non-numeric stored value', () => {
      localStorage.setItem(KEY, 'garbage');
      const { pane, api } = setup();
      expect(api.restoreSize()).toBeNull();
      expect(pane.style.flexBasis).toBe('');
    });
  });

  describe('collapse', () => {
    test('restores the size held before collapsing, not a default', () => {
      // The whole point: a collapse that restores to `min` or to the
      // stylesheet default reads as a broken gesture.
      const { api } = setup({ collapsedSize: 0 });
      api.applySize(420);
      api.toggleCollapse();
      expect(api.isCollapsed()).toBe(true);
      api.toggleCollapse();
      expect(api.isCollapsed()).toBe(false);
      expect(stored().size).toBe(420);
    });

    test('a double-click on the handle toggles it', () => {
      const { handle, pane, api } = setup({ collapsedSize: 0 });
      api.applySize(420);
      handle.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
      expect(pane.style.flexBasis).toBe('0px');
      handle.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
      expect(pane.style.flexBasis).toBe('420px');
    });

    test('collapse persists the pre-collapse size alongside the flag', () => {
      const { api } = setup({ collapsedSize: 0 });
      api.applySize(300);
      api.collapse();
      expect(stored()).toEqual({ size: 300, collapsed: true });
    });
  });

  describe('keyboard', () => {
    test('Arrow keys nudge along the axis and persist', () => {
      const { handle, pane } = setup();
      // happy-dom reports a 0-size rect, so the first nudge clamps up to min.
      handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
      expect(pane.style.flexBasis).toBe('180px');
      expect(stored().size).toBe(180);
    });

    test("axis 'y' listens for Up/Down, not Left/Right", () => {
      const { handle, pane } = setup({ axis: 'y', sizing: 'box' });
      handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
      expect(pane.style.height).toBe('');
      handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
      expect(pane.style.height).toBe('180px');
    });

    test('Enter and Space toggle collapse', () => {
      const { handle, api } = setup({ collapsedSize: 0 });
      api.applySize(420);
      handle.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      expect(api.isCollapsed()).toBe(true);
      handle.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
      expect(api.isCollapsed()).toBe(false);
    });
  });

  describe('pointer drag', () => {
    test('resizes live, flags .dragging, and persists on release', () => {
      const { handle, pane } = setup();

      handle.dispatchEvent(pointer('pointerdown', 0));
      expect(handle.classList.contains('dragging')).toBe(true);
      expect(pane.hasAttribute('data-splitter-dragging')).toBe(true);

      handle.dispatchEvent(pointer('pointermove', 300));
      expect(pane.style.flexBasis).toBe('300px');
      // Live drag must not thrash storage — only the release persists.
      expect(localStorage.getItem(KEY)).toBeNull();

      handle.dispatchEvent(pointer('pointerup', 300));
      expect(handle.classList.contains('dragging')).toBe(false);
      expect(pane.hasAttribute('data-splitter-dragging')).toBe(false);
      expect(stored().size).toBe(300);

      // Move handlers are detached on release: a stray move does nothing.
      handle.dispatchEvent(pointer('pointermove', 500));
      expect(pane.style.flexBasis).toBe('300px');
    });

    test("anchor 'end' inverts the delta — dragging back grows the pane", () => {
      // The settings drawer sits on the right, so dragging left widens it.
      const { handle, pane } = setup({ anchor: 'end', sizing: 'box' });
      handle.dispatchEvent(pointer('pointerdown', 0));
      handle.dispatchEvent(pointer('pointermove', -300));
      expect(pane.style.width).toBe('300px');
    });

    test('a press that never moves is a click, not a zero-distance drag', () => {
      // A double-click is two full pointerdown/up pairs BEFORE the dblclick
      // event. If those committed as drags, applySize would clear the
      // collapsed flag and overwrite the remembered expanded size, so the
      // collapse gesture could never restore.
      const { handle, api } = setup({ collapsedSize: 0 });
      api.applySize(420);
      api.collapse();
      localStorage.clear();

      handle.dispatchEvent(pointer('pointerdown', 0));
      handle.dispatchEvent(pointer('pointerup', 0));

      expect(api.isCollapsed()).toBe(true);
      expect(localStorage.getItem(KEY)).toBeNull();

      handle.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
      expect(api.isCollapsed()).toBe(false);
      expect(stored().size).toBe(420);
    });

    test('pointercancel ends the drag like a release', () => {
      const { handle } = setup();
      handle.dispatchEvent(pointer('pointerdown', 0));
      handle.dispatchEvent(pointer('pointermove', 240));
      handle.dispatchEvent(pointer('pointercancel', 240));
      expect(handle.classList.contains('dragging')).toBe(false);
      expect(stored().size).toBe(240);
    });

    test('a release before the pending frame runs still lands the final size', () => {
      // The release path must cancel the pending frame, apply the last
      // candidate synchronously, THEN drop data-splitter-dragging, THEN
      // persist. Get that order wrong and the drag persists a stale size and
      // animates its last few pixels through the very transition this
      // attribute exists to suppress.
      /** @type {Array<() => void>} */
      const queued = [];
      vi.stubGlobal('requestAnimationFrame', (/** @type {() => void} */ cb) => {
        queued.push(cb);
        return queued.length;
      });
      vi.stubGlobal('cancelAnimationFrame', () => {
        queued.length = 0;
      });

      const { handle, pane } = setup();
      handle.dispatchEvent(pointer('pointerdown', 0));
      handle.dispatchEvent(pointer('pointermove', 260));
      // Nothing applied yet — the frame has not run.
      expect(pane.style.flexBasis).toBe('');

      handle.dispatchEvent(pointer('pointerup', 260));
      expect(pane.style.flexBasis).toBe('260px');
      expect(pane.hasAttribute('data-splitter-dragging')).toBe(false);
      expect(stored().size).toBe(260);

      // The cancelled frame must not fire late and re-write geometry after the
      // suppression attribute is gone.
      expect(queued).toHaveLength(0);
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
  });

  describe('bounds', () => {
    test('a function-valued max is resolved per clamp, not captured', () => {
      // The drawer's ceiling is 90vw, which moves when the window resizes; a
      // number captured at construction would let it be dragged past the limit.
      let ceiling = 400;
      const { api } = setup({ max: () => ceiling });
      expect(api.applySize(9999)).toBe(400);
      ceiling = 700;
      expect(api.applySize(9999)).toBe(700);
    });
  });

  describe('host-owned persistence', () => {
    test('storageKey null never touches storage and reports through onCommit', () => {
      // The event dashboard keeps six layout fields plus a preset system in one
      // record; a second source of truth would let a stale splitter value win
      // over a freshly applied preset.
      const commits = /** @type {Array<[number, boolean]>} */ ([]);
      const { handle } = setup({
        storageKey: null,
        onCommit: (/** @type {number} */ s, /** @type {boolean} */ c) => commits.push([s, c]),
      });

      handle.dispatchEvent(pointer('pointerdown', 0));
      handle.dispatchEvent(pointer('pointermove', 300));
      handle.dispatchEvent(pointer('pointerup', 300));

      expect(localStorage.getItem(KEY)).toBeNull();
      expect(commits).toEqual([[300, false]]);
    });

    test('seeds from initialSize so the host owns the starting geometry', () => {
      const { api } = setup({ storageKey: null, initialSize: 240 });
      expect(api.restoreSize()).toBe(240);
    });
  });

  describe('destroy', () => {
    test('detaches interaction and does not persist an interrupted drag', () => {
      // A drawer closed mid-drag was never a deliberate choice of width, and
      // persisting it would corrupt the next restore.
      const { handle, pane, api } = setup();
      handle.dispatchEvent(pointer('pointerdown', 0));
      handle.dispatchEvent(pointer('pointermove', 300));

      api.destroy();
      expect(localStorage.getItem(KEY)).toBeNull();
      // A leftover attribute would suppress the drawer's own close animation.
      expect(pane.hasAttribute('data-splitter-dragging')).toBe(false);
      expect(handle.classList.contains('dragging')).toBe(false);

      handle.dispatchEvent(pointer('pointerdown', 0));
      expect(handle.classList.contains('dragging')).toBe(false);
    });
  });

  test('sets the ARIA separator semantics from the axis', () => {
    const { handle } = setup();
    expect(handle.getAttribute('role')).toBe('separator');
    expect(handle.getAttribute('aria-orientation')).toBe('vertical');

    const { handle: vertical } = setup({ axis: 'y' });
    expect(vertical.getAttribute('aria-orientation')).toBe('horizontal');
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
      expect(api.restoreSize()).toBeNull();
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
