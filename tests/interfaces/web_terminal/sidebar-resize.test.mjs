// @ts-check
/**
 * The sidebar's resize handle (sidebar-resize.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-resize.test.mjs
 *
 * The handle writes the labelled width to `--sidebar-user-width` on <html>,
 * clamps it to 180–400 px (less on a window too narrow for a 480 px
 * workspace), stores it under the scoped `osprey-sidebar-width` key, restores
 * it at boot, snaps to the collapsed state (collapsible.js, the real module)
 * past the minimum, and steps it from the keyboard. sidebar-boot.js's pre-paint
 * half of the restore is pinned at the end, against the same key.
 *
 * Module state is isolated by vi.resetModules() + dynamic import per test, and
 * each wired handle is destroyed after its test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { runBootScript } from '../design_system/js/boot-harness.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const SCOPE_ATTR = 'data-osprey-storage-scope';
const KEY = 'osprey-sidebar-width';
const PROP = '--sidebar-user-width';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-resize.js')} */
let resize;
/** @type {import('../../../src/osprey/interfaces/design_system/static/js/collapsible.js').Collapsible} */
let collapsible;
/** @type {{width: () => number, destroy: () => void} | null} */
let wired = null;

const root = document.documentElement;
const handle = () => /** @type {HTMLElement} */ (document.getElementById('sidebar-resize'));
const shownProp = () => root.style.getPropertyValue(PROP);
const collapsed = () => root.getAttribute('data-sidebar') === 'collapsed';

/** @param {number} width */
function setViewport(width) {
  Object.defineProperty(window, 'innerWidth', { value: width, configurable: true });
}

/** Render the handle and wire it the way sidebar.js does. */
async function boot() {
  document.body.innerHTML = `
    <nav class="sidebar" id="sidebar">
      <button id="sidebar-toggle" type="button"></button>
      <div class="sidebar-resize" id="sidebar-resize" role="separator" tabindex="0"></div>
    </nav>`;
  vi.resetModules();
  resize = await import(`${JS}/sidebar-resize.js`);
  const { initCollapsible } = await import('/design-system/js/collapsible.js');
  collapsible = initCollapsible({
    storageKey: 'osprey-sidebar',
    toggle: document.getElementById('sidebar-toggle'),
  });
  wired = resize.initSidebarResize({ handle: handle(), collapsible });
  if (!wired) throw new Error('initSidebarResize found no handle');
  return wired;
}

/**
 * @param {string} type
 * @param {number} clientX
 */
function pointer(type, clientX) {
  handle().dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, pointerId: 1, clientX })
  );
}

/**
 * Drag the handle from `from` through each of `path` (clientX), then release.
 * @param {number} from
 * @param {...number} path
 */
function drag(from, ...path) {
  pointer('pointerdown', from);
  for (const x of path) pointer('pointermove', x);
  pointer('pointerup', path.at(-1) ?? from);
}

/**
 * @param {string} key
 * @param {boolean} [shiftKey]
 */
function press(key, shiftKey = false) {
  handle().dispatchEvent(new KeyboardEvent('keydown', { key, shiftKey, bubbles: true }));
}

beforeEach(() => {
  window.localStorage.clear();
  root.removeAttribute('data-sidebar');
  root.removeAttribute('data-sidebar-resizing');
  root.removeAttribute(SCOPE_ATTR);
  root.style.removeProperty(PROP);
  setViewport(1280);
});

afterEach(() => {
  wired?.destroy();
  wired = null;
});

describe('clamp', () => {
  test('the width stays in 180–400 px on a wide window', async () => {
    await boot();
    expect(resize.clampWidth(100, 1600)).toBe(180);
    expect(resize.clampWidth(250.4, 1600)).toBe(250);
    expect(resize.clampWidth(900, 1600)).toBe(400);
  });

  test('a narrow window caps the width so the workspace keeps 480 px', async () => {
    await boot();
    expect(resize.maxWidth(760)).toBe(280);
    expect(resize.clampWidth(400, 760)).toBe(280);
    // Never below the minimum, however narrow.
    expect(resize.maxWidth(500)).toBe(180);
  });

  test('a window resize re-caps the shown width but keeps the chosen one', async () => {
    window.localStorage.setItem(KEY, '380');
    await boot();
    expect(shownProp()).toBe('380px');

    setViewport(700);
    window.dispatchEvent(new Event('resize'));
    expect(shownProp()).toBe('220px');
    expect(handle().getAttribute('aria-valuemax')).toBe('220');

    setViewport(1400);
    window.dispatchEvent(new Event('resize'));
    expect(shownProp()).toBe('380px');
  });
});

describe('drag and persist', () => {
  test('a drag sets the width live with the transition off, and stores it on release', async () => {
    await boot();
    expect(shownProp()).toBe('232px');

    pointer('pointerdown', 232);
    expect(root.hasAttribute('data-sidebar-resizing')).toBe(true);
    pointer('pointermove', 300);
    expect(shownProp()).toBe('300px');
    expect(handle().getAttribute('aria-valuenow')).toBe('300');
    expect(window.localStorage.getItem(KEY)).toBeNull();
    pointer('pointerup', 300);

    expect(root.hasAttribute('data-sidebar-resizing')).toBe(false);
    expect(window.localStorage.getItem(KEY)).toBe('300');
  });

  test('a drag past a limit clamps to it', async () => {
    await boot();
    drag(232, 900);
    expect(shownProp()).toBe('400px');
    drag(400, 150);
    expect(shownProp()).toBe('180px');
    expect(collapsed()).toBe(false);
  });

  test('a press that never moves stores nothing', async () => {
    await boot();
    drag(232, 232);
    expect(window.localStorage.getItem(KEY)).toBeNull();
  });

  test('the width is stored under the per-user key on a scoped page', async () => {
    root.setAttribute(SCOPE_ATTR, 'bob');
    await boot();
    drag(232, 260);
    expect(window.localStorage.getItem(`${KEY}--bob`)).toBe('260');
    expect(window.localStorage.getItem(KEY)).toBeNull();
  });
});

describe('restore', () => {
  test('a stored width is shown at boot', async () => {
    window.localStorage.setItem(KEY, '310');
    const { width } = await boot();
    expect(width()).toBe(310);
    expect(shownProp()).toBe('310px');
  });

  test.each(['abc', '12', '9000', '250.5', '-200', ''])(
    'a stored %j is ignored for the default',
    async (raw) => {
      window.localStorage.setItem(KEY, raw);
      await boot();
      expect(shownProp()).toBe('232px');
    }
  );

  test('a double-click resets to the default and forgets the stored width', async () => {
    window.localStorage.setItem(KEY, '350');
    await boot();
    handle().dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(shownProp()).toBe('232px');
    expect(window.localStorage.getItem(KEY)).toBeNull();
  });
});

describe('snap', () => {
  test('a drag well below the minimum collapses, keeping the stored width', async () => {
    window.localStorage.setItem(KEY, '300');
    await boot();
    drag(300, 200, 120);
    expect(collapsed()).toBe(true);
    expect(window.localStorage.getItem('osprey-sidebar')).toBe('collapsed');
    expect(window.localStorage.getItem(KEY)).toBe('300');
    expect(handle().getAttribute('aria-valuetext')).toBe('Icons only');
  });

  test('just below the minimum clamps rather than collapsing', async () => {
    await boot();
    drag(232, 150);
    expect(collapsed()).toBe(false);
    expect(shownProp()).toBe('180px');
  });

  test('a collapsed sidebar dragged outward expands to the stored width', async () => {
    window.localStorage.setItem(KEY, '320');
    window.localStorage.setItem('osprey-sidebar', 'collapsed');
    root.setAttribute('data-sidebar', 'collapsed');
    await boot();

    pointer('pointerdown', 56);
    pointer('pointermove', 80);
    expect(collapsed()).toBe(true);
    pointer('pointermove', 110);
    expect(collapsed()).toBe(false);
    expect(shownProp()).toBe('320px');
    // The drag goes on from the stored width, not from the pointer.
    pointer('pointermove', 130);
    expect(shownProp()).toBe('340px');
    pointer('pointerup', 130);
    expect(window.localStorage.getItem(KEY)).toBe('340');
  });
});

describe('keyboard', () => {
  test('arrows step 8 px, 32 px with Shift, and store the width', async () => {
    await boot();
    press('ArrowRight');
    expect(shownProp()).toBe('240px');
    press('ArrowLeft', true);
    expect(shownProp()).toBe('208px');
    expect(window.localStorage.getItem(KEY)).toBe('208');
  });

  test('Home and End go to the limits', async () => {
    setViewport(800);
    await boot();
    press('End');
    expect(shownProp()).toBe('320px');
    press('Home');
    expect(shownProp()).toBe('180px');
  });

  test('ArrowRight expands a collapsed sidebar to its width; ArrowLeft leaves it', async () => {
    window.localStorage.setItem(KEY, '300');
    root.setAttribute('data-sidebar', 'collapsed');
    await boot();
    press('ArrowLeft');
    expect(collapsed()).toBe(true);
    press('ArrowRight');
    expect(collapsed()).toBe(false);
    expect(shownProp()).toBe('300px');
  });

  test('the handle carries the separator value attributes', async () => {
    await boot();
    expect(handle().getAttribute('aria-valuemin')).toBe('180');
    expect(handle().getAttribute('aria-valuemax')).toBe('400');
    expect(handle().getAttribute('aria-valuenow')).toBe('232');
  });
});

describe('sidebar-boot.js restores the width before first paint', () => {
  test('a valid stored width is set, capped to the window', () => {
    window.localStorage.setItem(KEY, '350');
    runBootScript('sidebar-boot.js');
    expect(shownProp()).toBe('350px');

    root.style.removeProperty(PROP);
    setViewport(700);
    runBootScript('sidebar-boot.js');
    expect(shownProp()).toBe('220px');
  });

  test('an invalid stored width, or another user’s, sets nothing', () => {
    window.localStorage.setItem(KEY, '9000');
    runBootScript('sidebar-boot.js');
    expect(shownProp()).toBe('');

    window.localStorage.setItem(KEY, '300');
    root.setAttribute(SCOPE_ATTR, 'bob');
    runBootScript('sidebar-boot.js');
    expect(shownProp()).toBe('');
  });
});
