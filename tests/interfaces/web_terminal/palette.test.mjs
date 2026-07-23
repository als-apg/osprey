/**
 * Unit tests for the command-palette overlay (palette.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/palette.test.mjs
 *
 * happy-dom weakly supports scrollIntoView/focus, so these tests assert only
 * CLASS/ATTRIBUTE state — never scroll position or document.activeElement.
 * scrollIntoView is stubbed to a no-op and a fake `fetchConfig` is injected so
 * no real network is hit.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply to it.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  openPalette,
  closePalette,
  isOpen,
} from '../../../src/osprey/interfaces/web_terminal/static/js/palette.js';

/** A sections fixture: three leaf dot-keys under two sections. */
const SECTIONS = {
  control_system: { type: 'epics', write_verification: true },
  ui: { theme: 'dark' },
};

/** Flush the requestAnimationFrame that adds the `.visible` class. */
function flushRaf() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
}

/** Flush pending microtasks (config-fetch resolution). */
function flushMicro() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Build a deps bundle with spied navigation callbacks and a config fetch that
 * resolves to SECTIONS. Pass `fetchConfig` to override the config flow.
 * @param {Record<string, any>} [over]
 */
function makeDeps(over = {}) {
  return {
    getHiddenPanels: () => [{ id: 'ariel', label: 'ARIEL' }],
    getVisiblePanels: () => [{ id: 'lattice', label: 'Lattice' }],
    getPresets: () => [{ name: 'Focus', panels: ['ariel'] }],
    showPanel: vi.fn(),
    focusPanel: vi.fn(),
    applyPreset: vi.fn(),
    revealSetting: vi.fn(),
    actions: [{ label: 'Restart terminal', run: vi.fn() }],
    fetchConfig: () => Promise.resolve({ sections: SECTIONS }),
    ...over,
  };
}

/** @returns {HTMLElement} */
function overlay() {
  return /** @type {HTMLElement} */ (document.querySelector('.command-palette-overlay'));
}

/** @returns {HTMLInputElement} */
function input() {
  return /** @type {HTMLInputElement} */ (document.querySelector('.command-palette-input'));
}

/** Type into the palette input and fire the input event.
 * @param {string} value */
function typeQuery(value) {
  const el = input();
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Dispatch a bubbling keydown from the input (so capture-phase nav sees it).
 * @param {string} key */
function pressKey(key) {
  input().dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

beforeEach(() => {
  closePalette();
  document.body.innerHTML = '';
  // happy-dom does not implement scrollIntoView — stub to a no-op.
  Element.prototype.scrollIntoView = () => {};
});

afterEach(() => {
  closePalette();
});

describe('open / close lifecycle', () => {
  test('open creates a visible overlay; close tears it down', async () => {
    openPalette(makeDeps());
    expect(isOpen()).toBe(true);
    expect(overlay()).toBeTruthy();
    await flushRaf();
    expect(overlay().classList.contains('visible')).toBe(true);

    closePalette();
    expect(isOpen()).toBe(false);
    expect(overlay().classList.contains('visible')).toBe(false);
    // transitionend drives node removal (no real transitions in happy-dom).
    overlay().dispatchEvent(new Event('transitionend'));
    expect(document.querySelector('.command-palette-overlay')).toBeNull();
  });

  test('opening twice does not create a second overlay', () => {
    openPalette(makeDeps());
    openPalette(makeDeps());
    expect(document.querySelectorAll('.command-palette-overlay').length).toBe(1);
  });
});

describe('filtering + highlight', () => {
  test('a matching query renders items with highlight spans', async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('control');
    const items = document.querySelectorAll('.command-palette-item');
    expect(items.length).toBeGreaterThan(0);
    expect(document.querySelectorAll('.command-palette-match').length).toBeGreaterThan(0);
  });

  test('a non-matching query renders the empty state', async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('zzzzznope');
    expect(document.querySelector('.command-palette-empty')).toBeTruthy();
    expect(document.querySelectorAll('.command-palette-item').length).toBe(0);
  });

  test('group order Settings -> Panels -> Layouts -> Actions is preserved', async () => {
    openPalette(makeDeps());
    await flushMicro();
    typeQuery('');
    const headings = [...document.querySelectorAll('.command-palette-group-heading')].map(
      (h) => h.textContent,
    );
    expect(headings).toEqual(['Settings', 'Panels', 'Layouts', 'Actions']);
  });
});

describe('keyboard navigation', () => {
  test('Arrow keys move the active row and skip status rows', async () => {
    // Loading state: a status decoration coexists with panel/layout/action rows.
    /** @type {(value: any) => void} */
    let resolveConfig = () => {};
    const pending = new Promise((r) => {
      resolveConfig = r;
    });
    openPalette(makeDeps({ fetchConfig: () => pending }));
    typeQuery('');

    const status = /** @type {HTMLElement} */ (document.querySelector('.command-palette-status'));
    expect(status).toBeTruthy();
    expect(status.getAttribute('role')).toBeNull();

    const options = () => [...document.querySelectorAll('[role="option"]')];
    expect(options().length).toBeGreaterThan(0);

    pressKey('ArrowDown');
    const active1 = /** @type {HTMLElement} */ (document.querySelector('.command-palette-item--active'));
    expect(active1).toBeTruthy();
    expect(active1.getAttribute('aria-selected')).toBe('true');
    // aria-activedescendant always points at a real option id (never the status row).
    const ad1 = input().getAttribute('aria-activedescendant');
    expect(options().some((o) => o.id === ad1)).toBe(true);

    pressKey('ArrowUp');
    const ad2 = input().getAttribute('aria-activedescendant');
    expect(options().some((o) => o.id === ad2)).toBe(true);
    expect(document.querySelectorAll('.command-palette-item--active').length).toBe(1);

    resolveConfig({ sections: SECTIONS });
  });

  test('Enter runs the active item and closes', async () => {
    const run = vi.fn();
    openPalette(makeDeps({ actions: [{ label: 'Restart terminal', run }] }));
    await flushMicro();
    typeQuery('restart terminal');
    pressKey('Enter');
    expect(run).toHaveBeenCalledTimes(1);
    expect(isOpen()).toBe(false);
  });

  test('Escape closes and is handled in capture phase (stops propagation)', async () => {
    const bubbleSpy = vi.fn();
    document.addEventListener('keydown', bubbleSpy); // bubble-phase document listener
    try {
      openPalette(makeDeps());
      await flushMicro();
      pressKey('Escape');
      expect(isOpen()).toBe(false);
      // Capture-phase palette handler stopped propagation before the bubble listener.
      expect(bubbleSpy).not.toHaveBeenCalled();
    } finally {
      document.removeEventListener('keydown', bubbleSpy);
    }
  });
});

describe('concurrent config fetch', () => {
  test('pending -> loading status; resolved -> settings items appear', async () => {
    /** @type {(value: any) => void} */
    let resolveConfig = () => {};
    const pending = new Promise((r) => {
      resolveConfig = r;
    });
    openPalette(makeDeps({ fetchConfig: () => pending }));
    typeQuery('control');

    // While pending, the Settings group shows the loading decoration.
    const status = /** @type {HTMLElement} */ (document.querySelector('.command-palette-status'));
    expect(status).toBeTruthy();
    expect(status.textContent).toMatch(/loading/i);

    resolveConfig({ sections: SECTIONS });
    await flushMicro();

    // Status row gone; real settings items now match the query.
    expect(document.querySelector('.command-palette-status')).toBeNull();
    const labels = [...document.querySelectorAll('.command-palette-item-label')].map(
      (l) => l.textContent,
    );
    expect(labels.some((t) => t.includes('control_system'))).toBe(true);
  });

  test('a rejecting fetch yields the unavailable status row', async () => {
    openPalette(makeDeps({ fetchConfig: () => Promise.reject(new Error('boom')) }));
    typeQuery('control');
    await flushMicro();
    const status = /** @type {HTMLElement} */ (document.querySelector('.command-palette-status'));
    expect(status).toBeTruthy();
    expect(status.textContent).toMatch(/unavailable/i);
  });
});

describe('welcome-overlay guard', () => {
  test('openPalette is a no-op when a visible #welcome-overlay exists', () => {
    const welcome = document.createElement('div');
    welcome.id = 'welcome-overlay';
    document.body.appendChild(welcome);

    openPalette(makeDeps());
    expect(isOpen()).toBe(false);
    expect(document.querySelector('.command-palette-overlay')).toBeNull();
  });

  test('a hidden welcome overlay does not block opening', () => {
    const welcome = document.createElement('div');
    welcome.id = 'welcome-overlay';
    welcome.classList.add('hidden');
    document.body.appendChild(welcome);

    openPalette(makeDeps());
    expect(isOpen()).toBe(true);
  });
});
