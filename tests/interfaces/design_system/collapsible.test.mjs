// @ts-check
/**
 * Unit tests for the shared collapsible region
 * (design_system/static/js/collapsible.js):
 *   npx vitest run tests/interfaces/design_system/collapsible.test.mjs
 *
 * collapsible.js is the WRITER of the key sidebar-boot.js reads before first
 * paint, so the storage assertions pin the exact scoped key both derive.
 */

import { test, expect, describe, beforeEach, afterEach } from 'vitest';

import { initCollapsible } from '/design-system/js/collapsible.js';

const ATTR = 'data-sidebar';
const SCOPE_ATTR = 'data-osprey-storage-scope';
const KEY = 'osprey-sidebar';

/** @returns {HTMLButtonElement} */
const toggleButton = () => /** @type {HTMLButtonElement} */ (document.getElementById('toggle'));

/** @param {Partial<KeyboardEventInit>} init @returns {KeyboardEvent} */
function pressBackslash(init) {
  const event = new KeyboardEvent('keydown', { key: '\\', bubbles: true, cancelable: true, ...init });
  document.body.dispatchEvent(event);
  return event;
}

/** @param {string} platform */
function setPlatform(platform) {
  Object.defineProperty(navigator, 'platform', { value: platform, configurable: true });
}

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute(ATTR);
  document.documentElement.removeAttribute(SCOPE_ATTR);
  document.body.innerHTML = '<button id="toggle" type="button"></button>';
  setPlatform('Linux x86_64');
});

afterEach(() => {
  setPlatform('');
});

describe('initCollapsible', () => {
  test('syncs the toggle to the expanded state on init', () => {
    initCollapsible({ storageKey: KEY, toggle: toggleButton() });

    expect(toggleButton().getAttribute('aria-pressed')).toBe('false');
    expect(toggleButton().getAttribute('aria-label')).toBe('Hide sidebar labels');
    expect(toggleButton().title).toBe('Hide labels (Ctrl+\\)');
  });

  test('picks up a state sidebar-boot.js stamped before init', () => {
    document.documentElement.setAttribute(ATTR, 'collapsed');

    const region = initCollapsible({ storageKey: KEY, toggle: toggleButton() });

    expect(region.isCollapsed()).toBe(true);
    expect(toggleButton().getAttribute('aria-pressed')).toBe('true');
  });

  test('a toggle click flips the attribute, the storage and the button', () => {
    const region = initCollapsible({ storageKey: KEY, toggle: toggleButton() });

    toggleButton().click();

    expect(document.documentElement.getAttribute(ATTR)).toBe('collapsed');
    expect(region.isCollapsed()).toBe(true);
    expect(window.localStorage.getItem(KEY)).toBe('collapsed');
    expect(toggleButton().getAttribute('aria-pressed')).toBe('true');
    expect(toggleButton().getAttribute('aria-label')).toBe('Show sidebar labels');
    expect(toggleButton().title).toBe('Show labels (Ctrl+\\)');

    toggleButton().click();

    expect(document.documentElement.getAttribute(ATTR)).toBe('expanded');
    expect(window.localStorage.getItem(KEY)).toBe('expanded');
    expect(toggleButton().getAttribute('aria-pressed')).toBe('false');
  });

  test('the title names ⌘ on macOS', () => {
    setPlatform('MacIntel');

    initCollapsible({ storageKey: KEY, toggle: toggleButton() });

    expect(toggleButton().title).toBe('Hide labels (⌘\\)');
  });

  test('a scoped page writes only its own key', () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    const region = initCollapsible({ storageKey: KEY, toggle: toggleButton() });

    region.set('collapsed');

    expect(window.localStorage.getItem(`${KEY}--bob`)).toBe('collapsed');
    expect(window.localStorage.getItem(KEY)).toBeNull();
  });

  test('a custom attribute holds the state', () => {
    const region = initCollapsible({ attribute: 'data-drawer', storageKey: 'osprey-drawer' });

    region.toggle();

    expect(document.documentElement.getAttribute('data-drawer')).toBe('collapsed');
    expect(document.documentElement.hasAttribute(ATTR)).toBe(false);
    document.documentElement.removeAttribute('data-drawer');
  });

  test('set() rejects an unknown state and init requires a storage key', () => {
    const region = initCollapsible({ storageKey: KEY });

    expect(() => region.set(/** @type {any} */ ('half'))).toThrow(TypeError);
    expect(() => initCollapsible(/** @type {any} */ ({}))).toThrow(TypeError);
  });
});

describe('the Cmd/Ctrl+\\ shortcut', () => {
  test('Ctrl+\\ and Cmd+\\ toggle and prevent the default', () => {
    const region = initCollapsible({ storageKey: KEY, toggle: toggleButton() });

    const ctrl = pressBackslash({ ctrlKey: true });
    expect(ctrl.defaultPrevented).toBe(true);
    expect(region.isCollapsed()).toBe(true);

    pressBackslash({ metaKey: true });
    expect(region.isCollapsed()).toBe(false);
  });

  test('re-initialising rebinds the shortcut instead of toggling twice', () => {
    initCollapsible({ storageKey: KEY });
    const region = initCollapsible({ storageKey: KEY });

    pressBackslash({ ctrlKey: true });

    expect(region.isCollapsed()).toBe(true);
  });

  test('a bare \\, an extra modifier or a consumed event does nothing', () => {
    const region = initCollapsible({ storageKey: KEY });

    pressBackslash({});
    pressBackslash({ ctrlKey: true, shiftKey: true });
    pressBackslash({ ctrlKey: true, altKey: true });
    const consumed = new KeyboardEvent('keydown', { key: '\\', ctrlKey: true, bubbles: true, cancelable: true });
    consumed.preventDefault();
    document.body.dispatchEvent(consumed);

    expect(region.isCollapsed()).toBe(false);
  });
});

describe('the change event', () => {
  test('fires on document with the attribute and the new state, once per change', () => {
    const region = initCollapsible({ storageKey: KEY });
    /** @type {any[]} */
    const seen = [];
    const listener = (/** @type {Event} */ event) => seen.push(/** @type {CustomEvent} */ (event).detail);
    document.addEventListener('osprey-collapsible-change', listener);

    region.set('collapsed');
    region.set('collapsed');
    region.set('expanded');

    document.removeEventListener('osprey-collapsible-change', listener);
    expect(seen).toEqual([
      { attribute: ATTR, state: 'collapsed' },
      { attribute: ATTR, state: 'expanded' },
    ]);
  });
});
