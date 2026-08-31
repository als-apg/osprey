/**
 * Writer-side counterpart to mode-boot-scope.test.mjs: <osprey-display-menu>'s
 * View row must persist the pick under the SAME per-persona key mode-boot.js
 * reads. A reader that scopes and a writer that does not would be worse than
 * neither — the pick would land in the shared slot and then never be read back.
 *
 * Kept out of display-menu-component.test.mjs because that file's beforeEach
 * establishes an unscoped document, which is exactly the baseline its other
 * assertions depend on.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/display-menu-scope.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { qs } from '../../_support/dom.mjs';
import { runBootScript } from './boot-harness.mjs';

import * as ThemeManager from '/design-system/js/theme-manager.js';
import '/design-system/js/components/osprey-display-menu.js';

const SCOPE_ATTR = 'data-osprey-storage-scope';
const LEGACY_KEY = 'osprey-ui-mode';

function mount() {
  const el = document.createElement('osprey-display-menu');
  document.body.appendChild(el);
  return /** @type {HTMLElement} */ (el);
}

/** @param {HTMLElement} el @param {'expert'|'simple'} mode */
function pickView(el, mode) {
  qs(el, `.display-menu-view .display-seg-option[data-mode="${mode}"]`).click();
}

describe('<osprey-display-menu> View row storage scoping', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute('data-ui-mode');
    document.documentElement.removeAttribute(SCOPE_ATTR);
    document.body.innerHTML = '';
    document.body.className = '';
    window.history.replaceState({}, '', '/');
    ThemeManager.initTheme({ role: 'hub' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = '';
    document.body.className = '';
  });

  test('a scoped page writes the scoped key and leaves the shared slot alone', () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    const el = mount();

    pickView(el, 'simple');

    expect(window.localStorage.getItem(`${LEGACY_KEY}--bob`)).toBe('simple');
    expect(window.localStorage.getItem(LEGACY_KEY)).toBeNull();
  });

  test("one persona's pick does not overwrite another's", () => {
    window.localStorage.setItem(`${LEGACY_KEY}--carol`, 'simple');
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    const el = mount();

    pickView(el, 'expert');

    expect(window.localStorage.getItem(`${LEGACY_KEY}--bob`)).toBe('expert');
    expect(window.localStorage.getItem(`${LEGACY_KEY}--carol`)).toBe('simple');
  });

  test('an existing polluted shared slot is not repaired, and not read back either', () => {
    // Writing the scoped key deliberately leaves the legacy value in place:
    // other, still-unscoped surfaces on this origin may depend on it, and no
    // scoped reader will ever consult it.
    window.localStorage.setItem(LEGACY_KEY, 'simple');
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    const el = mount();

    pickView(el, 'expert');

    expect(window.localStorage.getItem(LEGACY_KEY)).toBe('simple');
    expect(window.localStorage.getItem(`${LEGACY_KEY}--bob`)).toBe('expert');
  });

  test('an unscoped page still writes the bare legacy key', () => {
    const el = mount();

    pickView(el, 'simple');

    expect(window.localStorage.getItem(LEGACY_KEY)).toBe('simple');
  });

  test('the scope is read at write time, so it tracks the document it lands in', () => {
    const el = mount();
    pickView(el, 'simple');
    expect(window.localStorage.getItem(LEGACY_KEY)).toBe('simple');

    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    pickView(el, 'expert');

    expect(window.localStorage.getItem(`${LEGACY_KEY}--bob`)).toBe('expert');
    // The earlier unscoped write is untouched by the scoped one.
    expect(window.localStorage.getItem(LEGACY_KEY)).toBe('simple');
  });

  // The writer and mode-boot.js each spell the scoped key inline. The two
  // key-name pins above would both stay green if the spelling drifted in step;
  // a real write-then-boot round trip fails whatever the literal is.
  test("a pick under one scope is what mode-boot.js reads back on that persona's next load", () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    const el = mount();
    pickView(el, 'simple');

    document.documentElement.setAttribute('data-ui-mode', 'expert');
    runBootScript('mode-boot.js');

    expect(document.documentElement.getAttribute('data-ui-mode')).toBe('simple');
  });

  test("a pick under one scope does not reach another persona's boot", () => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
    const el = mount();
    pickView(el, 'simple');

    document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    runBootScript('mode-boot.js');

    expect(document.documentElement.getAttribute('data-ui-mode')).toBe('expert');
  });
});
