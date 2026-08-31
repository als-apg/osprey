/**
 * Unit tests for theme-boot.js's per-persona scoping of the storage rungs.
 *
 * The full ladder (?theme= > stored {family,mode} JSON > legacy bare token >
 * server data-theme > 'auto') is pinned in
 * tests/interfaces/design_system/theme-boot.test.mjs. This file pins only what
 * the scope attribute changes about the two storage rungs:
 *
 *   - a scoped page reads `osprey-theme--<scope>`
 *   - a scoped page NEVER falls back to the bare `osprey-theme`, in either
 *     storage format — that shared origin-wide slot is the cross-persona bug
 *     the scoping exists to escape, and consulting it would hand the last
 *     writer's theme to every persona who has not yet picked one
 *   - with no attribute, the bare key is honoured exactly as before, legacy
 *     bare token included
 *
 * theme-boot.js is generated (generator/emit_js.py) as a non-module,
 * dependency-free IIFE, so each scenario arranges the DOM/storage and re-runs
 * the on-disk source through the shared boot harness.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/theme-boot-scope.test.mjs
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import { runBootScript } from './boot-harness.mjs';

const SCOPE_ATTR = 'data-osprey-storage-scope';
const LEGACY_KEY = 'osprey-theme';

function runBoot() {
  runBootScript('theme-boot.js');
}

function currentTheme() {
  return document.documentElement.getAttribute('data-theme');
}

/**
 * Force the OS preference so every 'auto' fall-through resolves to a known id
 * rather than to whatever the host machine happens to prefer.
 * @param {boolean} prefersDark
 */
function stubOSPreference(prefersDark) {
  vi.stubGlobal('matchMedia', (/** @type {string} */ query) => ({
    matches: query.includes('dark') ? prefersDark : !prefersDark,
    media: query,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
  }));
}

/** @param {string} key @param {{family: string, mode: string}|string} preference */
function store(key, preference) {
  window.localStorage.setItem(
    key,
    typeof preference === 'string' ? preference : JSON.stringify(preference)
  );
}

describe('theme-boot.js storage scoping', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    document.documentElement.removeAttribute(SCOPE_ATTR);
    window.history.replaceState({}, '', '/');
    stubOSPreference(true);
  });

  describe('a polluted legacy key does not leak across personas', () => {
    test("bob boots his server theme, not the shared slot's retro", () => {
      // The origin-wide slot whoever picked last wrote.
      store(LEGACY_KEY, { family: 'retro', mode: 'dark' });
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-theme', 'light');

      runBoot();

      expect(currentTheme()).toBe('light');
    });

    test('carol, same polluted slot, boots HER server theme', () => {
      store(LEGACY_KEY, { family: 'retro', mode: 'dark' });
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      document.documentElement.setAttribute('data-theme', 'desy-dark');

      runBoot();

      expect(currentTheme()).toBe('desy-dark');
    });

    test('a polluted legacy BARE token is ignored on a scoped page too', () => {
      // The legacy bare-token rung is part of the same shared slot; scoping
      // has to close both storage rungs, not just the structured one.
      store(LEGACY_KEY, 'retro-light');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-theme', 'light');

      runBoot();

      expect(currentTheme()).toBe('light');
    });

    test('with no server rung either, a scoped page falls through to auto', () => {
      // The point of "no legacy fallback": absent a scoped value there is no
      // stored preference at all, so resolution continues down the ladder to
      // 'auto' within DEFAULT_FAMILY rather than borrowing the last writer's.
      store(LEGACY_KEY, { family: 'retro', mode: 'light' });
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

      runBoot();

      expect(currentTheme()).toBe('dark');
    });
  });

  describe('a scoped page reads its own key', () => {
    test("bob's scoped preference applies over a differing server attribute", () => {
      store(`${LEGACY_KEY}--bob`, { family: 'retro', mode: 'dark' });
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-theme', 'light');

      runBoot();

      expect(currentTheme()).toBe('retro-dark');
    });

    test('a scoped legacy bare token is honoured under the scoped key', () => {
      store(`${LEGACY_KEY}--bob`, 'high-contrast-light');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-theme', 'light');

      runBoot();

      expect(currentTheme()).toBe('high-contrast-light');
    });

    test("carol's key is not bob's — one persona's pick leaves the other alone", () => {
      store(`${LEGACY_KEY}--bob`, { family: 'retro', mode: 'dark' });
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      document.documentElement.setAttribute('data-theme', 'light');

      runBoot();

      expect(currentTheme()).toBe('light');
    });

    test('an unparseable scoped value still falls through to the server rung', () => {
      store(`${LEGACY_KEY}--bob`, 'not-a-theme');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-theme', 'desy-light');

      runBoot();

      expect(currentTheme()).toBe('desy-light');
    });

    test('?theme= still outranks the scoped storage rung', () => {
      store(`${LEGACY_KEY}--bob`, { family: 'retro', mode: 'dark' });
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      window.history.replaceState({}, '', '/?theme=light');

      runBoot();

      expect(currentTheme()).toBe('light');
    });
  });

  describe('unscoped serving is unchanged', () => {
    test('with no attribute the legacy structured preference is honoured', () => {
      store(LEGACY_KEY, { family: 'retro', mode: 'dark' });

      runBoot();

      expect(currentTheme()).toBe('retro-dark');
    });

    test('with no attribute the legacy bare token is honoured', () => {
      store(LEGACY_KEY, 'high-contrast-light');

      runBoot();

      expect(currentTheme()).toBe('high-contrast-light');
    });

    test('an unscoped page ignores a scoped key left behind by a scoped visit', () => {
      store(`${LEGACY_KEY}--bob`, { family: 'retro', mode: 'dark' });
      document.documentElement.setAttribute('data-theme', 'light');

      runBoot();

      expect(currentTheme()).toBe('light');
    });

    test('an empty attribute value is treated as unscoped, not as scope ""', () => {
      // The server omits the attribute rather than emitting `=""`; this pins
      // the boot script's inline guard against minting `osprey-theme--`.
      store(LEGACY_KEY, { family: 'retro', mode: 'dark' });
      document.documentElement.setAttribute(SCOPE_ATTR, '');

      runBoot();

      expect(currentTheme()).toBe('retro-dark');
    });
  });
});
