/**
 * Unit tests for mode-boot.js's pre-paint resolution ladder:
 *   1. ?mode= URL query param
 *   2. localStorage['osprey-ui-mode']
 *   3. the data-ui-mode attribute the server already rendered on <html>
 *   4. 'expert' (the default)
 * with invalid values at any rung falling through to the next, and a
 * no-clobber guard that leaves an already-correct server attribute
 * untouched.
 *
 * mode-boot.js is a non-module, dependency-free IIFE (a sibling of the
 * generated theme-boot.js) — it exports nothing and runs on load — so
 * rather than importing it, each scenario sets up
 * window.location/localStorage/data-ui-mode and then re-executes the exact
 * on-disk source against those happy-dom globals, via the shared
 * ./js/boot-harness.mjs. Pure DOM/logic guard, happy-dom environment
 * (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/design_system/mode-boot.test.mjs
 *
 * The per-persona scoping of the storage rung is pinned separately, in
 * ./js/mode-boot-scope.test.mjs.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import { runBootScript } from './js/boot-harness.mjs';

/** Execute the pre-paint boot IIFE against the current happy-dom globals. */
function runBoot() {
  runBootScript('mode-boot.js');
}

/** @param {string} search e.g. '' or '?mode=simple' */
function setSearch(search) {
  window.history.replaceState({}, '', `/${search}`);
}

function currentMode() {
  return document.documentElement.getAttribute('data-ui-mode');
}

describe('mode-boot.js resolution ladder', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-ui-mode');
    setSearch('');
  });

  describe('rung 1 — ?mode= query param', () => {
    test('a valid ?mode= wins over everything below it', () => {
      window.localStorage.setItem('osprey-ui-mode', 'expert');
      document.documentElement.setAttribute('data-ui-mode', 'expert');
      setSearch('?mode=simple');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test('an invalid ?mode= is ignored and resolution falls through', () => {
      window.localStorage.setItem('osprey-ui-mode', 'simple');
      setSearch('?mode=bogus');

      runBoot();

      expect(currentMode()).toBe('simple');
    });
  });

  describe('rung 2 — localStorage', () => {
    test('a valid stored mode applies when no query param is present', () => {
      window.localStorage.setItem('osprey-ui-mode', 'simple');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test('stored mode outranks a differing server attribute', () => {
      window.localStorage.setItem('osprey-ui-mode', 'simple');
      document.documentElement.setAttribute('data-ui-mode', 'expert');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test('an invalid stored mode is ignored and resolution falls through to the server attr', () => {
      window.localStorage.setItem('osprey-ui-mode', 'bogus');
      document.documentElement.setAttribute('data-ui-mode', 'simple');

      runBoot();

      expect(currentMode()).toBe('simple');
    });
  });

  describe('rung 3 — server-rendered data-ui-mode', () => {
    test('a valid server attribute applies when no query/storage rung matches', () => {
      document.documentElement.setAttribute('data-ui-mode', 'simple');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test('an invalid server attribute falls through to the default', () => {
      document.documentElement.setAttribute('data-ui-mode', 'bogus');

      runBoot();

      expect(currentMode()).toBe('expert');
    });
  });

  describe('rung 4 — default', () => {
    test("resolves to 'expert' when no rung yields a valid mode", () => {
      runBoot();

      expect(currentMode()).toBe('expert');
    });

    test("stamps the default even when the page rendered no data-ui-mode (artifacts)", () => {
      // artifacts/static/index.html renders no data-ui-mode; boot must
      // still stamp the attribute so mode-scoped CSS has something to match.
      expect(document.documentElement.hasAttribute('data-ui-mode')).toBe(false);

      runBoot();

      expect(currentMode()).toBe('expert');
    });
  });

  describe('no-clobber', () => {
    test('an already-correct server attribute is not rewritten', () => {
      document.documentElement.setAttribute('data-ui-mode', 'simple');
      const spy = vi.spyOn(document.documentElement, 'setAttribute');

      runBoot();

      expect(spy).not.toHaveBeenCalled();
      expect(currentMode()).toBe('simple');
      spy.mockRestore();
    });

    test('a differing server attribute IS rewritten to the resolved mode', () => {
      document.documentElement.setAttribute('data-ui-mode', 'expert');
      setSearch('?mode=simple');
      const spy = vi.spyOn(document.documentElement, 'setAttribute');

      runBoot();

      expect(spy).toHaveBeenCalledWith('data-ui-mode', 'simple');
      expect(currentMode()).toBe('simple');
      spy.mockRestore();
    });
  });
});
