/**
 * Unit tests for theme-boot.js's pre-paint resolution ladder:
 *   1. ?theme= URL query param (bare token: 'auto' or a concrete id)
 *   2a. localStorage['osprey-theme'] holding the structured
 *       {"family":...,"mode":...} JSON theme-manager persists
 *   2b. the same key holding a legacy bare token ('auto' or a concrete id)
 *   3. the data-theme attribute the server already rendered on <html>
 *   4. 'auto' — the OS preference, within the server attribute's family
 *      when there is one, else DEFAULT_FAMILY
 * with unrecognized values at any rung falling through to the next, and a
 * no-clobber guard that leaves an already-correct server attribute
 * untouched.
 *
 * theme-boot.js is generated (by generator/emit_js.py) as a non-module,
 * dependency-free IIFE — it exports nothing and runs on load — so rather
 * than importing it, each scenario sets up
 * window.location/localStorage/data-theme and then re-executes the exact
 * on-disk source against those happy-dom globals, exactly as the sibling
 * mode-boot.test.mjs does. Pure DOM/logic guard, happy-dom environment
 * (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/design_system/theme-boot.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

// Resolve the on-disk source relative to this test file. `import.meta.dirname`
// is a plain string, so it sidesteps happy-dom's override of the global URL
// (which breaks `fileURLToPath(new URL(...))` under this environment).
const THEME_BOOT_PATH = join(
  import.meta.dirname,
  '../../../src/osprey/interfaces/design_system/static/js/theme-boot.js'
);
const themeBootSource = readFileSync(THEME_BOOT_PATH, 'utf8');

const STORAGE_KEY = 'osprey-theme';

/**
 * Execute the pre-paint boot IIFE against the current happy-dom globals.
 * `window`/`document` are passed as explicit parameters so the source's
 * free references bind to the test's DOM without a global `eval`.
 */
function runBoot() {
  const boot = new Function('window', 'document', themeBootSource);
  boot(globalThis.window, globalThis.document);
}

/** @param {string} search e.g. '' or '?theme=light' */
function setSearch(search) {
  window.history.replaceState({}, '', `/${search}`);
}

function currentTheme() {
  return document.documentElement.getAttribute('data-theme');
}

/** @param {{family: string, mode: string}|string} preference */
function storePreference(preference) {
  window.localStorage.setItem(
    STORAGE_KEY,
    typeof preference === 'string' ? preference : JSON.stringify(preference)
  );
}

/**
 * Force the OS preference so 'auto' resolution can be asserted in both
 * directions. Without this the assertions would pass for the wrong reason
 * whenever the host's own OS preference happened to agree.
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

describe('theme-boot.js resolution ladder', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    setSearch('');
    stubOSPreference(true);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('rung 1 — ?theme= query param', () => {
    test('a valid ?theme= wins over every storage format below it', () => {
      storePreference({ family: 'retro', mode: 'light' });
      document.documentElement.setAttribute('data-theme', 'desy-dark');
      setSearch('?theme=high-contrast-light');

      runBoot();

      expect(currentTheme()).toBe('high-contrast-light');
    });

    test('an invalid ?theme= is ignored and resolution falls through to storage', () => {
      storePreference({ family: 'retro', mode: 'light' });
      setSearch('?theme=bogus');

      runBoot();

      expect(currentTheme()).toBe('retro-light');
    });
  });

  describe('rung 2a — structured {family, mode} storage', () => {
    test("a stored dark preference resolves to that family's dark id", () => {
      storePreference({ family: 'retro', mode: 'dark' });

      runBoot();

      expect(currentTheme()).toBe('retro-dark');
    });

    test("a stored light preference resolves to that family's light id", () => {
      // Forced OS dark, so a 'light' mode that leaked into auto resolution
      // would visibly resolve to the wrong id here.
      storePreference({ family: 'main', mode: 'light' });

      runBoot();

      expect(currentTheme()).toBe('light');
    });

    test("mode 'auto' follows the OS preference within the stored family", () => {
      storePreference({ family: 'desy', mode: 'auto' });

      runBoot();

      expect(currentTheme()).toBe('desy-dark');
    });

    test("mode 'auto' resolves to the stored family's light id when the OS prefers light", () => {
      stubOSPreference(false);
      storePreference({ family: 'desy', mode: 'auto' });

      runBoot();

      expect(currentTheme()).toBe('desy-light');
    });

    test("mode 'auto' stays in the stored family even when the server pinned another", () => {
      // The stored family, not FAMILY_BY_ID[serverTheme], supplies auto's
      // family — storage outranks the server attribute.
      document.documentElement.setAttribute('data-theme', 'high-contrast-light');
      storePreference({ family: 'retro', mode: 'auto' });

      runBoot();

      expect(currentTheme()).toBe('retro-dark');
    });

    test('a stored preference outranks a differing server attribute', () => {
      document.documentElement.setAttribute('data-theme', 'desy-dark');
      storePreference({ family: 'main', mode: 'light' });

      runBoot();

      expect(currentTheme()).toBe('light');
    });
  });

  describe('rung 2b — legacy bare-token storage', () => {
    test('a bare concrete id still applies', () => {
      storePreference('light');

      runBoot();

      expect(currentTheme()).toBe('light');
    });

    test("a bare 'auto' resolves within the server attribute's family", () => {
      document.documentElement.setAttribute('data-theme', 'retro-light');
      storePreference('auto');

      runBoot();

      expect(currentTheme()).toBe('retro-dark');
    });

    test("a bare 'auto' falls back to DEFAULT_FAMILY with no server attribute", () => {
      storePreference('auto');

      runBoot();

      expect(currentTheme()).toBe('dark');
    });
  });

  describe('unrecognized storage values fall through', () => {
    test('non-JSON junk is ignored at both storage rungs', () => {
      storePreference('not-a-theme');
      document.documentElement.setAttribute('data-theme', 'desy-light');

      runBoot();

      expect(currentTheme()).toBe('desy-light');
    });

    test('JSON naming an unknown family falls through to the server attribute', () => {
      storePreference({ family: 'nonesuch', mode: 'dark' });
      document.documentElement.setAttribute('data-theme', 'desy-light');

      runBoot();

      expect(currentTheme()).toBe('desy-light');
    });

    test('JSON naming an unknown mode falls through to the server attribute', () => {
      storePreference({ family: 'retro', mode: 'sepia' });
      document.documentElement.setAttribute('data-theme', 'desy-light');

      runBoot();

      expect(currentTheme()).toBe('desy-light');
    });

    test('JSON that is not an object falls through', () => {
      window.localStorage.setItem(STORAGE_KEY, '"light"');
      document.documentElement.setAttribute('data-theme', 'desy-light');

      runBoot();

      expect(currentTheme()).toBe('desy-light');
    });
  });

  describe('rungs 3 and 4 — server attribute, then auto', () => {
    test('a valid server attribute applies when no query/storage rung matches', () => {
      document.documentElement.setAttribute('data-theme', 'high-contrast-light');

      runBoot();

      expect(currentTheme()).toBe('high-contrast-light');
    });

    test('an unknown server attribute falls through to auto in DEFAULT_FAMILY', () => {
      document.documentElement.setAttribute('data-theme', 'bogus');

      runBoot();

      expect(currentTheme()).toBe('dark');
    });

    test('a bare page with no rung at all resolves to auto in DEFAULT_FAMILY', () => {
      stubOSPreference(false);

      runBoot();

      expect(currentTheme()).toBe('light');
    });
  });

  describe('no-clobber', () => {
    test('an already-correct server attribute is not rewritten', () => {
      document.documentElement.setAttribute('data-theme', 'desy-light');
      storePreference({ family: 'desy', mode: 'light' });
      const spy = vi.spyOn(document.documentElement, 'setAttribute');

      runBoot();

      expect(spy).not.toHaveBeenCalled();
      expect(currentTheme()).toBe('desy-light');
      spy.mockRestore();
    });

    test('a differing server attribute IS rewritten to the resolved theme', () => {
      document.documentElement.setAttribute('data-theme', 'desy-light');
      storePreference({ family: 'retro', mode: 'dark' });
      const spy = vi.spyOn(document.documentElement, 'setAttribute');

      runBoot();

      expect(spy).toHaveBeenCalledWith('data-theme', 'retro-dark');
      expect(currentTheme()).toBe('retro-dark');
      spy.mockRestore();
    });
  });
});
