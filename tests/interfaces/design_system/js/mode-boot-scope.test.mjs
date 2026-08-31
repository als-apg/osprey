/**
 * Unit tests for mode-boot.js's per-persona scoping of the storage rung.
 *
 * The ladder itself (?mode= > storage > server data-ui-mode > 'expert') is
 * pinned in tests/interfaces/design_system/mode-boot.test.mjs. This file pins
 * only what the scope attribute changes about rung 2:
 *
 *   - scoped pages read `osprey-ui-mode--<scope>`
 *   - a scoped page NEVER falls back to the bare `osprey-ui-mode`, even when it
 *     holds a valid mode — that shared slot is the cross-persona bug the
 *     scoping exists to escape, and consulting it would re-inflict the bug once
 *     per persona
 *   - with no attribute, the bare key is honoured exactly as before
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/mode-boot-scope.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { runBootScript } from './boot-harness.mjs';

const SCOPE_ATTR = 'data-osprey-storage-scope';
const LEGACY_KEY = 'osprey-ui-mode';

function runBoot() {
  runBootScript('mode-boot.js');
}

function currentMode() {
  return document.documentElement.getAttribute('data-ui-mode');
}

describe('mode-boot.js storage scoping', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-ui-mode');
    document.documentElement.removeAttribute(SCOPE_ATTR);
    window.history.replaceState({}, '', '/');
  });

  describe('a polluted legacy key does not leak across personas', () => {
    test("bob boots his server mode, not the shared slot's 'simple'", () => {
      // The shared origin-wide slot someone else wrote last.
      window.localStorage.setItem(LEGACY_KEY, 'simple');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-ui-mode', 'expert');

      runBoot();

      expect(currentMode()).toBe('expert');
    });

    test("carol, same polluted slot, boots HER server mode", () => {
      window.localStorage.setItem(LEGACY_KEY, 'expert');
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      document.documentElement.setAttribute('data-ui-mode', 'simple');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test('with no server rung either, a scoped page falls through to the default', () => {
      // The point of "no legacy fallback": absent a scoped value there is no
      // stored preference at all, so resolution continues down the ladder
      // rather than borrowing the last writer's.
      window.localStorage.setItem(LEGACY_KEY, 'simple');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

      runBoot();

      expect(currentMode()).toBe('expert');
    });
  });

  describe('a scoped page reads its own key', () => {
    test("bob's scoped 'simple' applies over a differing server attribute", () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'simple');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-ui-mode', 'expert');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test("carol's key is not bob's — one persona's pick leaves the other alone", () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'simple');
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      document.documentElement.setAttribute('data-ui-mode', 'expert');

      runBoot();

      expect(currentMode()).toBe('expert');
    });

    test('an invalid scoped value still falls through to the server rung', () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'bogus');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-ui-mode', 'simple');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test('?mode= still outranks the scoped storage rung', () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'simple');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      window.history.replaceState({}, '', '/?mode=expert');

      runBoot();

      expect(currentMode()).toBe('expert');
    });
  });

  describe('unscoped serving is unchanged', () => {
    test('with no attribute the legacy key is honoured as before', () => {
      window.localStorage.setItem(LEGACY_KEY, 'simple');

      runBoot();

      expect(currentMode()).toBe('simple');
    });

    test('an unscoped page ignores a scoped key left behind by a scoped visit', () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'simple');
      document.documentElement.setAttribute('data-ui-mode', 'expert');

      runBoot();

      expect(currentMode()).toBe('expert');
    });

    test('an empty attribute value is treated as unscoped, not as scope ""', () => {
      // The server omits the attribute rather than emitting `=""`; this pins
      // the boot script's inline guard against minting `osprey-ui-mode--`.
      window.localStorage.setItem(LEGACY_KEY, 'simple');
      document.documentElement.setAttribute(SCOPE_ATTR, '');

      runBoot();

      expect(currentMode()).toBe('simple');
    });
  });
});
