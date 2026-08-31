/**
 * Unit tests for rail-boot.js's per-persona scoping of the storage rung.
 *
 * The ladder itself (?rail= > storage > server data-rail-position > 'left') is
 * pinned in tests/interfaces/design_system/rail-boot.test.mjs. This file pins
 * only what the scope attribute changes about rung 2, exactly as
 * mode-boot-scope.test.mjs does for the ui-mode axis:
 *
 *   - scoped pages read `osprey-rail-position--<scope>`
 *   - a scoped page NEVER falls back to the bare `osprey-rail-position`, even
 *     when it holds a valid position — that shared slot is the cross-persona
 *     bug the scoping exists to escape
 *   - with no attribute, the bare key is honoured exactly as before
 *
 * rail-boot.js is the READER half of a pair: rail-position.js (web_terminal)
 * writes the key this script reads, and the two must derive byte-identical
 * keys for the same scope. The writer's half is pinned in
 * tests/interfaces/web_terminal/js/storage-scope-keys.test.js.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/rail-boot-scope.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { runBootScript } from './boot-harness.mjs';

const SCOPE_ATTR = 'data-osprey-storage-scope';
const LEGACY_KEY = 'osprey-rail-position';

function runBoot() {
  runBootScript('rail-boot.js');
}

function currentPosition() {
  return document.documentElement.getAttribute('data-rail-position');
}

describe('rail-boot.js storage scoping', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-rail-position');
    document.documentElement.removeAttribute(SCOPE_ATTR);
    window.history.replaceState({}, '', '/');
  });

  describe('a polluted legacy key does not leak across personas', () => {
    test("bob boots his server rail, not the shared slot's 'top'", () => {
      // The shared origin-wide slot whichever persona flipped the rail last.
      window.localStorage.setItem(LEGACY_KEY, 'top');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-rail-position', 'left');

      runBoot();

      expect(currentPosition()).toBe('left');
    });

    test('carol, same polluted slot, boots HER server rail', () => {
      window.localStorage.setItem(LEGACY_KEY, 'left');
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      document.documentElement.setAttribute('data-rail-position', 'top');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test('with no server rung either, a scoped page falls through to the default', () => {
      window.localStorage.setItem(LEGACY_KEY, 'top');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

      runBoot();

      expect(currentPosition()).toBe('left');
    });
  });

  describe('a scoped page reads its own key', () => {
    test("bob's scoped 'top' applies over a differing server attribute", () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'top');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-rail-position', 'left');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test("carol's key is not bob's — one persona's pick leaves the other alone", () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'top');
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      document.documentElement.setAttribute('data-rail-position', 'left');

      runBoot();

      expect(currentPosition()).toBe('left');
    });

    test('an invalid scoped value still falls through to the server rung', () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'bogus');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      document.documentElement.setAttribute('data-rail-position', 'top');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test('?rail= still outranks the scoped storage rung', () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'top');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      window.history.replaceState({}, '', '/?rail=left');

      runBoot();

      expect(currentPosition()).toBe('left');
    });
  });

  describe('unscoped serving is unchanged', () => {
    test('with no attribute the legacy key is honoured as before', () => {
      window.localStorage.setItem(LEGACY_KEY, 'top');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test('an unscoped page ignores a scoped key left behind by a scoped visit', () => {
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'top');
      document.documentElement.setAttribute('data-rail-position', 'left');

      runBoot();

      expect(currentPosition()).toBe('left');
    });

    test('an empty attribute value is treated as unscoped, not as scope ""', () => {
      // The server omits the attribute rather than emitting `=""`; this pins
      // the boot script's inline guard against minting `osprey-rail-position--`.
      window.localStorage.setItem(LEGACY_KEY, 'top');
      document.documentElement.setAttribute(SCOPE_ATTR, '');

      runBoot();

      expect(currentPosition()).toBe('top');
    });
  });
});
