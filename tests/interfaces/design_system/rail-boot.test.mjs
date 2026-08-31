/**
 * Unit tests for rail-boot.js's pre-paint resolution ladder:
 *   1. ?rail= URL query param
 *   2. localStorage['osprey-rail-position']
 *   3. the data-rail-position attribute the server already rendered on <html>
 *   4. 'left' (the default)
 * with invalid values at any rung falling through to the next, and a
 * no-clobber guard that leaves an already-correct server attribute
 * untouched.
 *
 * rail-boot.js is a non-module, dependency-free IIFE (a sibling of
 * mode-boot.js) — it exports nothing and runs on load — so rather than
 * importing it, each scenario sets up window.location/localStorage/
 * data-rail-position and then re-executes the exact on-disk source against
 * those happy-dom globals. Pure DOM/logic guard, happy-dom environment
 * (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/design_system/rail-boot.test.mjs
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';
import { runBootScript } from './js/boot-harness.mjs';

/** Re-execute the on-disk rail-boot.js against the current happy-dom globals. */
const runBoot = () => runBootScript('rail-boot.js');

/** @param {string} search e.g. '' or '?rail=top' */
function setSearch(search) {
  window.history.replaceState({}, '', `/${search}`);
}

function currentPosition() {
  return document.documentElement.getAttribute('data-rail-position');
}

describe('rail-boot.js resolution ladder', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-rail-position');
    setSearch('');
  });

  describe('rung 1 — ?rail= query param', () => {
    test('a valid ?rail= wins over everything below it', () => {
      window.localStorage.setItem('osprey-rail-position', 'left');
      document.documentElement.setAttribute('data-rail-position', 'left');
      setSearch('?rail=top');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test('an invalid ?rail= is ignored and resolution falls through', () => {
      window.localStorage.setItem('osprey-rail-position', 'top');
      setSearch('?rail=diagonal');

      runBoot();

      expect(currentPosition()).toBe('top');
    });
  });

  describe('rung 2 — localStorage', () => {
    test('a valid stored position applies when no query param is present', () => {
      window.localStorage.setItem('osprey-rail-position', 'top');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test('stored position outranks a differing server attribute', () => {
      window.localStorage.setItem('osprey-rail-position', 'top');
      document.documentElement.setAttribute('data-rail-position', 'left');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test('an invalid stored position falls through to the server attr', () => {
      window.localStorage.setItem('osprey-rail-position', 'banana');
      document.documentElement.setAttribute('data-rail-position', 'top');

      runBoot();

      expect(currentPosition()).toBe('top');
    });
  });

  describe('rung 3 — server-rendered data-rail-position', () => {
    test('a valid server attribute applies when no query/storage rung matches', () => {
      document.documentElement.setAttribute('data-rail-position', 'top');

      runBoot();

      expect(currentPosition()).toBe('top');
    });

    test('an invalid server attribute falls through to the default', () => {
      document.documentElement.setAttribute('data-rail-position', 'bogus');

      runBoot();

      expect(currentPosition()).toBe('left');
    });
  });

  describe('rung 4 — default', () => {
    test("resolves to 'left' when no rung yields a valid position", () => {
      runBoot();

      expect(currentPosition()).toBe('left');
    });
  });

  describe('no-clobber', () => {
    test('an already-correct server attribute is not rewritten', () => {
      document.documentElement.setAttribute('data-rail-position', 'top');
      const spy = vi.spyOn(document.documentElement, 'setAttribute');

      runBoot();

      expect(spy).not.toHaveBeenCalled();
      expect(currentPosition()).toBe('top');
      spy.mockRestore();
    });

    test('a differing server attribute IS rewritten to the resolved position', () => {
      document.documentElement.setAttribute('data-rail-position', 'left');
      setSearch('?rail=top');
      const spy = vi.spyOn(document.documentElement, 'setAttribute');

      runBoot();

      expect(spy).toHaveBeenCalledWith('data-rail-position', 'top');
      expect(currentPosition()).toBe('top');
      spy.mockRestore();
    });
  });
});
