/**
 * Unit tests for sidebar-boot.js, the pre-paint stamp of the sidebar width:
 * localStorage['osprey-sidebar'] (per-render scoped) === 'collapsed' stamps
 * data-sidebar="collapsed" on <html>; anything else leaves <html> alone,
 * because an absent attribute means expanded.
 *
 * Like rail-boot.js, sidebar-boot.js is a non-module, dependency-free IIFE, so
 * each scenario arranges localStorage and the <html> attributes and then
 * re-executes the exact on-disk source against the happy-dom globals.
 * collapsible.js is the WRITER of the key; collapsible.test.mjs pins its half.
 *   npx vitest run tests/interfaces/design_system/sidebar-boot.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';
import { runBootScript } from './js/boot-harness.mjs';

const SCOPE_ATTR = 'data-osprey-storage-scope';
const LEGACY_KEY = 'osprey-sidebar';

const runBoot = () => runBootScript('sidebar-boot.js');

function currentState() {
  return document.documentElement.getAttribute('data-sidebar');
}

describe('sidebar-boot.js', () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute('data-sidebar');
    document.documentElement.removeAttribute(SCOPE_ATTR);
  });

  describe('unscoped page', () => {
    test("a stored 'collapsed' stamps data-sidebar=collapsed", () => {
      window.localStorage.setItem(LEGACY_KEY, 'collapsed');

      runBoot();

      expect(currentState()).toBe('collapsed');
    });

    test("a stored 'expanded' leaves the attribute absent", () => {
      window.localStorage.setItem(LEGACY_KEY, 'expanded');

      runBoot();

      expect(currentState()).toBeNull();
    });

    test('no stored value or an invalid one leaves the attribute absent', () => {
      runBoot();
      expect(currentState()).toBeNull();

      window.localStorage.setItem(LEGACY_KEY, 'narrow');
      runBoot();
      expect(currentState()).toBeNull();
    });
  });

  describe('scoped page', () => {
    test("bob's scoped 'collapsed' applies", () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      window.localStorage.setItem(`${LEGACY_KEY}--bob`, 'collapsed');

      runBoot();

      expect(currentState()).toBe('collapsed');
    });

    test('a polluted bare key is never read when the page is scoped', () => {
      window.localStorage.setItem(LEGACY_KEY, 'collapsed');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

      runBoot();

      expect(currentState()).toBeNull();
    });

    test("another user's scoped value does not apply", () => {
      window.localStorage.setItem(`${LEGACY_KEY}--alice`, 'collapsed');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

      runBoot();

      expect(currentState()).toBeNull();
    });

    test('an empty scope attribute counts as unscoped', () => {
      window.localStorage.setItem(LEGACY_KEY, 'collapsed');
      document.documentElement.setAttribute(SCOPE_ATTR, '');

      runBoot();

      expect(currentState()).toBe('collapsed');
    });
  });
});
