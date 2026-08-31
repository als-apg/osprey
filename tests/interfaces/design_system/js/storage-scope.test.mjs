/**
 * Unit tests for storage-scope.js — the one written-down definition of how a
 * localStorage key is derived from `<html data-osprey-storage-scope>`.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/storage-scope.test.mjs
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { storageScope, scopedStorageKey } from '/design-system/js/storage-scope.js';

const SCOPE_ATTR = 'data-osprey-storage-scope';

describe('storage-scope.js', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute(SCOPE_ATTR);
  });

  describe('storageScope()', () => {
    test('returns null when the attribute is absent (single-user serving)', () => {
      expect(storageScope()).toBeNull();
    });

    test('returns the attribute value when the server stamped one', () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      expect(storageScope()).toBe('bob');
    });

    test('treats an empty attribute value as unscoped', () => {
      // The server omits the attribute rather than rendering `=""`, so this is
      // defensive: an empty scope would otherwise mint `<base>--`, a slot
      // belonging to no persona.
      document.documentElement.setAttribute(SCOPE_ATTR, '');
      expect(storageScope()).toBeNull();
    });
  });

  describe('scopedStorageKey()', () => {
    test('returns the bare legacy key when unscoped', () => {
      expect(scopedStorageKey('osprey-ui-mode')).toBe('osprey-ui-mode');
    });

    test('suffixes the scope with a double dash when scoped', () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      expect(scopedStorageKey('osprey-ui-mode')).toBe('osprey-ui-mode--bob');
    });

    test('an empty attribute value never mints a trailing-dash key', () => {
      document.documentElement.setAttribute(SCOPE_ATTR, '');
      expect(scopedStorageKey('osprey-ui-mode')).toBe('osprey-ui-mode');
    });

    test('two personas on one origin get two distinct keys', () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      const bob = scopedStorageKey('osprey-ui-mode');
      document.documentElement.setAttribute(SCOPE_ATTR, 'carol');
      const carol = scopedStorageKey('osprey-ui-mode');

      expect(bob).not.toBe(carol);
      // ...and neither of them is the shared slot they exist to escape.
      expect(bob).not.toBe('osprey-ui-mode');
      expect(carol).not.toBe('osprey-ui-mode');
    });

    test('the scope is read per call, not captured at module load', () => {
      expect(scopedStorageKey('osprey-rail-position')).toBe('osprey-rail-position');
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      expect(scopedStorageKey('osprey-rail-position')).toBe('osprey-rail-position--bob');
    });

    test('is base-agnostic — every axis derives its key the same way', () => {
      document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
      expect(scopedStorageKey('osprey-theme')).toBe('osprey-theme--bob');
      expect(scopedStorageKey('osprey-dock-layout')).toBe('osprey-dock-layout--bob');
    });
  });
});
