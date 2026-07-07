/**
 * Smoke test for the `/design-system/js` Vitest alias (see vitest.config.js).
 *
 * Design-system modules are imported at runtime via absolute
 * `/design-system/js/*` URLs (served by each panel's HTTP server). This test
 * proves the alias resolves that same absolute specifier to the on-disk
 * module under Node/Vitest by exercising a real function through it, rather
 * than just asserting the import doesn't throw.
 *
 *   npx vitest run tests/interfaces/design_system/js/alias-smoke.test.mjs
 */

import { test, expect } from 'vitest';

import { escapeHtml } from '/design-system/js/dom.js';

test('escapeHtml resolves through the /design-system/js alias and escapes HTML-significant characters', () => {
  expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
  expect(escapeHtml('a & b')).toBe('a &amp; b');
  expect(escapeHtml('<img src=x onerror=alert(1)>')).toBe('&lt;img src=x onerror=alert(1)&gt;');
});

test('escapeHtml leaves double quotes as-is (matches the call sites it replaces)', () => {
  expect(escapeHtml('"quoted"')).toBe('"quoted"');
});

test('escapeHtml treats nullish input as an empty string', () => {
  expect(escapeHtml(null)).toBe('');
  expect(escapeHtml(undefined)).toBe('');
});
