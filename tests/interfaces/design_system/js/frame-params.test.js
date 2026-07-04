/**
 * Unit tests for the design-system micro-frontend query-param contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/frame-params.test.js
 *
 * Covers CONTRACT_VERSION and applyEmbedded().
 *
 * NOTE: frame-params.js is imported by RELATIVE path, not the absolute
 * `/design-system/js/frame-params.js` runtime specifier — Vitest/Vite
 * resolves against the repo root with no alias configured, so the absolute
 * path would not load. This mirrors tests/interfaces/design_system/js/dom.test.js.
 */

import { test, expect, describe, afterEach } from 'vitest';

import {
  applyEmbedded,
  CONTRACT_VERSION,
} from '../../../../src/osprey/interfaces/design_system/static/js/frame-params.js';

describe('applyEmbedded', () => {
  afterEach(() => {
    document.body.className = '';
  });

  test('?embedded=true adds the embedded class to document.body', () => {
    window.history.replaceState({}, '', '?embedded=true');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(true);
  });

  test('no embedded param leaves the embedded class absent', () => {
    window.history.replaceState({}, '', '?');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(false);
  });

  test('?embedded=false leaves the embedded class absent', () => {
    window.history.replaceState({}, '', '?embedded=false');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(false);
  });

  test('?embedded=1 leaves the embedded class absent', () => {
    window.history.replaceState({}, '', '?embedded=1');

    applyEmbedded();

    expect(document.body.classList.contains('embedded')).toBe(false);
  });
});

describe('CONTRACT_VERSION', () => {
  test('is a non-empty string', () => {
    expect(typeof CONTRACT_VERSION).toBe('string');
    expect(CONTRACT_VERSION.length).toBeGreaterThan(0);
  });
});
