/**
 * Unit tests for the design-system micro-frontend query-param contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/frame-params.test.js
 *
 * Covers CONTRACT_VERSION, applyEmbedded(), and onModeChange() (the
 * receive side of the host's `osprey-mode-change` postMessage broadcast:
 * origin check, mode normalization, data-ui-mode stamping, callback hook).
 *
 * NOTE: frame-params.js is imported by RELATIVE path, not the absolute
 * `/design-system/js/frame-params.js` runtime specifier — Vitest/Vite
 * resolves against the repo root with no alias configured, so the absolute
 * path would not load. This mirrors tests/interfaces/design_system/js/dom.test.js.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  applyEmbedded,
  onModeChange,
  CONTRACT_VERSION,
} from '../../../../src/osprey/interfaces/design_system/static/js/frame-params.js';

/**
 * Deliver a message event synchronously (window.postMessage queues a task,
 * which awkwardly interleaves with test teardown; dispatchEvent is the
 * established pattern for exercising message listeners in this suite).
 * @param {any} data
 * @param {string} [origin]
 */
function deliverMessage(data, origin = window.location.origin) {
  window.dispatchEvent(new MessageEvent('message', { data, origin }));
}

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

describe('onModeChange', () => {
  afterEach(() => {
    document.documentElement.removeAttribute('data-ui-mode');
  });

  test('stamps data-ui-mode and invokes the callback with the mode', () => {
    const callback = vi.fn();
    onModeChange(callback);

    deliverMessage({ type: 'osprey-mode-change', mode: 'simple' });

    expect(document.documentElement.getAttribute('data-ui-mode')).toBe('simple');
    expect(callback).toHaveBeenCalledWith('simple');
  });

  test('normalizes any non-"simple" mode to "expert"', () => {
    const callback = vi.fn();
    onModeChange(callback);

    deliverMessage({ type: 'osprey-mode-change', mode: 'bogus' });

    expect(document.documentElement.getAttribute('data-ui-mode')).toBe('expert');
    expect(callback).toHaveBeenCalledWith('expert');
  });

  test('ignores messages from a foreign origin', () => {
    const callback = vi.fn();
    onModeChange(callback);

    deliverMessage({ type: 'osprey-mode-change', mode: 'simple' }, 'https://evil.example');

    expect(document.documentElement.hasAttribute('data-ui-mode')).toBe(false);
    expect(callback).not.toHaveBeenCalled();
  });

  test('ignores unrelated message types and missing modes', () => {
    const callback = vi.fn();
    onModeChange(callback);

    deliverMessage({ type: 'osprey-session-change', session_id: 'abc' });
    deliverMessage({ type: 'osprey-mode-change' });
    deliverMessage(null);

    expect(document.documentElement.hasAttribute('data-ui-mode')).toBe(false);
    expect(callback).not.toHaveBeenCalled();
  });

  test('works without a callback (CSS-only pages)', () => {
    onModeChange();

    deliverMessage({ type: 'osprey-mode-change', mode: 'simple' });

    expect(document.documentElement.getAttribute('data-ui-mode')).toBe('simple');
  });
});
