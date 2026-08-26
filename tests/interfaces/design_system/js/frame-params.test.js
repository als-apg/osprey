/**
 * Unit tests for the design-system micro-frontend query-param contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/frame-params.test.js
 *
 * Covers CONTRACT_VERSION, applyEmbedded(), stripQueryMode(), and
 * onModeChange() (the receive side of the host's `osprey-mode-change`
 * postMessage broadcast: origin check, mode normalization, data-ui-mode
 * stamping, callback hook).
 *
 * NOTE: frame-params.js is imported by RELATIVE path, not the absolute
 * `/design-system/js/frame-params.js` runtime specifier — matching the
 * sibling suites in this directory (see
 * tests/interfaces/design_system/js/dom.test.js).
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  applyEmbedded,
  isEmbedded,
  onModeChange,
  stripQueryMode,
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

describe('isEmbedded', () => {
  afterEach(() => {
    document.body.className = '';
    window.history.replaceState({}, '', '/');
  });

  test('is true only for ?embedded=true, exactly as applyEmbedded reads it', () => {
    for (const [query, expected] of [
      ['?embedded=true', true],
      ['?embedded=false', false],
      ['?embedded=1', false],
      ['', false],
    ]) {
      window.history.replaceState({}, '', `/${query}`);
      expect(isEmbedded()).toBe(expected);
    }
  });

  test('answers from the URL, so a page can branch on it before applyEmbedded()', () => {
    // The theme role is picked at module load; <body> may not carry the class yet.
    window.history.replaceState({}, '', '/?embedded=true');
    expect(document.body.classList.contains('embedded')).toBe(false);
    expect(isEmbedded()).toBe(true);
  });
});

describe('CONTRACT_VERSION', () => {
  test('is a non-empty string', () => {
    expect(typeof CONTRACT_VERSION).toBe('string');
    expect(CONTRACT_VERSION.length).toBeGreaterThan(0);
  });
});

describe('stripQueryMode', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.history.replaceState({}, '', '/');
  });

  test('drops ?mode= when present', () => {
    window.history.replaceState({}, '', '?mode=simple');

    stripQueryMode();

    expect(window.location.search).toBe('');
  });

  test('no-ops when no mode param is present', () => {
    window.history.replaceState({}, '', '?other=x');
    const replaceState = vi.spyOn(window.history, 'replaceState');

    stripQueryMode();

    expect(replaceState).not.toHaveBeenCalled();
    expect(window.location.search).toBe('?other=x');
  });

  test('preserves the other params and the hash', () => {
    window.history.replaceState({}, '', '?mode=simple&other=x#frag');

    stripQueryMode();

    expect(window.location.search).toBe('?other=x');
    expect(window.location.hash).toBe('#frag');
  });

  test('rewrites the URL in place rather than adding a history entry', () => {
    window.history.replaceState({}, '', '?mode=simple&other=x');
    const replaceState = vi.spyOn(window.history, 'replaceState');
    const pushState = vi.spyOn(window.history, 'pushState');

    stripQueryMode();

    expect(replaceState).toHaveBeenCalledTimes(1);
    expect(pushState).not.toHaveBeenCalled();
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
