/**
 * Unit tests for the design-system DOM utility helpers.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/design_system/js/dom.test.js
 *
 * Covers el(), escapeHtml(), and the trailing-edge debounce().
 *
 * NOTE: dom.js is imported by RELATIVE path here rather than the absolute
 * `/design-system/js/dom.js` runtime specifier. vitest.config.js does
 * configure that alias (see alias-smoke.test.mjs), but this file predates it
 * and mirrors tests/interfaces/channel_finder/chunk-filter.test.mjs.
 */

import { test, expect, vi, describe, afterEach } from 'vitest';

import {
  el,
  escapeHtml,
  debounce,
  panelApiPrefix,
} from '../../../../src/osprey/interfaces/design_system/static/js/dom.js';

describe('escapeHtml', () => {
  test('undefined yields the empty string (SC4)', () => {
    expect(escapeHtml(undefined)).toBe("");
  });

  test('escapes angle brackets and ampersands (SC4)', () => {
    expect(escapeHtml("<b>&")).toBe("&lt;b&gt;&amp;");
  });

  test('escapes a <script> tag', () => {
    expect(escapeHtml("<script>alert(1)</script>")).toBe(
      "&lt;script&gt;alert(1)&lt;/script&gt;"
    );
  });

  test('escapes a bare ampersand', () => {
    expect(escapeHtml("&")).toBe("&amp;");
  });

  test('null yields the empty string', () => {
    expect(escapeHtml(null)).toBe("");
  });

  test('coerces a number to its string form', () => {
    expect(escapeHtml(42)).toBe("42");
  });

  test('coerces 0 to "0" (not the empty string)', () => {
    expect(escapeHtml(0)).toBe("0");
  });

  test('coerces false to "false"', () => {
    expect(escapeHtml(false)).toBe("false");
  });

  test('escapes a double-quote to &quot;', () => {
    expect(escapeHtml('a"b')).toBe('a&quot;b');
  });

  test('escapes a single-quote to &#39;', () => {
    expect(escapeHtml("a'b")).toBe('a&#39;b');
  });
});

describe('el', () => {
  test("el('div', 'foo') returns a DIV element with the class applied", () => {
    const node = el('div', 'foo');
    expect(node).toBeInstanceOf(HTMLElement);
    expect(node.tagName).toBe('DIV');
    expect(node.className).toBe('foo');
  });

  test("el('span') with no class returns a SPAN with empty className", () => {
    const node = el('span');
    expect(node).toBeInstanceOf(HTMLElement);
    expect(node.tagName).toBe('SPAN');
    expect(node.className).toBe('');
  });
});

describe('debounce', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test('rapid successive calls coalesce into a single invocation', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const debounced = debounce(fn, 100);

    debounced();
    debounced();
    debounced();

    expect(fn).toHaveBeenCalledTimes(0);
    vi.advanceTimersByTime(150);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('the trailing call fires with the LAST arguments', () => {
    vi.useFakeTimers();
    const fn = vi.fn();
    const debounced = debounce(fn, 100);

    debounced('first');
    debounced('second');
    debounced('third');

    vi.advanceTimersByTime(150);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith('third');
  });
});

describe('panelApiPrefix', () => {
  /** @param {string} path */
  function atPath(path) {
    window.history.replaceState(null, '', path);
  }

  afterEach(() => {
    atPath('/');
  });

  test('single-user reverse proxy: /panel/<id>/… yields /panel/<id>', () => {
    atPath('/panel/results/results/');
    expect(panelApiPrefix()).toBe('/panel/results');
  });

  test('multi-user reverse proxy: /u/<user>/panel/<id>/… keeps the outer prefix', () => {
    atPath('/u/alice/panel/results/results/');
    expect(panelApiPrefix()).toBe('/u/alice/panel/results');
  });

  test('multi-user deep path: nested panel-internal segments do not extend the prefix', () => {
    atPath('/u/bob/panel/health/health-panel/index.html');
    expect(panelApiPrefix()).toBe('/u/bob/panel/health');
  });

  test('standalone (sidecar-mounted) panel: no /panel/ segment yields the empty string', () => {
    atPath('/results/');
    expect(panelApiPrefix()).toBe('');
  });

  test('a panel-internal path segment merely containing "panel" is not a mount marker', () => {
    atPath('/health-panel/index.html');
    expect(panelApiPrefix()).toBe('');
  });
});
