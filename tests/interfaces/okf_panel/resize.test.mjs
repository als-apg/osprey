// @ts-check
/**
 * Unit tests for the OKF Knowledge Panel's sidebar resizer (resize.js):
 *   npx vitest run tests/interfaces/okf_panel/resize.test.mjs
 *
 * Pointer-drag mechanics ride pointer capture (browser-level; exercised
 * manually / via the hub), so these tests pin the pure clamp, the stored-width
 * restore, and the keyboard nudge path — the parts happy-dom can host.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import {
  clampWidth,
  initSidebarResize,
} from '../../../src/osprey/interfaces/okf_panel/static/js/resize.js';

const STORAGE_KEY = 'osprey-okf-sidebar-width';

function renderApp() {
  document.body.innerHTML = `
    <div id="app">
      <aside id="sidebar"></aside>
      <div id="sidebar-resizer" role="separator" tabindex="0"></div>
      <main id="reader"></main>
    </div>
  `;
}

describe('clampWidth', () => {
  test('passes through in-range widths, rounded', () => {
    expect(clampWidth(300)).toBe(300);
    expect(clampWidth(300.6)).toBe(301);
  });

  test('clamps below the minimum and above the maximum', () => {
    expect(clampWidth(10)).toBe(180);
    expect(clampWidth(5000)).toBe(560);
  });
});

describe('initSidebarResize', () => {
  beforeEach(() => {
    localStorage.clear();
    renderApp();
  });

  test('no-ops when the resizer or sidebar is absent', () => {
    document.body.innerHTML = '<div id="app"></div>';
    expect(() => initSidebarResize()).not.toThrow();
  });

  test('restores a stored width (clamped) onto the sidebar', () => {
    localStorage.setItem(STORAGE_KEY, '420');
    initSidebarResize();
    const sidebar = document.getElementById('sidebar');
    expect(sidebar?.style.flexBasis).toBe('420px');
    expect(sidebar?.style.maxWidth).toBe('420px');
  });

  test('ignores a non-numeric stored value', () => {
    localStorage.setItem(STORAGE_KEY, 'garbage');
    initSidebarResize();
    expect(document.getElementById('sidebar')?.style.flexBasis).toBe('');
  });

  test('Arrow keys nudge the width and persist it', () => {
    initSidebarResize();
    const resizer = /** @type {HTMLElement} */ (
      document.getElementById('sidebar-resizer')
    );
    // happy-dom reports a 0-width rect, so the nudge clamps up to the minimum.
    resizer.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    const sidebar = document.getElementById('sidebar');
    expect(sidebar?.style.flexBasis).toBe('180px');
    // The shared splitter persists a {size, collapsed} record, not a bare
    // width — a collapse has to remember what to restore to.
    expect(JSON.parse(String(localStorage.getItem(STORAGE_KEY))).size).toBe(180);
  });
});
