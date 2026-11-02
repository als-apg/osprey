// @ts-check
/**
 * Unit tests for the OKF Knowledge Panel's sidebar resizer (resize.js):
 *   npx vitest run tests/interfaces/okf_panel/resize.test.mjs
 *
 * Pointer-drag mechanics ride pointer capture (browser-level; exercised
 * manually / via the hub), so these tests pin the pure clamp, the stored
 * record, and the keyboard nudge path — the parts happy-dom can host. When
 * the sidebar shows is contents.js's (contents.test.mjs).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  clampWidth,
  initSidebarResize,
  readSidebarRecord,
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

describe('readSidebarRecord', () => {
  beforeEach(() => localStorage.clear());

  test('a bare width from before the collapse existed reads as a width, shown', () => {
    localStorage.setItem(STORAGE_KEY, '420');
    expect(readSidebarRecord()).toEqual({ size: 420, collapsed: false });
  });

  test('a stored hide reads back with its width', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ size: 300, collapsed: true }));
    expect(readSidebarRecord()).toEqual({ size: 300, collapsed: true });
  });

  test('an unreadable value reads as nothing stored', () => {
    localStorage.setItem(STORAGE_KEY, 'garbage');
    expect(readSidebarRecord()).toEqual({ size: null, collapsed: false });
  });
});

describe('initSidebarResize', () => {
  beforeEach(() => {
    localStorage.clear();
    renderApp();
  });

  test('no-ops when the sidebar is absent', () => {
    document.body.innerHTML = '<div id="app"></div>';
    const split = initSidebarResize({ onCommit: () => {} });
    expect(() => split.expand()).not.toThrow();
  });

  test('starts collapsed, and opens at the stored width, clamped', () => {
    localStorage.setItem(STORAGE_KEY, '5000');
    const split = initSidebarResize({ onCommit: () => {} });
    const sidebar = document.getElementById('sidebar');
    expect(split.isCollapsed()).toBe(true);
    expect(sidebar?.style.flexBasis).toBe('');
    split.expand();
    expect(sidebar?.style.flexBasis).toBe('560px');
    expect(sidebar?.style.maxWidth).toBe('560px');
  });

  test('an Arrow-key nudge reports the new width and stores nothing itself', () => {
    localStorage.setItem(STORAGE_KEY, '300');
    const onCommit = vi.fn();
    const split = initSidebarResize({ onCommit });
    split.expand();
    onCommit.mockClear();
    const resizer = /** @type {HTMLElement} */ (document.getElementById('sidebar-resizer'));
    resizer.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(document.getElementById('sidebar')?.style.flexBasis).toBe('316px');
    expect(onCommit).toHaveBeenCalledWith(316, false);
    // The caller owns the record: an automatic hide must not overwrite it.
    expect(localStorage.getItem(STORAGE_KEY)).toBe('300');
  });
});

describe('on a multi-user mount (storage scope)', () => {
  const SCOPE_ATTR = 'data-osprey-storage-scope';

  beforeEach(() => {
    localStorage.clear();
    renderApp();
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
  });

  afterEach(() => {
    document.documentElement.removeAttribute(SCOPE_ATTR);
  });

  test('a scoped page reads its own record, not the shared one', () => {
    localStorage.setItem(STORAGE_KEY, '420');
    localStorage.setItem(STORAGE_KEY + '--bob', '300');
    expect(readSidebarRecord().size).toBe(300);
  });

  test('a scoped page with no record of its own has no stored width', () => {
    localStorage.setItem(STORAGE_KEY, '420');
    expect(readSidebarRecord().size).toBeNull();
  });
});
