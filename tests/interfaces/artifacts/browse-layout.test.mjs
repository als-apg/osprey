/**
 * Unit tests for the Artifact Gallery browse-view layout module
 * (browse-layout.js): orientation state + splitter.
 *
 *   npx vitest run tests/interfaces/artifacts/browse-layout.test.mjs
 *
 * Pointer-drag mechanics ride pointer capture (browser-level; exercised
 * manually / via the hub), so — like the OKF panel's resize.test.mjs —
 * these tests pin the pure clamp/normalize helpers, the stored restore
 * paths, the orientation toggle, and the keyboard nudge path, which
 * happy-dom can host.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import {
  normalizeOrient,
  clampWidth,
  getOrient,
  initBrowseLayout,
} from '../../../src/osprey/interfaces/artifacts/static/js/browse-layout.js';
import { byId } from '../_support/dom.mjs';

const ORIENT_KEY = 'osprey-artifacts-browse-orient';
const WIDTH_KEY = 'osprey-artifacts-browse-sidebar-width';

/** Minimal DOM fixture matching artifacts/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `
    <div class="browse-layout">
      <aside class="browse-sidebar" id="browse-sidebar"></aside>
      <div class="resize-handle osprey-splitter" id="resize-handle"
           role="separator" tabindex="0"></div>
      <div class="browse-preview-pane" id="browse-preview-pane"></div>
    </div>
    <button id="orient-toggle-btn"><span class="orient-label"></span></button>
  `;
}

function initAll() {
  initBrowseLayout({
    handle: byId('resize-handle'),
    sidebar: byId('browse-sidebar'),
    toggle: byId('orient-toggle-btn'),
  });
}

beforeEach(() => {
  localStorage.clear();
  // index.html ships the row default on <html>; mirror that starting state.
  document.documentElement.dataset.browseOrient = 'row';
  mountFixture();
});

describe('pure helpers', () => {
  test('normalizeOrient treats anything but "column" as "row"', () => {
    expect(normalizeOrient('column')).toBe('column');
    expect(normalizeOrient('row')).toBe('row');
    expect(normalizeOrient('sideways')).toBe('row');
    expect(normalizeOrient(null)).toBe('row');
    expect(normalizeOrient(undefined)).toBe('row');
  });

  test('clampWidth passes through in-range widths (rounded) and clamps extremes', () => {
    expect(clampWidth(300)).toBe(300);
    expect(clampWidth(300.6)).toBe(301);
    expect(clampWidth(10)).toBe(200);
    expect(clampWidth(5000)).toBe(560);
  });
});

describe('init / restore', () => {
  test('no stored state keeps the shipped row default, no inline sizes', () => {
    initAll();
    expect(getOrient()).toBe('row');
    expect(byId('browse-sidebar').style.flexBasis).toBe('');
  });

  test('a persisted "column" choice is re-applied at init', () => {
    localStorage.setItem(ORIENT_KEY, 'column');
    initAll();
    expect(getOrient()).toBe('column');
    expect(document.documentElement.dataset.browseOrient).toBe('column');
  });

  test('a persisted garbage orientation falls back to row', () => {
    localStorage.setItem(ORIENT_KEY, 'diagonal');
    initAll();
    expect(getOrient()).toBe('row');
  });

  test('a stored width is restored (clamped) as inline flex-basis in row mode', () => {
    localStorage.setItem(WIDTH_KEY, '340');
    initAll();
    const sidebar = byId('browse-sidebar');
    expect(sidebar.style.flexBasis).toBe('340px');
    expect(sidebar.style.maxWidth).toBe('340px');
  });

  test('a stored width is NOT applied when the stored orientation is column', () => {
    localStorage.setItem(ORIENT_KEY, 'column');
    localStorage.setItem(WIDTH_KEY, '340');
    initAll();
    expect(byId('browse-sidebar').style.flexBasis).toBe('');
  });

  test('no-ops safely when elements are absent', () => {
    expect(() => initBrowseLayout({ handle: null, sidebar: null, toggle: null })).not.toThrow();
  });
});

describe('orientation toggle', () => {
  test('click flips row -> column: stamps <html>, clears inline sizes, persists', () => {
    localStorage.setItem(WIDTH_KEY, '340');
    initAll();
    const sidebar = byId('browse-sidebar');
    expect(sidebar.style.flexBasis).toBe('340px');

    byId('orient-toggle-btn').click();

    expect(getOrient()).toBe('column');
    expect(sidebar.style.flexBasis).toBe('');
    expect(sidebar.style.maxWidth).toBe('');
    expect(localStorage.getItem(ORIENT_KEY)).toBe('column');
  });

  test('click flips column -> row and re-applies the stored width', () => {
    localStorage.setItem(ORIENT_KEY, 'column');
    localStorage.setItem(WIDTH_KEY, '340');
    initAll();

    byId('orient-toggle-btn').click();

    expect(getOrient()).toBe('row');
    expect(byId('browse-sidebar').style.flexBasis).toBe('340px');
    expect(localStorage.getItem(ORIENT_KEY)).toBe('row');
  });

  test('aria-label/title/visible label always name the layout a click switches TO', () => {
    initAll();
    const toggle = byId('orient-toggle-btn');
    expect(toggle.getAttribute('aria-label')).toBe('Switch to stacked layout');
    expect(toggle.title).toBe('Switch to stacked layout');
    expect(toggle.querySelector('.orient-label')?.textContent).toBe('Stacked layout');

    toggle.click();
    expect(toggle.getAttribute('aria-label')).toBe('Switch to side-by-side layout');
    expect(toggle.title).toBe('Switch to side-by-side layout');
    expect(toggle.querySelector('.orient-label')?.textContent).toBe('Side-by-side layout');
  });

  test('a label-less toggle host (plain icon button) still syncs without throwing', () => {
    document.body.innerHTML += '<button id="bare-toggle"></button>';
    expect(() => initBrowseLayout({
      handle: null,
      sidebar: byId('browse-sidebar'),
      toggle: byId('bare-toggle'),
    })).not.toThrow();
    expect(byId('bare-toggle').title).toBe('Switch to stacked layout');
  });
});

describe('keyboard resize (ARIA separator pattern)', () => {
  /** @param {string} key */
  function pressArrow(key) {
    byId('resize-handle').dispatchEvent(
      new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true })
    );
  }

  // happy-dom reports 0-width rects, so pin the live width explicitly to
  // exercise the real nudge arithmetic instead of only the low-end clamp.
  /** @param {number} px */
  function mockSidebarWidth(px) {
    byId('browse-sidebar').getBoundingClientRect = () =>
      /** @type {DOMRect} */ (/** @type {unknown} */ ({ width: px }));
  }

  test('Arrow keys nudge the split by the step and persist, from the live width', () => {
    initAll();
    const sidebar = byId('browse-sidebar');

    mockSidebarWidth(300);
    pressArrow('ArrowRight');
    expect(sidebar.style.flexBasis).toBe('316px');
    // The shared splitter persists a {size, collapsed} record, not a bare
    // width — a collapse has to remember what to restore to.
    expect(JSON.parse(String(localStorage.getItem(WIDTH_KEY))).size).toBe(316);

    mockSidebarWidth(316);
    pressArrow('ArrowLeft');
    expect(sidebar.style.flexBasis).toBe('300px');
    expect(JSON.parse(String(localStorage.getItem(WIDTH_KEY))).size).toBe(300);
  });

  test('clamps at the minimum', () => {
    initAll();
    mockSidebarWidth(208);
    pressArrow('ArrowLeft');
    expect(byId('browse-sidebar').style.flexBasis).toBe('200px');
    mockSidebarWidth(200);
    pressArrow('ArrowLeft');
    expect(byId('browse-sidebar').style.flexBasis).toBe('200px');
  });

  test('inert in column mode — the stacked band has no splitter', () => {
    localStorage.setItem(ORIENT_KEY, 'column');
    localStorage.setItem(WIDTH_KEY, '300');
    initAll();
    pressArrow('ArrowRight');
    expect(byId('browse-sidebar').style.flexBasis).toBe('');
    expect(localStorage.getItem(WIDTH_KEY)).toBe('300');
  });
});
