// @ts-check
/**
 * When the OKF Knowledge Panel's contents list (the panel sidebar) shows
 * (contents.js, on the design system's panel-sidebar.js):
 *   npx vitest run tests/interfaces/okf_panel/contents.test.mjs
 *
 * The overview is a table of contents itself, so the list starts closed
 * there; a page opens it unless the operator hid it on a page, which is
 * remembered; a view change is never stored; a search opens it; a narrow tile
 * makes it an overlay. Each test mounts the real page shell and imports a
 * fresh module, as a page load would. happy-dom does no layout, so the panel
 * root's width comes from a stubbed ResizeObserver.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { mountShell } from './page-shell.mjs';

const STORAGE_KEY = 'osprey-okf-sidebar-width';

/** @typedef {typeof import('../../../src/osprey/interfaces/okf_panel/static/js/contents.js')} Contents */

/** @type {Set<{cb: ResizeObserverCallback}>} */
const observers = new Set();

class FakeResizeObserver {
  /** @param {ResizeObserverCallback} cb */
  constructor(cb) {
    this.cb = cb;
    observers.add(this);
  }

  observe() {}

  disconnect() {
    observers.delete(this);
  }
}

/**
 * Report a new content width for the panel root.
 * @param {number} width
 */
function resizePanel(width) {
  for (const observer of observers) {
    const entry = /** @type {ResizeObserverEntry} */ (
      /** @type {unknown} */ ({ contentRect: { width } })
    );
    observer.cb([entry], /** @type {ResizeObserver} */ (/** @type {unknown} */ (observer)));
  }
}

/** A page load: the shell in its closed start state, a fresh module. @returns {Promise<Contents>} */
async function load() {
  observers.clear();
  mountShell();
  vi.resetModules();
  const contents = await import('../../../src/osprey/interfaces/okf_panel/static/js/contents.js');
  contents.initContents();
  return contents;
}

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

function isOpen() {
  return $('sidebar').getAttribute('data-open') === 'true';
}

function clickToggle() {
  $('contents-toggle').click();
}

/** @returns {{size?: number, collapsed?: boolean} | null} */
function stored() {
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw === null ? null : JSON.parse(raw);
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('the overview and pages', () => {
  test('the overview starts with the list closed, and the toggle offers to show it', async () => {
    await load();
    expect(isOpen()).toBe(false);
    expect($('app').getAttribute('data-open')).toBe('false');
    expect($('contents-toggle').getAttribute('aria-expanded')).toBe('false');
    expect($('contents-toggle').getAttribute('aria-label')).toBe('Show sidebar');
  });

  test('a page opens the list at a width stored before the shared sidebar', async () => {
    localStorage.setItem(STORAGE_KEY, '420');
    const contents = await load();
    contents.showContentsFor('page');
    expect(isOpen()).toBe(true);
    expect($('sidebar').style.width).toBe('420px');
    expect($('contents-toggle').getAttribute('aria-label')).toBe('Hide sidebar');
  });

  test('back on the overview the list closes again, and nothing is stored', async () => {
    const contents = await load();
    contents.showContentsFor('page');
    contents.showContentsFor('browse');
    expect(isOpen()).toBe(false);
    expect(stored()).toBeNull();
  });
});

describe("the operator's toggle on a page", () => {
  test('a hide holds on the next page and after a reload', async () => {
    let contents = await load();
    contents.showContentsFor('page');
    clickToggle();
    expect(isOpen()).toBe(false);
    expect(stored()?.collapsed).toBe(true);

    contents.showContentsFor('page');
    expect(isOpen()).toBe(false);
    contents.showContentsFor('browse');
    contents.showContentsFor('page');
    expect(isOpen()).toBe(false);

    contents = await load();
    contents.showContentsFor('page');
    expect(isOpen()).toBe(false);
  });

  test('a show on a later page is remembered too', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ size: 300, collapsed: true }));
    let contents = await load();
    contents.showContentsFor('page');
    expect(isOpen()).toBe(false);
    clickToggle();
    expect(isOpen()).toBe(true);
    expect(stored()).toEqual({ size: 300, collapsed: false });

    contents = await load();
    contents.showContentsFor('page');
    expect(isOpen()).toBe(true);
  });

  test("the divider's double-click hides it the same way", async () => {
    const contents = await load();
    contents.showContentsFor('page');
    $('sidebar-resizer').dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(isOpen()).toBe(false);
    expect(stored()?.collapsed).toBe(true);
  });
});

describe('a hide or show on the overview', () => {
  test('is not remembered: pages still open the list, the next overview starts closed', async () => {
    const contents = await load();
    clickToggle();
    expect(isOpen()).toBe(true);
    clickToggle();
    expect(isOpen()).toBe(false);
    expect(stored()?.collapsed).not.toBe(true);

    contents.showContentsFor('page');
    expect(isOpen()).toBe(true);
    contents.showContentsFor('browse');
    expect(isOpen()).toBe(false);
  });
});

describe('search', () => {
  test('a search opens the list on the overview; clearing it closes it again', async () => {
    const contents = await load();
    contents.setContentsSearching(true);
    expect(isOpen()).toBe(true);
    contents.setContentsSearching(false);
    expect(isOpen()).toBe(false);
    expect(stored()).toBeNull();
  });

  test("a search opens it over a hide; clearing it brings the operator's hide back", async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ size: 300, collapsed: true }));
    const contents = await load();
    contents.showContentsFor('page');
    contents.setContentsSearching(true);
    expect(isOpen()).toBe(true);
    contents.setContentsSearching(false);
    expect(isOpen()).toBe(false);
    expect(stored()?.collapsed).toBe(true);
  });

  test('the list stays open for the results while the view changes', async () => {
    const contents = await load();
    contents.setContentsSearching(true);
    contents.showContentsFor('page');
    expect(isOpen()).toBe(true);
    contents.showContentsFor('browse');
    expect(isOpen()).toBe(true);
  });
});

describe('a narrow tile', () => {
  test('makes the list an overlay, closed on pages; a show is not remembered, a pick closes it', async () => {
    const contents = await load();
    resizePanel(600);
    contents.showContentsFor('page');
    expect(isOpen()).toBe(false);
    expect($('sidebar').hasAttribute('data-overlay')).toBe(true);

    clickToggle();
    expect(isOpen()).toBe(true);
    expect(stored()).toBeNull();
    contents.contentsPicked();
    expect(isOpen()).toBe(false);

    resizePanel(1000);
    expect($('sidebar').hasAttribute('data-overlay')).toBe(false);
    expect(isOpen()).toBe(true);
  });

  test('a pick in a wide tile leaves the list as it is', async () => {
    const contents = await load();
    contents.showContentsFor('page');
    contents.contentsPicked();
    expect(isOpen()).toBe(true);
  });
});

describe('Reset contents width', () => {
  test('goes back to the default width and forgets the stored one', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ size: 400 }));
    const contents = await load();
    contents.showContentsFor('page');
    expect($('sidebar').style.width).toBe('400px');
    contents.resetContentsWidth();
    expect($('sidebar').style.width).toBe('260px');
    expect(isOpen()).toBe(true);
    expect(stored()).toBeNull();
  });
});
