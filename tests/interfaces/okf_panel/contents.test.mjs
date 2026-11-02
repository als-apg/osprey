// @ts-check
/**
 * When the OKF Knowledge Panel's contents list (the sidebar) shows
 * (contents.js):
 *   npx vitest run tests/interfaces/okf_panel/contents.test.mjs
 *
 * The overview is a table of contents itself, so the list starts hidden
 * there; a page opens it unless the operator hid it on a page, which is
 * remembered; a search shows it; a narrow tile keeps it hidden. Each test
 * imports a fresh module, as a page load would.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const STORAGE_KEY = 'osprey-okf-sidebar-width';

/** @typedef {typeof import('../../../src/osprey/interfaces/okf_panel/static/js/contents.js')} Contents */

/** A page load: fresh DOM in the closed start state, fresh module. @returns {Promise<Contents>} */
async function load() {
  document.body.innerHTML = `
    <div id="app" data-contents="closed">
      <aside id="sidebar"></aside>
      <div id="sidebar-resizer" role="separator" tabindex="0"></div>
      <main id="reader">
        <div class="reader-bar"><button id="contents-toggle" type="button"></button></div>
        <div id="reader-content"></div>
      </main>
    </div>`;
  vi.resetModules();
  const contents = await import('../../../src/osprey/interfaces/okf_panel/static/js/contents.js');
  contents.initContents();
  return contents;
}

function isOpen() {
  return document.getElementById('app')?.getAttribute('data-contents') === 'open';
}

function clickToggle() {
  document.getElementById('contents-toggle')?.click();
}

/** @returns {{size: number | null, collapsed: boolean} | null} */
function stored() {
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw === null ? null : JSON.parse(raw);
}

/** @param {boolean} narrow */
function setNarrow(narrow) {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    (query) =>
      /** @type {MediaQueryList} */ (
        /** @type {unknown} */ ({
          matches: narrow,
          media: query,
          addEventListener: () => {},
          removeEventListener: () => {},
        })
      )
  );
}

beforeEach(() => {
  localStorage.clear();
  setNarrow(false);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('the overview and pages', () => {
  test('the overview starts with the list hidden, and the toggle offers to show it', async () => {
    await load();
    const toggle = document.getElementById('contents-toggle');
    expect(isOpen()).toBe(false);
    expect(toggle?.getAttribute('aria-expanded')).toBe('false');
    expect(toggle?.getAttribute('aria-label')).toBe('Show contents');
  });

  test('a page opens the list at the stored width', async () => {
    localStorage.setItem(STORAGE_KEY, '420');
    const contents = await load();
    contents.showContentsFor('page');
    expect(isOpen()).toBe(true);
    expect(document.getElementById('sidebar')?.style.flexBasis).toBe('420px');
    expect(document.getElementById('contents-toggle')?.getAttribute('aria-label')).toBe(
      'Hide contents'
    );
  });

  test('back on the overview the list hides again, and nothing is stored', async () => {
    const contents = await load();
    contents.showContentsFor('page');
    contents.showContentsFor('browse');
    expect(isOpen()).toBe(false);
    expect(stored()).toBeNull();
  });
});

describe("the operator's hide on a page", () => {
  test('holds on the next page and after a reload', async () => {
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
    document.getElementById('sidebar-resizer')?.dispatchEvent(new MouseEvent('dblclick'));
    expect(isOpen()).toBe(false);
    expect(stored()?.collapsed).toBe(true);
  });
});

describe('a hide or show on the overview', () => {
  test('is not remembered: pages still open the list, the next overview starts hidden', async () => {
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
  test('a search shows the list on the overview; clearing it hides it again', async () => {
    const contents = await load();
    contents.setContentsSearching(true);
    expect(isOpen()).toBe(true);
    contents.setContentsSearching(false);
    expect(isOpen()).toBe(false);
  });

  test("a search shows it over a hide; clearing it brings the operator's hide back", async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ size: 300, collapsed: true }));
    const contents = await load();
    contents.showContentsFor('page');
    contents.setContentsSearching(true);
    expect(isOpen()).toBe(true);
    contents.setContentsSearching(false);
    expect(isOpen()).toBe(false);
    expect(stored()?.collapsed).toBe(true);
  });
});

describe('a narrow tile', () => {
  test('keeps the list hidden on pages, and a show there is not remembered', async () => {
    setNarrow(true);
    const contents = await load();
    contents.showContentsFor('page');
    expect(isOpen()).toBe(false);
    clickToggle();
    expect(isOpen()).toBe(true);
    expect(stored()?.collapsed).not.toBe(true);
    clickToggle();
    expect(stored()?.collapsed).not.toBe(true);

    setNarrow(false);
    const wide = await load();
    wide.showContentsFor('page');
    expect(isOpen()).toBe(true);
  });
});
