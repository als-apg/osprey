// @ts-check
/**
 * Embedded in the web-terminal hub, the KNOWLEDGE panel's one header is the
 * tile bar, in Simple as in Expert (app.js `publishHeaderContribution`):
 *   npx vitest run tests/interfaces/okf_panel/app-header.test.mjs
 *
 * The panel boots in Simple here — the mode that used to contribute nothing —
 * and must still put the search box and the Browse view in the bar, draw no
 * "Browse all pages" bar inside the pane, and answer both items' actions.
 *
 * header-contrib.js is mocked: under happy-dom `window.parent === window`, so
 * the real `contributeHeader()` is a no-op and the contribution's shape could
 * not be asserted. The mocks record into a hoisted object rather than leaning
 * on `mock.calls`: app.js registers and publishes once, at boot in
 * `beforeAll`, and the runner clears mock call history between tests. app.js
 * boots on evaluation, so the URL, mode, DOM, `marked` and `fetch` are in place
 * before the one dynamic import below.
 */

import { beforeAll, describe, expect, test, vi } from 'vitest';

const header = vi.hoisted(() => ({
  /** Every contribution app.js sent, oldest first. */
  contributions: /** @type {any[][]} */ ([]),
  /** The callback app.js registered for the hub's header actions. */
  onAction: /** @type {((id: string, value?: string) => void) | null} */ (null),
}));

vi.mock('/design-system/js/header-contrib.js', () => ({
  contributeHeader: (/** @type {any[]} */ items) => {
    header.contributions.push(items);
  },
  onHeaderAction: (/** @type {(id: string, value?: string) => void} */ callback) => {
    header.onAction = callback;
  },
}));

/** @type {string[]} */
const fetched = [];

/**
 * @param {unknown} body
 */
function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

/**
 * @param {string} url
 */
async function fakeFetch(url) {
  fetched.push(url);
  if (url === '/api/concepts') {
    return jsonResponse({
      groups: [
        { id: 'devices', label: 'devices', concepts: [{ id: 'devices/bpm', title: 'BPM' }] },
      ],
    });
  }
  if (url === '/api/bundle_health') return jsonResponse({ ok: true, counts: {}, total: 0 });
  if (url === '/api/structure') return jsonResponse({ markdown: '# Facility Knowledge Base' });
  if (url.startsWith('/api/search?q=')) {
    return jsonResponse({ results: [{ id: 'devices/bpm', title: 'BPM', score: null }] });
  }
  if (url.startsWith('/api/concept?id=')) {
    const id = decodeURIComponent(url.slice('/api/concept?id='.length));
    return jsonResponse({ id, frontmatter: { title: id }, body: '' });
  }
  throw new Error(`unstubbed fetch: ${url}`);
}

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/** The items of the most recent contribution. @returns {any[]} */
function lastContribution() {
  expect(header.contributions.length).toBeGreaterThan(0);
  return header.contributions[header.contributions.length - 1];
}

/** @param {string} id @returns {any} */
function contributedItem(id) {
  return lastContribution().find((/** @type {any} */ item) => item.id === id);
}

/**
 * Fire the hub's action round-trip for a contributed item.
 * @param {string} id
 * @param {string} [value]
 */
async function headerAction(id, value) {
  if (!header.onAction) throw new Error('app.js registered no header-action callback');
  header.onAction(id, value);
  await flush();
}

beforeAll(async () => {
  history.replaceState(null, '', '/?embedded=true#devices/bpm');
  document.documentElement.setAttribute('data-ui-mode', 'simple');
  document.body.innerHTML = `
    <div id="app">
      <aside id="sidebar">
        <div id="search-box">
          <form id="search-form"><input id="search-input" type="search" /></form>
        </div>
        <a id="structure-link" href="#__structure">Knowledge Base</a>
        <nav id="tree"></nav>
        <div id="search-results" hidden></div>
        <footer id="bundle-health" hidden></footer>
      </aside>
      <main id="reader">
        <div id="reader-topbar" class="reader-topbar" hidden>
          <button type="button" id="browse-all">Browse all pages</button>
        </div>
        <div id="reader-content"></div>
      </main>
    </div>`;
  vi.stubGlobal('marked', { parse: (/** @type {string} */ md) => md });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel tile-bar contribution (embedded, Simple)', () => {
  test('the pane draws no "Browse all pages" bar', () => {
    expect(document.body.classList.contains('embedded')).toBe(true);
    expect(document.getElementById('reader-topbar')).toBeNull();
    expect(document.getElementById('browse-all')).toBeNull();
    expect(document.body.textContent).not.toContain('Browse all pages');
  });

  test('Simple contributes the search box and the Browse view, nav last', () => {
    const items = lastContribution();
    expect(items.map((/** @type {any} */ item) => `${item.kind}:${item.id}`)).toEqual([
      'search:search',
      'nav:view',
    ]);
    expect(contributedItem('search').placeholder).toBe('Search concepts…');
    expect(contributedItem('view').items).toEqual([
      { id: 'browse', label: 'Browse', active: false },
    ]);
  });

  test('Browse opens the structure overview and marks itself active', async () => {
    await headerAction('view', 'browse');
    expect(fetched).toContain('/api/structure');
    expect(contributedItem('view').items[0].active).toBe(true);
    expect(document.getElementById('reader-content')?.textContent).toContain(
      'Facility Knowledge Base'
    );
  });

  test('opening a page clears the Browse entry', async () => {
    const link = /** @type {HTMLElement} */ (
      document.querySelector('.concept-link[data-concept-id="devices/bpm"]')
    );
    link.click();
    await flush();
    expect(contributedItem('view').items[0].active).toBe(false);
  });

  test('the bar search runs the panel search and shows the results', async () => {
    await headerAction('search', 'bpm');
    expect(fetched).toContain('/api/search?q=bpm');
    expect(document.getElementById('search-results')?.hidden).toBe(false);
  });
});
