// @ts-check
/**
 * Embedded in the web-terminal hub, the KNOWLEDGE panel gives the tile bar its
 * identity and its menu, and keeps every control in its own toolbar (app.js):
 *   npx vitest run tests/interfaces/okf_panel/app-header.test.mjs
 *
 * The tile bar gets the open concept's id as its subtitle and the panel's
 * menu (protocol v3: `text` and `menu`, nothing else). The toolbar's search
 * runs the panel search, and its breadcrumb's Browse opens the overview.
 *
 * header-contrib.js is mocked: under happy-dom `window.parent === window`, so
 * the real `contributeHeader()` is a no-op and the contribution's shape could
 * not be asserted. The mocks record into a hoisted object rather than leaning
 * on `mock.calls`: app.js registers and publishes at boot in `beforeAll`, and
 * the runner clears mock call history between tests. app.js boots on
 * evaluation, so the URL, DOM, storage, `marked` and `fetch` are in place
 * before the one dynamic import below.
 */

import { beforeAll, describe, expect, test, vi } from 'vitest';

import { mountShell } from './page-shell.mjs';

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

const STORAGE_KEY = 'osprey-okf-sidebar-width';

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
        { id: 'devices', label: 'Devices', concepts: [{ id: 'devices/bpm', title: 'BPM' }] },
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
    return jsonResponse({ id, frontmatter: { title: 'BPM' }, body: '' });
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

/** @returns {string | null} the sidebar's open state */
function contentsOpen() {
  return document.getElementById('sidebar')?.getAttribute('data-open') ?? null;
}

/** The crumbs' texts, in order. @returns {string[]} */
function crumbs() {
  return Array.from(document.querySelectorAll('#crumb > :not(.osprey-crumb-sep)')).map(
    (crumb) => crumb.textContent?.trim() ?? ''
  );
}

/**
 * Type into the toolbar's search field and wait out its debounce.
 * @param {string} text
 */
async function typeSearch(text) {
  const input = /** @type {HTMLInputElement} */ (document.getElementById('search-input'));
  input.value = text;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 250));
  await flush();
}

beforeAll(async () => {
  history.replaceState(null, '', '/?embedded=true#devices/bpm');
  mountShell();
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ size: 320 }));
  vi.stubGlobal('marked', { parse: (/** @type {string} */ md) => md });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel tile-bar contribution (embedded)', () => {
  test('on a page the bar gets the concept id as its subtitle and the panel menu, only', () => {
    expect(lastContribution()).toEqual([
      { kind: 'text', id: 'concept', text: 'devices/bpm' },
      {
        kind: 'menu',
        id: 'knowledge',
        items: [{ id: 'reset-width', label: 'Reset contents width' }],
      },
    ]);
    for (const contribution of header.contributions) {
      for (const item of contribution) expect(['text', 'icon', 'menu']).toContain(item.kind);
    }
  });

  test('the toolbar names the page: Browse, its group, its title', () => {
    expect(crumbs()).toEqual(['Browse', 'Devices', 'BPM']);
    expect(contentsOpen()).toBe('true');
  });

  test('the bundle check shows as a status chip', () => {
    const chip = /** @type {HTMLButtonElement} */ (document.getElementById('bundle-health'));
    expect(chip.textContent?.trim()).toBe('Bundle healthy');
    expect(chip.classList.contains('osprey-chip--ok')).toBe(true);
    expect(chip.disabled).toBe(false);
  });

  test('the tile ⋯ entry resets the contents width', () => {
    const sidebar = /** @type {HTMLElement} */ (document.getElementById('sidebar'));
    expect(sidebar.style.width).toBe('320px');
    if (!header.onAction) throw new Error('app.js registered no header-action callback');
    header.onAction('knowledge', 'reset-width');
    expect(sidebar.style.width).toBe('260px');
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  test('embedded, the toolbar has no ⋯ of its own: the tile has the one ⋯', () => {
    expect(document.getElementById('panel-menu')).toBeNull();
  });

  test('Browse opens the overview, and the bar drops the subtitle', async () => {
    const browse = /** @type {HTMLElement} */ (document.querySelector('#crumb a'));
    expect(browse.textContent).toBe('Browse');
    browse.click();
    await flush();
    expect(fetched).toContain('/api/structure');
    expect(document.getElementById('reader-content')?.textContent).toContain(
      'Facility Knowledge Base'
    );
    expect(location.hash).toBe('#__structure');
    expect(crumbs()).toEqual(['Browse']);
    expect(document.querySelector('#crumb a')).toBeNull();
    expect(lastContribution().map((/** @type {any} */ item) => item.kind)).toEqual(['menu']);
    expect(contentsOpen()).toBe('false');
  });

  test('the toolbar search runs the panel search and opens the contents list for it', async () => {
    await typeSearch('bpm');
    expect(fetched).toContain('/api/search?q=bpm');
    expect(document.getElementById('search-results')?.hidden).toBe(false);
    expect(contentsOpen()).toBe('true');
    await typeSearch('');
    expect(document.getElementById('search-results')?.hidden).toBe(true);
    expect(contentsOpen()).toBe('false');
  });
});
