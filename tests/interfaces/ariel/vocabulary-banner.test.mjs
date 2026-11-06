// @ts-check
/**
 * Front-end proofs for the ARIEL vocabulary-expansion surface, on the list
 * pane (list.js):
 *
 *   - the degraded-configuration banners `initList(root, {capabilities})`
 *     renders above the list: blocking `#config-invalid-banner` (search box and
 *     mode chip disabled) on `configuration_invalid`, non-blocking
 *     `#config-warning-banner` on `configuration_warning`, nothing at all
 *     otherwise — embedded too, where the search box belongs to the host;
 *   - the query-expansion strip: ONE `.expansion-pair` per
 *     `expanded_terms` group, with a group's several canonicals comma-joined
 *     inside that single pair (`ts → troubleshoot, timing system`) — never one
 *     pair per canonical — absent when nothing expanded, and
 *     escaped like every other render sink.
 *
 *   npx vitest run tests/interfaces/ariel/vocabulary-banner.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). Only api.js is mocked (the network
 * boundary); components.js, search.js and list.js run for real so the actual
 * render sinks are exercised end-to-end.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    searchApi: { ...actual.searchApi, search: vi.fn() },
    entriesApi: { ...actual.entriesApi, list: vi.fn(), get: vi.fn() },
  };
});

const API_PATH = '../../../src/osprey/interfaces/ariel/static/js/api.js';
const STORE_PATH = '../../../src/osprey/interfaces/ariel/static/js/store.js';
const LIST_PATH = '../../../src/osprey/interfaces/ariel/static/js/list.js';

/** @type {any} */
let api;
/** @type {any} */
let store;

/** The list pane root, as index.html provides it. */
function mountFixture() {
  document.body.innerHTML = '<section id="ariel-list"></section>';
}

/**
 * Mount the list pane from a fresh module graph.
 * @param {any} [capabilities]
 */
async function initList(capabilities) {
  api = await import(API_PATH);
  store = await import(STORE_PATH);
  const list = await import(LIST_PATH);
  vi.mocked(api.entriesApi.list).mockResolvedValue({
    entries: [],
    total: 0,
    page: 1,
    total_pages: 1,
  });
  list.initList(/** @type {HTMLElement} */ (document.getElementById('ariel-list')), {
    capabilities,
  });
  await flush();
}

/**
 * Run a query the way the search box's Enter does, and let it paint.
 * @param {string} query
 */
async function performSearch(query) {
  store.setState({ query, rerank: 'now' }, { force: true });
  await flush();
  await flush();
}

/** Let every pending microtask (and the awaits chained behind it) run. */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** @returns {HTMLElement} */
function results() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list-body'));
}

/** @returns {HTMLButtonElement} */
function modeChip() {
  return /** @type {HTMLButtonElement} */ (document.getElementById('ariel-mode-chip'));
}

/** @returns {HTMLInputElement} */
function searchInput() {
  return /** @type {HTMLInputElement} */ (document.getElementById('ariel-search-input'));
}

/**
 * Collapse the render's template whitespace so a pair's text can be compared
 * to the exact string the proposal pins.
 * @param {Element} el
 * @returns {string}
 */
function text(el) {
  return (el.textContent ?? '').replace(/\s+/g, ' ').trim();
}

/**
 * A search response carrying the given expansions.
 * @param {{original: string, alternatives: string[]}[]} expandedTerms
 * @returns {any}
 */
function searchResponse(expandedTerms) {
  return {
    answer: '',
    sources: [],
    search_modes_used: ['keyword'],
    execution_time_ms: 5,
    total_results: 0,
    entries: [],
    diagnostics: [],
    expanded_terms: expandedTerms,
  };
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  history.replaceState(null, '', '/');
  mountFixture();
});

afterEach(() => {
  history.replaceState(null, '', '/');
});

describe('degraded-configuration banner (initList)', () => {
  test('configuration_invalid blocks search: banner + disabled controls', async () => {
    await initList(
      /** @type {any} */ ({
        status: 'configuration_invalid',
        config_errors: ['ariel.vocabulary.path: /x/vocabulary.yml — not found'],
        remedy: 'disable ariel.vocabulary.enabled or repoint ariel.vocabulary.path, then restart',
        categories: { direct: { modes: [] } },
        default_mode: null,
        shared_parameters: [],
        vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
      })
    );

    const banner = document.getElementById('config-invalid-banner');
    expect(banner, 'blocking banner rendered').not.toBeNull();
    const el = /** @type {HTMLElement} */ (banner);
    expect(el.classList.contains('config-banner')).toBe(true);
    expect(el.classList.contains('config-banner-invalid')).toBe(true);
    expect(el.getAttribute('role')).toBe('alert');
    expect(text(el)).toContain('ariel.vocabulary.path: /x/vocabulary.yml — not found');
    expect(text(el)).toContain('disable ariel.vocabulary.enabled');

    // The banner sits in the list pane, above the rows; the controls survive
    // (disabled), so the browser tests can still see them.
    const pane = /** @type {HTMLElement} */ (document.getElementById('ariel-list'));
    expect(pane.contains(/** @type {Node} */ (banner))).toBe(true);
    expect(results().contains(/** @type {Node} */ (banner)), 'not wiped by a list paint').toBe(
      false
    );
    expect(modeChip(), 'mode chip still in the DOM').not.toBeNull();
    expect(modeChip().disabled, 'mode chip disabled').toBe(true);
    expect(searchInput().disabled, 'search input disabled').toBe(true);
  });

  test('configuration_warning is non-blocking: banner rendered, controls live', async () => {
    await initList(
      /** @type {any} */ ({
        status: 'configuration_warning',
        config_errors: ['search_modules.semantic.model: not set'],
        remedy: 'fix the named key in config.yml and restart',
        categories: { direct: { label: 'Direct', modes: [] } },
        default_mode: 'keyword',
        shared_parameters: [],
        vocabulary: { enabled: true, concepts: 12, expand_by_default: true },
      })
    );

    expect(document.getElementById('config-invalid-banner'), 'not the blocking banner').toBeNull();
    const banner = document.getElementById('config-warning-banner');
    expect(banner, 'warning banner rendered').not.toBeNull();
    const el = /** @type {HTMLElement} */ (banner);
    expect(el.classList.contains('config-banner')).toBe(true);
    expect(el.classList.contains('config-banner-warning')).toBe(true);
    expect(text(el)).toContain('search_modules.semantic.model: not set');
    expect(text(el)).toContain('fix the named key in config.yml and restart');

    expect(modeChip().disabled, 'mode chip stays enabled').toBe(false);
    expect(searchInput().disabled, 'search input stays enabled').toBe(false);
  });

  test('a healthy payload renders no banner', async () => {
    await initList(
      /** @type {any} */ ({
        categories: { direct: { label: 'Direct', modes: [] } },
        default_mode: 'keyword',
        shared_parameters: [],
        vocabulary: { enabled: true, concepts: 12, expand_by_default: true },
      })
    );

    expect(document.querySelector('[data-testid="config-banner"]')).toBeNull();
    expect(searchInput().disabled).toBe(false);
  });

  test('a failed capabilities fetch (null) renders no banner and does not throw', async () => {
    await expect(initList(null)).resolves.toBeUndefined();
    expect(document.querySelector('[data-testid="config-banner"]')).toBeNull();
    expect(searchInput().disabled).toBe(false);
  });

  test('no capabilities at all keeps the call site working', async () => {
    await expect(initList()).resolves.toBeUndefined();
    expect(document.querySelector('[data-testid="config-banner"]')).toBeNull();
  });

  test('embedded, with no search box of its own, the list still shows the banner', async () => {
    history.replaceState(null, '', '/?embedded=true');
    await initList(
      /** @type {any} */ ({
        status: 'configuration_invalid',
        config_errors: ['boom'],
        remedy: null,
      })
    );
    expect(searchInput(), 'the search box belongs to the host').toBeNull();
    const banner = document.getElementById('config-invalid-banner');
    expect(banner, 'banner rendered in the list pane').not.toBeNull();
    expect(banner?.textContent ?? '').toContain('boom');
  });
});

describe('query-expansion strip (results header)', () => {
  beforeEach(async () => {
    await initList(/** @type {any} */ (null));
  });

  test('a group with several canonicals renders ONE comma-joined pair', async () => {
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([{ original: 'ts', alternatives: ['troubleshoot', 'timing system'] }])
    );

    await performSearch('ts fault');

    const banner = results().querySelector('[data-testid="expansion-banner"]');
    expect(banner, 'expansion banner rendered').not.toBeNull();
    const pairs = results().querySelectorAll('.expansion-pair');
    expect(pairs.length, 'one pair per expanded_terms group').toBe(1);
    expect(text(pairs[0])).toBe('ts → troubleshoot, timing system');
  });

  test('two groups render two pairs', async () => {
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([
        { original: 'ts', alternatives: ['troubleshoot'] },
        { original: 'bpm', alternatives: ['beam position monitor'] },
      ])
    );

    await performSearch('ts bpm');

    const pairs = results().querySelectorAll('.expansion-pair');
    expect(pairs.length).toBe(2);
    expect(text(pairs[0])).toBe('ts → troubleshoot');
    expect(text(pairs[1])).toBe('bpm → beam position monitor');
  });

  test('no banner when nothing expanded', async () => {
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(searchResponse([]));
    await performSearch('nothing to expand');
    expect(results().querySelector('[data-testid="expansion-banner"]')).toBeNull();
  });

  test('no banner when the response omits expanded_terms entirely', async () => {
    const response = searchResponse([]);
    delete response.expanded_terms;
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(response);
    await performSearch('older backend');
    expect(results().querySelector('[data-testid="expansion-banner"]')).toBeNull();
  });

  test('hostile expansion terms are escaped, not parsed', async () => {
    const payload = "'><img src=x onerror=alert(1)>";
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([{ original: payload, alternatives: [payload] }])
    );

    await performSearch('xss');

    const banner = /** @type {HTMLElement} */ (
      results().querySelector('[data-testid="expansion-banner"]')
    );
    expect(banner, 'banner rendered').not.toBeNull();
    expect(banner.querySelector('img'), 'no live <img> injected').toBeNull();
    expect(banner.querySelector('script'), 'no live <script> injected').toBeNull();
    expect(banner.textContent).toContain(payload);
    for (const node of [banner, ...Array.from(banner.querySelectorAll('*'))]) {
      expect(node.getAttributeNames().filter((n) => n.startsWith('on'))).toEqual([]);
    }
  });
});
