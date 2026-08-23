// @ts-check
/**
 * Front-end proofs for the ARIEL vocabulary-expansion surface (Task 3.5):
 *
 *   - the degraded-configuration banners `initSearch(capabilities)` renders
 *     above the search form: blocking `#config-invalid-banner` (search controls
 *     disabled) on `configuration_invalid`, non-blocking `#config-warning-banner`
 *     on `configuration_warning`, nothing at all otherwise;
 *   - the Expert-mode query-expansion strip: ONE `.expansion-pair` per
 *     `expanded_terms` group, with a group's several canonicals comma-joined
 *     inside that single pair (`ts → troubleshoot, timing system`) — never one
 *     pair per canonical — absent in Simple mode and when nothing expanded, and
 *     escaped like every other render sink.
 *
 *   npx vitest run tests/interfaces/ariel/vocabulary-banner.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). Only api.js is mocked (the network
 * boundary); components.js and search.js run for real so the actual render
 * sinks are exercised end-to-end.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    searchApi: { ...actual.searchApi, search: vi.fn() },
  };
});

import { searchApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import { initSearch, performSearch } from '../../../src/osprey/interfaces/ariel/static/js/search.js';

/** The search view as index.html lays it out: the form container plus its controls. */
function mountFixture() {
  document.body.innerHTML = `
    <div class="search-form" id="search-form">
      <div class="search-input-wrapper">
        <input type="text" id="search-input" class="input">
        <div class="search-input-actions">
          <button id="search-btn" class="btn btn-primary">Search</button>
        </div>
      </div>
    </div>
    <div id="search-results"></div>
  `;
}

/** @returns {HTMLElement} */
function results() {
  return /** @type {HTMLElement} */ (document.getElementById('search-results'));
}

/** @returns {HTMLButtonElement} */
function searchBtn() {
  return /** @type {HTMLButtonElement} */ (document.getElementById('search-btn'));
}

/** @returns {HTMLInputElement} */
function searchInput() {
  return /** @type {HTMLInputElement} */ (document.getElementById('search-input'));
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
  mountFixture();
  vi.clearAllMocks();
});

afterEach(() => {
  document.documentElement.removeAttribute('data-ui-mode');
});

describe('degraded-configuration banner (initSearch)', () => {
  test('configuration_invalid blocks the form: banner + disabled controls', () => {
    initSearch(/** @type {any} */ ({
      status: 'configuration_invalid',
      config_errors: ['ariel.vocabulary.path: /x/vocabulary.yml — not found'],
      remedy: 'disable ariel.vocabulary.enabled or repoint ariel.vocabulary.path, then restart',
      categories: { direct: { modes: [] } },
      default_mode: null,
      shared_parameters: [],
      vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
    }));

    const banner = document.getElementById('config-invalid-banner');
    expect(banner, 'blocking banner rendered').not.toBeNull();
    const el = /** @type {HTMLElement} */ (banner);
    expect(el.classList.contains('config-banner')).toBe(true);
    expect(el.classList.contains('config-banner-invalid')).toBe(true);
    expect(el.getAttribute('role')).toBe('alert');
    expect(text(el)).toContain('ariel.vocabulary.path: /x/vocabulary.yml — not found');
    expect(text(el)).toContain('disable ariel.vocabulary.enabled');

    // The banner is the form's first child; the button survives (disabled), so
    // the browser tests can still see it.
    const form = /** @type {HTMLElement} */ (document.getElementById('search-form'));
    expect(form.firstElementChild).toBe(banner);
    expect(searchBtn(), 'search button still in the DOM').not.toBeNull();
    expect(searchBtn().disabled, 'search button disabled').toBe(true);
    expect(searchInput().disabled, 'search input disabled').toBe(true);
  });

  test('configuration_warning is non-blocking: banner rendered, controls live', () => {
    initSearch(/** @type {any} */ ({
      status: 'configuration_warning',
      config_errors: ['search_modules.semantic.model: not set'],
      remedy: 'fix the named key in config.yml and restart',
      categories: { direct: { label: 'Direct', modes: [] } },
      default_mode: 'keyword',
      shared_parameters: [],
      vocabulary: { enabled: true, concepts: 12, expand_by_default: true },
    }));

    expect(document.getElementById('config-invalid-banner'), 'not the blocking banner').toBeNull();
    const banner = document.getElementById('config-warning-banner');
    expect(banner, 'warning banner rendered').not.toBeNull();
    const el = /** @type {HTMLElement} */ (banner);
    expect(el.classList.contains('config-banner')).toBe(true);
    expect(el.classList.contains('config-banner-warning')).toBe(true);
    expect(text(el)).toContain('search_modules.semantic.model: not set');
    expect(text(el)).toContain('fix the named key in config.yml and restart');

    expect(searchBtn().disabled, 'search button stays enabled').toBe(false);
    expect(searchInput().disabled, 'search input stays enabled').toBe(false);
  });

  test('a healthy payload renders no banner', () => {
    initSearch(/** @type {any} */ ({
      categories: { direct: { label: 'Direct', modes: [] } },
      default_mode: 'keyword',
      shared_parameters: [],
      vocabulary: { enabled: true, concepts: 12, expand_by_default: true },
    }));

    expect(document.querySelector('[data-testid="config-banner"]')).toBeNull();
    expect(searchBtn().disabled).toBe(false);
  });

  test('a failed capabilities fetch (null) renders no banner and does not throw', () => {
    expect(() => initSearch(null)).not.toThrow();
    expect(document.querySelector('[data-testid="config-banner"]')).toBeNull();
    expect(searchBtn().disabled).toBe(false);
  });

  test('no argument at all keeps the pre-existing call site working', () => {
    expect(() => initSearch()).not.toThrow();
    expect(document.querySelector('[data-testid="config-banner"]')).toBeNull();
  });

  test('a missing #search-form (embedded/partial DOM) is a no-op, not a crash', () => {
    document.body.innerHTML = '<div id="search-results"></div>';
    expect(() => initSearch(/** @type {any} */ ({
      status: 'configuration_invalid',
      config_errors: ['boom'],
      remedy: null,
    }))).not.toThrow();
    expect(document.querySelector('[data-testid="config-banner"]')).toBeNull();
  });
});

describe('query-expansion strip (Expert results header)', () => {
  test('a group with several canonicals renders ONE comma-joined pair', async () => {
    vi.mocked(searchApi.search).mockResolvedValueOnce(
      searchResponse([{ original: 'ts', alternatives: ['troubleshoot', 'timing system'] }])
    );

    await performSearch('ts fault');

    const banner = results().querySelector('[data-testid="expansion-banner"]');
    expect(banner, 'expansion banner rendered in Expert mode').not.toBeNull();
    const pairs = results().querySelectorAll('.expansion-pair');
    expect(pairs.length, 'one pair per expanded_terms group').toBe(1);
    expect(text(pairs[0])).toBe('ts → troubleshoot, timing system');
  });

  test('two groups render two pairs', async () => {
    vi.mocked(searchApi.search).mockResolvedValueOnce(
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
    vi.mocked(searchApi.search).mockResolvedValueOnce(searchResponse([]));
    await performSearch('nothing to expand');
    expect(results().querySelector('[data-testid="expansion-banner"]')).toBeNull();
  });

  test('no banner when the response omits expanded_terms entirely', async () => {
    const response = searchResponse([]);
    delete response.expanded_terms;
    vi.mocked(searchApi.search).mockResolvedValueOnce(response);
    await performSearch('older backend');
    expect(results().querySelector('[data-testid="expansion-banner"]')).toBeNull();
  });

  test('no banner in Simple mode', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    vi.mocked(searchApi.search).mockResolvedValueOnce(
      searchResponse([{ original: 'ts', alternatives: ['troubleshoot', 'timing system'] }])
    );

    await performSearch('ts fault');

    expect(results().querySelector('[data-testid="expansion-banner"]')).toBeNull();
    expect(results().querySelector('.expansion-pair')).toBeNull();
  });

  test('hostile expansion terms are escaped, not parsed', async () => {
    const payload = "'><img src=x onerror=alert(1)>";
    vi.mocked(searchApi.search).mockResolvedValueOnce(
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
