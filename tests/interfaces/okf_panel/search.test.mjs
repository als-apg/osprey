// @ts-check
/**
 * Unit tests for the OKF Knowledge Panel's search-results module (search.js):
 *   npx vitest run tests/interfaces/okf_panel/search.test.mjs
 *
 * `/api/search` answers from one of two backends and says which in the `score`
 * field of every hit: a number when the qmd sidecar ranked the results, `null`
 * when the substring fallback produced them. Both are normal deployment
 * states, so both presentations are pinned here:
 *
 *   ranked   -> <ol> + a per-item rank + a caption naming the ordering
 *   unranked -> the plain <ul> of link-plus-snippet cards, unchanged
 *
 * The rank-derived score itself must never reach the page as a number: read
 * as "0.33" or "33%" beside a document, it would claim a calibrated
 * confidence qmd does not compute. The last test in the ranked block is the
 * standing gate on that.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

/** @typedef {{id: string, title?: string, snippet?: string, score?: number|null}} SearchResultItem */

/** @type {typeof import('../../../src/osprey/interfaces/okf_panel/static/js/search.js')} */
let search;
/** @type {HTMLElement} */
let containerEl;
/** @type {import('vitest').Mock<(id: string) => void>} */
let onSelect;

/** @type {SearchResultItem[]} */
const RANKED = [
  { id: 'devices/rf-system', title: 'RF System', snippet: 'The RF system provides', score: 1 },
  { id: 'devices/bpm', title: 'Beam Position Monitor', snippet: 'Measures position', score: 0.5 },
  { id: 'references/glossary', title: 'Glossary', snippet: 'Facility terminology.', score: 0.33 },
];

/** @type {SearchResultItem[]} */
const UNRANKED = [
  { id: 'devices/rf-system', title: 'RF System', snippet: 'The RF system provides', score: null },
  { id: 'devices/bpm', title: 'Beam Position Monitor', snippet: 'Measures position', score: null },
];

// `search.js` keeps its container handle and select callback as module-private
// `let`s, exactly as `tree.js` does. A fresh module instance per test (rather
// than a re-`init` on a shared one) keeps that state from leaking between
// tests -- see the same note in tree.test.mjs.
beforeEach(async () => {
  vi.resetModules();
  document.body.innerHTML = '<div id="search-results" hidden></div>';
  containerEl = /** @type {HTMLElement} */ (document.getElementById('search-results'));
  onSelect = vi.fn();
  search = await import('../../../src/osprey/interfaces/okf_panel/static/js/search.js');
  search.initSearchResults({ containerEl, onSelect });
});

describe('isRanked', () => {
  test('a scored result set is ranked', () => {
    expect(search.isRanked(RANKED)).toBe(true);
  });

  test('a null-scored result set is not ranked', () => {
    expect(search.isRanked(UNRANKED)).toBe(false);
  });

  test('a result set with no score field at all is not ranked', () => {
    expect(search.isRanked([{ id: 'devices/bpm' }])).toBe(false);
  });

  test('a zero score still counts as ranked', () => {
    // 0 is a legitimate qmd score; only a non-number means "unranked".
    expect(search.isRanked([{ id: 'devices/bpm', score: 0 }])).toBe(true);
  });
});

describe('unranked results (no sidecar)', () => {
  test('renders the plain <ul> card list', () => {
    search.renderSearchResults(UNRANKED);

    const list = containerEl.querySelector('.result-list');
    expect(list && list.tagName).toBe('UL');
    expect(containerEl.querySelectorAll('.result-item').length).toBe(2);
    expect(containerEl.querySelector('.result-link')?.textContent).toBe('RF System');
    expect(containerEl.querySelector('.result-snippet')?.textContent).toBe(
      'The RF system provides'
    );
    expect(containerEl.hidden).toBe(false);
  });

  test('shows no rank and no ordering caption', () => {
    search.renderSearchResults(UNRANKED);

    expect(containerEl.querySelector('.result-rank')).toBe(null);
    expect(containerEl.querySelector('.result-mode')).toBe(null);
    expect(containerEl.querySelector('.result-list-ranked')).toBe(null);
  });
});

describe('ranked results (sidecar answering)', () => {
  test('renders an ordered list in the order given', () => {
    search.renderSearchResults(RANKED);

    const list = containerEl.querySelector('.result-list');
    expect(list && list.tagName).toBe('OL');
    expect(list?.classList.contains('result-list-ranked')).toBe(true);
    const titles = [...containerEl.querySelectorAll('.result-link')].map((a) => a.textContent);
    expect(titles).toEqual(['RF System', 'Beam Position Monitor', 'Glossary']);
  });

  test('each item carries its 1-based rank', () => {
    search.renderSearchResults(RANKED);

    const ranks = [...containerEl.querySelectorAll('.result-rank')].map((s) => s.textContent);
    expect(ranks).toEqual(['1', '2', '3']);
  });

  test('the rank span is hidden from assistive tech', () => {
    // The <ol> already conveys position; an exposed span would double-announce.
    search.renderSearchResults(RANKED);

    const ranks = [...containerEl.querySelectorAll('.result-rank')];
    expect(ranks.length).toBe(3);
    expect(ranks.every((s) => s.getAttribute('aria-hidden') === 'true')).toBe(true);
  });

  test('a caption names the ordering', () => {
    search.renderSearchResults(RANKED);

    expect(containerEl.querySelector('.result-mode')?.textContent).toBe('Ranked by relevance');
  });

  test('the raw score never reaches the page', () => {
    search.renderSearchResults(RANKED);

    const text = containerEl.textContent || '';
    expect(text).not.toContain('0.5');
    expect(text).not.toContain('0.33');
    expect(text).not.toContain('%');
  });
});

describe('shared behaviour', () => {
  test('an empty result set reports no matches in either mode', () => {
    search.renderSearchResults([]);

    expect(containerEl.textContent).toBe('No matches.');
    expect(containerEl.querySelector('.result-list')).toBe(null);
  });

  test('clicking a result selects its concept without navigating', () => {
    search.renderSearchResults(RANKED);

    const link = /** @type {HTMLAnchorElement} */ (containerEl.querySelector('.result-link'));
    const ev = new MouseEvent('click', { bubbles: true, cancelable: true });
    link.dispatchEvent(ev);

    expect(onSelect).toHaveBeenCalledWith('devices/rf-system');
    expect(ev.defaultPrevented).toBe(true);
  });

  test('a result with no snippet renders no snippet line', () => {
    search.renderSearchResults([{ id: 'devices/bpm', title: 'BPM', score: 1 }]);

    expect(containerEl.querySelector('.result-snippet')).toBe(null);
  });

  test('a result with no title falls back to its concept id', () => {
    search.renderSearchResults([{ id: 'devices/bpm', score: null }]);

    expect(containerEl.querySelector('.result-link')?.textContent).toBe('devices/bpm');
  });

  test('re-rendering replaces the previous results', () => {
    search.renderSearchResults(RANKED);
    search.renderSearchResults(UNRANKED);

    expect(containerEl.querySelectorAll('.result-item').length).toBe(2);
    expect(containerEl.querySelector('.result-mode')).toBe(null);
    expect(containerEl.querySelector('.result-list')?.tagName).toBe('UL');
  });

  test('clearSearchResults hides and empties the panel', () => {
    search.renderSearchResults(RANKED);
    search.clearSearchResults();

    expect(containerEl.hidden).toBe(true);
    expect(containerEl.innerHTML).toBe('');
  });

  test('renderSearchMessage shows one muted line', () => {
    search.renderSearchMessage('Search failed.');

    expect(containerEl.hidden).toBe(false);
    expect(containerEl.querySelector('.muted')?.textContent).toBe('Search failed.');
  });
});
