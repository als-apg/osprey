// @ts-check
/**
 * Sink-security regression for the ARIEL entries surface (Phase 4, Task 4.2):
 * drives a hostile payload through the five render sinks that survived the
 * entries.js -> entries-detail.js/entries-form.js/entries-helpers.js split
 * (2.2-2.4) and asserts the parsed DOM is inert on each of them:
 *   - renderEntriesList (entries.js, via the exported loadEntries)
 *   - renderEntryDetail (entries-detail.js, via the exported showEntry)
 *   - addTag (entries-form.js, via the exported loadDraft)
 *   - showImageLightbox (entries-detail.js, exported directly)
 *   - renderSessionInfoPanel (entries-form.js, via the exported loadDraft)
 *
 * Task 4.3 extends this file with the onclick -> delegated-listener
 * conversion's own regression: every site that used to interpolate a value
 * into an inline `onclick="..."` string (a JS-string-context sink that
 * escapeHtml's HTML-context escaping cannot make safe — see dom.js) now
 * carries that value only as a `data-*` attribute, read back via
 * `.dataset` and never re-parsed as JavaScript. The describes below drive a
 * JS-string-breakout payload through each converted sink and assert both
 * halves of the fix: (a) no element anywhere in the rendered surface carries
 * an `on*` attribute, and (b) the original click behavior still works,
 * unchanged, via the delegated listener.
 *
 *   npx vitest run tests/interfaces/ariel/render-xss.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). Only api.js is mocked (the network
 * boundary); components.js/entries-helpers.js/utils.js run for real so the
 * actual escapeHtml sinks are exercised end-to-end.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    entriesApi: { ...actual.entriesApi, list: vi.fn(), get: vi.fn() },
    draftsApi: { ...actual.draftsApi, get: vi.fn() },
    searchApi: { ...actual.searchApi, search: vi.fn() },
  };
});

import { entriesApi, draftsApi, searchApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import { loadEntries, initEntriesListDelegation } from '../../../src/osprey/interfaces/ariel/static/js/entries.js';
import { showEntry, showImageLightbox, initEntryDetail } from '../../../src/osprey/interfaces/ariel/static/js/entries-detail.js';
import { loadDraft, initEntryTags } from '../../../src/osprey/interfaces/ariel/static/js/entries-form.js';
import { performSearch, initSearchResultsDelegation } from '../../../src/osprey/interfaces/ariel/static/js/search.js';

/**
 * Single-quote-based payload: breaks out of any `"`-quoted HTML attribute
 * and would inject a live element if reflected unescaped, but — unlike a
 * double-quote payload — stays inert if it round-trips through the
 * addTag() duplicate-check selector (`[data-value="${value}"]`), so the
 * same constant is safe to drive through every sink in this file.
 */
const PAYLOAD = "'><img src=x onerror=alert(1)>";

/**
 * The payload that demonstrates the JS-string-context breakout the onclick
 * conversion (Task 4.3) fixes: reflected unescaped into
 * `onclick="window.app.showEntry('${id}')"`, the browser HTML-decodes the
 * attribute *before* the JS parser runs, so this closes the string early and
 * executes `alert(1)` on click — see the module docstring above. Every
 * converted sink must now carry it inertly as a `data-*` attribute instead.
 */
const JS_BREAKOUT_PAYLOAD = "'-alert(1)-'";

/** Both payload shapes the onclick-conversion tests drive through the converted sinks. */
const HOSTILE_PAYLOADS = [JS_BREAKOUT_PAYLOAD, PAYLOAD];

/**
 * Assert the PAYLOAD reached `root` as inert text: none of these templates
 * legitimately produce an <img>/<svg>/<script> element on the fixtures used
 * in this file, so any occurrence would mean the payload's own
 * `<img src=x onerror=alert(1)>` parsed as a live node instead of being
 * escaped. (The one sink that legitimately renders a real <img> —
 * showImageLightbox — asserts its own exact-count invariant separately,
 * below, rather than reusing this helper.)
 * @param {Element|null} root
 */
function assertPayloadInert(root) {
  expect(root, 'sink root exists').not.toBeNull();
  const el = /** @type {Element} */ (root);
  expect(el.querySelector('img'), 'no live <img> injected by the payload').toBeNull();
  expect(el.querySelector('svg'), 'no live <svg> injected by the payload').toBeNull();
  expect(el.querySelector('script'), 'no live <script> injected by the payload').toBeNull();
  // The sink was still reached: escaped entities decode back to the raw
  // payload in textContent.
  expect(el.textContent).toContain(PAYLOAD);
}

/**
 * Assert no element in `root` (inclusive) carries any `on*` attribute —
 * the direct proof that no interpolated value is reachable as an inline
 * event-handler string anymore, regardless of how it was escaped.
 * @param {Element|null} root
 */
function assertNoEventHandlerAttributes(root) {
  expect(root, 'root exists').not.toBeNull();
  const el = /** @type {Element} */ (root);
  const all = [el, ...Array.from(el.querySelectorAll('*'))];
  for (const node of all) {
    const offenders = node.getAttributeNames().filter(n => n.startsWith('on'));
    expect(offenders, `<${node.tagName.toLowerCase()}> has no on* attributes`).toEqual([]);
  }
}

/** Fresh DOM fixture covering every container the sinks under test read. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="entries-list"></div>
    <div id="search-results"></div>
    <div id="entry-modal" class="hidden"><div id="entry-modal-body"></div></div>
    <div id="create-entry-form">
      <div id="entry-tags"></div>
      <div id="file-preview"></div>
    </div>
  `;
}

beforeEach(() => {
  mountFixture();
  // Attach the delegated listeners once per test, matching production init
  // (initEntries/initSearch) — the fixture's containers exist before each
  // render, so this proves the listeners survive the innerHTML replacement
  // every loadEntries()/showEntry()/performSearch() call performs.
  initEntriesListDelegation();
  initEntryDetail();
  initEntryTags();
  initSearchResultsDelegation();
  vi.clearAllMocks();
});

describe('renderEntriesList (entries.js, via loadEntries)', () => {
  test('escapes hostile entry fields in the list and pagination header', async () => {
    vi.mocked(entriesApi.list).mockResolvedValueOnce({
      entries: [{
        entry_id: PAYLOAD,
        timestamp: '2026-01-01T00:00:00Z',
        author: PAYLOAD,
        source_system: PAYLOAD,
        raw_text: `${PAYLOAD}\nbody text`,
        score: null,
        attachments: [],
        keywords: [PAYLOAD],
        highlights: [],
      }],
      total: 1,
      page: 1,
      total_pages: 1,
    });

    await loadEntries();

    assertPayloadInert(document.getElementById('entries-list'));
  });
});

describe('renderEntryDetail (entries-detail.js, via showEntry)', () => {
  test('escapes hostile entry, attachment, and session-metadata fields', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce({
      entry_id: PAYLOAD,
      timestamp: '2026-01-01T00:00:00Z',
      author: PAYLOAD,
      source_system: PAYLOAD,
      raw_text: `${PAYLOAD}\ndetails ${PAYLOAD}`,
      keywords: [PAYLOAD],
      // filename/type deliberately don't resolve as an image, so the only
      // legitimate <img> in this suite's fixtures is the lightbox test's.
      attachments: [{ filename: PAYLOAD, url: PAYLOAD, type: PAYLOAD }],
      summary: PAYLOAD,
      metadata: {
        logbook: PAYLOAD,
        shift: PAYLOAD,
        session_metadata: {
          operator: PAYLOAD,
          session_id: PAYLOAD,
          git_branch: PAYLOAD,
          model_name: PAYLOAD,
          created_via: PAYLOAD,
          session_start_time: PAYLOAD,
        },
      },
    });

    await showEntry('entry-1');

    assertPayloadInert(document.getElementById('entry-modal-body'));
  });
});

describe('addTag + renderSessionInfoPanel (entries-form.js, via loadDraft)', () => {
  test('escapes a hostile tag and hostile session-metadata fields', async () => {
    vi.mocked(draftsApi.get).mockResolvedValueOnce({
      subject: '', details: '', author: '', logbook: '', shift: '',
      tags: [PAYLOAD],
      metadata: {
        session_metadata: {
          operator: PAYLOAD,
          git_branch: PAYLOAD,
          session_id: PAYLOAD,
          model: PAYLOAD,
          created_via: PAYLOAD,
        },
      },
    });

    await loadDraft('draft-1');

    assertPayloadInert(document.getElementById('entry-tags'));

    const panel = document.getElementById('session-info-panel');
    expect(panel, 'session info panel was rendered').not.toBeNull();
    assertPayloadInert(panel);
  });
});

describe('showImageLightbox (entries-detail.js, direct)', () => {
  test('escapes a hostile url and filename', () => {
    showImageLightbox(PAYLOAD, PAYLOAD);

    const overlay = document.getElementById('image-lightbox');
    expect(overlay, 'lightbox overlay rendered').not.toBeNull();
    const el = /** @type {Element} */ (overlay);
    expect(el.querySelector('svg'), 'no live <svg> injected by the payload').toBeNull();
    expect(el.querySelector('script'), 'no live <script> injected by the payload').toBeNull();
    // Exactly the lightbox's own structural <img> — the payload's embedded
    // `<img src=x ...>` did not parse as a second, live element.
    expect(el.querySelectorAll('img').length, 'exactly one legitimate <img>').toBe(1);
    expect(el.textContent).toContain(PAYLOAD);
  });
});

// ---------------------------------------------------------------------------
// Task 4.3: onclick -> delegated-listener conversion. Each describe below
// covers one converted sink: no on* attribute anywhere, the hostile value
// round-trips through `.dataset` unmangled, and the original click behavior
// still fires through the delegated listener attached in beforeEach.
// ---------------------------------------------------------------------------

describe('entry card (components.js renderEntryCard, via loadEntries + delegated click)', () => {
  test.each(HOSTILE_PAYLOADS)('carries entry_id %j as data only and dispatches showEntry via the delegated click', async (payload) => {
    vi.mocked(entriesApi.list).mockResolvedValueOnce({
      entries: [{
        entry_id: payload,
        timestamp: '2026-01-01T00:00:00Z',
        author: 'author',
        source_system: 'sys',
        raw_text: 'body',
        score: null,
        attachments: [],
        keywords: [],
        highlights: [],
      }],
      total: 1,
      page: 1,
      total_pages: 1,
    });

    await loadEntries();

    const list = /** @type {HTMLElement} */ (document.getElementById('entries-list'));
    assertNoEventHandlerAttributes(list);

    const card = /** @type {HTMLElement|null} */ (list.querySelector('[data-entry-id]'));
    expect(card, 'entry card rendered').not.toBeNull();
    // Exact round-trip through dataset — the value is carried as data, never
    // re-parsed as JavaScript (the bug: escapeHtml's `'` -> `&#39;` decodes
    // back to a literal `'` inside an inline onclick's JS string, breaking
    // out of it).
    expect(/** @type {HTMLElement} */ (card).dataset.entryId).toBe(payload);

    vi.mocked(entriesApi.get).mockResolvedValueOnce({
      entry_id: payload,
      timestamp: '2026-01-01T00:00:00Z',
      author: 'author',
      source_system: 'sys',
      raw_text: 'body',
      keywords: [],
      attachments: [],
    });

    /** @type {HTMLElement} */ (card).dispatchEvent(new MouseEvent('click', { bubbles: true }));

    // Proves the click reached the delegated handler AND carried the exact,
    // unmangled id through — not a JS-string-parsed fragment of it.
    await vi.waitFor(() => expect(entriesApi.get).toHaveBeenCalledWith(payload));
  });
});

describe('pagination (entries.js renderPagination, via delegated click)', () => {
  test('clicking Next re-loads the next page through the listener attached on #entries-list itself', async () => {
    vi.mocked(entriesApi.list).mockResolvedValueOnce({
      entries: [{
        entry_id: 'e1',
        timestamp: '2026-01-01T00:00:00Z',
        author: 'a',
        source_system: 's',
        raw_text: 'body',
        score: null,
        attachments: [],
        keywords: [],
        highlights: [],
      }],
      total: 40,
      page: 1,
      total_pages: 2,
    });

    await loadEntries();

    const list = /** @type {HTMLElement} */ (document.getElementById('entries-list'));
    assertNoEventHandlerAttributes(list);

    const nextBtn = /** @type {HTMLElement|null} */ (list.querySelector('[data-page="2"]'));
    expect(nextBtn, 'Next button rendered with data-page').not.toBeNull();

    vi.mocked(entriesApi.list).mockResolvedValueOnce({ entries: [], total: 40, page: 2, total_pages: 2 });

    // list.innerHTML was replaced wholesale by the loadEntries() call above —
    // this click only works if the listener lives on #entries-list itself
    // (attached once, in beforeEach) rather than on the discarded button.
    /** @type {HTMLElement} */ (nextBtn).dispatchEvent(new MouseEvent('click', { bubbles: true }));

    await vi.waitFor(() => expect(entriesApi.list).toHaveBeenCalledTimes(2));
    expect(vi.mocked(entriesApi.list).mock.calls[1][0]).toMatchObject({ page: 2 });
  });
});

describe('cited-source link (components.js renderAnswerBox, via performSearch + delegated click)', () => {
  test.each(HOSTILE_PAYLOADS)('carries source id %j as data only, suppresses navigation, and dispatches showEntry', async (payload) => {
    vi.mocked(searchApi.search).mockResolvedValueOnce({
      answer: 'The answer.',
      sources: [payload],
      search_modes_used: ['keyword'],
      execution_time_ms: 5,
      total_results: 0,
      entries: [],
      diagnostics: [],
    });

    await performSearch('some query');

    const results = /** @type {HTMLElement} */ (document.getElementById('search-results'));
    assertNoEventHandlerAttributes(results);

    const link = /** @type {HTMLElement|null} */ (results.querySelector('a[data-entry-id]'));
    expect(link, 'cited-source link rendered').not.toBeNull();
    expect(/** @type {HTMLElement} */ (link).dataset.entryId).toBe(payload);

    vi.mocked(entriesApi.get).mockResolvedValueOnce({
      entry_id: payload,
      timestamp: '2026-01-01T00:00:00Z',
      author: 'a',
      source_system: 's',
      raw_text: 'body',
      keywords: [],
      attachments: [],
    });

    const clickEvent = new MouseEvent('click', { bubbles: true, cancelable: true });
    /** @type {HTMLElement} */ (link).dispatchEvent(clickEvent);

    // The link is `href="#"`; the delegated handler must call preventDefault()
    // to preserve the old inline handler's `return false` (no navigation).
    expect(clickEvent.defaultPrevented, 'navigation suppressed').toBe(true);
    await vi.waitFor(() => expect(entriesApi.get).toHaveBeenCalledWith(payload));
  });
});

describe('attachment thumbnail (entries-detail.js renderEntryDetail, via showEntry + delegated click)', () => {
  test.each(HOSTILE_PAYLOADS)('carries lightbox url/filename %j as data only and opens the lightbox via delegated click', async (payload) => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce({
      entry_id: 'entry-1',
      timestamp: '2026-01-01T00:00:00Z',
      author: 'author',
      source_system: 'sys',
      raw_text: 'body',
      keywords: [],
      attachments: [{ filename: payload, url: payload, type: 'image/png' }],
    });

    await showEntry('entry-1');

    const modalBody = /** @type {HTMLElement} */ (document.getElementById('entry-modal-body'));
    assertNoEventHandlerAttributes(modalBody);

    const thumb = /** @type {HTMLElement|null} */ (modalBody.querySelector('[data-lightbox-url]'));
    expect(thumb, 'attachment thumbnail rendered').not.toBeNull();
    expect(/** @type {HTMLElement} */ (thumb).dataset.lightboxUrl).toBe(payload);
    expect(/** @type {HTMLElement} */ (thumb).dataset.lightboxName).toBe(payload);

    /** @type {HTMLElement} */ (thumb).dispatchEvent(new MouseEvent('click', { bubbles: true }));

    const overlay = document.getElementById('image-lightbox');
    expect(overlay, 'lightbox overlay opened via delegated click').not.toBeNull();
    expect(/** @type {Element} */ (overlay).textContent).toContain(payload);
  });
});

describe('tag remove button (entries-form.js addTag, via loadDraft + delegated click)', () => {
  test.each(HOSTILE_PAYLOADS)('renders hostile tag %j with no on* attributes and removes the chip via delegated click', async (payload) => {
    vi.mocked(draftsApi.get).mockResolvedValueOnce({
      subject: '', details: '', author: '', logbook: '', shift: '',
      tags: [payload],
      metadata: null,
    });

    await loadDraft('draft-1');

    const tagsContainer = /** @type {HTMLElement} */ (document.getElementById('entry-tags'));
    assertNoEventHandlerAttributes(tagsContainer);

    expect(tagsContainer.querySelector('.tag'), 'tag chip rendered').not.toBeNull();
    const removeBtn = /** @type {HTMLElement|null} */ (tagsContainer.querySelector('[data-tag-remove]'));
    expect(removeBtn, 'tag remove button rendered').not.toBeNull();

    /** @type {HTMLElement} */ (removeBtn).dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(tagsContainer.querySelector('.tag'), 'tag chip removed by the delegated click').toBeNull();
  });
});
