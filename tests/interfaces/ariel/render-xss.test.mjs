// @ts-check
/**
 * Sink-security regression for the ARIEL entries surface (Phase 4, Task 4.2):
 * drives a hostile payload through the entries surface's render sinks and
 * asserts the parsed DOM is inert on each of them:
 *   - the list rows (list.js, via initList and the empty-query listing)
 *   - renderEntryDetail (reader.js, via showInReader)
 *   - addTag (entries-form.js, via a draft opened in the compose host)
 *   - showImageLightbox (reader.js, exported directly)
 *   - renderSessionInfoPanel (entries-form.js, via a draft opened in the
 *     compose host)
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

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    entriesApi: {
      ...actual.entriesApi,
      list: vi.fn(),
      get: vi.fn(),
      getPublishInfo: vi.fn(),
      thread: vi.fn(),
    },
    draftsApi: { ...actual.draftsApi, get: vi.fn() },
    searchApi: { ...actual.searchApi, search: vi.fn() },
  };
});

import { entriesApi, draftsApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import { getState, setState } from '../../../src/osprey/interfaces/ariel/static/js/store.js';
import { initList } from '../../../src/osprey/interfaces/ariel/static/js/list.js';
import { setEnhancementModules } from '../../../src/osprey/interfaces/ariel/static/js/components.js';
import {
  showInReader,
  showImageLightbox,
} from '../../../src/osprey/interfaces/ariel/static/js/reader.js';
import { initEntries } from '../../../src/osprey/interfaces/ariel/static/js/entries.js';
import { entryFieldsRendered } from '../../../src/osprey/interfaces/ariel/static/js/entry-fields.js';

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

/** A well-formed rendition URL, the only shape a viewable thumbnail is drawn from. */
const RENDITION_URL = '/attachments/att-0123456789ab/rendition';

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
    const offenders = node.getAttributeNames().filter((n) => n.startsWith('on'));
    expect(offenders, `<${node.tagName.toLowerCase()}> has no on* attributes`).toEqual([]);
  }
}

/** Fresh DOM fixture covering every container the sinks under test read. */
function mountFixture() {
  document.body.innerHTML = `
    <section id="ariel-list"></section>
    <div id="ariel-reader-body"></div>
    <section id="ariel-compose" hidden>
      <form id="create-entry-form">
        <div id="entry-tags"></div>
        <div id="file-preview"></div>
      </form>
    </section>
  `;
}

/** @returns {HTMLElement} The list pane's body, where rows are painted. */
function listBody() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list-body'));
}

/** @returns {HTMLElement} The reader body showInReader renders into. */
function readerBody() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-reader-body'));
}

/** Let every pending microtask (and the awaits chained behind it) run. */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Re-run the empty-query listing (as clearing the search box does) and let it
 * paint.
 */
async function loadListing() {
  setState({ query: '', rerank: 'now' }, { force: true });
  await flush();
  await flush();
}

/**
 * Open `draftId` the way the panel does — through the store's `compose` key,
 * which the compose host turns into a draft load — and let it render.
 * @param {string} draftId
 */
async function openDraft(draftId) {
  setState({ compose: { draft: draftId } });
  for (let i = 0; i < 4; i++) await flush();
}

/** @type {{destroy(): void}|null} */
let composeHost = null;

/**
 * A listing page.
 * @param {any[]} entries
 * @param {{next_cursor?: string|null}} [paging]
 * @returns {any}
 */
function listPage(entries, { next_cursor = null } = {}) {
  return { entries, total: entries.length, page: null, total_pages: 1, next_cursor };
}

beforeEach(async () => {
  mountFixture();
  setState({
    query: '',
    mode: null,
    filters: {},
    mailbox: 'all',
    sel: null,
    narrowView: 'list',
    compose: null,
  });
  vi.mocked(entriesApi.list).mockResolvedValue(listPage([]));
  // The list pane binds its delegated listener on its stable body once; every
  // listing replaces the body's content, so the cases below prove the listener
  // survives the innerHTML replacement.
  initList(/** @type {HTMLElement} */ (document.getElementById('ariel-list')), {
    capabilities: null,
  });
  // The compose host renders the declared entry fields (none here) at init;
  // a draft load waits for that render to settle.
  vi.mocked(entriesApi.getPublishInfo).mockResolvedValue({
    supports_write: false,
    requires_auth: false,
    source_system: null,
    entry_fields: [],
  });
  composeHost = initEntries();
  await entryFieldsRendered();
  await flush();
  vi.clearAllMocks();
  vi.mocked(entriesApi.list).mockResolvedValue(listPage([]));
});

afterEach(() => {
  composeHost?.destroy();
  composeHost = null;
});

describe('list rows (list.js, via the empty-query listing)', () => {
  test('escapes hostile entry fields in the rows and the day separator', async () => {
    vi.mocked(entriesApi.list).mockResolvedValueOnce(
      listPage([
        {
          entry_id: PAYLOAD,
          timestamp: '2026-01-01T00:00:00Z',
          author: PAYLOAD,
          source_system: PAYLOAD,
          subject: PAYLOAD,
          snippet: `body ${PAYLOAD}`,
          score: null,
          attachment_count: 0,
          enhancements: { [PAYLOAD]: PAYLOAD, image_caption: PAYLOAD },
          keywords: [PAYLOAD],
          highlights: [],
          matched_via: [],
        },
      ])
    );

    await loadListing();

    assertPayloadInert(listBody());
  });

  test('a subject-less row escapes the snippet it falls back to', async () => {
    vi.mocked(entriesApi.list).mockResolvedValueOnce(
      listPage([
        {
          entry_id: 'e1',
          timestamp: '2026-01-01T00:00:00Z',
          author: 'a',
          source_system: 's',
          subject: null,
          snippet: `${PAYLOAD}\n${PAYLOAD}`,
          score: null,
          keywords: [],
          highlights: [],
        },
      ])
    );

    await loadListing();

    assertPayloadInert(listBody());
  });

  test('escapes hostile module names, labels and states through the badge strip', async () => {
    setEnhancementModules([
      { name: PAYLOAD, label: PAYLOAD },
      { name: 'image_caption', label: PAYLOAD },
    ]);
    try {
      vi.mocked(entriesApi.list).mockResolvedValueOnce(
        listPage([
          {
            entry_id: 'e1',
            timestamp: '2026-01-01T00:00:00Z',
            author: 'a',
            source_system: 's',
            subject: 'subject',
            snippet: 'body',
            enhancements: { [PAYLOAD]: PAYLOAD, image_caption: 'stale' },
            keywords: [],
            highlights: [],
          },
        ])
      );

      await loadListing();

      const list = listBody();
      const strip = /** @type {Element} */ (list?.querySelector('.ariel-strip'));
      expect(strip, 'strip rendered').not.toBeNull();
      expect(list?.querySelector('img'), 'no live <img> injected by the payload').toBeNull();
      assertNoEventHandlerAttributes(list);
      const cells = Array.from(strip.querySelectorAll('.ariel-strip-cell'));
      expect(cells.map((c) => /** @type {HTMLElement} */ (c).dataset.module)).toEqual([
        PAYLOAD,
        'image_caption',
      ]);
      expect(cells[0].getAttribute('aria-label')).toBe(`${PAYLOAD}: n/a`);
      expect(cells[1].getAttribute('title')).toBe(`${PAYLOAD}: stale`);
    } finally {
      setEnhancementModules([]);
    }
  });
});

describe('renderEntryDetail (reader.js, via showInReader)', () => {
  test('escapes hostile entry, attachment, and session-metadata fields', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce({
      entry_id: PAYLOAD,
      timestamp: '2026-01-01T00:00:00Z',
      author: PAYLOAD,
      source_system: PAYLOAD,
      raw_text: `${PAYLOAD}\ndetails ${PAYLOAD}`,
      keywords: [PAYLOAD],
      // No `viewable: true`, so this is a file card and the only legitimate
      // <img> in this suite's fixtures is the lightbox test's.
      attachments: [{ filename: PAYLOAD, url: PAYLOAD, mime_type: PAYLOAD }],
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

    await showInReader(readerBody(), 'entry-1');

    assertPayloadInert(readerBody());
  });
});

describe('addTag + renderSessionInfoPanel (entries-form.js, via a draft opened in compose)', () => {
  test('escapes a hostile tag and hostile session-metadata fields', async () => {
    vi.mocked(draftsApi.get).mockResolvedValueOnce({
      subject: '',
      details: '',
      author: '',
      logbook: '',
      shift: '',
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

    await openDraft('draft-1');

    assertPayloadInert(document.getElementById('entry-tags'));

    const panel = document.getElementById('session-info-panel');
    expect(panel, 'session info panel was rendered').not.toBeNull();
    assertPayloadInert(panel);
  });
});

describe('showImageLightbox (reader.js, direct)', () => {
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

describe('list row (list.js, via the listing + delegated click)', () => {
  test.each(HOSTILE_PAYLOADS)(
    'carries entry_id %j as data only and selects it via the delegated click',
    async (payload) => {
      vi.mocked(entriesApi.list).mockResolvedValueOnce(
        listPage([
          {
            entry_id: payload,
            timestamp: '2026-01-01T00:00:00Z',
            author: 'author',
            source_system: 'sys',
            snippet: 'body',
            score: null,
            attachment_count: 0,
            keywords: [],
            highlights: [],
          },
        ])
      );

      await loadListing();

      const list = listBody();
      assertNoEventHandlerAttributes(list);

      const row = /** @type {HTMLElement|null} */ (list.querySelector('[data-entry-id]'));
      expect(row, 'list row rendered').not.toBeNull();
      // Exact round-trip through dataset — the value is carried as data, never
      // re-parsed as JavaScript (the bug: escapeHtml's `'` -> `&#39;` decodes
      // back to a literal `'` inside an inline onclick's JS string, breaking
      // out of it).
      expect(/** @type {HTMLElement} */ (row).dataset.entryId).toBe(payload);

      /** @type {HTMLElement} */ (row).dispatchEvent(new MouseEvent('click', { bubbles: true }));

      // Proves the click reached the delegated handler AND carried the exact,
      // unmangled id through — not a JS-string-parsed fragment of it.
      expect(getState().sel).toBe(payload);
      expect(getState().narrowView).toBe('reader');
    }
  );
});

describe('load older (list.js, via delegated click)', () => {
  /** @param {string} id */
  const row = (id) => ({
    entry_id: id,
    timestamp: '2026-01-01T00:00:00Z',
    author: 'a',
    source_system: 's',
    snippet: 'body',
    score: null,
    attachment_count: 0,
    keywords: [],
    highlights: [],
  });

  test('clicking Load older appends the next page through the listener on the list body itself', async () => {
    vi.mocked(entriesApi.list).mockResolvedValueOnce(listPage([row('e1')], { next_cursor: 'C1' }));

    await loadListing();

    const list = listBody();
    assertNoEventHandlerAttributes(list);

    const older = /** @type {HTMLElement|null} */ (list.querySelector('[data-load-more]'));
    expect(older, 'Load older rendered').not.toBeNull();

    vi.mocked(entriesApi.list).mockResolvedValueOnce(listPage([row('e2')]));

    // The body's innerHTML was replaced wholesale by the listing above — this
    // click only works if the listener lives on the body itself (bound once,
    // at initList) rather than on the discarded button.
    /** @type {HTMLElement} */ (older).dispatchEvent(new MouseEvent('click', { bubbles: true }));

    await vi.waitFor(() => expect(entriesApi.list).toHaveBeenCalledTimes(2));
    expect(vi.mocked(entriesApi.list).mock.calls[1][0]).toMatchObject({ cursor: 'C1' });
    await vi.waitFor(() => expect(list.querySelector('[data-entry-id="e2"]')).not.toBeNull());
    expect(list.querySelector('[data-entry-id="e1"]'), 'older rows are appended').not.toBeNull();
  });

  test('Load older renders while older pages remain and is gone at the end', async () => {
    const list = listBody();

    vi.mocked(entriesApi.list).mockResolvedValueOnce(listPage([row('e1')], { next_cursor: 'C1' }));
    await loadListing();
    expect(list.querySelectorAll('[data-load-more]').length, 'one control at the end').toBe(1);

    vi.mocked(entriesApi.list).mockResolvedValueOnce(listPage([row('e2')]));
    /** @type {HTMLElement} */ (list.querySelector('[data-load-more]')).click();
    await vi.waitFor(() => expect(list.querySelector('[data-entry-id="e2"]')).not.toBeNull());
    expect(list.querySelector('[data-load-more]'), 'no control past the last page').toBeNull();

    vi.mocked(entriesApi.list).mockResolvedValueOnce(listPage([row('only')]));
    await loadListing();
    expect(list.querySelector('[data-load-more]'), 'no control on a single page').toBeNull();
  });
});

describe('attachment thumbnail (reader.js renderEntryDetail, via showInReader + delegated click)', () => {
  test.each(HOSTILE_PAYLOADS)(
    'carries lightbox filename %j as data only and opens the lightbox via delegated click',
    async (payload) => {
      vi.mocked(entriesApi.get).mockResolvedValueOnce({
        entry_id: 'entry-1',
        timestamp: '2026-01-01T00:00:00Z',
        author: 'author',
        source_system: 'sys',
        raw_text: 'body',
        keywords: [],
        attachments: [
          { filename: payload, viewable: true, display_url: RENDITION_URL, mime_type: 'image/png' },
        ],
      });

      await showInReader(readerBody(), 'entry-1');

      const modalBody = readerBody();
      assertNoEventHandlerAttributes(modalBody);

      const thumb = /** @type {HTMLElement|null} */ (
        modalBody.querySelector('[data-lightbox-url]')
      );
      expect(thumb, 'attachment thumbnail rendered').not.toBeNull();
      expect(/** @type {HTMLElement} */ (thumb).dataset.lightboxUrl).toBe(RENDITION_URL);
      expect(/** @type {HTMLElement} */ (thumb).dataset.lightboxName).toBe(payload);

      /** @type {HTMLElement} */ (thumb).dispatchEvent(new MouseEvent('click', { bubbles: true }));

      const overlay = document.getElementById('image-lightbox');
      expect(overlay, 'lightbox overlay opened via delegated click').not.toBeNull();
      expect(/** @type {Element} */ (overlay).textContent).toContain(payload);
    }
  );

  test.each(HOSTILE_PAYLOADS)(
    'a hostile display_url %j renders no thumbnail and no href',
    async (payload) => {
      vi.mocked(entriesApi.get).mockResolvedValueOnce({
        entry_id: 'entry-1',
        timestamp: '2026-01-01T00:00:00Z',
        author: 'author',
        source_system: 'sys',
        raw_text: 'body',
        keywords: [],
        attachments: [
          { filename: 'beam.png', viewable: true, display_url: payload, mime_type: 'image/png' },
        ],
      });

      await showInReader(readerBody(), 'entry-1');

      const modalBody = readerBody();
      assertNoEventHandlerAttributes(modalBody);
      expect(modalBody.querySelector('img'), 'no thumbnail').toBeNull();
      expect(modalBody.querySelector('[data-lightbox-url]'), 'no lightbox target').toBeNull();
      expect(modalBody.querySelector('[href]'), 'no href').toBeNull();
      expect(modalBody.textContent).toContain('beam.png');
    }
  );
});

describe('tag remove button (entries-form.js addTag, via a draft opened in compose + delegated click)', () => {
  test.each(HOSTILE_PAYLOADS)(
    'renders hostile tag %j with no on* attributes and removes the chip via delegated click',
    async (payload) => {
      vi.mocked(draftsApi.get).mockResolvedValueOnce({
        subject: '',
        details: '',
        author: '',
        logbook: '',
        shift: '',
        tags: [payload],
        metadata: null,
      });

      await openDraft('draft-1');

      const tagsContainer = /** @type {HTMLElement} */ (document.getElementById('entry-tags'));
      assertNoEventHandlerAttributes(tagsContainer);

      expect(tagsContainer.querySelector('.tag'), 'tag chip rendered').not.toBeNull();
      const removeBtn = /** @type {HTMLElement|null} */ (
        tagsContainer.querySelector('[data-tag-remove]')
      );
      expect(removeBtn, 'tag remove button rendered').not.toBeNull();

      /** @type {HTMLElement} */ (removeBtn).dispatchEvent(
        new MouseEvent('click', { bubbles: true })
      );

      expect(
        tagsContainer.querySelector('.tag'),
        'tag chip removed by the delegated click'
      ).toBeNull();
    }
  );
});
