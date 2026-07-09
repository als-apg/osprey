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
  };
});

import { entriesApi, draftsApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import { loadEntries } from '../../../src/osprey/interfaces/ariel/static/js/entries.js';
import { showEntry, showImageLightbox } from '../../../src/osprey/interfaces/ariel/static/js/entries-detail.js';
import { loadDraft } from '../../../src/osprey/interfaces/ariel/static/js/entries-form.js';

/**
 * Single-quote-based payload: breaks out of any `"`-quoted HTML attribute
 * and would inject a live element if reflected unescaped, but — unlike a
 * double-quote payload — stays inert if it round-trips through the
 * addTag() duplicate-check selector (`[data-value="${value}"]`), so the
 * same constant is safe to drive through every sink in this file.
 */
const PAYLOAD = "'><img src=x onerror=alert(1)>";

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

/** Fresh DOM fixture covering every container the five sinks under test read. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="entries-list"></div>
    <div id="entry-modal" class="hidden"><div id="entry-modal-body"></div></div>
    <div id="create-entry-form">
      <div id="entry-tags"></div>
      <div id="file-preview"></div>
    </div>
  `;
}

beforeEach(() => {
  mountFixture();
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
