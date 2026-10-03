// @ts-check
/**
 * Attachment rendering in the ARIEL web UI:
 *   npx vitest run tests/interfaces/ariel/attachments-render.test.mjs
 *
 * The entry detail view draws an <img> only for attachments the server marks
 * `viewable: true`, always from `display_url`, and every href/src built from
 * attachment data passes `safeHref` (http(s) or `/api/attachments/` only).
 * Everything else is a file card. The entry card shows a "picture match"
 * marker only when the search matched through an image.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    entriesApi: { ...actual.entriesApi, get: vi.fn() },
  };
});

import { entriesApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import {
  showEntry,
  initEntryDetail,
  safeHref,
} from '../../../src/osprey/interfaces/ariel/static/js/entries-detail.js';
import { renderEntryCard } from '../../../src/osprey/interfaces/ariel/static/js/components.js';

const RENDITION = '/api/attachments/att-0123456789ab/rendition';
const ORIGINAL = '/api/attachments/att-0123456789ab';

/**
 * Render one entry with the given attachments into the detail modal.
 * @param {any[]} attachments
 * @returns {Promise<HTMLElement>} the modal body
 */
async function renderDetail(attachments) {
  vi.mocked(entriesApi.get).mockResolvedValueOnce({
    entry_id: 'entry-1',
    timestamp: '2026-01-01T00:00:00Z',
    author: 'author',
    source_system: 'sys',
    raw_text: 'subject\nbody',
    keywords: [],
    attachments,
  });
  await showEntry('entry-1');
  return /** @type {HTMLElement} */ (document.getElementById('entry-modal-body'));
}

beforeEach(() => {
  document.body.innerHTML = `
    <div id="entry-modal" class="hidden"><div id="entry-modal-body"></div></div>
  `;
  initEntryDetail();
  vi.clearAllMocks();
});

describe('safeHref', () => {
  test.each([
    'http://example.org/a.png',
    'https://example.org/a.png',
    'HTTPS://example.org/a.png',
    RENDITION,
    ORIGINAL,
  ])('keeps %j', (url) => {
    expect(safeHref(url)).toBe(url);
  });

  test.each([
    'javascript:alert(1)',
    'JavaScript:alert(1)',
    ' javascript:alert(1)',
    'data:image/svg+xml;base64,PHN2Zz4=',
    'vbscript:msgbox(1)',
    '//evil.example/a.png',
    '/api/other',
    'relative/path.png',
    '',
    null,
    undefined,
    42,
  ])('rejects %j', (url) => {
    expect(safeHref(/** @type {any} */ (url))).toBeNull();
  });
});

describe('entry detail attachments', () => {
  test('a viewable attachment draws an <img> from display_url and prints mime_type', async () => {
    const body = await renderDetail([
      { filename: 'beam.png', viewable: true, display_url: RENDITION, mime_type: 'image/png', url: 'https://elog.example/beam.png' },
    ]);
    const img = body.querySelector('img');
    expect(img, 'thumbnail rendered').not.toBeNull();
    expect(/** @type {HTMLImageElement} */ (img).getAttribute('src')).toBe(RENDITION);
    const thumb = /** @type {HTMLElement} */ (body.querySelector('[data-lightbox-url]'));
    expect(thumb.dataset.lightboxUrl).toBe(RENDITION);
    expect(body.textContent).toContain('image/png');
  });

  test('image/svg+xml with viewable:false renders the file card, not an <img>', async () => {
    const body = await renderDetail([
      { filename: 'diagram.svg', viewable: false, display_url: ORIGINAL, mime_type: 'image/svg+xml' },
    ]);
    expect(body.querySelector('img'), 'no thumbnail for a non-viewable image').toBeNull();
    expect(body.querySelector('[data-lightbox-url]')).toBeNull();
    const link = body.querySelector('a[href]');
    expect(link, 'file card links to the download').not.toBeNull();
    expect(/** @type {HTMLAnchorElement} */ (link).getAttribute('href')).toBe(ORIGINAL);
    expect(body.textContent).toContain('image/svg+xml');
    expect(body.textContent).toContain('diagram.svg');
  });

  test('viewable must be exactly true; a truthy non-boolean draws no <img>', async () => {
    const body = await renderDetail([
      { filename: 'a.png', viewable: 'true', display_url: RENDITION, mime_type: 'image/png' },
    ]);
    expect(body.querySelector('img')).toBeNull();
  });

  test('legacy image shapes without viewable never draw an <img>', async () => {
    const body = await renderDetail([
      { filename: 'photo.jpg', url: 'https://elog.example/photo.jpg', type: 'image/jpeg' },
    ]);
    expect(body.querySelector('img')).toBeNull();
  });

  test('a javascript: display_url renders no link and no thumbnail', async () => {
    const body = await renderDetail([
      { filename: 'x.png', viewable: true, display_url: 'javascript:alert(1)', mime_type: 'image/png' },
      { filename: 'y.pdf', viewable: false, display_url: 'javascript:alert(1)', mime_type: 'application/pdf' },
    ]);
    expect(body.querySelector('img')).toBeNull();
    expect(body.querySelector('[data-lightbox-url]')).toBeNull();
    expect(body.querySelector('a[href]'), 'no href for an unsafe url').toBeNull();
    expect(body.innerHTML).not.toContain('javascript:');
    expect(body.textContent).toContain('x.png');
    expect(body.textContent).toContain('y.pdf');
  });

  test('a null display_url renders a file card without a link', async () => {
    const body = await renderDetail([
      { filename: 'pending.pdf', viewable: false, display_url: null, mime_type: 'application/pdf' },
    ]);
    expect(body.querySelector('a[href]')).toBeNull();
    expect(body.querySelector('img')).toBeNull();
    expect(body.textContent).toContain('pending.pdf');
    expect(body.textContent).toContain('application/pdf');
    expect(body.textContent).toContain('Attachments (1)');
  });

  test('viewable with a null display_url falls back to a link-less file card', async () => {
    const body = await renderDetail([
      { filename: 'gone.png', viewable: true, display_url: null, mime_type: 'image/png' },
    ]);
    expect(body.querySelector('img')).toBeNull();
    expect(body.querySelector('a[href]')).toBeNull();
    expect(body.textContent).toContain('gone.png');
  });

  test('an absolute https display_url on a file card is kept as the download link', async () => {
    const body = await renderDetail([
      { filename: 'r.pdf', viewable: false, display_url: 'https://elog.example/r.pdf', mime_type: 'application/pdf' },
    ]);
    const link = /** @type {HTMLAnchorElement} */ (body.querySelector('a[href]'));
    expect(link.getAttribute('href')).toBe('https://elog.example/r.pdf');
  });

  test('clicking a thumbnail opens the lightbox on the rendition', async () => {
    const body = await renderDetail([
      { filename: 'beam.png', viewable: true, display_url: RENDITION, mime_type: 'image/png' },
    ]);
    const thumb = /** @type {HTMLElement} */ (body.querySelector('[data-lightbox-url]'));
    thumb.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const overlay = document.getElementById('image-lightbox');
    expect(overlay).not.toBeNull();
    expect(/** @type {Element} */ (overlay).querySelector('img')?.getAttribute('src')).toBe(RENDITION);
    overlay?.remove();
  });

  test('a delegated click on a hand-made thumbnail with an unsafe url opens no lightbox', () => {
    const body = /** @type {HTMLElement} */ (document.getElementById('entry-modal-body'));
    body.innerHTML = '<div data-lightbox-url="javascript:alert(1)" data-lightbox-name="x"></div>';
    /** @type {HTMLElement} */ (body.firstElementChild).dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(document.getElementById('image-lightbox')).toBeNull();
  });
});

describe('entry card picture-match marker', () => {
  /** @param {any} extra */
  const card = (extra) => {
    const host = document.createElement('div');
    host.innerHTML = renderEntryCard({
      entry_id: 'e-1',
      timestamp: '2026-01-01T00:00:00Z',
      author: 'a',
      source_system: 's',
      raw_text: 'subject\nbody',
      score: null,
      attachments: [{ filename: 'a.png' }, { filename: 'b.pdf' }],
      ...extra,
    });
    return host;
  };

  test('appears when matched_via contains image', () => {
    const host = card({ matched_via: ['text', 'image'] });
    expect(host.querySelector('[data-match-marker="image"]'), 'marker rendered').not.toBeNull();
    expect(host.textContent).toContain('picture match');
  });

  test.each([
    [undefined],
    [null],
    [[]],
    [['text']],
    [['semantic', 'keyword']],
  ])('is absent when matched_via is %j', (matchedVia) => {
    const host = card({ matched_via: matchedVia });
    expect(host.querySelector('[data-match-marker]')).toBeNull();
    expect(host.textContent).not.toContain('picture match');
  });

  test('the paperclip count is attachments.length', () => {
    expect(card({}).textContent).toContain('📎 2');
  });
});
