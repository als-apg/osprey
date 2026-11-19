// @ts-check
/**
 * The reader's facility references:
 *   npx vitest run tests/interfaces/ariel/reader-facility.test.mjs
 *
 * The title and body mark each quoted id the way `renderEntryText` does; the
 * title only when it is raw_text's first line. The sidebar list names each
 * target once with one line per record, or the reason it has none.
 *
 * An entry that does not declare `metadata.body_format: markdown` renders the
 * HTML it rendered before entries could declare a format, byte for byte; the
 * Markdown path itself is entry-markdown.test.mjs.
 */

import { test, expect, describe } from 'vitest';
import {
  renderEntryTextHtml,
  renderFacilityReferencesList,
} from '../../../src/osprey/interfaces/ariel/static/js/reader-facility.js';
import { renderEntryDetail } from '../../../src/osprey/interfaces/ariel/static/js/reader.js';

const RECORD = { kind: 'channel', unit: 'A', description: 'Main current' };

/** @param {string} html */
function dom(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  return div;
}

describe('renderEntryTextHtml', () => {
  const rawText = 'SR:DEV tripped\nreset SR:DEV and GONE';
  const references = [
    { start: 0, end: 6, target: 'SR:DEV', records: [RECORD] },
    { start: 21, end: 27, target: 'SR:DEV', records: [RECORD] },
    { start: 32, end: 36, target: 'GONE', records: [] },
  ];

  test('marks the body and the first-line title', () => {
    const { titleHtml, bodyHtml } = renderEntryTextHtml({
      raw_text: rawText,
      facility_references: references,
    });
    expect(dom(titleHtml).querySelectorAll('.entry-facility-link')).toHaveLength(1);
    const body = dom(bodyHtml);
    expect(body.textContent).toBe('reset SR:DEV and GONE');
    expect(body.querySelector('.entry-facility-link')?.textContent).toBe('SR:DEV');
    expect(body.querySelector('.entry-facility-link--dangling')?.textContent).toBe('GONE');
  });

  test('a stored subject equal to the first line is marked', () => {
    const { titleHtml } = renderEntryTextHtml({
      raw_text: rawText,
      subject: 'SR:DEV tripped',
      facility_references: references,
    });
    expect(dom(titleHtml).querySelector('.entry-facility-link')?.textContent).toBe('SR:DEV');
  });

  test('a stored subject that differs shows plain and escaped', () => {
    const { titleHtml } = renderEntryTextHtml({
      raw_text: rawText,
      subject: '<b>SR:DEV</b> trip',
      facility_references: references,
    });
    expect(titleHtml).toBe('&lt;b&gt;SR:DEV&lt;/b&gt; trip');
  });

  test('empty text titles Untitled', () => {
    expect(renderEntryTextHtml({ raw_text: '' }).titleHtml).toBe('Untitled');
  });
});

describe('an entry that declares no Markdown renders as before', () => {
  // Markup characters, indentation and a facility id: what a stored plain
  // entry holds and a Markdown renderer would change.
  const rawText =
    '# Shift <b>log</b>\n\n- reset SR:DEV\n  * twice\n| a | b |\n[x](javascript:1) & done';
  const start = rawText.indexOf('SR:DEV');
  const references = [{ start, end: start + 6, target: 'SR:DEV', records: [RECORD] }];
  /** The body's HTML as the reader produced it before `body_format` was read. */
  const BODY =
    '- reset <span class="entry-facility-link" title="channel · A · Main current">SR:DEV</span>\n' +
    '  * twice\n| a | b |\n[x](javascript:1) &amp; done';

  /** @param {object|null} [metadata] */
  const entry = (metadata) => ({
    entry_id: 'e-1',
    timestamp: '2026-01-01T00:00:00Z',
    author: 'author',
    source_system: 'sys',
    raw_text: rawText,
    facility_references: references,
    ...(metadata === undefined ? {} : { metadata }),
  });

  /** @param {any} e */
  function detail(e) {
    const container = document.createElement('div');
    renderEntryDetail(container, e);
    return container.innerHTML;
  }

  const NOT_MARKDOWN = [
    undefined,
    null,
    {},
    { body_format: null },
    { body_format: '' },
    { body_format: 'html' },
    { body_format: 'plain' },
    { body_format: 'text/markdown' },
    { body_format: ' markdown' },
    { body_format: 7 },
    { body_format: ['markdown'] },
    { logbook: 'Operations' },
  ];

  test.each(NOT_MARKDOWN)('metadata %j: the body is the stored text, escaped', (metadata) => {
    const { titleHtml, bodyHtml, rendered } = renderEntryTextHtml(
      /** @type {any} */ (entry(metadata))
    );
    expect(bodyHtml).toBe(BODY);
    expect(titleHtml).toBe('# Shift &lt;b&gt;log&lt;/b&gt;');
    expect(rendered).toBe(false);
  });

  test.each(NOT_MARKDOWN.filter((m) => !m || !('logbook' in m)))(
    'metadata %j: the whole reader is the same HTML as with no metadata',
    (metadata) => {
      const html = detail(entry(metadata));
      expect(html).toBe(detail(entry()));
      expect(html).toContain(`<div class="entry-detail-text">${BODY}</div>`);
      expect(html).not.toContain('entry-detail-text--rendered');
    }
  );

  test('a Markdown entry is the same HTML too when the page has no renderer', () => {
    // This suite loads neither marked nor DOMPurify, as a page whose vendored
    // scripts failed to load.
    expect(/** @type {any} */ (globalThis).marked).toBeUndefined();
    expect(/** @type {any} */ (globalThis).DOMPurify).toBeUndefined();
    expect(detail(entry({ body_format: 'markdown' }))).toBe(detail(entry()));
  });
});

describe('renderFacilityReferencesList', () => {
  test('is empty with no references', () => {
    expect(renderFacilityReferencesList(null)).toBe('');
    expect(renderFacilityReferencesList([])).toBe('');
  });

  test('names each target once with one line per record', () => {
    const list = dom(
      renderFacilityReferencesList([
        { start: 0, end: 3, target: 'DEV', records: [RECORD, { kind: 'device' }] },
        { start: 8, end: 11, target: 'DEV', records: [RECORD, { kind: 'device' }] },
      ])
    );
    const targets = list.querySelectorAll('.facility-ref-target');
    expect([...targets].map((t) => t.textContent)).toEqual(['DEV']);
    expect([...list.querySelectorAll('.facility-ref-record')].map((r) => r.textContent)).toEqual([
      'channel · A · Main current',
      'device',
    ]);
  });

  test('an empty records list reads not in the facility file', () => {
    const none = dom(
      renderFacilityReferencesList([{ start: 0, end: 4, target: 'GONE', records: [] }])
    ).querySelector('.facility-ref-none');
    expect(none?.textContent).toBe('not in the facility file');
  });

  test('null records read facility file not loaded', () => {
    const none = dom(
      renderFacilityReferencesList([{ start: 0, end: 3, target: 'DEV', records: null }])
    ).querySelector('.facility-ref-none');
    expect(none?.textContent).toBe('facility file not loaded');
  });

  test('escapes a hostile target', () => {
    const html = renderFacilityReferencesList([
      { start: 0, end: 1, target: '<img src=x onerror=1>', records: [] },
    ]);
    expect(dom(html).querySelector('img')).toBeNull();
  });
});
