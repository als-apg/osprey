// @ts-check
/**
 * Unit tests for the pure, DOM-free entries helpers (entries-helpers.js),
 * extracted from entries.js/components.js during the P4 retrofit:
 *   npx vitest run tests/interfaces/ariel/entries-helpers.test.mjs
 */

import { test, expect } from 'vitest';

import {
  formatFileSize,
  parseEntryText,
  renderEntryText,
} from '../../../src/osprey/interfaces/ariel/static/js/entries-helpers.js';

test('formatFileSize renders bytes below 1024 as B', () => {
  expect(formatFileSize(0)).toBe('0 B');
  expect(formatFileSize(1)).toBe('1 B');
  expect(formatFileSize(1023)).toBe('1023 B');
});

test('formatFileSize renders 1024..1MB-1 as KB', () => {
  expect(formatFileSize(1024)).toBe('1.0 KB');
  expect(formatFileSize(1536)).toBe('1.5 KB');
  expect(formatFileSize(1024 * 1024 - 1)).toBe('1024.0 KB');
});

test('formatFileSize renders 1MB and above as MB', () => {
  expect(formatFileSize(1024 * 1024)).toBe('1.0 MB');
  expect(formatFileSize(1024 * 1024 * 2.5)).toBe('2.5 MB');
});

test('parseEntryText splits the first line as subject and the rest as details', () => {
  expect(parseEntryText('Subject line\nBody line one\nBody line two')).toEqual({
    subject: 'Subject line',
    details: 'Body line one\nBody line two',
    detailsStart: 13,
  });
});

test('parseEntryText falls back to the full text as details when there is no remainder', () => {
  expect(parseEntryText('Just one line')).toEqual({
    subject: 'Just one line',
    details: 'Just one line',
    detailsStart: 0,
  });
});

test('parseEntryText handles empty/malformed input', () => {
  expect(parseEntryText('')).toEqual({ subject: 'Untitled', details: '', detailsStart: 0 });
  expect(parseEntryText(undefined)).toEqual({ subject: 'Untitled', details: '', detailsStart: 0 });
  expect(parseEntryText('\n\n')).toEqual({
    subject: 'Untitled',
    details: '\n\n',
    detailsStart: 0,
  });
});

test('parseEntryText counts detailsStart in code points past removed whitespace', () => {
  const text = 'S\u{1F680}\n\n  body';
  const { details, detailsStart } = parseEntryText(text);

  expect(details).toBe('body');
  expect(Array.from(text).slice(detailsStart).join('')).toBe('body');
});

const RECORD = [{ kind: 'channel', unit: 'A' }];

test('renderEntryText wraps exactly the target of a reference', () => {
  const references = [{ start: 4, end: 7, target: 'DEV', records: RECORD }];

  expect(renderEntryText('see DEV now', references)).toBe(
    'see <span class="entry-facility-link" title="channel · A">DEV</span> now'
  );
});

test('renderEntryText keeps HTML escaped inside and outside a span', () => {
  const text = '<b> <i> <u>';
  const html = renderEntryText(text, [{ start: 4, end: 7, target: '<i>', records: RECORD }]);

  expect(html).toContain('&lt;b&gt; <span');
  expect(html).toContain('>&lt;i&gt;</span> &lt;u&gt;');
  expect(html).not.toContain('<b>');
  expect(html).not.toContain('<i>');
});

test('renderEntryText escapes the title of a hostile target', () => {
  const target = '"><img src=x>';
  const html = renderEntryText(target, [
    { start: 0, end: Array.from(target).length, target, records: null },
  ]);

  expect(html).not.toContain('<img');
});

test('a dangling reference gets the dangling class and title', () => {
  expect(renderEntryText('GONE', [{ start: 0, end: 4, target: 'GONE', records: [] }])).toBe(
    '<span class="entry-facility-link--dangling" title="GONE · not in the facility file">GONE</span>'
  );
});

test('a reference with null records gets the plain class and the not-loaded title', () => {
  expect(renderEntryText('DEV', [{ start: 0, end: 3, target: 'DEV', records: null }])).toBe(
    '<span class="entry-facility-link" title="DEV · facility file not loaded">DEV</span>'
  );
});

test('offsets after an emoji land on the target', () => {
  const html = renderEntryText('\u{1F680} DEV', [
    { start: 2, end: 5, target: 'DEV', records: RECORD },
  ]);

  expect(html).toContain('>DEV</span>');
});

test('a reference outside the window is not rendered', () => {
  const references = [
    { start: 0, end: 3, target: 'ONE', records: RECORD },
    { start: 4, end: 7, target: 'TWO', records: RECORD },
  ];

  expect(renderEntryText('ONE TWO', references, 0, 3)).toBe(
    '<span class="entry-facility-link" title="channel · A">ONE</span>'
  );
  expect(renderEntryText('ONE TWO', references, 0, 5)).toBe(
    '<span class="entry-facility-link" title="channel · A">ONE</span> T'
  );
});

test('null references render plain text', () => {
  expect(renderEntryText('a<b', null)).toBe('a&lt;b');
});
