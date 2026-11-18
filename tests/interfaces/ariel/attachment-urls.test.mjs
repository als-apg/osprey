// @ts-check
/**
 * The URLs the ARIEL page builds from attachment data:
 *   npx vitest run tests/interfaces/ariel/attachment-urls.test.mjs
 *
 * `safeHref` keeps an absolute http(s) URL, joins one of this API's attachment
 * routes to the API base, and refuses everything else. `originalHref` gives
 * the stored original behind a rendition route.
 */

import { test, expect, describe } from 'vitest';

import {
  safeHref,
  originalHref,
} from '../../../src/osprey/interfaces/ariel/static/js/attachment-urls.js';

/** `display_url` values as the server sends them: routes relative to the API base. */
const RENDITION = '/attachments/att-0123456789ab/rendition';
const ORIGINAL = '/attachments/att-0123456789ab';
/** The same routes joined to the page's API base, as the page requests them. */
const RENDITION_SRC = '/api/attachments/att-0123456789ab/rendition';
const ORIGINAL_SRC = '/api/attachments/att-0123456789ab';

describe('safeHref', () => {
  test.each(['http://example.org/a.png', 'https://example.org/a.png', 'HTTPS://example.org/a.png'])(
    'keeps %j',
    (url) => {
      expect(safeHref(url)).toBe(url);
    }
  );

  test.each([
    [RENDITION, RENDITION_SRC],
    [ORIGINAL, ORIGINAL_SRC],
  ])('joins the attachment route %j to the API base', (route, src) => {
    expect(safeHref(route)).toBe(src);
  });

  test.each([
    'javascript:alert(1)',
    'JavaScript:alert(1)',
    ' javascript:alert(1)',
    'data:image/svg+xml;base64,PHN2Zz4=',
    'vbscript:msgbox(1)',
    '//evil.example/a.png',
    '/api/other',
    '/api/attachments/att-0123456789ab/rendition',
    '/attachments/',
    '/attachments/../rendition',
    '/attachments/./rendition',
    '/attachments/a/b',
    '/attachments/a/rendition/x',
    '/attachments/a?x=1',
    'relative/path.png',
    '',
    null,
    undefined,
    42,
  ])('rejects %j', (url) => {
    expect(safeHref(/** @type {any} */ (url))).toBeNull();
  });
});

describe('originalHref', () => {
  test('gives the original behind a rendition route', () => {
    expect(originalHref(RENDITION)).toBe(ORIGINAL_SRC);
  });

  test.each([ORIGINAL, 'https://example.org/a.png/rendition', '/other/rendition', null])(
    'has none for %j',
    (url) => {
      expect(originalHref(url)).toBeNull();
    }
  );
});
