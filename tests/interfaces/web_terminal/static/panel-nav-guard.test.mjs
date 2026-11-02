// @ts-check
/**
 * The panel navigation guard: the client half.
 *
 * buildEmbedSrc (panel-iframe-sync.js) is the one choke point every panel
 * iframe src passes: panel-manager's navigatePanel, panel-iframe-factory's
 * createPanelIframe and panel-lifecycle's sidecar start. A url is read the way
 * the WHATWG URL parser reads it (edge C0 controls and spaces stripped, tabs
 * and newlines removed, `\` read as `/`) and must then be a root-relative path
 * or an http(s) URL; anything else becomes `about:blank`, never a src that
 * runs script in the host origin or loads another host's document. The server
 * half answers the same urls 422 (test_panel_nav_guard.py); the case lists
 * here mirror its HOSTILE_URLS / NAVIGABLE_URLS.
 *
 * Every case runs with window.__OSPREY_PREFIX__ '' (single-user) and
 * '/u/alice' (multi-user); the urls arrive already server-prefixed.
 *
 *   npx vitest run tests/interfaces/web_terminal/static/panel-nav-guard.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const SYNC_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js';
const LIFECYCLE_SOURCE =
  '../../../../src/osprey/interfaces/web_terminal/static/js/panel-lifecycle.js';

// panel-iframe-sync reads the active session from terminal.js; buildEmbedSrc
// does not, so a stub at the boundary keeps the PTY module out of this test.
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => null,
}));

const {
  buildEmbedSrc,
  isNavigableUrl,
  isUrlWithinMount,
  normaliseNavigationUrl,
  REFUSED_EMBED_SRC,
} = await import(SYNC_PATH);

const HOSTILE_URLS = [
  'javascript:alert(1)',
  'JavaScript:alert(1)',
  'JAVASCRIPT:alert(1)',
  '  javascript:alert(1)',
  '\u0000\u0001javascript:alert(1)',
  '\tjava\tscript:alert(1)',
  'java\nscript:alert(1)',
  'java\r\nscript:alert(1)',
  'javascript:alert(1)  ',
  'data:text/html,<script>alert(1)</script>',
  ' DATA:text/html,x',
  'vbscript:msgbox(1)',
  'VBScript:msgbox(1)',
  'blob:http://localhost/0b1c',
  'file:///etc/passwd',
  '//evil.example/x',
  '  //evil.example/x',
  '/\\evil.example/x',
  '\\\\evil.example/x',
  '/\t/evil.example/x',
  '/\n/evil.example/x',
  'panel/ariel',
  '#fragment',
  '?q=1',
  '',
  '   ',
  'http:///no-host',
  'http:grafana.lan/d',
  'https:',
];

const NAVIGABLE_PATHS = [
  '/panel/ariel/',
  '/panel/ariel/#/draft/7',
  '/ariel/logbook/draft/7',
  '/panel-static/reference/index.html?x=1',
];

const PREFIXES = [
  ['single-user', ''],
  ['/u/alice', '/u/alice'],
];

describe.each(PREFIXES)('buildEmbedSrc navigation guard (%s)', (_name, prefix) => {
  /** @type {import('vitest').MockInstance} */
  let warn;

  beforeEach(() => {
    /** @type {any} */ (window).__OSPREY_PREFIX__ = prefix;
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    delete (/** @type {any} */ (window).__OSPREY_PREFIX__);
    warn.mockRestore();
  });

  test.each(HOSTILE_URLS)('refuses %j', (raw) => {
    expect(buildEmbedSrc(raw)).toBe(REFUSED_EMBED_SRC);
    expect(isNavigableUrl(raw)).toBe(false);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  test.each(NAVIGABLE_PATHS)('a root-relative path still embeds: %s', (path) => {
    const src = buildEmbedSrc(`${prefix}${path}`);
    const parsed = new URL(src);

    expect(parsed.origin).toBe(window.location.origin);
    expect(parsed.pathname.startsWith(`${prefix}/`)).toBe(true);
    expect(parsed.searchParams.get('embedded')).toBe('true');
    expect(parsed.searchParams.get('theme')).toBeTruthy();
    expect(warn).not.toHaveBeenCalled();
  });

  test('a fragment survives on a root-relative path', () => {
    const src = buildEmbedSrc(`${prefix}/panel/ariel/#/draft/7`);
    expect(new URL(src).hash).toBe('#/draft/7');
  });

  test.each([
    ['https://grafana.lan:3000/d/abc', 'https://grafana.lan:3000/d/abc'],
    ['HTTPS://grafana.lan/d', 'https://grafana.lan/d'],
    ['  http://grafana.lan/d  ', 'http://grafana.lan/d'],
  ])('an http(s) url embeds: %j', (raw, expected) => {
    const src = new URL(buildEmbedSrc(raw));
    expect(`${src.origin}${src.pathname}`).toBe(expected);
    expect(src.searchParams.get('embedded')).toBe('true');
  });

  test('a whitespace-padded root-relative path embeds normalised', () => {
    const src = new URL(buildEmbedSrc(` \t${prefix}/panel/ar\tiel/\n `));
    expect(src.origin).toBe(window.location.origin);
    expect(src.pathname).toBe(`${prefix}/panel/ariel/`);
  });

  test('a non-string url is refused', () => {
    expect(buildEmbedSrc(/** @type {any} */ (undefined))).toBe(REFUSED_EMBED_SRC);
    expect(buildEmbedSrc(/** @type {any} */ (null))).toBe(REFUSED_EMBED_SRC);
  });
});

// Paths that pass the scheme rule and leave ARIEL's mount; the server's
// OUTSIDE_ARIEL_PATHS (test_panel_nav_guard.py), judged here against the
// prefixed mount the config route hands the page.
const OUTSIDE_ARIEL_PATHS = [
  '/api/panels',
  '/',
  '/panel/grafana/d/1',
  '/panel/arielx/',
  '/panel/ariel/../grafana/',
  '/panel/ariel/%2e%2e/grafana',
  '/panel/ariel/..%2Fgrafana',
  '/panel/ariel/%2e./grafana',
  '/panel/ariel/%252e%252e/grafana',
  '/panel/ariel//evil',
  '/panel/ariel/%2F%2Fevil.example/x',
  '/panel/ariel/%252F%252Fevil.example/x',
  '/panel/ariel/%5C%5Cevil.example/x',
  '/panel/ariel/%5Cevil.example/x',
  '/panel/ariel/x;/../../api',
  '/panel/ariel/./%2e%2e/%2e%2e/api/panels',
];

// http(s) urls never pass in the browser: the server has already rewritten an
// admitted one into the mount, so a url that still names a host left the panel.
const FOREIGN_URLS = [
  'http://evil.example/panel/ariel/',
  `${window.location.origin}/api/panels`,
  'https://127.0.0.1:10300/',
  'http://127.0.0.1:10300@evil.example/',
];

describe.each(PREFIXES)('isUrlWithinMount, the member rule (%s)', (_name, prefix) => {
  const mount = `${prefix}/panel/ariel`;

  test.each(HOSTILE_URLS)('a hostile url is outside every mount: %j', (raw) => {
    expect(isUrlWithinMount(raw, mount)).toBe(false);
  });

  test.each(OUTSIDE_ARIEL_PATHS)('a path leaving the mount is refused: %s', (path) => {
    expect(isUrlWithinMount(`${prefix}${path}`, mount)).toBe(false);
  });

  test.each(FOREIGN_URLS)('a url naming a host is refused: %s', (url) => {
    expect(isUrlWithinMount(url, mount)).toBe(false);
  });

  test.each([
    '/panel/ariel',
    '/panel/ariel/',
    '/panel/ariel/#/draft/7',
    '/panel/ariel/logbook/7?x=1',
    '/panel/ariel/a%20b/c%3Fd',
  ])('a url under the mount is admitted: %j', (path) => {
    expect(isUrlWithinMount(`${prefix}${path}`, mount)).toBe(true);
  });

  test('padding and tabs are read as the parser reads them', () => {
    expect(isUrlWithinMount(` \t${prefix}/panel/ar\tiel/#/dr\naft/7\n `, mount)).toBe(true);
  });

  test('the same-origin absolute spelling of a mount path is admitted', () => {
    expect(isUrlWithinMount(`${window.location.origin}${mount}/x`, mount)).toBe(true);
  });

  test('a discovered bundle admits only its own files', () => {
    const bundle = `${prefix}/panel-static/reference/`;
    expect(isUrlWithinMount(`${prefix}/panel-static/reference/index.html?x=1`, bundle)).toBe(true);
    expect(isUrlWithinMount(`${prefix}/panel-static/reference`, bundle)).toBe(true);
    expect(isUrlWithinMount(`${prefix}/panel-static/ref/index.html`, bundle)).toBe(false);
    expect(isUrlWithinMount(`${prefix}/panel/reference/`, bundle)).toBe(false);
  });

  test('a panel with no mount admits nothing', () => {
    expect(isUrlWithinMount(`${prefix}/panel/ariel/`, null)).toBe(false);
    expect(isUrlWithinMount(`${prefix}/panel/ariel/`, '')).toBe(false);
  });

  test('a prefixed mount never admits the unprefixed path of another user', () => {
    if (!prefix) return;
    expect(isUrlWithinMount('/panel/ariel/', mount)).toBe(false);
    expect(isUrlWithinMount('/u/bob/panel/ariel/', mount)).toBe(false);
  });
});

describe('normaliseNavigationUrl reads as WHATWG does', () => {
  test.each([
    ['  JavaScript:alert(1)', 'javascript:alert(1)'],
    ['\u0000\u001fhttp://h/', 'http://h/'],
    ['java\tscr\nipt:x', 'javascript:x'],
    ['/\\evil.example', '//evil.example'],
    ['HTTPS://Host/Path', 'https://Host/Path'],
    ['/panel/ariel/#/x', '/panel/ariel/#/x'],
  ])('%j -> %j', (raw, expected) => {
    expect(normaliseNavigationUrl(raw)).toBe(expected);
  });

  test('agrees with the URL parser on what a hostile url would resolve to', () => {
    // The reason for the guard: these pass a naive startsWith('/') test, yet
    // the real parser reads them as another host.
    for (const raw of ['/\\evil.example/x', '/\t/evil.example/x']) {
      expect(new URL(raw, window.location.origin).host).toBe('evil.example');
      expect(isNavigableUrl(raw)).toBe(false);
    }
  });
});

describe('every iframe src goes through the choke point', () => {
  test('panel-lifecycle assigns iframe.src only via buildEmbedSrc', async () => {
    const { readFile } = await import('node:fs/promises');
    const source = await readFile(new URL(LIFECYCLE_SOURCE, import.meta.url), 'utf8');
    const assignments = source.match(/\.src\s*=\s*[^;]+;/g) ?? [];
    expect(assignments.length).toBeGreaterThan(0);
    for (const line of assignments) expect(line).toContain('buildEmbedSrc(');
  });
});
