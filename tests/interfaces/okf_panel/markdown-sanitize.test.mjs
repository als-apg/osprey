// @ts-check
// @vitest-environment jsdom
/**
 * The KNOWLEDGE panel's markdown reaches the DOM only through DOMPurify:
 *   npx vitest run tests/interfaces/okf_panel/markdown-sanitize.test.mjs
 *
 * Both `marked.parse` sinks — a concept's body and the structure overview —
 * render through app.js's one helper, `renderMarkdownInto`. A bundle is files
 * on disk an agent can write, so raw `marked` output is untrusted HTML.
 *
 * The same helper points every image at the panel's asset route or replaces it
 * with its alt text (sanitizeMarkdownHtml in helpers.js), so no concept makes
 * the panel contact another host.
 *
 * jsdom, not the suite's happy-dom: the XSS cases run the REAL DOMPurify and
 * the real `marked`, and DOMPurify mis-parses under happy-dom (see
 * chat-render.test.mjs).
 */

import createDOMPurify from 'dompurify';
import { marked } from 'marked';
import { afterEach, beforeAll, describe, expect, test, vi } from 'vitest';

import { mountShell } from './page-shell.mjs';
import { sanitizeMarkdownHtml } from '../../../src/osprey/interfaces/okf_panel/static/js/helpers.js';

const HOSTILE = [
  '# Title',
  '',
  '<img src="x" onerror="globalThis.__okfPwned = true">',
  '',
  '<script>globalThis.__okfPwned = true</script>',
  '',
  '[click](javascript:globalThis.__okfPwned=true)',
].join('\n');

/** @type {typeof import('../../../src/osprey/interfaces/okf_panel/static/js/app.js')} */
let app;

/**
 * @param {unknown} body
 */
function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

/**
 * @param {string} url
 */
async function fakeFetch(url) {
  if (url === '/api/concepts') {
    return jsonResponse({
      groups: [
        { id: 'devices', label: 'devices', concepts: [{ id: 'devices/bpm', title: 'BPM' }] },
      ],
    });
  }
  if (url === '/api/bundle_health') return jsonResponse({ ok: true, counts: {}, total: 0 });
  if (url === '/api/structure') return jsonResponse({ markdown: HOSTILE });
  if (url.startsWith('/api/concept?id=')) {
    return jsonResponse({ id: 'devices/bpm', frontmatter: { title: 'BPM' }, body: HOSTILE });
  }
  throw new Error(`unstubbed fetch: ${url}`);
}

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * @param {Element} root
 */
function assertInert(root) {
  expect(root.querySelector('script')).toBeNull();
  for (const node of root.querySelectorAll('*')) {
    for (const attr of node.getAttributeNames()) {
      expect(attr.startsWith('on'), `${node.tagName} carries ${attr}`).toBe(false);
    }
  }
  for (const link of root.querySelectorAll('a')) {
    expect((link.getAttribute('href') || '').toLowerCase()).not.toContain('javascript:');
  }
  expect(/** @type {any} */ (globalThis).__okfPwned).toBeUndefined();
}

beforeAll(async () => {
  mountShell();
  // jsdom has no layout, so no scrollIntoView (the tree calls it on select).
  Element.prototype.scrollIntoView = () => {};
  vi.stubGlobal('marked', marked);
  vi.stubGlobal('DOMPurify', createDOMPurify(window));
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  app = await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

afterEach(() => {
  vi.stubGlobal('marked', marked);
  vi.stubGlobal('DOMPurify', createDOMPurify(window));
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
});

describe('renderMarkdownInto', () => {
  test('strips <img onerror>, <script> and javascript: links', () => {
    const target = document.createElement('div');
    app.renderMarkdownInto(target, HOSTILE);

    expect(target.querySelector('h1')?.textContent).toBe('Title');
    expect(target.querySelector('img')?.getAttribute('src')).toBe('/api/asset?target=x&from=');
    assertInert(target);
  });

  test('only DOMPurify output reaches innerHTML', () => {
    vi.stubGlobal('DOMPurify', {
      sanitize: () => '<em>sanitised</em>',
      addHook() {},
      removeHook() {},
    });
    const target = document.createElement('div');
    app.renderMarkdownInto(target, HOSTILE);

    expect(target.innerHTML).toBe('<em>sanitised</em>');
  });

  test('without any DOMPurify the source is inert text, never markup', () => {
    vi.stubGlobal('DOMPurify', undefined);
    const target = document.createElement('div');
    app.renderMarkdownInto(target, HOSTILE);

    expect(target.children).toHaveLength(0);
    expect(target.textContent).toBe(HOSTILE);
  });

  test('without marked the source is inert text', () => {
    vi.stubGlobal('marked', undefined);
    const target = document.createElement('div');
    app.renderMarkdownInto(target, '<b>x</b>');

    expect(target.children).toHaveLength(0);
    expect(target.textContent).toBe('<b>x</b>');
  });

  test('a sanitiser that throws leaves inert text', () => {
    vi.stubGlobal('DOMPurify', {
      sanitize: () => {
        throw new Error('boom');
      },
    });
    const target = document.createElement('div');
    app.renderMarkdownInto(target, '<b>x</b>');

    expect(target.textContent).toBe('<b>x</b>');
  });

  test('a sanitiser without hooks leaves inert text, never markup', () => {
    vi.stubGlobal('DOMPurify', { sanitize: () => '<img src="https://evil.example/x.png">' });
    const target = document.createElement('div');
    const source = '![x](https://evil.example/x.png)';
    app.renderMarkdownInto(target, source);

    expect(target.children).toHaveLength(0);
    expect(target.textContent).toBe(source);
  });

  test('null renders empty', () => {
    const target = document.createElement('div');
    target.textContent = 'stale';
    app.renderMarkdownInto(target, null);

    expect(target.textContent).toBe('');
  });
});

describe('resolveSanitizer', () => {
  test("uses this page's DOMPurify global", () => {
    const own = { sanitize: (/** @type {string} */ s) => s };
    const win = /** @type {any} */ ({ DOMPurify: own });
    expect(app.resolveSanitizer(win)).toBe(own);
  });

  test('a page with no DOMPurify yields no sanitiser', () => {
    const win = /** @type {any} */ ({ DOMPurify: undefined });
    expect(app.resolveSanitizer(win)).toBeNull();
  });
});

describe('both panel sinks', () => {
  test('the structure overview is sanitised', () => {
    // The boot (no fragment) renders the structure overview.
    const reader = /** @type {HTMLElement} */ (document.getElementById('reader-content'));
    expect(reader.querySelector('.osprey-md-rendered h1')?.textContent).toBe('Title');
    assertInert(reader);
  });

  test('a concept body is sanitised', async () => {
    location.hash = '#devices/bpm';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    await flush();

    const reader = /** @type {HTMLElement} */ (document.getElementById('reader-content'));
    expect(reader.querySelector('.concept-title')?.textContent).toBe('BPM');
    expect(reader.querySelector('.osprey-md-rendered img')).not.toBeNull();
    assertInert(reader);
  });
});

/** Where the corpus points; nothing may reach it. */
const EVIL = 'evil.example';

/**
 * Attributes that hold a URL, wider than the sanitiser's own list so the
 * outcome is checked independently of it.
 */
const URL_ATTRIBUTES = [
  'action',
  'archive',
  'background',
  'cite',
  'classid',
  'codebase',
  'data',
  'dynsrc',
  'formaction',
  'href',
  'icon',
  'longdesc',
  'lowsrc',
  'manifest',
  'ping',
  'poster',
  'profile',
  'src',
  'srcset',
  'usemap',
  'xlink:href',
];

const REMOTE_CORPUS = [
  'A [link](https://evil.example/link) stays a link.',
  '',
  '<img src="figures/a.png" alt="Set" srcset="https://evil.example/s1.png 2x">',
  '',
  '<p style="background:url(https://evil.example/s2.png)">styled</p>',
  '',
  '<style>body { background: url(https://evil.example/s3.png) }</style>',
  '',
  '<table background="https://evil.example/s4.png"><tr><td background="https://evil.example/s4b.png">t</td></tr></table>',
  '',
  '<input type="image" src="https://evil.example/s5.png">',
  '',
  '<video poster="https://evil.example/s6.png" src="https://evil.example/s6.mp4"><source src="https://evil.example/s6b.mp4"></video>',
  '',
  '<audio src="https://evil.example/s7.mp3" controls></audio>',
  '',
  '<iframe src="https://evil.example/s8.html"></iframe>',
  '',
  '<object data="https://evil.example/s9.swf"></object>',
  '',
  '<svg><image href="https://evil.example/s10.png"/><image xlink:href="https://evil.example/s10b.png"/><filter><feImage href="https://evil.example/s11.png"/></filter></svg>',
  '',
  '<blockquote cite="https://evil.example/cite">quoted</blockquote>',
  '',
  '<form action="https://evil.example/f"><button formaction="https://evil.example/g">b</button></form>',
  '',
  '<map name="m"><area href="https://evil.example/area" shape="rect" coords="0,0,1,1"></map>',
  '',
  '<picture><source srcset="https://evil.example/p.png"><img src="https://evil.example/p2.png" alt="Pic"></picture>',
  '',
  '<embed src="https://evil.example/e.swf">',
  '',
  '<link rel="stylesheet" href="https://evil.example/l.css">',
].join('\n');

/**
 * @param {string} html
 * @returns {DocumentFragment}
 */
function fragmentOf(html) {
  const template = document.createElement('template');
  template.innerHTML = html;
  return template.content;
}

/**
 * Every URL-bearing attribute left in *html*, other than `a[href]` and an
 * `img[src]` on the asset route, and every attribute value that names the
 * corpus host or loads through CSS.
 * @param {string} html
 * @returns {string[]}
 */
function strayUrls(html) {
  const stray = [];
  for (const node of fragmentOf(html).querySelectorAll('*')) {
    const tag = node.localName;
    for (const name of node.getAttributeNames()) {
      const value = node.getAttribute(name) || '';
      if (tag === 'a' && name === 'href') continue;
      if (tag === 'img' && name === 'src') {
        if (!value.startsWith('/api/asset?target=')) stray.push(`img[src=${value}]`);
        continue;
      }
      if (URL_ATTRIBUTES.includes(name) || value.includes(EVIL) || /url\(/i.test(value)) {
        stray.push(`${tag}[${name}=${value}]`);
      }
    }
  }
  return stray;
}

/**
 * @param {string} markdown
 * @param {string} conceptId
 */
function sanitized(markdown, conceptId) {
  const purify = createDOMPurify(window);
  return sanitizeMarkdownHtml(purify, String(marked.parse(markdown)), conceptId);
}

describe('images go to the asset route', () => {
  const THREE_SYNTAXES = [
    '![Inline figure](figures/a.png)',
    '',
    '![Reference figure][ref]',
    '',
    '[ref]: ../shared/b.png',
    '',
    '<img src="/devices/c.png" alt="Raw figure">',
  ].join('\n');

  test('inline, reference and raw-HTML images point at the asset route', () => {
    const target = document.createElement('div');
    app.renderMarkdownInto(target, THREE_SYNTAXES, 'devices/bpm');

    const images = Array.from(target.querySelectorAll('img'));
    expect(images.map((img) => [img.getAttribute('src'), img.getAttribute('alt')])).toEqual([
      ['/api/asset?target=figures%2Fa.png&from=devices%2Fbpm', 'Inline figure'],
      ['/api/asset?target=..%2Fshared%2Fb.png&from=devices%2Fbpm', 'Reference figure'],
      ['/api/asset?target=%2Fdevices%2Fc.png&from=devices%2Fbpm', 'Raw figure'],
    ]);
  });

  test('a percent-encoded name reaches the route as written', () => {
    const html = sanitized('![x](figures/a%20b.png)', 'devices/bpm');
    const img = fragmentOf(html).querySelector('img');

    expect(img?.getAttribute('src')).toBe(
      '/api/asset?target=figures%2Fa%2520b.png&from=devices%2Fbpm'
    );
  });

  test('the structure overview resolves from the bundle root', () => {
    const html = sanitized('![x](devices/figures/a.png)', '');

    expect(fragmentOf(html).querySelector('img')?.getAttribute('src')).toBe(
      '/api/asset?target=devices%2Ffigures%2Fa.png&from='
    );
  });

  test('remote, protocol-relative and data: images become their alt text', () => {
    const markdown = [
      '![Remote alt](https://evil.example/r.png)',
      '',
      '![Protocol alt](//evil.example/p.png)',
      '',
      '![Data alt](data:image/png;base64,iVBORw0KGgo=)',
    ].join('\n');
    const target = document.createElement('div');
    app.renderMarkdownInto(target, markdown, 'devices/bpm');

    expect(target.querySelector('img')).toBeNull();
    const fallbacks = Array.from(target.querySelectorAll('.okf-image-fallback'));
    expect(fallbacks).toHaveLength(3);
    expect(fallbacks[0].textContent).toContain('Remote alt');
    expect(fallbacks[1].textContent).toContain('Protocol alt');
    expect(fallbacks[2].textContent).toContain('Data alt');
    for (const box of fallbacks) expect(box.textContent).toContain('external image, not loaded');
    // A remote http(s) source is offered as a link; nothing else is.
    expect(Array.from(target.querySelectorAll('a')).map((a) => a.getAttribute('href'))).toEqual([
      'https://evil.example/r.png',
    ]);
    expect(strayUrls(target.innerHTML)).toEqual([]);
  });

  test('an image without a source becomes its alt text', () => {
    const target = document.createElement('div');
    app.renderMarkdownInto(target, '<img alt="No source"> <img src="" alt="Empty source">');

    expect(target.querySelector('img')).toBeNull();
    expect(target.textContent).toContain('No source');
    expect(target.textContent).toContain('Empty source');
  });

  test('alt text is text, never markup', () => {
    const target = document.createElement('div');
    app.renderMarkdownInto(
      target,
      '<img src="https://evil.example/x.png" alt="<b>bold</b><img src=x onerror=1>">'
    );

    expect(target.querySelector('b')).toBeNull();
    expect(target.querySelector('img')).toBeNull();
    expect(target.textContent).toContain('<b>bold</b>');
    assertInert(target);
  });

  test('the remote-reference corpus leaves only links and asset-route images', () => {
    const html = sanitized(REMOTE_CORPUS, 'devices/bpm');

    expect(strayUrls(html)).toEqual([]);
    expect(html).not.toMatch(/<style|<svg|<iframe|<object|<embed|<video|<audio|<input|<link/i);
    const fragment = fragmentOf(html);
    expect(Array.from(fragment.querySelectorAll('img')).map((i) => i.getAttribute('src'))).toEqual([
      '/api/asset?target=figures%2Fa.png&from=devices%2Fbpm',
    ]);
    expect(fragment.querySelector('a[href="https://evil.example/link"]')).not.toBeNull();
    expect(fragment.textContent).toContain('Pic');
  });

  test('the hook is on DOMPurify only for the call', () => {
    const purify = createDOMPurify(window);
    sanitizeMarkdownHtml(purify, '<p>x</p>', 'devices/bpm');

    const plain = purify.sanitize('<img src="https://evil.example/x.png">');
    expect(plain).toContain('https://evil.example/x.png');
  });

  test('a DOMPurify without hooks is refused', () => {
    expect(() =>
      sanitizeMarkdownHtml({ sanitize: (/** @type {string} */ s) => s }, '<p>x</p>', '')
    ).toThrow();
  });

  test('a concept body passes its id to the route', async () => {
    location.hash = '#devices/bpm';
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    await flush();

    const reader = /** @type {HTMLElement} */ (document.getElementById('reader-content'));
    expect(reader.querySelector('.osprey-md-rendered img')?.getAttribute('src')).toBe(
      '/api/asset?target=x&from=devices%2Fbpm'
    );
  });
});
