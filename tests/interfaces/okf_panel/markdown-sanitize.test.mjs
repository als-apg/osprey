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
 * jsdom, not the suite's happy-dom: the XSS cases run the REAL DOMPurify and
 * the real `marked`, and DOMPurify mis-parses under happy-dom (see
 * chat-render.test.mjs).
 */

import createDOMPurify from 'dompurify';
import { marked } from 'marked';
import { afterEach, beforeAll, describe, expect, test, vi } from 'vitest';

import { mountShell } from './page-shell.mjs';

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
});

describe('renderMarkdownInto', () => {
  test('strips <img onerror>, <script> and javascript: links', () => {
    const target = document.createElement('div');
    app.renderMarkdownInto(target, HOSTILE);

    expect(target.querySelector('h1')?.textContent).toBe('Title');
    expect(target.querySelector('img')).not.toBeNull();
    assertInert(target);
  });

  test('only DOMPurify output reaches innerHTML', () => {
    vi.stubGlobal('DOMPurify', { sanitize: () => '<em>sanitised</em>' });
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
