// @vitest-environment jsdom
/**
 * The gallery side of the artifact content sandbox (M10, point 1).
 *
 *   npx vitest run tests/interfaces/artifacts/content-sandbox.test.mjs
 *
 * jsdom, not the suite's happy-dom: the XSS cases run the REAL DOMPurify and
 * the real marked (both npm devDependencies, the vendored versions), and
 * DOMPurify mis-parses under happy-dom (see chat-render.test.mjs).
 *
 * Pinned here: rendered markdown reaches the DOM only through md-render.js's
 * sanitising sink (the preview pane and the standalone page share it); the
 * preview iframes run without `allow-same-origin`.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';
import createDOMPurify from 'dompurify';
import { marked } from 'marked';

import { renderMarkdownInto } from '../../../src/osprey/interfaces/artifacts/static/js/md-render.js';
import { renderMarkdownView } from '../../../src/osprey/interfaces/artifacts/static/js/preview-content.js';
import { artifactViewportHtml } from '../../../src/osprey/interfaces/artifacts/static/js/artifact-viewport.js';

const HOSTILE = [
  '# Report',
  '',
  '<img src="x" onerror="window.__pwned = 1">',
  '',
  '<script>window.__pwned = 2</script>',
  '',
  '[click](javascript:window.__pwned=3)',
  '',
  '<a href="#" onclick="window.__pwned = 4">x</a>',
  '',
  '<iframe src="/api/artifacts"></iframe>',
].join('\n');

function realLibraries() {
  vi.stubGlobal('marked', marked);
  vi.stubGlobal('DOMPurify', createDOMPurify(window));
}

/** @param {Element} el */
function assertInert(el) {
  expect(el.querySelector('script')).toBeNull();
  expect(el.querySelector('iframe')).toBeNull();
  expect(el.querySelector('[onerror]')).toBeNull();
  expect(el.querySelector('[onclick]')).toBeNull();
  for (const a of el.querySelectorAll('a')) {
    expect(a.getAttribute('href') ?? '').not.toMatch(/^\s*javascript:/i);
  }
  expect(/** @type {any} */ (window).__pwned).toBeUndefined();
}

afterEach(() => {
  vi.unstubAllGlobals();
  delete (/** @type {any} */ (window).__pwned);
});

describe('renderMarkdownInto (md-render.js, the one sink)', () => {
  test('strips <img onerror>, <script>, javascript: links and inline handlers', () => {
    realLibraries();
    const el = document.createElement('div');
    renderMarkdownInto(el, HOSTILE);
    assertInert(el);
    // The benign markup survives: the heading and the (handler-less) image.
    expect(el.querySelector('h1')?.textContent).toBe('Report');
    const img = el.querySelector('img');
    expect(img).not.toBeNull();
    expect(img?.hasAttribute('onerror')).toBe(false);
  });

  test('only DOMPurify output reaches innerHTML', () => {
    vi.stubGlobal('marked', marked);
    vi.stubGlobal('DOMPurify', { sanitize: () => '<em>sanitised</em>' });
    const el = document.createElement('div');
    renderMarkdownInto(el, HOSTILE);
    expect(el.innerHTML).toBe('<em>sanitised</em>');
  });

  test('without DOMPurify the source is inert text, never markup', () => {
    vi.stubGlobal('marked', marked);
    vi.stubGlobal('DOMPurify', undefined);
    const el = document.createElement('div');
    renderMarkdownInto(el, HOSTILE);
    expect(el.children.length).toBe(0);
    expect(el.textContent).toBe(HOSTILE);
  });

  test('a throwing sanitiser degrades to inert text', () => {
    vi.stubGlobal('marked', marked);
    vi.stubGlobal('DOMPurify', {
      sanitize: () => {
        throw new Error('boom');
      },
    });
    const el = document.createElement('div');
    renderMarkdownInto(el, HOSTILE);
    expect(el.children.length).toBe(0);
    expect(el.textContent).toBe(HOSTILE);
  });
});

describe('the gallery preview pane renders markdown through DOMPurify', () => {
  test('renderMarkdownView strips <img onerror> from a hostile artifact', async () => {
    realLibraries();
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, text: () => Promise.resolve(HOSTILE) })
    );
    const container = document.createElement('div');

    await renderMarkdownView(container, { id: 'm1', artifact_type: 'markdown', filename: 'r.md' });

    const rendered = container.querySelector('.osprey-md-rendered');
    expect(rendered).not.toBeNull();
    assertInert(/** @type {Element} */ (rendered));
    expect(rendered?.querySelector('img')).not.toBeNull();
  });
});

describe('gallery iframes run in an opaque origin', () => {
  test.each(['plot_html', 'table_html', 'dashboard_html', 'html', 'notebook'])(
    'the %s preview iframe is sandboxed without allow-same-origin',
    (artifact_type) => {
      const host = document.createElement('div');
      host.innerHTML = artifactViewportHtml({
        id: 'a1',
        artifact_type,
        filename: 'a.html',
        title: 't',
      });
      const iframe = host.querySelector('iframe');
      expect(iframe).not.toBeNull();
      const tokens = (iframe?.getAttribute('sandbox') ?? '').split(/\s+/);
      expect(tokens).toContain('allow-scripts');
      expect(tokens).toContain('allow-downloads');
      expect(tokens).not.toContain('allow-same-origin');
    }
  );
});
