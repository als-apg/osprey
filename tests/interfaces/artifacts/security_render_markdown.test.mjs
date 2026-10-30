// @vitest-environment jsdom
/**
 * MARKDOWN path of the hostile-content security suite (security_render.test.mjs
 * holds the other render paths): the gallery's markdown preview (md-render.js
 * via preview-content.js) driven with the vendored marked and DOMPurify builds
 * — byte-checked against vendor_manifest.json — so script-bearing agent
 * markdown is proven inert with the real sanitizer:
 *   npx vitest run tests/interfaces/artifacts/security_render_markdown.test.mjs
 *
 * jsdom rather than the suite's happy-dom: DOMPurify does not sanitise under
 * happy-dom, and jsdom cannot be loaded from inside a happy-dom test file under
 * the vmThreads pool (its CommonJS graph require()s the ESM-only @exodus/bytes,
 * which fails on Node 22). The environment loads jsdom outside the test VM, as
 * agent-links-vendored.test.mjs does.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { test, expect, describe, beforeAll, beforeEach, afterEach, vi } from 'vitest';

import {
  renderMathInMarkdown,
  renderMarkdownView,
} from '../../../src/osprey/interfaces/artifacts/static/js/preview-content.js';

// =========================================================================
// MARKDOWN path (md-render.js via preview-content.js) — agent-written
// markdown carrying script, rendered with the vendored marked + DOMPurify.
// =========================================================================

describe('MARKDOWN path (md-render.js / preview-content.js) — script-bearing markdown', () => {
  const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
  const MANIFEST = path.join(REPO_ROOT, 'src/osprey/interfaces/vendor_manifest.json');
  /** @type {Readonly<Record<string, string>>} */
  const BUILDS = Object.freeze({
    marked: 'node_modules/marked/marked.min.js',
    DOMPurify: 'node_modules/dompurify/dist/purify.min.js',
  });

  /** @param {string} name */
  function pinned(name) {
    const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
    const asset = manifest.assets.find((/** @type {{ name: string }} */ a) => a.name === name);
    if (!asset) throw new Error(`vendor_manifest.json names no asset ${name}`);
    return asset;
  }

  /** @param {string} name */
  function buildBytes(name) {
    return readFileSync(path.join(REPO_ROOT, BUILDS[name]));
  }

  /**
   * Evaluate a UMD build as a classic <script> would, then take its global back.
   * @param {string} name
   * @returns {any}
   */
  function loadClassicScript(name) {
    new Function('exports', 'module', 'define', 'require', buildBytes(name).toString('utf8'))();
    const g = /** @type {any} */ (globalThis);
    const lib = g[name];
    delete g[name];
    return lib;
  }

  /**
   * A KaTeX stand-in shaped like KaTeX's real output (quoted class/style
   * attributes, escaped source), so the test sees the markup the swap splices in.
   */
  const fakeKatex = {
    renderToString: (/** @type {string} */ expr, /** @type {{displayMode: boolean}} */ opts) =>
      `<span class="katex${opts.displayMode ? ' katex-display' : ''}">` +
      `<span class="katex-html" aria-hidden="true" style="height:0.8em">` +
      `<span class="mord mathnormal">${escapeForFake(expr)}</span></span></span>`,
  };

  /** @param {string} s */
  function escapeForFake(s) {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /** Hostile markdown bodies an agent could write into a .md artifact. */
  const HOSTILE_MD = {
    SCRIPT: '# Report\n\n<script>window.__pwned = 1</script>\n\nbody',
    IMG_ONERROR: 'see <img src=x onerror="window.__pwned=1"> here',
    JS_LINK: '[click me](javascript:window.__pwned=1)',
    RAW_JS_ANCHOR: '<a href="javascript:window.__pwned=1">x</a>',
    IFRAME: '<iframe srcdoc="<script>parent.__pwned=1</script>"></iframe>',
    SVG_ONLOAD: '<svg onload="window.__pwned=1"><circle r="4"/></svg>',
    // A math placeholder inside a link target: splicing KaTeX's quoted markup
    // into the href after sanitizing would break out of the attribute.
    MATH_IN_HREF: '[x](http://example.com/$a" onmouseover="window.__pwned=1$)',
  };

  /**
   * Nothing executable anywhere under `root`.
   * @param {ParentNode} root
   */
  function expectInert(root) {
    expect(root.querySelector('script')).toBeNull();
    expect(root.querySelector('iframe')).toBeNull();
    for (const el of root.querySelectorAll('*')) {
      for (const attr of Array.from(el.attributes)) {
        expect(attr.name.startsWith('on'), `${el.tagName} carries ${attr.name}`).toBe(false);
        if (attr.name === 'href' || attr.name === 'src' || attr.name === 'xlink:href') {
          expect(attr.value.trim().toLowerCase().startsWith('javascript:')).toBe(false);
        }
      }
    }
  }

  /** @type {{ marked: any, DOMPurify: any }} */
  let real;

  beforeAll(() => {
    real = {
      marked: loadClassicScript('marked'),
      DOMPurify: loadClassicScript('DOMPurify')(window),
    };
  });

  beforeEach(() => {
    vi.stubGlobal('marked', real.marked);
    vi.stubGlobal('DOMPurify', real.DOMPurify);
    vi.stubGlobal('katex', fakeKatex);
    delete (/** @type {any} */ (globalThis).__pwned);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
  });

  for (const name of Object.keys(BUILDS)) {
    test(`${name} in node_modules is the build the vendor manifest pins`, () => {
      const digest = createHash('sha256').update(buildBytes(name)).digest('hex');
      expect(digest).toBe(pinned(name).sha256);
    });
  }

  test('the vendor manifest delivers DOMPurify to the artifacts app', () => {
    expect(pinned('DOMPurify').targets).toContain('artifacts/static/js/vendor/');
  });

  test('the gallery page loads DOMPurify before its modules', () => {
    const page = readFileSync(
      path.join(REPO_ROOT, 'src/osprey/interfaces/artifacts/static/index.html'),
      'utf8'
    );
    const tag = page.indexOf("vendor_url('DOMPurify', '/static/js/vendor/purify.min.js')");
    expect(tag).toBeGreaterThan(-1);
    expect(tag).toBeLessThan(page.indexOf('type="module"'));
  });

  for (const [label, md] of Object.entries(HOSTILE_MD)) {
    test(`renderMathInMarkdown leaves ${label} inert`, () => {
      const root = document.createElement('div');
      root.innerHTML = renderMathInMarkdown(md);
      expectInert(root);
    });

    test(`the preview pane renders ${label} with nothing executable`, async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({ ok: true, text: () => Promise.resolve(md) })
      );
      const container = document.createElement('div');
      document.body.appendChild(container);

      await renderMarkdownView(container, {
        id: 'm1',
        filename: 'r.md',
        artifact_type: 'markdown',
      });

      const rendered = container.querySelector('.osprey-md-rendered');
      expect(rendered).not.toBeNull();
      expectInert(container);
      expect(/** @type {any} */ (globalThis).__pwned).toBeUndefined();
    });
  }

  test('ordinary markdown still renders as markup', () => {
    const root = document.createElement('div');
    root.innerHTML = renderMathInMarkdown(
      '# Title\n\n**bold** and [link](https://example.com)\n\n```py\nx = 1\n```'
    );
    expect(root.querySelector('h1')?.textContent).toBe('Title');
    expect(root.querySelector('strong')?.textContent).toBe('bold');
    expect(root.querySelector('a')?.getAttribute('href')).toBe('https://example.com');
    expect(root.querySelector('pre code')).not.toBeNull();
  });

  test('math still renders through the sanitizer', () => {
    const root = document.createElement('div');
    root.innerHTML = renderMathInMarkdown('Inline $x^2$ and display $$E = mc^2$$ done.');
    const math = root.querySelectorAll('.katex');
    expect(math.length).toBe(2);
    expect(root.querySelector('.katex-display .katex-html')?.getAttribute('style')).toBe(
      'height:0.8em'
    );
    expect(root.textContent).toContain('x^2');
    expect(root.textContent).toContain('E = mc^2');
    expect(root.innerHTML).not.toContain('\x00MATH');
  });

  test('without DOMPurify the preview shows the markdown as plain text', async () => {
    vi.stubGlobal('DOMPurify', undefined);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        text: () => Promise.resolve(HOSTILE_MD.IMG_ONERROR),
      })
    );
    const container = document.createElement('div');

    await renderMarkdownView(container, { id: 'm1', filename: 'r.md', artifact_type: 'markdown' });

    expect(container.querySelector('img')).toBeNull();
    expect(container.textContent).toContain('onerror');
  });
});
