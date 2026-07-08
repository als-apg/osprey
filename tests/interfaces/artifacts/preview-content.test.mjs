// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * Unit tests for the Artifact Gallery preview-content module
 * (preview-content.js, preview.js's sibling — kept separate purely to stay
 * under the 450-line module cap): the markdown+KaTeX render pipeline and the
 * recursive JSON viewer.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal — mirrors preview.test.mjs/state.test.mjs.
 *   npx vitest run tests/interfaces/artifacts/preview-content.test.mjs
 *
 * Both exports here take a container element and an artifact object as
 * plain args (no shared mutable state with preview.js, no DOM fixture
 * beyond a bare `document.createElement('div')`). `marked`/`hljs`/`katex`
 * are vendored classic-script globals (see vendor-globals.d.ts); tests stub
 * them via vi.stubGlobal.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  renderMathInMarkdown,
  renderMarkdownView,
  renderJsonHtml,
  renderJsonView,
} from '../../../src/osprey/interfaces/artifacts/static/js/preview-content.js';

/** @returns {any} */
function makeArtifact(overrides = {}) {
  return {
    id: 'a1',
    title: 'Beam Profile',
    filename: 'beam_profile.png',
    artifact_type: 'plot_png',
    category: 'visualization',
    pinned: false,
    timestamp: '2026-07-01T10:00:00Z',
    size_bytes: 2048,
    ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('renderJsonHtml (recursive JSON viewer)', () => {
  test('renders primitives', () => {
    expect(renderJsonHtml(null, 0)).toBe('<span class="json-null">null</span>');
    expect(renderJsonHtml(true, 0)).toBe('<span class="json-bool">true</span>');
    expect(renderJsonHtml(42, 0)).toBe('<span class="json-num">42</span>');
    expect(renderJsonHtml('hi', 0)).toBe('<span class="json-str">"hi"</span>');
  });

  test('truncates strings past 200 characters', () => {
    const long = 'x'.repeat(250);
    const html = renderJsonHtml(long, 0);
    expect(html).toContain('x'.repeat(200) + '...');
    expect(html).not.toContain('x'.repeat(201));
  });

  test('renders empty and populated arrays, truncating past 20 items', () => {
    expect(renderJsonHtml([], 0)).toBe('<span class="json-bracket">[]</span>');

    const small = renderJsonHtml([1, 2], 0);
    expect((small.match(/json-item/g) || []).length).toBe(2);

    const big = renderJsonHtml(Array.from({ length: 25 }, (_, i) => i), 0);
    expect(big).toContain('... 5 more');
  });

  test('renders empty and populated objects with escaped keys', () => {
    expect(renderJsonHtml({}, 0)).toBe('<span class="json-bracket">{}</span>');
    const html = renderJsonHtml({ '<k>': 1 }, 0);
    expect(html).toContain('&lt;k&gt;');
    expect(html).toContain('json-num');
  });

  test('truncates objects past 50 keys, mirroring the array truncation shape', () => {
    /** @type {Record<string, number>} */
    const manyKeys = {};
    for (let i = 0; i < 75; i++) manyKeys[`key${i}`] = i;

    const html = renderJsonHtml(manyKeys, 0);

    expect((html.match(/json-item/g) || []).length).toBe(51); // 50 rendered keys + the truncation marker row
    expect(html).toContain('key0');
    expect(html).toContain('key49');
    expect(html).not.toContain('key50');
    expect(html).toContain('<span class="json-truncated">... 25 more</span>');
  });

  test('truncates recursion past depth 6', () => {
    /** @type {any} */
    let nested = 'leaf';
    for (let i = 0; i < 8; i++) nested = [nested];
    expect(renderJsonHtml(nested, 0)).toContain('json-truncated');
  });
});

describe('renderJsonView', () => {
  test('fetches the artifact file, parses JSON, and renders via renderJsonHtml', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ a: 1 }),
    }));
    const container = document.createElement('div');

    await renderJsonView(container, makeArtifact({ artifact_type: 'json', filename: 'x.json' }));

    expect(container.innerHTML).toBe(renderJsonHtml({ a: 1 }, 0));
  });

  test('on fetch failure: shows a friendly fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    const container = document.createElement('div');

    await renderJsonView(container, makeArtifact());

    expect(container.textContent).toContain('Failed to load JSON');
  });
});

describe('renderMathInMarkdown', () => {
  test('delegates straight to marked.parse when katex is undefined', () => {
    vi.stubGlobal('marked', { parse: vi.fn((t) => `<p>${t}</p>`) });
    expect(renderMathInMarkdown('hello')).toBe('<p>hello</p>');
  });

  test('renders display ($$...$$) and inline ($...$) math via katex, then runs marked on the rest', () => {
    const renderToString = vi.fn((expr, opts) => `<math data-display="${opts.displayMode}">${expr}</math>`);
    vi.stubGlobal('katex', { renderToString });
    vi.stubGlobal('marked', { parse: (t) => `<p>${t}</p>` });

    const html = renderMathInMarkdown('Inline $x^2$ and display $$y = mx + b$$ done.');

    expect(renderToString).toHaveBeenCalledWith('x^2', expect.objectContaining({ displayMode: false }));
    expect(renderToString).toHaveBeenCalledWith('y = mx + b', expect.objectContaining({ displayMode: true }));
    expect(html).toContain('<math data-display="false">x^2</math>');
    expect(html).toContain('<math data-display="true">y = mx + b</math>');
    expect(html).not.toContain('\x00MATH');
  });

  test('falls back to an escaped error span when katex.renderToString throws', () => {
    vi.stubGlobal('katex', { renderToString: vi.fn(() => { throw new Error('bad expr'); }) });
    vi.stubGlobal('marked', { parse: (t) => t });

    const html = renderMathInMarkdown('$broken$');
    expect(html).toContain('katex-error-inline');
  });
});

describe('renderMarkdownView', () => {
  test('fetches the artifact file and renders it through the marked+KaTeX pipeline', async () => {
    vi.stubGlobal('marked', {
      use: vi.fn(),
      parse: vi.fn((t) => `<p>${t}</p>`),
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, text: () => Promise.resolve('# Hi') }));
    const container = document.createElement('div');

    await renderMarkdownView(container, makeArtifact({ artifact_type: 'markdown' }));

    const rendered = container.querySelector('.osprey-md-rendered');
    expect(rendered).not.toBeNull();
    expect(rendered.innerHTML).toBe('<p># Hi</p>');
  });

  test('falls back to plain textContent when marked is unavailable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, text: () => Promise.resolve('raw text') }));
    const container = document.createElement('div');

    await renderMarkdownView(container, makeArtifact());

    expect(container.querySelector('.osprey-md-rendered').textContent).toBe('raw text');
  });

  test('on fetch failure: shows a friendly fallback', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    const container = document.createElement('div');

    await renderMarkdownView(container, makeArtifact());

    expect(container.textContent).toContain('Failed to load markdown');
  });
});
