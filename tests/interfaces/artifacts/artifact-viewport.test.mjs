/**
 * Unit tests for the Artifact Gallery's single per-type viewport dispatch
 * (artifact-viewport.js).
 *
 * This module exists to be the ONE answer to "how does artifact X render?".
 * Before it, the gallery had two full-size surfaces with a dispatch each —
 * one a full 9-branch switch with async markdown/JSON/timeseries mounting,
 * the other a borrowed `thumbnailHtml()` that only knew <img>/<iframe> types
 * and dropped markdown, JSON, text, PDF and timeseries to a placeholder. The
 * preview pane (preview.js) now calls exactly these two functions, standalone
 * and as the embedded viewer alike, so a type that renders here renders
 * wherever an artifact is shown full-size.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally) —
 * mirrors preview.test.mjs/preview-content.test.mjs:
 *   npx vitest run tests/interfaces/artifacts/artifact-viewport.test.mjs
 *
 * preview-content.js is mocked: `mountArtifactViewport`'s job is *which*
 * renderer fires against *which* element, not what marked/KaTeX or the JSON
 * viewer then produce (preview-content.test.mjs owns that). The module is
 * otherwise stateless — a container element and an artifact object in, no
 * shared state with preview.js or gallery.js — so there is no fixture beyond
 * a bare `document.createElement('div')`.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test, expect, describe, afterEach, beforeEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/artifacts/static/js/preview-content.js', () => ({
  renderMarkdownView: vi.fn(),
  renderJsonView: vi.fn(),
}));

// theme-manager is mocked so the theme the viewport stamps onto embedded
// pages is under test control: `null` (not initialised -- the default for
// every test below) leaves srcs bare, a concrete id must ride along as
// `?theme=` so the page first-paints in the gallery's theme.
const themeState = vi.hoisted(() => ({ current: /** @type {string|null} */ (null) }));
vi.mock('/design-system/js/theme-manager.js', () => ({
  getTheme: () => themeState.current,
}));

import {
  artifactViewportHtml,
  mountArtifactViewport,
  setViewerImporter,
} from '../../../src/osprey/interfaces/artifacts/static/js/artifact-viewport.js';
import { initTypeRegistry } from '../../../src/osprey/interfaces/artifacts/static/js/types.js';
import {
  renderMarkdownView,
  renderJsonView,
} from '../../../src/osprey/interfaces/artifacts/static/js/preview-content.js';
import { qs } from '../_support/dom.mjs';

/** @returns {any} */
function makeArtifact(overrides = {}) {
  return {
    id: 'a1',
    title: 'Beam Profile',
    filename: 'beam_profile.png',
    artifact_type: 'plot_png',
    category: 'visualization',
    timestamp: '2026-07-01T10:00:00Z',
    ...overrides,
  };
}

/**
 * Build a detached container holding `artifact`'s viewport markup — the exact
 * two-step both call sites perform (innerHTML, then mount).
 * @param {any} artifact
 * @param {any} [onTimeseriesNeeded]
 * @returns {HTMLElement}
 */
function mount(artifact, onTimeseriesNeeded = vi.fn()) {
  const container = document.createElement('div');
  container.innerHTML = artifactViewportHtml(artifact);
  mountArtifactViewport(container, artifact, { onTimeseriesNeeded });
  return container;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
  themeState.current = null;
});

describe('artifactViewportHtml: embedded pages carry the gallery theme', () => {
  // The old in-page bridge read the parent gallery's data-theme at load, so a
  // freshly mounted preview always matched. The served page now resolves via
  // theme-boot.js, whose top rung is `?theme=` -- the viewport must supply it,
  // or a new preview can boot in the OS theme inside a gallery showing another.
  test.each(['plot_html', 'table_html', 'dashboard_html', 'html'])(
    'artifact_type "%s" iframe src carries the theme in force as ?theme=',
    (artifact_type) => {
      themeState.current = 'retro-light';
      const container = document.createElement('div');
      container.innerHTML = artifactViewportHtml(makeArtifact({ id: 'a1', artifact_type }));

      expect(qs(container, 'iframe').getAttribute('src')).toMatch(/\?theme=retro-light$/);
    }
  );

  test('the notebook rendered endpoint carries it too', () => {
    themeState.current = 'high-contrast-dark';
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(
      makeArtifact({ id: 'nb1', artifact_type: 'notebook' })
    );

    expect(qs(container, 'iframe').getAttribute('src')).toBe(
      '/api/notebooks/nb1/rendered?theme=high-contrast-dark'
    );
  });

  test('images are not iframes and get no theme param', () => {
    themeState.current = 'retro-light';
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(makeArtifact({ artifact_type: 'plot_png' }));

    expect(qs(container, 'img').getAttribute('src')).not.toContain('theme=');
  });
});

// C-OPAQUE / C-MSG: an iframe of a type the server serves sandboxed carries the
// same token set as its `sandbox` attribute (the two intersect, and printing
// needs `allow-modals` in both); never `allow-same-origin`, which would hand
// agent-written script the console's origin. Inert types (PDF, text) carry no
// attribute at all -- Chrome's PDF viewer does not render sandboxed.
const OPAQUE_TOKENS = ['allow-downloads', 'allow-modals', 'allow-popups', 'allow-scripts'];

/** @param {Element} el */
function sandboxTokens(el) {
  return (el.getAttribute('sandbox') ?? '').split(/\s+/).filter(Boolean).sort();
}

describe('artifactViewportHtml: iframe sandbox follows the served type', () => {
  test.each([
    ['plot_html', 'text/html'],
    ['table_html', 'text/html'],
    ['dashboard_html', 'text/html'],
    ['html', 'text/html'],
    ['html', undefined],
    ['notebook', 'application/x-ipynb+json'],
  ])(
    'artifact_type "%s" (%s) is framed with exactly the C-OPAQUE tokens',
    (artifact_type, mime_type) => {
      const container = document.createElement('div');
      container.innerHTML = artifactViewportHtml(makeArtifact({ artifact_type, mime_type }));

      const frame = qs(container, 'iframe');
      expect(sandboxTokens(frame)).toEqual(OPAQUE_TOKENS);
      expect(frame.getAttribute('sandbox')).not.toContain('allow-same-origin');
    }
  );

  test('a PDF frame carries no sandbox attribute', () => {
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(
      makeArtifact({ artifact_type: 'weird_type', mime_type: 'application/pdf' })
    );

    expect(qs(container, 'iframe').hasAttribute('sandbox')).toBe(false);
  });

  test('a text frame (served text/plain whatever its own type) carries no sandbox attribute', () => {
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(
      makeArtifact({ artifact_type: 'text', mime_type: 'application/x-tex' })
    );

    expect(qs(container, 'iframe').hasAttribute('sandbox')).toBe(false);
  });
});

describe('artifactViewportHtml: per-type dispatch', () => {
  test.each([
    ['plot_html', 'iframe.preview-iframe-light'],
    ['table_html', 'iframe.preview-iframe-light'],
    ['dashboard_html', 'iframe.preview-iframe-light'],
    ['html', 'iframe.preview-iframe-light'],
    ['plot_png', 'img'],
    ['image', 'img'],
    ['markdown', '.md-preview-container'],
    ['json', '.json-viewer'],
    ['text', 'iframe.preview-iframe-dark'],
  ])('artifact_type "%s" renders %s', (artifact_type, selector) => {
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(makeArtifact({ artifact_type }));

    expect(container.querySelector(selector)).not.toBeNull();
  });

  test('artifact_type "notebook" renders the rendered-notebook iframe endpoint', () => {
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(
      makeArtifact({ id: 'nb1', artifact_type: 'notebook' })
    );

    expect(qs(container, 'iframe.preview-iframe-light').getAttribute('src')).toBe(
      '/api/notebooks/nb1/rendered'
    );
  });

  test('artifact_type "notebook" percent-encodes a hostile id in the iframe src', () => {
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(
      makeArtifact({ id: 'a/../b?x="y"', artifact_type: 'notebook' })
    );

    expect(qs(container, 'iframe').getAttribute('src')).toBe(
      '/api/notebooks/a%2F..%2Fb%3Fx%3D%22y%22/rendered'
    );
  });

  test('an unrecognized type with mime_type application/pdf renders a light iframe', () => {
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(
      makeArtifact({ artifact_type: 'weird_type', mime_type: 'application/pdf' })
    );

    expect(container.querySelector('iframe.preview-iframe-light')).not.toBeNull();
  });

  test('a fully unrecognized type falls back to a download link naming the file', () => {
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(
      makeArtifact({ artifact_type: 'mystery', filename: 'blob.bin' })
    );

    expect(qs(container, '.preview-download a').textContent).toContain('blob.bin');
  });

  test.each([
    ['metadata.data_type', { metadata: { data_type: 'timeseries', data_file: 'x.parquet' } }],
    ['archiver_read tool_source', { tool_source: 'archiver_read', data_file: 'y.parquet' }],
  ])('timeseries by %s wins over the artifact_type switch', (_label, overrides) => {
    const container = document.createElement('div');
    // artifact_type would otherwise route to the <img> branch.
    container.innerHTML = artifactViewportHtml(makeArtifact({ ...overrides }));

    expect(container.querySelector('.ts-viewport-container')).not.toBeNull();
    expect(container.querySelector('img')).toBeNull();
  });

  test('emits no ids — each mount is found through its own container', () => {
    for (const artifact_type of ['markdown', 'json', 'text', 'html', 'image']) {
      const html = artifactViewportHtml(makeArtifact({ artifact_type }));
      expect(html).not.toMatch(/\sid=/);
    }
    expect(
      artifactViewportHtml(makeArtifact({ data_file: 'x.parquet', tool_source: 'archiver_read' }))
    ).not.toMatch(/\sid=/);
  });
});

describe('mountArtifactViewport: post-mount renderers', () => {
  test('markdown mounts renderMarkdownView against the md container', () => {
    const container = mount(makeArtifact({ artifact_type: 'markdown' }));

    expect(renderMarkdownView).toHaveBeenCalledTimes(1);
    expect(vi.mocked(renderMarkdownView).mock.calls[0][0]).toBe(
      container.querySelector('.md-preview-container')
    );
  });

  test('json mounts renderJsonView against the json container', () => {
    const container = mount(makeArtifact({ artifact_type: 'json' }));

    expect(renderJsonView).toHaveBeenCalledTimes(1);
    expect(vi.mocked(renderJsonView).mock.calls[0][0]).toBe(
      container.querySelector('.json-viewer')
    );
  });

  test('timeseries fires onTimeseriesNeeded with the ts container and the artifact', () => {
    const artifact = makeArtifact({ tool_source: 'archiver_read', data_file: 'y.parquet' });
    const onTimeseriesNeeded = vi.fn();
    const container = mount(artifact, onTimeseriesNeeded);

    expect(onTimeseriesNeeded).toHaveBeenCalledTimes(1);
    expect(onTimeseriesNeeded.mock.calls[0][0]).toBe(
      container.querySelector('.ts-viewport-container')
    );
    expect(onTimeseriesNeeded.mock.calls[0][1]).toBe(artifact);
  });

  test('a timeseries artifact with no data file falls through to the type switch', () => {
    const onTimeseriesNeeded = vi.fn();
    const container = mount(
      makeArtifact({ artifact_type: 'image', metadata: { data_type: 'timeseries' } }),
      onTimeseriesNeeded
    );

    expect(onTimeseriesNeeded).not.toHaveBeenCalled();
    expect(container.querySelector('img')).not.toBeNull();
  });

  test('types with no post-mount step mount nothing', () => {
    const onTimeseriesNeeded = vi.fn();
    mount(makeArtifact({ artifact_type: 'html' }), onTimeseriesNeeded);

    expect(renderMarkdownView).not.toHaveBeenCalled();
    expect(renderJsonView).not.toHaveBeenCalled();
    expect(onTimeseriesNeeded).not.toHaveBeenCalled();
  });

  test('onTimeseriesNeeded is optional — a caller that omits it does not throw', () => {
    const artifact = makeArtifact({ tool_source: 'archiver_read', data_file: 'y.parquet' });
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(artifact);

    expect(() => mountArtifactViewport(container, artifact, {})).not.toThrow();
  });
});

describe('mountArtifactViewport: container scoping', () => {
  test('mounting two viewports targets each container, not the first in the document', () => {
    // The post-mount step must find its element through the container it is
    // handed. A dispatch that looked its containers up by document id would
    // resolve both mounts below to whichever came first in document order —
    // the whole reason this markup carries classes instead.
    const firstPane = document.createElement('div');
    const secondPane = document.createElement('div');
    document.body.append(firstPane, secondPane);

    const firstArtifact = makeArtifact({ id: 'first', artifact_type: 'markdown' });
    const secondArtifact = makeArtifact({ id: 'second', artifact_type: 'markdown' });

    firstPane.innerHTML = artifactViewportHtml(firstArtifact);
    mountArtifactViewport(firstPane, firstArtifact, {});
    secondPane.innerHTML = artifactViewportHtml(secondArtifact);
    mountArtifactViewport(secondPane, secondArtifact, {});

    const calls = vi.mocked(renderMarkdownView).mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[0][0]).toBe(firstPane.querySelector('.md-preview-container'));
    expect(calls[1][0]).toBe(secondPane.querySelector('.md-preview-container'));
    expect(calls[0][0]).not.toBe(calls[1][0]);

    firstPane.remove();
    secondPane.remove();
  });
});

// ---------------------------------------------------------------------------
// The viewer hook: an extension type's own ES module renders its artifacts
// ---------------------------------------------------------------------------

/** The shared /api/type-registry fixture: `beam_log` has a viewer, `orbit_fit` none. */
const REGISTRY = JSON.parse(
  readFileSync(
    resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/type-registry.json'),
    'utf8'
  )
);

/** @param {any} registry */
async function loadRegistry(registry) {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ json: () => Promise.resolve(registry) }));
  await initTypeRegistry();
  vi.unstubAllGlobals();
}

/**
 * Mount the way preview.js does: the viewport sits inside pane chrome the
 * dispatch must never touch, and the container handed to mount is the pane.
 * @param {any} artifact
 */
async function mountInPane(artifact) {
  const pane = document.createElement('div');
  pane.innerHTML = `<div class="pane-chrome">chrome</div><div class="preview-viewport">${artifactViewportHtml(artifact)}</div>`;
  await mountArtifactViewport(pane, artifact, {});
  return pane;
}

describe('the viewer hook', () => {
  /** @type {import('vitest').Mock} */
  let importer;
  /** @type {import('vitest').MockInstance} */
  let warn;

  beforeEach(async () => {
    await loadRegistry(REGISTRY);
    importer = vi.fn();
    setViewerImporter(importer);
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(async () => {
    setViewerImporter(null);
    warn.mockRestore();
    await loadRegistry({});
    document.documentElement.style.removeProperty('--bg-primary');
  });

  const beamLog = () => makeArtifact({ kind: 'beam_log', artifact_type: 'table_html' });

  test('the fixture declares beam_log with a viewer and orbit_fit without', () => {
    const byKey = Object.fromEntries(
      REGISTRY.extension_types.map((/** @type {{key: string}} */ e) => [e.key, e])
    );
    expect(byKey.beam_log.viewer_url).toBeTruthy();
    expect(byKey.orbit_fit.viewer_url).toBeNull();
  });

  test('render receives the container, the artifact and ctx {fileUrl, theme, tokens}', async () => {
    themeState.current = 'retro-light';
    document.documentElement.style.setProperty('--bg-primary', '#123456');
    /** @type {import('vitest').Mock<(el: HTMLElement, artifact: unknown, ctx: {fileUrl: string, theme: string, tokens: Record<string, string>}) => void>} */
    const render = vi.fn((el) => {
      el.textContent = 'rendered by beam_log';
    });
    importer.mockResolvedValue({ render });
    const artifact = beamLog();

    const pane = await mountInPane(artifact);

    expect(importer).toHaveBeenCalledTimes(1);
    expect(importer).toHaveBeenCalledWith('/viewers/beam_log.js');
    expect(render).toHaveBeenCalledTimes(1);
    const [el, passed, ctx] = render.mock.calls[0];
    expect(el).toBe(pane.querySelector('.ext-viewer-container'));
    expect(passed).toBe(artifact);
    expect(ctx.fileUrl).toBe('/files/a1/beam_profile.png');
    expect(ctx.theme).toBe('retro-light');
    expect(ctx.tokens['--bg-primary']).toBe('#123456');
    expect(qs(pane, '.pane-chrome').textContent).toBe('chrome');
    expect(qs(pane, '.ext-viewer-container').textContent).toBe('rendered by beam_log');
    expect(warn).not.toHaveBeenCalled();
  });

  test('the key is URL-encoded into the import path', async () => {
    await loadRegistry({
      extension_types: [{ key: 'a b', label: 'AB', base_form: 'data', viewer_url: '/x' }],
    });
    importer.mockResolvedValue({ render: vi.fn() });
    await mountInPane(makeArtifact({ kind: 'a b' }));
    expect(importer).toHaveBeenCalledWith('/viewers/a%20b.js');
  });

  /** @type {[string, () => void][]} */
  const failures = [
    ['a rejected import', () => importer.mockRejectedValue(new Error('404'))],
    ['a module without a render export', () => importer.mockResolvedValue({ default: () => {} })],
    [
      'a render that throws',
      () =>
        importer.mockResolvedValue({
          render: () => {
            throw new Error('boom');
          },
        }),
    ],
    [
      'a render whose promise rejects',
      () => importer.mockResolvedValue({ render: () => Promise.reject(new Error('late boom')) }),
    ],
  ];

  test.each(failures)('%s falls back to the base dispatch and warns once', async (_name, arm) => {
    arm();
    const artifact = beamLog();

    const pane = await mountInPane(artifact);

    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toContain('beam_log');
    // The base dispatch for a table_html: the opaque iframe, inside the
    // viewport — the pane's chrome is untouched.
    const iframe = qs(pane, '.ext-viewer-container iframe');
    expect(iframe.getAttribute('src')).toBe('/files/a1/beam_profile.png');
    expect(qs(pane, '.pane-chrome').textContent).toBe('chrome');
  });

  test('a fallback runs the base post-mount step too', async () => {
    importer.mockRejectedValue(new Error('404'));
    const artifact = makeArtifact({ kind: 'beam_log', artifact_type: 'markdown' });

    const pane = await mountInPane(artifact);

    expect(warn).toHaveBeenCalledTimes(1);
    expect(renderMarkdownView).toHaveBeenCalledTimes(1);
    expect(vi.mocked(renderMarkdownView).mock.calls[0][0]).toBe(
      pane.querySelector('.ext-viewer-container .md-preview-container')
    );
  });

  test('a kind without a viewer takes the base dispatch, no import', async () => {
    const artifact = makeArtifact({ kind: 'orbit_fit', artifact_type: 'plot_png' });
    expect(artifactViewportHtml(artifact)).toBe(artifactViewportHtml(makeArtifact()));
    await mountInPane(artifact);
    expect(importer).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
  });

  test('an unregistered kind takes the base dispatch, no import', async () => {
    const artifact = makeArtifact({ kind: 'no_such_type' });
    expect(artifactViewportHtml(artifact)).toBe(artifactViewportHtml(makeArtifact()));
    await mountInPane(artifact);
    expect(importer).not.toHaveBeenCalled();
  });

  test('an artifact without a kind takes the base dispatch, no import', async () => {
    const artifact = makeArtifact({ artifact_type: 'markdown' });
    const pane = await mountInPane(artifact);
    expect(importer).not.toHaveBeenCalled();
    expect(pane.querySelector('.ext-viewer-container')).toBeNull();
    expect(renderMarkdownView).toHaveBeenCalledTimes(1);
  });

  test('a viewer outranks the timeseries branch for its kind', async () => {
    const render = vi.fn();
    importer.mockResolvedValue({ render });
    const onTimeseriesNeeded = vi.fn();
    const artifact = makeArtifact({
      kind: 'beam_log',
      artifact_type: 'plot_html',
      tool_source: 'archiver_read',
      data_file: 'data.json',
    });
    const pane = document.createElement('div');
    pane.innerHTML = artifactViewportHtml(artifact);
    await mountArtifactViewport(pane, artifact, { onTimeseriesNeeded });
    expect(render).toHaveBeenCalledTimes(1);
    expect(onTimeseriesNeeded).not.toHaveBeenCalled();
  });
});
