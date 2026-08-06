/**
 * Unit tests for the Artifact Gallery's single per-type viewport dispatch
 * (artifact-viewport.js).
 *
 * This module exists to be the ONE answer to "how does artifact X render?".
 * Before it, Expert's preview pane (preview.js) and Simple's result card
 * (gallery.js) each had their own dispatch — Expert's a full 9-branch switch
 * with async markdown/JSON/timeseries mounting, Simple's a borrowed
 * `thumbnailHtml()` that only knew <img>/<iframe> types and dropped markdown,
 * JSON, text, PDF and timeseries to a placeholder. Everything below is
 * therefore a parity guard as much as a dispatch guard: both panes call
 * exactly these two functions, so a type that renders here renders in both.
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

import { test, expect, describe, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/artifacts/static/js/preview-content.js', () => ({
  renderMarkdownView: vi.fn(),
  renderJsonView: vi.fn(),
}));

import {
  artifactViewportHtml,
  mountArtifactViewport,
} from '../../../src/osprey/interfaces/artifacts/static/js/artifact-viewport.js';
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
    ['archiver_data category', { category: 'archiver_data', data_file: 'y.parquet' }],
  ])('timeseries by %s wins over the artifact_type switch', (_label, overrides) => {
    const container = document.createElement('div');
    // artifact_type would otherwise route to the <img> branch.
    container.innerHTML = artifactViewportHtml(makeArtifact({ ...overrides }));

    expect(container.querySelector('.ts-viewport-container')).not.toBeNull();
    expect(container.querySelector('img')).toBeNull();
  });

  test('emits no ids — both panes mount this markup, so ids would collide', () => {
    for (const artifact_type of ['markdown', 'json', 'text', 'html', 'image']) {
      const html = artifactViewportHtml(makeArtifact({ artifact_type }));
      expect(html).not.toMatch(/\sid=/);
    }
    expect(
      artifactViewportHtml(makeArtifact({ data_file: 'x.parquet', category: 'archiver_data' }))
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
    const artifact = makeArtifact({ category: 'archiver_data', data_file: 'y.parquet' });
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
    const artifact = makeArtifact({ category: 'archiver_data', data_file: 'y.parquet' });
    const container = document.createElement('div');
    container.innerHTML = artifactViewportHtml(artifact);

    expect(() => mountArtifactViewport(container, artifact, {})).not.toThrow();
  });
});

describe('mountArtifactViewport: container scoping (the parity property)', () => {
  test('mounting two viewports targets each container, not the first in the document', () => {
    // The Expert pane and the Simple card can both hold a mounted viewport at
    // once (renderPreview still runs while Simple is on screen). A dispatch
    // that looked its containers up by document id would resolve both mounts
    // to whichever came first in document order — the whole reason this markup
    // carries classes instead.
    const expertPane = document.createElement('div');
    const simpleCard = document.createElement('div');
    document.body.append(expertPane, simpleCard);

    const expertArtifact = makeArtifact({ id: 'expert', artifact_type: 'markdown' });
    const simpleArtifact = makeArtifact({ id: 'simple', artifact_type: 'markdown' });

    expertPane.innerHTML = artifactViewportHtml(expertArtifact);
    mountArtifactViewport(expertPane, expertArtifact, {});
    simpleCard.innerHTML = artifactViewportHtml(simpleArtifact);
    mountArtifactViewport(simpleCard, simpleArtifact, {});

    const calls = vi.mocked(renderMarkdownView).mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[0][0]).toBe(expertPane.querySelector('.md-preview-container'));
    expect(calls[1][0]).toBe(simpleCard.querySelector('.md-preview-container'));
    expect(calls[0][0]).not.toBe(calls[1][0]);

    expertPane.remove();
    simpleCard.remove();
  });
});
