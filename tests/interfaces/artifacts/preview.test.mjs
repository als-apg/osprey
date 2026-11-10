/**
 * Unit tests for the Artifact Gallery preview pane shell (preview.js, task
 * 5.4 extraction: renderPreview's per-type viewport dispatch). Delete is the
 * hub's (artifact-verbs.test.mjs).
 *
 * The markdown+KaTeX render pipeline and the JSON viewer moved to the
 * sibling preview-content.js module (split purely to stay under the
 * 450-line cap) and are covered by preview-content.test.mjs instead — the
 * "artifact_type renders the right viewport container" dispatch case for
 * markdown/json stays here since that's preview.js's own renderPreview
 * behavior, not preview-content.js's.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal — mirrors state.test.mjs:
 *   npx vitest run tests/interfaces/artifacts/preview.test.mjs
 *
 * preview.js reads/writes state.js's module-singleton artifact/selection
 * state (no vi.resetModules, just call the setters fresh per test), and
 * formats via types.js (real implementations).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  setArtifacts,
  setSelectedArtifact,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';

import { createPreviewRenderer } from '../../../src/osprey/interfaces/artifacts/static/js/preview.js';
import { qs, byId } from '../_support/dom.mjs';

/** Minimal DOM fixture matching artifacts/static/index.html's structure. */
function mountFixture() {
  document.body.className = '';
  document.body.innerHTML = `
    <div class="browse-preview-pane" id="browse-preview-pane">
      <div class="preview-empty" id="preview-empty"></div>
      <div class="preview-content hidden" id="preview-content"></div>
    </div>
  `;
}

/** @returns {any} */
function makeArtifact(overrides = {}) {
  return {
    id: 'a1',
    title: 'Beam Profile',
    filename: 'beam_profile.png',
    artifact_type: 'plot_png',
    category: 'visualization',
    form: 'plots',
    form_label: 'Plots',
    timestamp: '2026-07-01T10:00:00Z',
    size_bytes: 2048,
    ...overrides,
  };
}

function makeCallbacks() {
  return {
    onTimeseriesNeeded: vi.fn(),
  };
}

beforeEach(() => {
  mountFixture();
  setArtifacts([]);
  setSelectedArtifact(null);
  vi.stubGlobal(
    'confirm',
    vi.fn(() => true)
  );
  Object.defineProperty(window.navigator, 'clipboard', {
    value: { writeText: vi.fn(() => Promise.resolve()) },
    configurable: true,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('renderPreview: empty state', () => {
  test('shows the empty placeholder and hides the content pane when nothing is selected', () => {
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    expect(byId('preview-empty').classList.contains('hidden')).toBe(false);
    expect(byId('preview-content').classList.contains('hidden')).toBe(true);
  });

  test('the empty pane is the centred state: no artifact selected, pick one in the sidebar', () => {
    createPreviewRenderer(makeCallbacks()).renderPreview();

    const state = qs(byId('preview-empty'), '.osprey-state');
    expect(qs(state, '.osprey-state-message').textContent).toBe('No artifact selected');
    expect(qs(state, '.osprey-state-detail').textContent).toBe('Pick one in the sidebar.');
    expect(state.querySelector('.osprey-ic')).not.toBeNull();
  });

  test('embedded, the list to pick from is the hub sidebar', () => {
    createPreviewRenderer(makeCallbacks()).renderPreview();

    expect(qs(byId('preview-empty'), '.osprey-state-detail').textContent).toBe(
      'Pick one in the sidebar.'
    );
  });
});

describe('renderPreview: per-type viewport dispatch', () => {
  // The markdown and json viewports fetch the artifact's file to fill the
  // container renderPreview just created. Nothing serves this environment, so
  // that read is answered here; the rows below assert on the container, not on
  // what lands inside it. Any other URL is a dependency these cases have not
  // declared, and fails loudly.
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url !== '/files/a1/beam_profile.png') throw new Error(`unstubbed fetch: ${url}`);
        return { ok: true, status: 200, text: async () => '', json: async () => ({}) };
      })
    );
  });

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
    setSelectedArtifact(makeArtifact({ artifact_type }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    expect(byId('preview-empty').classList.contains('hidden')).toBe(true);
    expect(byId('preview-content').classList.contains('hidden')).toBe(false);
    expect(document.querySelector(`.preview-viewport ${selector}`)).not.toBeNull();
  });

  test('artifact_type "notebook" renders the rendered-notebook iframe endpoint', () => {
    setSelectedArtifact(makeArtifact({ id: 'nb1', artifact_type: 'notebook' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    const iframe = qs(document, '.preview-viewport iframe.preview-iframe-light');
    expect(iframe.getAttribute('src')).toBe('/api/notebooks/nb1/rendered');
  });

  test('artifact_type "notebook" percent-encodes a hostile id in the iframe src', () => {
    setSelectedArtifact(makeArtifact({ id: 'a/../b?x="y"', artifact_type: 'notebook' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    const iframe = qs(document, '.preview-viewport iframe.preview-iframe-light');
    const src = iframe.getAttribute('src');
    if (src === null) throw new Error('unreachable: iframe has no src');
    const idSegment = src.split('/')[3];
    expect(idSegment).not.toMatch(/[/?"]/);
    expect(src).toBe('/api/notebooks/a%2F..%2Fb%3Fx%3D%22y%22/rendered');
  });

  test('an unrecognized type with mime_type application/pdf renders a light iframe of the file', () => {
    setSelectedArtifact(
      makeArtifact({ artifact_type: 'weird_type', mime_type: 'application/pdf' })
    );
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    expect(document.querySelector('.preview-viewport iframe.preview-iframe-light')).not.toBeNull();
  });

  test('an opaque artifact previews in a frame without allow-same-origin', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'html', mime_type: 'text/html' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    const sandbox = qs(document, '.preview-viewport iframe').getAttribute('sandbox') ?? '';
    expect(sandbox.split(/\s+/).sort()).toEqual([
      'allow-downloads',
      'allow-modals',
      'allow-popups',
      'allow-scripts',
    ]);
  });

  test('a PDF previews in a frame with no sandbox attribute', () => {
    setSelectedArtifact(
      makeArtifact({ artifact_type: 'weird_type', mime_type: 'application/pdf' })
    );
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    expect(qs(document, '.preview-viewport iframe').hasAttribute('sandbox')).toBe(false);
  });

  test('a fully unrecognized type falls back to a download link', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'mystery', filename: 'blob.bin' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    const link = document.querySelector('.preview-download a');
    expect(link).not.toBeNull();
    if (link === null) throw new Error('unreachable: download link missing');
    expect(link.textContent).toContain('blob.bin');
  });

  test('timeseries data (metadata.data_type) renders the timeseries container and fires onTimeseriesNeeded', () => {
    const artifact = makeArtifact({
      artifact_type: 'plot_html',
      metadata: { data_type: 'timeseries', data_file: 'x.parquet' },
    });
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    const renderer = createPreviewRenderer(callbacks);
    renderer.renderPreview();

    const tsEl = document.querySelector('.preview-viewport .ts-viewport-container');
    expect(tsEl).not.toBeNull();
    expect(callbacks.onTimeseriesNeeded).toHaveBeenCalledTimes(1);
    expect(callbacks.onTimeseriesNeeded.mock.calls[0][0]).toBe(tsEl);
    expect(callbacks.onTimeseriesNeeded.mock.calls[0][1]).toBe(artifact);
  });

  test('an archiver entry saved before data_type existed (tool_source archiver_read) routes to the timeseries viewport', () => {
    const artifact = makeArtifact({
      artifact_type: 'json',
      tool_source: 'archiver_read',
      data_file: 'y.parquet',
    });
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    const renderer = createPreviewRenderer(callbacks);
    renderer.renderPreview();

    expect(document.querySelector('.preview-viewport .ts-viewport-container')).not.toBeNull();
    expect(callbacks.onTimeseriesNeeded).toHaveBeenCalledTimes(1);
  });
});

describe('renderPreview: timeseries is told by data_type and the archiver, never the category', () => {
  /** @param {any} artifact */
  function render(artifact) {
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    createPreviewRenderer(callbacks).renderPreview();
    return callbacks;
  }

  test('an array channel read (data_type timeseries, no category) gets the timeseries viewport', () => {
    const callbacks = render(
      makeArtifact({
        artifact_type: 'json',
        category: '',
        tool_source: 'channel_read',
        metadata: { data_type: 'timeseries', channel: 'SR:BPM:X', data_file: 'w.parquet' },
      })
    );
    expect(document.querySelector('.preview-viewport .ts-viewport-container')).not.toBeNull();
    expect(callbacks.onTimeseriesNeeded).toHaveBeenCalledTimes(1);
  });

  test('a scalar channel snapshot does not, even when it carries the archiver_data category', () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, status: 200, text: async () => '', json: async () => ({}) }))
    );
    for (const category of ['channel_values', 'archiver_data']) {
      const callbacks = render(
        makeArtifact({
          artifact_type: 'json',
          filename: 'snapshot.json',
          category,
          tool_source: 'channel_read',
          data_file: 'snap.parquet',
          metadata: { channel: 'SR:DCCT' },
        })
      );
      expect(document.querySelector('.ts-viewport-container')).toBeNull();
      expect(callbacks.onTimeseriesNeeded).not.toHaveBeenCalled();
    }
  });
});

describe('renderPreview: the artifact alone (the hub has the bar and the details)', () => {
  test('renders the viewport, and no header, description, meta strip or path', () => {
    setSelectedArtifact(
      makeArtifact({ description: 'Horizontal beam size', tool_source: 'channel_finder' })
    );
    createPreviewRenderer(makeCallbacks()).renderPreview();

    const content = byId('preview-content');
    expect(Array.from(content.children, (el) => el.className)).toEqual(['preview-viewport']);
    expect(document.querySelector('.preview-header')).toBeNull();
    expect(document.querySelector('.preview-desc')).toBeNull();
    expect(document.querySelector('.preview-meta-strip')).toBeNull();
    expect(document.getElementById('preview-copy-path')).toBeNull();
    expect(content.textContent).not.toContain('Horizontal beam size');
    expect(document.querySelector('.preview-viewport img')).not.toBeNull();
  });

  test('mounts a timeseries viewport through onTimeseriesNeeded', () => {
    const artifact = makeArtifact({ tool_source: 'archiver_read', data_file: 'y.parquet' });
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    createPreviewRenderer(callbacks).renderPreview();

    expect(callbacks.onTimeseriesNeeded).toHaveBeenCalledTimes(1);
    expect(callbacks.onTimeseriesNeeded.mock.calls[0][0]).toBe(
      document.querySelector('.preview-viewport .ts-viewport-container')
    );
  });
});
