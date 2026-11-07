/**
 * Unit tests for the Artifact Gallery preview pane shell (preview.js, task
 * 5.4 extraction: renderPreview's per-type viewport dispatch), and the
 * `deleteArtifact` action the viewer runs.
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
  getArtifacts,
  setSelectedArtifact,
  getSelectedArtifact,
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

describe('renderPreview: meta content', () => {
  test('omits the Source meta item when tool_source is absent', () => {
    setSelectedArtifact(makeArtifact());
    createPreviewRenderer(makeCallbacks()).renderPreview();

    expect(
      Array.from(document.querySelectorAll('.preview-meta-label')).map((el) => el.textContent)
    ).not.toContain('Source');
  });
});

describe('renderPreview: the pane has no header (the hub renders it)', () => {
  test('renders the description, the meta strip and the viewport, and no header', () => {
    setSelectedArtifact(
      makeArtifact({ description: 'Horizontal beam size', tool_source: 'channel_finder' })
    );
    createPreviewRenderer(makeCallbacks()).renderPreview();

    expect(document.querySelector('.preview-header')).toBeNull();
    expect(qs(document, '.preview-desc').textContent).toBe('Horizontal beam size');
    expect(qs(document, '.preview-meta-strip').textContent).toContain('channel_finder');
    expect(document.querySelector('.preview-meta-strip .preview-path-text')).not.toBeNull();
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

describe('deleteArtifact', () => {
  test('confirms, DELETEs, updates local state, and re-renders to empty', async () => {
    const confirmStub = vi.fn(() => true);
    vi.stubGlobal('confirm', confirmStub);
    const artifact = makeArtifact();
    setArtifacts([artifact, makeArtifact({ id: 'a2' })]);
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    const renderer = createPreviewRenderer(callbacks);
    renderer.renderPreview();
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) });
    vi.stubGlobal('fetch', fetchMock);

    await renderer.deleteArtifact(artifact);

    expect(confirmStub).toHaveBeenCalledWith('Delete "Beam Profile"? This cannot be undone.');
    expect(fetchMock).toHaveBeenCalledWith('/api/artifacts/a1', { method: 'DELETE' });
    expect(getArtifacts().map((a) => a.id)).toEqual(['a2']);
    expect(getSelectedArtifact()).toBeNull();
    expect(byId('preview-empty').classList.contains('hidden')).toBe(false);
  });

  test('deleting an artifact other than the one shown keeps the selection', async () => {
    const doomed = makeArtifact();
    const shown = makeArtifact({ id: 'a2', title: 'Orbit', filename: 'orbit.png' });
    setArtifacts([doomed, shown]);
    setSelectedArtifact(shown);
    const renderer = createPreviewRenderer(makeCallbacks());
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) })
    );

    await renderer.deleteArtifact(doomed);

    expect(getArtifacts().map((a) => a.id)).toEqual(['a2']);
    expect(getSelectedArtifact()).toBe(shown);
    expect(qs(document, '.preview-path-text').textContent).toContain('orbit.png');
  });

  test('does nothing when the confirm dialog is declined', async () => {
    vi.stubGlobal(
      'confirm',
      vi.fn(() => false)
    );
    const artifact = makeArtifact();
    setArtifacts([artifact]);
    const callbacks = makeCallbacks();
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await createPreviewRenderer(callbacks).deleteArtifact(artifact);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(getArtifacts()).toEqual([artifact]);
  });

  test('a refused DELETE leaves the list, the selection and the focus as they were', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const artifact = makeArtifact();
    setArtifacts([artifact]);
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }));

    await createPreviewRenderer(callbacks).deleteArtifact(artifact);

    expect(getArtifacts()).toEqual([artifact]);
    expect(getSelectedArtifact()).toBe(artifact);
  });
});

describe('renderPreview: copy path', () => {
  test('copies the store\'s artifact path and toggles a "copied" class', async () => {
    const meta = document.createElement('meta');
    meta.setAttribute('name', 'osprey-artifact-dir');
    meta.setAttribute('content', 'state/agent/artifacts');
    document.head.appendChild(meta);
    vi.useFakeTimers();
    try {
      setSelectedArtifact(makeArtifact());
      createPreviewRenderer(makeCallbacks()).renderPreview();

      expect(qs(document, '.preview-path-text').textContent).toBe(
        'state/agent/artifacts/beam_profile.png'
      );

      const btn = byId('preview-copy-path');
      btn.click();
      await vi.waitFor(() =>
        expect(window.navigator.clipboard.writeText).toHaveBeenCalledWith(
          'state/agent/artifacts/beam_profile.png'
        )
      );

      expect(btn.classList.contains('copied')).toBe(true);
      vi.advanceTimersByTime(1500);
      expect(btn.classList.contains('copied')).toBe(false);
    } finally {
      vi.useRealTimers();
      meta.remove(); // this file's document is shared across tests
    }
  });
});
