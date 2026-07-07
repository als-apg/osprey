/**
 * Unit tests for the Artifact Gallery preview pane shell (preview.js, task
 * 5.4 extraction: renderPreview's per-type viewport dispatch, pin toggling,
 * fullscreen enter/exit + its "N new" badge, and agent focus).
 *
 * The markdown+KaTeX render pipeline and the JSON viewer moved to the
 * sibling preview-content.js module (split purely to stay under the
 * 450-line cap) and are covered by preview-content.test.mjs instead — the
 * "artifact_type renders the right viewport container" dispatch case for
 * markdown/json stays here since that's preview.js's own renderPreview
 * behavior, not preview-content.js's.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal — mirrors state.test.mjs/render.test.mjs:
 *   npx vitest run tests/interfaces/artifacts/preview.test.mjs
 *
 * preview.js reads/writes state.js's module-singleton artifact/selection/
 * focus state (no vi.resetModules, just call the setters fresh per test —
 * same convention as render.test.mjs), and formats via types.js (real
 * implementations). logbook.js/print.js are mocked via vi.mock
 * (preview.js imports and calls injectLogbookButtons/injectPrintButton
 * directly — there is no `window.*` bridge) — the real modules
 * pull in their own DOM-modal/print-window machinery that's out of scope
 * for a preview-pane test; only the call itself matters here.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  setArtifacts,
  getArtifacts,
  setSelectedArtifact,
  getSelectedArtifact,
  setFocusedArtifact,
  getFocusedArtifact,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';

vi.mock('../../../src/osprey/interfaces/artifacts/static/js/logbook.js', () => ({
  injectLogbookButtons: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/artifacts/static/js/print.js', () => ({
  injectPrintButton: vi.fn(),
}));

import { createPreviewRenderer } from '../../../src/osprey/interfaces/artifacts/static/js/preview.js';
import { injectLogbookButtons } from '../../../src/osprey/interfaces/artifacts/static/js/logbook.js';
import { injectPrintButton } from '../../../src/osprey/interfaces/artifacts/static/js/print.js';

/** Minimal DOM fixture matching artifacts/static/index.html's structure. */
function mountFixture() {
  document.body.className = '';
  document.body.innerHTML = `
    <aside class="browse-sidebar" id="browse-sidebar">
      <div class="sidebar-body" id="sidebar-body"></div>
    </aside>
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
    pinned: false,
    timestamp: '2026-07-01T10:00:00Z',
    size_bytes: 2048,
    ...overrides,
  };
}

function makeCallbacks() {
  return {
    onArtifactDeleted: vi.fn(),
    onPinToggled: vi.fn(),
    onFullscreenExit: vi.fn(),
    onTimeseriesNeeded: vi.fn(),
  };
}

beforeEach(() => {
  mountFixture();
  setArtifacts([]);
  setSelectedArtifact(null);
  setFocusedArtifact(null);
  vi.stubGlobal('confirm', vi.fn(() => true));
  Object.defineProperty(window.navigator, 'clipboard', {
    value: { writeText: vi.fn(() => Promise.resolve()) },
    configurable: true,
  });
  injectLogbookButtons.mockClear();
  injectPrintButton.mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('renderPreview: empty state', () => {
  test('shows the empty placeholder and hides the content pane when nothing is selected', () => {
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    expect(document.getElementById('preview-empty').classList.contains('hidden')).toBe(false);
    expect(document.getElementById('preview-content').classList.contains('hidden')).toBe(true);
  });
});

describe('renderPreview: per-type viewport dispatch', () => {
  test.each([
    ['plot_html', 'iframe.preview-iframe-light'],
    ['table_html', 'iframe.preview-iframe-light'],
    ['dashboard_html', 'iframe.preview-iframe-light'],
    ['html', 'iframe.preview-iframe-light'],
    ['plot_png', 'img'],
    ['image', 'img'],
    ['markdown', '#md-viewport.md-preview-container'],
    ['json', '#json-viewport.json-viewer'],
    ['text', 'iframe.preview-iframe-dark'],
  ])('artifact_type "%s" renders %s', (artifact_type, selector) => {
    setSelectedArtifact(makeArtifact({ artifact_type }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    expect(document.getElementById('preview-empty').classList.contains('hidden')).toBe(true);
    expect(document.getElementById('preview-content').classList.contains('hidden')).toBe(false);
    expect(document.querySelector(`.preview-viewport ${selector}`)).not.toBeNull();
  });

  test('artifact_type "notebook" renders the rendered-notebook iframe endpoint', () => {
    setSelectedArtifact(makeArtifact({ id: 'nb1', artifact_type: 'notebook' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    const iframe = document.querySelector('.preview-viewport iframe.preview-iframe-light');
    expect(iframe.getAttribute('src')).toBe('/api/notebooks/nb1/rendered');
  });

  test('artifact_type "notebook" percent-encodes a hostile id in the iframe src', () => {
    setSelectedArtifact(makeArtifact({ id: 'a/../b?x="y"', artifact_type: 'notebook' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    const iframe = document.querySelector('.preview-viewport iframe.preview-iframe-light');
    const src = iframe.getAttribute('src');
    const idSegment = src.split('/')[3];
    expect(idSegment).not.toMatch(/[/?"]/);
    expect(src).toBe('/api/notebooks/a%2F..%2Fb%3Fx%3D%22y%22/rendered');
  });

  test('an unrecognized type with mime_type application/pdf renders a light iframe of the file', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'weird_type', mime_type: 'application/pdf' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    expect(document.querySelector('.preview-viewport iframe.preview-iframe-light')).not.toBeNull();
  });

  test('a fully unrecognized type falls back to a download link', () => {
    setSelectedArtifact(makeArtifact({ artifact_type: 'mystery', filename: 'blob.bin' }));
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.renderPreview();

    const link = document.querySelector('.preview-download a');
    expect(link).not.toBeNull();
    expect(link.textContent).toContain('blob.bin');
  });

  test('timeseries data (metadata.data_type) renders the ts-viewport container and fires onTimeseriesNeeded', () => {
    const artifact = makeArtifact({
      artifact_type: 'plot_html',
      metadata: { data_type: 'timeseries', data_file: 'x.parquet' },
    });
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    const renderer = createPreviewRenderer(callbacks);
    renderer.renderPreview();

    const tsEl = document.getElementById('ts-viewport');
    expect(tsEl).not.toBeNull();
    expect(callbacks.onTimeseriesNeeded).toHaveBeenCalledTimes(1);
    expect(callbacks.onTimeseriesNeeded.mock.calls[0][0]).toBe(tsEl);
    expect(callbacks.onTimeseriesNeeded.mock.calls[0][1]).toBe(artifact);
  });

  test('archiver_data category with a data_file also routes to the timeseries viewport', () => {
    const artifact = makeArtifact({ category: 'archiver_data', data_file: 'y.parquet' });
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    const renderer = createPreviewRenderer(callbacks);
    renderer.renderPreview();

    expect(document.getElementById('ts-viewport')).not.toBeNull();
    expect(callbacks.onTimeseriesNeeded).toHaveBeenCalledTimes(1);
  });
});

describe('renderPreview: header/meta content', () => {
  test('renders title, size, created time, and (when present) tool_source', () => {
    setSelectedArtifact(makeArtifact({ tool_source: 'channel_finder', size_bytes: 1024 }));
    createPreviewRenderer(makeCallbacks()).renderPreview();

    expect(document.querySelector('.preview-header-title').textContent).toBe('Beam Profile');
    expect(document.querySelector('.preview-meta-value').textContent).toBe('1.0 KB');
    expect(document.body.innerHTML).toContain('channel_finder');
  });

  test('omits the Source meta item when tool_source is absent', () => {
    setSelectedArtifact(makeArtifact());
    createPreviewRenderer(makeCallbacks()).renderPreview();

    expect(Array.from(document.querySelectorAll('.preview-meta-label')).map((el) => el.textContent)).not.toContain('Source');
  });

  test('pin button reflects the artifact\'s pinned state', () => {
    setSelectedArtifact(makeArtifact({ pinned: true }));
    createPreviewRenderer(makeCallbacks()).renderPreview();

    const pinBtn = document.getElementById('preview-toggle-pin');
    expect(pinBtn.classList.contains('btn-action-pinned')).toBe(true);
    expect(pinBtn.title).toBe('Unpin');
  });

  test('calls injectLogbookButtons/injectPrintButton directly (real imports, no window bridge)', () => {
    setSelectedArtifact(makeArtifact());

    createPreviewRenderer(makeCallbacks()).renderPreview();

    expect(injectLogbookButtons).toHaveBeenCalledTimes(1);
    expect(injectPrintButton).toHaveBeenCalledTimes(1);
  });
});

describe('renderPreview: delete flow', () => {
  test('confirms, DELETEs, updates local state, fires onArtifactDeleted, and re-renders to empty', async () => {
    const artifact = makeArtifact();
    setArtifacts([artifact, makeArtifact({ id: 'a2' })]);
    setSelectedArtifact(artifact);
    setFocusedArtifact(artifact);
    const callbacks = makeCallbacks();
    createPreviewRenderer(callbacks).renderPreview();

    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) });
    vi.stubGlobal('fetch', fetchMock);

    document.getElementById('preview-delete').click();
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    expect(fetchMock).toHaveBeenCalledWith('/api/artifacts/a1', { method: 'DELETE' });
    expect(getArtifacts().map((a) => a.id)).toEqual(['a2']);
    expect(getSelectedArtifact()).toBeNull();
    expect(getFocusedArtifact()).toBeNull();
    expect(callbacks.onArtifactDeleted).toHaveBeenCalledTimes(1);
    // renderPreview() ran again after clearing selection -> back to empty state.
    expect(document.getElementById('preview-empty').classList.contains('hidden')).toBe(false);
  });

  test('does nothing when the confirm dialog is declined', () => {
    vi.stubGlobal('confirm', vi.fn(() => false));
    setArtifacts([makeArtifact()]);
    setSelectedArtifact(makeArtifact());
    const callbacks = makeCallbacks();
    createPreviewRenderer(callbacks).renderPreview();

    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    document.getElementById('preview-delete').click();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(callbacks.onArtifactDeleted).not.toHaveBeenCalled();
  });
});

describe('renderPreview: copy path', () => {
  test('copies the _agent_data path and toggles a "copied" class', async () => {
    vi.useFakeTimers();
    setSelectedArtifact(makeArtifact());
    createPreviewRenderer(makeCallbacks()).renderPreview();

    const btn = document.getElementById('preview-copy-path');
    btn.click();
    await vi.waitFor(() => expect(window.navigator.clipboard.writeText).toHaveBeenCalledWith('_agent_data/artifacts/beam_profile.png'));

    expect(btn.classList.contains('copied')).toBe(true);
    vi.advanceTimersByTime(1500);
    expect(btn.classList.contains('copied')).toBe(false);
    vi.useRealTimers();
  });
});

describe('pin toggling', () => {
  test('on success: flips artifact.pinned, fires onPinToggled, and re-renders', async () => {
    const artifact = makeArtifact({ pinned: false });
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    createPreviewRenderer(callbacks).renderPreview();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    document.getElementById('preview-toggle-pin').click();
    await new Promise((r) => setTimeout(r, 0));

    expect(artifact.pinned).toBe(true);
    expect(callbacks.onPinToggled).toHaveBeenCalledTimes(1);
    // renderPreview() ran again -> pin button now reflects the flipped state.
    expect(document.getElementById('preview-toggle-pin').classList.contains('btn-action-pinned')).toBe(true);
  });

  test('on a non-OK response: leaves pinned unchanged and does not fire the callback', async () => {
    const artifact = makeArtifact({ pinned: false });
    setSelectedArtifact(artifact);
    const callbacks = makeCallbacks();
    createPreviewRenderer(callbacks).renderPreview();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }));
    document.getElementById('preview-toggle-pin').click();
    await new Promise((r) => setTimeout(r, 0));

    expect(artifact.pinned).toBe(false);
    expect(callbacks.onPinToggled).not.toHaveBeenCalled();
  });

  test('on a network failure: does not throw', async () => {
    setSelectedArtifact(makeArtifact());
    createPreviewRenderer(makeCallbacks()).renderPreview();
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));

    expect(() => document.getElementById('preview-toggle-pin').click()).not.toThrow();
    await new Promise((r) => setTimeout(r, 0));
  });
});

describe('fullscreen mode', () => {
  test('enterFullscreen sets isFullscreen, adds body classes, and renders the preview', () => {
    const artifact = makeArtifact();
    const renderer = createPreviewRenderer(makeCallbacks());

    renderer.enterFullscreen(artifact);

    expect(renderer.isFullscreen()).toBe(true);
    expect(getSelectedArtifact()).toBe(artifact);
    expect(document.body.classList.contains('fullscreen-mode')).toBe(true);
    expect(document.getElementById('preview-content').classList.contains('hidden')).toBe(false);
  });

  test('enterFullscreen with no artifact and none already selected is a no-op', () => {
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.enterFullscreen();
    expect(renderer.isFullscreen()).toBe(false);
  });

  test('enterFullscreen is idempotent while already fullscreen', () => {
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.enterFullscreen(makeArtifact());
    expect(() => renderer.enterFullscreen(makeArtifact({ id: 'a2' }))).not.toThrow();
    expect(renderer.isFullscreen()).toBe(true);
    // The second call was a no-op: selection stays on the first artifact.
    expect(getSelectedArtifact().id).toBe('a1');
  });

  test('exitFullscreen resets isFullscreen, removes body classes, and fires onFullscreenExit', () => {
    const callbacks = makeCallbacks();
    const renderer = createPreviewRenderer(callbacks);
    renderer.enterFullscreen(makeArtifact());

    renderer.exitFullscreen();

    expect(renderer.isFullscreen()).toBe(false);
    expect(document.body.classList.contains('fullscreen-mode')).toBe(false);
    expect(callbacks.onFullscreenExit).toHaveBeenCalledTimes(1);
  });

  test('exitFullscreen is a no-op when not fullscreen', () => {
    const callbacks = makeCallbacks();
    const renderer = createPreviewRenderer(callbacks);
    renderer.exitFullscreen();
    expect(callbacks.onFullscreenExit).not.toHaveBeenCalled();
  });

  test('noteNewArtifactArrival/updateNewArtifactBadge track and display the "N new" count', () => {
    const renderer = createPreviewRenderer(makeCallbacks());
    renderer.enterFullscreen(makeArtifact());

    renderer.noteNewArtifactArrival();
    renderer.noteNewArtifactArrival();
    renderer.updateNewArtifactBadge();

    const badge = document.getElementById('fullscreen-new-badge');
    expect(badge.textContent).toBe('2 new');
    expect(badge.dataset.count).toBe('2');
  });
});

describe('XSS hardening (Task 1.3 — escape-metadata-sinks)', () => {
  test('a hostile category is escaped in the badge class attribute — no element injection, no attribute breakout', () => {
    const hostile = '"><img src=x onerror=alert(1)>';
    setSelectedArtifact(makeArtifact({ category: hostile, artifact_type: hostile }));
    createPreviewRenderer(makeCallbacks()).renderPreview();

    expect(document.querySelector('.preview-header img')).toBeNull();
    expect(document.querySelector('.preview-header [onerror]')).toBeNull();
    expect(document.getElementById('preview-content').innerHTML).not.toMatch(/"><img/);

    const badge = document.querySelector('.badge');
    expect(badge).not.toBeNull();
  });

  test('benign category renders the badge class byte-identical (regression guard)', () => {
    setSelectedArtifact(makeArtifact({ category: 'visualization' }));
    createPreviewRenderer(makeCallbacks()).renderPreview();

    const badge = document.querySelector('.badge');
    expect(badge.className).toBe('badge badge-visualization');
  });
});

describe('setAsFocus', () => {
  test('on success: POSTs to /api/focus and sets focusedArtifact', async () => {
    const artifact = makeArtifact();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
    const renderer = createPreviewRenderer(makeCallbacks());

    await renderer.setAsFocus(artifact);

    expect(fetch).toHaveBeenCalledWith('/api/focus', expect.objectContaining({ method: 'POST' }));
    expect(getFocusedArtifact()).toBe(artifact);
  });

  test('on a non-OK response: leaves focusedArtifact untouched', async () => {
    setFocusedArtifact(null);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }));
    const renderer = createPreviewRenderer(makeCallbacks());

    await renderer.setAsFocus(makeArtifact());

    expect(getFocusedArtifact()).toBeNull();
  });

  test('on a network failure: does not throw', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));
    const renderer = createPreviewRenderer(makeCallbacks());
    await expect(renderer.setAsFocus(makeArtifact())).resolves.toBeUndefined();
  });
});
