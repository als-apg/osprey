// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Unit tests for the Scaffold Gallery detail-view modules
 * (scaffold/detail.js -- the shell: open/create/header/modes/dispatch --
 * and scaffold/detail-content.js -- the Preview/Diff content renderers).
 *
 * Pure-logic + DOM guard, happy-dom environment (configured globally),
 * `fetch`/`confirm`/`prompt`/`alert` mocked via vi.stubGlobal -- mirrors the
 * pattern in scaffold-data.test.mjs and scaffold-view.test.mjs:
 *   npx vitest run tests/interfaces/web_terminal/scaffold-detail.test.mjs
 *
 * A fake `gallery` host object stands in for the ArtifactGallery instance,
 * the same "pass `this`" shape the real class uses. Where a test needs
 * gallery.renderDetailModes() to actually re-render (the settings.json
 * dirty-callback path, and the mode-tab click handler), it's wired to the
 * factory's own renderDetailModes -- exactly how ArtifactGallery's
 * `renderDetailModes() { return this._detail.renderDetailModes(); }`
 * delegator wires it in scaffold-gallery.js.
 *
 * Covers: the mode-switch state machine (which tabs are offered by
 * ownership/custom status, the framework-edit redirect, and the
 * unsaved-changes confirm guard), renderDetailContent's mode dispatch, and
 * renderDiff's delegation to diff-utils (grouped blocks, word-level
 * highlighting, the no-diff empty state).
 *
 * NOTE: imported by RELATIVE path -- these modules live under web_terminal,
 * not design-system, so the `/design-system/js/*` alias does not apply.
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import { createScaffoldGalleryDetail } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/detail.js';
import { createScaffoldGalleryDetailContent } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/detail-content.js';

/** @param {object} [overrides] */
function makeGallery(overrides = {}) {
  return {
    selectedArtifact: null,
    currentView: 'gallery',
    detailMode: 'preview',
    editDirty: false,
    artifacts: [],
    galleryView: document.createElement('div'),
    detailView: document.createElement('div'),
    detailHeaderEl: document.createElement('div'),
    detailModesEl: document.createElement('div'),
    detailContentEl: document.createElement('div'),
    onDetailOpen: null,
    load: () => Promise.resolve(),
    closeDetail: vi.fn(),
    releaseToFramework: vi.fn(),
    takeOwnership: vi.fn(),
    handleEditFramework: vi.fn(),
    discardEdits: vi.fn(),
    saveOverride: vi.fn(),
    renderEdit: vi.fn(() => Promise.resolve()),
    ...overrides,
  };
}

beforeEach(() => {
  vi.stubGlobal('confirm', vi.fn(() => true));
  vi.stubGlobal('prompt', vi.fn(() => null));
  vi.stubGlobal('alert', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Mode availability by ownership
// ---------------------------------------------------------------------------

describe('renderDetailModes -- tab availability by ownership', () => {
  function modeLabels(gallery, detail) {
    detail.renderDetailModes();
    return [...gallery.detailModesEl.querySelectorAll('.prompts-mode-btn')].map((b) => b.textContent);
  }

  test('a framework artifact offers Preview and Edit, but no Diff', () => {
    const gallery = makeGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    const detail = createScaffoldGalleryDetail(gallery);
    expect(modeLabels(gallery, detail)).toEqual(['Preview', 'Edit']);
  });

  test('a user-owned, non-custom artifact offers Preview, Diff, and Edit', () => {
    const gallery = makeGallery({ selectedArtifact: { name: 'a', status: 'user-owned', custom: false } });
    const detail = createScaffoldGalleryDetail(gallery);
    expect(modeLabels(gallery, detail)).toEqual(['Preview', 'Diff', 'Edit']);
  });

  test('a user-owned, custom artifact has no framework default to diff against -- no Diff tab', () => {
    const gallery = makeGallery({ selectedArtifact: { name: 'a', status: 'user-owned', custom: true } });
    const detail = createScaffoldGalleryDetail(gallery);
    expect(modeLabels(gallery, detail)).toEqual(['Preview', 'Edit']);
  });

  test('Discard/Save action buttons only appear in edit mode', () => {
    const gallery = makeGallery({
      selectedArtifact: { name: 'a', status: 'user-owned', custom: false },
      detailMode: 'edit',
      editDirty: true,
    });
    const detail = createScaffoldGalleryDetail(gallery);
    detail.renderDetailModes();

    const discardBtn = gallery.detailModesEl.querySelector('.prompts-discard-btn');
    const saveBtn = gallery.detailModesEl.querySelector('.prompts-save-btn');
    expect(discardBtn).toBeTruthy();
    expect(saveBtn).toBeTruthy();
    expect(discardBtn.disabled).toBe(false);
    expect(saveBtn.disabled).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Mode-switch state machine
// ---------------------------------------------------------------------------

describe('mode-switch state machine', () => {
  test('clicking Edit on a framework artifact redirects to handleEditFramework instead of switching', () => {
    const gallery = makeGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    const detail = createScaffoldGalleryDetail(gallery);
    detail.renderDetailModes();

    const editBtn = [...gallery.detailModesEl.querySelectorAll('.prompts-mode-btn')]
      .find((b) => b.textContent === 'Edit');
    editBtn.dispatchEvent(new Event('click'));

    expect(gallery.handleEditFramework).toHaveBeenCalledOnce();
    expect(gallery.detailMode).toBe('preview');
  });

  test('clicking the already-active mode is a no-op', () => {
    const gallery = makeGallery({
      selectedArtifact: { name: 'a', status: 'user-owned', custom: false },
      detailMode: 'preview',
    });
    const detail = createScaffoldGalleryDetail(gallery);
    detail.renderDetailModes();

    const previewBtn = [...gallery.detailModesEl.querySelectorAll('.prompts-mode-btn')]
      .find((b) => b.textContent === 'Preview');
    previewBtn.dispatchEvent(new Event('click'));

    expect(confirm).not.toHaveBeenCalled();
    expect(gallery.detailMode).toBe('preview');
  });

  test('switching modes with unsaved edits prompts confirm; confirming clears editDirty and switches', () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ has_diff: false }),
    })));

    const gallery = makeGallery({
      selectedArtifact: { name: 'a', status: 'user-owned', custom: false },
      detailMode: 'preview',
      editDirty: true,
    });
    const detail = createScaffoldGalleryDetail(gallery);
    gallery.renderDetailModes = detail.renderDetailModes;
    detail.renderDetailModes();

    const diffBtn = [...gallery.detailModesEl.querySelectorAll('.prompts-mode-btn')]
      .find((b) => b.textContent === 'Diff');
    diffBtn.dispatchEvent(new Event('click'));

    expect(confirm).toHaveBeenCalledOnce();
    expect(gallery.editDirty).toBe(false);
    expect(gallery.detailMode).toBe('diff');
  });

  test('switching modes with unsaved edits and cancelling confirm leaves mode and dirty flag unchanged', () => {
    vi.stubGlobal('confirm', vi.fn(() => false));

    const gallery = makeGallery({
      selectedArtifact: { name: 'a', status: 'user-owned', custom: false },
      detailMode: 'preview',
      editDirty: true,
    });
    const detail = createScaffoldGalleryDetail(gallery);
    detail.renderDetailModes();

    const diffBtn = [...gallery.detailModesEl.querySelectorAll('.prompts-mode-btn')]
      .find((b) => b.textContent === 'Diff');
    diffBtn.dispatchEvent(new Event('click'));

    expect(confirm).toHaveBeenCalledOnce();
    expect(gallery.editDirty).toBe(true);
    expect(gallery.detailMode).toBe('preview');
  });
});

// ---------------------------------------------------------------------------
// openDetail
// ---------------------------------------------------------------------------

describe('openDetail', () => {
  test('resets detail state, swaps view visibility, and fires onDetailOpen', () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ content: 'hello', language: 'text' }),
    })));

    const onDetailOpen = vi.fn();
    const gallery = makeGallery({ detailMode: 'edit', editDirty: true, onDetailOpen });
    const detail = createScaffoldGalleryDetail(gallery);

    const artifact = { name: 'my-artifact', status: 'framework' };
    detail.openDetail(artifact);

    expect(gallery.selectedArtifact).toBe(artifact);
    expect(gallery.currentView).toBe('detail');
    expect(gallery.detailMode).toBe('preview');
    expect(gallery.editDirty).toBe(false);
    expect(gallery.galleryView.style.display).toBe('none');
    expect(gallery.detailView.style.display).toBe('');
    expect(onDetailOpen).toHaveBeenCalledOnce();
    expect(gallery.detailHeaderEl.querySelector('.prompts-detail-name').textContent).toBe('my-artifact');
  });
});

// ---------------------------------------------------------------------------
// renderDetailContent -- mode dispatch
// ---------------------------------------------------------------------------

describe('renderDetailContent -- mode dispatch', () => {
  test('edit mode delegates to gallery.renderEdit()', async () => {
    const gallery = makeGallery({
      selectedArtifact: { name: 'a', status: 'user-owned' },
      detailMode: 'edit',
    });
    const detail = createScaffoldGalleryDetail(gallery);

    await detail.renderDetailContent();

    expect(gallery.renderEdit).toHaveBeenCalledOnce();
  });

  test('preview mode fetches and renders artifact content', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ content: 'plain body', language: 'text' }),
    })));

    const gallery = makeGallery({
      selectedArtifact: { name: 'a', status: 'framework' },
      detailMode: 'preview',
    });
    const detail = createScaffoldGalleryDetail(gallery);

    await detail.renderDetailContent();

    expect(gallery.detailContentEl.querySelector('.prompts-preview-content').textContent).toBe('plain body');
  });

  test('a fetch failure renders an escaped error message instead of throwing', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('<script>boom</script>'))));

    const gallery = makeGallery({
      selectedArtifact: { name: 'a', status: 'framework' },
      detailMode: 'preview',
    });
    const detail = createScaffoldGalleryDetail(gallery);

    await detail.renderDetailContent();

    const errorEl = gallery.detailContentEl.querySelector('.prompts-content-error');
    expect(errorEl).toBeTruthy();
    expect(errorEl.querySelector('script')).toBeNull();
    expect(errorEl.textContent).toContain('<script>boom</script>');
  });
});

// ---------------------------------------------------------------------------
// renderDiff -- delegates to diff-utils with grouped blocks
// ---------------------------------------------------------------------------

describe('renderDiff', () => {
  test('renders stats, and groups hunk/context/change blocks with word-level highlighting', async () => {
    const unifiedDiff = [
      '@@ -1,3 +1,3 @@',
      ' context line',
      '-old line here',
      '+new line here',
    ].join('\n');

    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ has_diff: true, additions: 1, deletions: 1, unified_diff: unifiedDiff }),
    })));

    const gallery = makeGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const content = createScaffoldGalleryDetailContent(gallery);

    await content.renderDiff();

    expect(gallery.detailContentEl.querySelector('.prompts-diff-add').textContent).toBe('+1');
    expect(gallery.detailContentEl.querySelector('.prompts-diff-del').textContent).toBe('−1');

    const lines = [...gallery.detailContentEl.querySelectorAll('.prompts-diff-line')];
    expect(lines.map((l) => l.className)).toEqual([
      'prompts-diff-line hunk',
      'prompts-diff-line context',
      'prompts-diff-line del',
      'prompts-diff-line add',
    ]);

    const [, , delLine, addLine] = lines;
    // The common "line" token is kept on both sides; "old"/"new" differ --
    // proof that renderDiff delegates word-level highlighting to diff-utils
    // rather than doing a plain line-level render.
    expect(delLine.querySelector('.diff-word-del').textContent).toBe('old');
    expect(delLine.querySelector('.diff-word-keep')).toBeTruthy();
    expect(addLine.querySelector('.diff-word-add').textContent).toBe('new');
    expect(addLine.querySelector('.diff-word-keep')).toBeTruthy();
  });

  test('renders the empty state when the artifact has no diff from its framework default', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ has_diff: false }),
    })));

    const gallery = makeGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const content = createScaffoldGalleryDetailContent(gallery);

    await content.renderDiff();

    expect(gallery.detailContentEl.textContent).toContain('No differences from framework default.');
  });
});

// ---------------------------------------------------------------------------
// renderPreview
// ---------------------------------------------------------------------------

describe('renderPreview', () => {
  test('markdown content renders a front-matter table plus the body', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        content: '---\nname: my-agent\n---\nSome instructions.',
        language: 'markdown',
      }),
    })));

    const gallery = makeGallery({ selectedArtifact: { name: 'my-agent', status: 'framework' } });
    const content = createScaffoldGalleryDetailContent(gallery);

    await content.renderPreview();

    expect(gallery.detailContentEl.textContent).toContain('my-agent');
    expect(gallery.detailContentEl.textContent).toContain('Some instructions.');
  });

  test('settings-json mounts the structured editor and wires dirty state back through the gallery', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ content: '{"model":"anthropic/claude"}', language: 'json' }),
    })));

    const gallery = makeGallery({ selectedArtifact: { name: 'settings-json', status: 'user-owned' } });
    gallery.renderDetailModes = vi.fn();
    const content = createScaffoldGalleryDetailContent(gallery);

    await content.renderPreview();

    expect(gallery.detailContentEl._settingsEditor).toBeTruthy();
    expect(typeof gallery.detailContentEl._settingsEditor.getData).toBe('function');

    const modelInput = gallery.detailContentEl.querySelector('.config-edit-input');
    modelInput.value = 'anthropic/claude-opus';
    modelInput.dispatchEvent(new Event('input'));

    expect(gallery.editDirty).toBe(true);
    expect(gallery.renderDetailModes).toHaveBeenCalled();
  });
});
