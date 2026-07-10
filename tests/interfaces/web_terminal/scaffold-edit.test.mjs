// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Unit tests for the Scaffold Gallery edit-view modules
 * (scaffold/edit-form.js -- the edit-mode content renderers, and
 * scaffold/edit.js -- the write-side ownership/save/discard/close actions).
 *
 * Pure-logic + DOM guard, happy-dom environment (configured globally),
 * `fetch`/`confirm`/`alert` mocked via vi.stubGlobal -- mirrors the pattern
 * in scaffold-detail.test.mjs and scaffold-data.test.mjs:
 *   npx vitest run tests/interfaces/web_terminal/scaffold-edit.test.mjs
 *
 * A fake `gallery` host object stands in for the ArtifactGallery instance,
 * the same "pass `this`" shape the real class uses.
 *
 * Covers: front-matter form field generation per type (text/select/number),
 * the dirty-flag transitions on edit (typing in any field/textarea),
 * discard (clears dirty, forces preview mode), and save (clears dirty only
 * on a successful PUT); and the write-side actions -- take/release
 * ownership, the framework-edit-copy flow, the three saveOverride content
 * sources (plain textarea, front-matter form, settings.json structured
 * editor) plus its framework-settings ownership-warning gate, reset-to-
 * framework, reload+reopen, and closeDetail's unsaved-changes guard (the
 * same editDirty guard the drawer's unsaved-changes prompt reads).
 *
 * NOTE: imported by RELATIVE path -- these modules live under web_terminal,
 * not design-system, so the `/design-system/js/*` alias does not apply.
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import { createScaffoldGalleryEditForm } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit-form.js';
import { createScaffoldGalleryEdit } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit.js';
import { resetFetchCache } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/data.js';

/** @param {object} [overrides] */
function makeEditFormGallery(overrides = {}) {
  return {
    selectedArtifact: null,
    editDirty: false,
    detailContentEl: document.createElement('div'),
    renderDetailModes: vi.fn(),
    ...overrides,
  };
}

/** @param {object} [overrides] */
function makeEditGallery(overrides = {}) {
  return {
    selectedArtifact: null,
    artifacts: [],
    categoryFilter: () => true,
    categoryOverrides: {},
    categoryRemaps: {},
    summary: { total: 0, framework: 0, userOwned: 0 },
    currentView: 'detail',
    detailMode: 'edit',
    editDirty: false,
    detailContentEl: document.createElement('div'),
    errorEl: document.createElement('div'),
    galleryView: document.createElement('div'),
    detailView: document.createElement('div'),
    onDetailClose: null,
    openDetail: vi.fn(),
    renderDetailHeader: vi.fn(),
    renderDetailModes: vi.fn(),
    renderDetailContent: vi.fn(),
    renderGallery: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  resetFetchCache();
  vi.stubGlobal('confirm', vi.fn(() => true));
  vi.stubGlobal('alert', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// renderEdit -- dispatch + dirty-flag transitions
// ---------------------------------------------------------------------------

describe('renderEdit', () => {
  test('plain content (no model front matter) renders a textarea; typing marks dirty', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ content: 'plain body', language: 'text' }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);

    await form.renderEdit();

    const textarea = gallery.detailContentEl.querySelector('.prompts-edit-textarea');
    expect(textarea).toBeTruthy();
    expect(textarea.value).toBe('plain body');
    expect(gallery.editDirty).toBe(false);

    textarea.value = 'edited body';
    textarea.dispatchEvent(new Event('input'));

    expect(gallery.editDirty).toBe(true);
    expect(gallery.renderDetailModes).toHaveBeenCalled();
  });

  test('front matter with a model field renders the agent config form + instructions textarea', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        content: '---\nname: my-agent\nmodel: sonnet\n---\nDo the thing.',
        language: 'markdown',
      }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'my-agent', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);

    await form.renderEdit();

    expect(gallery.detailContentEl.textContent).toContain('AGENT CONFIGURATION');
    expect(gallery.detailContentEl.textContent).toContain('AGENT INSTRUCTIONS');
    const bodyTextarea = gallery.detailContentEl.querySelector('.prompts-edit-textarea');
    expect(bodyTextarea.value).toBe('Do the thing.');

    bodyTextarea.dispatchEvent(new Event('input'));
    expect(gallery.editDirty).toBe(true);
  });

  test('front matter without a model field falls back to the plain-text editor', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ content: '---\ndescription: no model here\n---\nBody text.', language: 'markdown' }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);

    await form.renderEdit();

    expect(gallery.detailContentEl.textContent).not.toContain('AGENT CONFIGURATION');
    const textarea = gallery.detailContentEl.querySelector('.prompts-edit-textarea');
    expect(textarea.value).toBe('---\ndescription: no model here\n---\nBody text.');
  });

  test('settings-json mounts the structured editor and clears any stale _settingsEditor first', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ content: '{"model":"anthropic/claude"}', language: 'json' }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'settings-json', status: 'user-owned' } });
    gallery.detailContentEl._settingsEditor = 'stale';
    const form = createScaffoldGalleryEditForm(gallery);

    await form.renderEdit();

    expect(gallery.detailContentEl._settingsEditor).toBeTruthy();
    expect(gallery.detailContentEl._settingsEditor).not.toBe('stale');
    expect(typeof gallery.detailContentEl._settingsEditor.getData).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Front-matter form field generation per type
// ---------------------------------------------------------------------------

describe('renderFrontMatterForm -- field generation per type', () => {
  test('text fields render as plain text inputs carrying the front-matter value', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        content: '---\nname: my-agent\ndescription: does things\nmodel: sonnet\n---\nBody.',
        language: 'markdown',
      }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'my-agent', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);
    await form.renderEdit();

    const fields = gallery.detailContentEl.querySelectorAll('.prompts-fm-field');
    const nameField = [...fields].find((f) => f.textContent.startsWith('name'));
    const nameInput = nameField.querySelector('input');
    expect(nameInput.type).toBe('text');
    expect(nameInput.value).toBe('my-agent');
  });

  test('the model field renders as a select populated with AGENT_MODEL_OPTIONS, current value selected', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ content: '---\nname: a\nmodel: sonnet\n---\nBody.', language: 'markdown' }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);
    await form.renderEdit();

    const fields = gallery.detailContentEl.querySelectorAll('.prompts-fm-field');
    const modelField = [...fields].find((f) => f.textContent.startsWith('model'));
    const select = modelField.querySelector('select');
    expect(select).toBeTruthy();
    expect(select.value).toBe('sonnet');
    expect(select.options.length).toBeGreaterThan(1);
  });

  test('maxTurns renders as a bounded number input', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        content: '---\nname: a\nmodel: sonnet\nmaxTurns: 5\n---\nBody.',
        language: 'markdown',
      }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);
    await form.renderEdit();

    const fields = gallery.detailContentEl.querySelectorAll('.prompts-fm-field');
    const maxTurnsField = [...fields].find((f) => f.textContent.startsWith('maxTurns'));
    const input = maxTurnsField.querySelector('input');
    expect(input.type).toBe('number');
    expect(input.min).toBe('1');
    expect(input.max).toBe('100');
    expect(input.value).toBe('5');
  });

  test('changing a select field (not just text inputs) marks the gallery dirty', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ content: '---\nname: a\nmodel: sonnet\n---\nBody.', language: 'markdown' }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);
    await form.renderEdit();

    const select = [...gallery.detailContentEl.querySelectorAll('.prompts-fm-field')]
      .find((f) => f.textContent.startsWith('model'))
      .querySelector('select');

    select.dispatchEvent(new Event('change'));
    expect(gallery.editDirty).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// discardEdits -- dirty-flag transition
// ---------------------------------------------------------------------------

describe('discardEdits', () => {
  test('clears the dirty flag, forces preview mode, and re-renders modes + content', () => {
    const gallery = makeEditGallery({ editDirty: true, detailMode: 'edit' });
    const edit = createScaffoldGalleryEdit(gallery);

    edit.discardEdits();

    expect(gallery.editDirty).toBe(false);
    expect(gallery.detailMode).toBe('preview');
    expect(gallery.renderDetailModes).toHaveBeenCalledOnce();
    expect(gallery.renderDetailContent).toHaveBeenCalledOnce();
  });

  test('is a no-op on the mode itself when already in preview', () => {
    const gallery = makeEditGallery({ editDirty: true, detailMode: 'preview' });
    const edit = createScaffoldGalleryEdit(gallery);

    edit.discardEdits();

    expect(gallery.detailMode).toBe('preview');
    expect(gallery.editDirty).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// saveOverride -- content sources + dirty-flag transition
// ---------------------------------------------------------------------------

describe('saveOverride', () => {
  test('reads from the plain textarea, PUTs it, and clears the dirty flag on success', async () => {
    const putCalls = [];
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      putCalls.push({ url, init });
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [{ name: 'a', status: 'user-owned' }] }) });
    }));

    const gallery = makeEditGallery({
      selectedArtifact: { name: 'a', status: 'user-owned' },
      editDirty: true,
    });
    const textarea = document.createElement('textarea');
    textarea.className = 'prompts-edit-textarea';
    textarea.value = 'new content';
    gallery.detailContentEl.appendChild(textarea);

    const edit = createScaffoldGalleryEdit(gallery);
    await edit.saveOverride();

    expect(putCalls.some((c) => c.url.includes('/override') && c.init.method === 'PUT'
      && JSON.parse(c.init.body).content === 'new content')).toBe(true);
    expect(gallery.editDirty).toBe(false);
    expect(gallery.openDetail).toHaveBeenCalledOnce();
  });

  test('reads from the front-matter form fields + body textarea, assembling YAML front matter', async () => {
    let savedBody = null;
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      if (init && init.method === 'PUT') savedBody = JSON.parse(init.body).content;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const nameInput = document.createElement('input');
    nameInput.value = 'my-agent';
    const bodyTextarea = document.createElement('textarea');
    bodyTextarea.value = 'Instructions here.';
    gallery.detailContentEl._frontMatterFields = { name: nameInput };
    gallery.detailContentEl._bodyTextarea = bodyTextarea;

    const edit = createScaffoldGalleryEdit(gallery);
    await edit.saveOverride();

    expect(savedBody).toContain('name: my-agent');
    expect(savedBody).toContain('Instructions here.');
  });

  test('reads from the settings.json structured editor via _settingsEditor.getData()', async () => {
    let savedBody = null;
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      if (init && init.method === 'PUT') savedBody = JSON.parse(init.body).content;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'settings-json', status: 'user-owned' } });
    gallery.detailContentEl._settingsEditor = { getData: () => '{"model":"x"}', isDirty: () => true };

    const edit = createScaffoldGalleryEdit(gallery);
    await edit.saveOverride();

    expect(savedBody).toBe('{"model":"x"}');
  });

  test('a framework-owned settings.json prompts an ownership warning before saving; cancelling aborts the save', async () => {
    const fetchSpy = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) }));
    vi.stubGlobal('fetch', fetchSpy);

    const gallery = makeEditGallery({ selectedArtifact: { name: 'settings-json', status: 'framework' } });
    gallery.detailContentEl._settingsEditor = { getData: () => '{}', isDirty: () => true };

    const edit = createScaffoldGalleryEdit(gallery);
    const savePromise = edit.saveOverride();

    // The ownership-warning dialog is a hand-rolled DOM overlay (not
    // `confirm()`) -- click Cancel.
    const cancelBtn = document.querySelector('.config-ownership-cancel');
    expect(cancelBtn).toBeTruthy();
    cancelBtn.dispatchEvent(new Event('click'));
    await savePromise;

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(document.querySelector('.config-ownership-overlay')).toBeNull();
  });

  test('confirming the ownership warning claims the file, then saves the override', async () => {
    const calls = [];
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      calls.push({ url, method: init?.method });
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'settings-json', status: 'framework' } });
    gallery.detailContentEl._settingsEditor = { getData: () => '{}', isDirty: () => true };

    const edit = createScaffoldGalleryEdit(gallery);
    const savePromise = edit.saveOverride();

    document.querySelector('.config-ownership-confirm').dispatchEvent(new Event('click'));
    await savePromise;

    expect(calls.some((c) => c.url.includes('/claim') && c.method === 'POST')).toBe(true);
    expect(calls.some((c) => c.url.includes('/override') && c.method === 'PUT')).toBe(true);
  });

  test('a failed save surfaces the error on gallery.errorEl and leaves editDirty untouched', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: false, status: 500, json: () => Promise.resolve({ detail: 'disk full' }),
    })));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'user-owned' }, editDirty: true });
    const textarea = document.createElement('textarea');
    textarea.className = 'prompts-edit-textarea';
    textarea.value = 'x';
    gallery.detailContentEl.appendChild(textarea);

    const edit = createScaffoldGalleryEdit(gallery);
    await edit.saveOverride();

    expect(gallery.editDirty).toBe(true);
    expect(gallery.errorEl.style.display).toBe('flex');
    expect(gallery.errorEl.textContent).toContain('disk full');
  });
});

// ---------------------------------------------------------------------------
// Ownership actions
// ---------------------------------------------------------------------------

describe('takeOwnership / releaseToFramework / handleEditFramework', () => {
  test('takeOwnership: cancelling the confirm dialog makes no network call', async () => {
    vi.stubGlobal('confirm', vi.fn(() => false));
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    const edit = createScaffoldGalleryEdit(gallery);
    await edit.takeOwnership();

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  test('takeOwnership: confirming claims the file then reloads + reopens the artifact', async () => {
    const calls = [];
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      calls.push({ url, method: init?.method });
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ name: 'a', category: 'agents', status: 'user-owned' }] }),
      });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    const edit = createScaffoldGalleryEdit(gallery);
    await edit.takeOwnership();

    expect(calls.some((c) => c.url.includes('/claim') && c.method === 'POST')).toBe(true);
    expect(gallery.openDetail).toHaveBeenCalledOnce();
  });

  test('releaseToFramework delegates to unoverrideArtifact (DELETE the override)', async () => {
    const calls = [];
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      calls.push({ url, method: init?.method });
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const edit = createScaffoldGalleryEdit(gallery);
    await edit.releaseToFramework();

    expect(calls.some((c) => c.url.includes('/override?delete_file=true') && c.method === 'DELETE')).toBe(true);
  });

  test('handleEditFramework claims the file, refetches artifacts, and switches to edit mode', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url.includes('/claim')) return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ name: 'a', category: 'agents', status: 'user-owned' }] }),
      });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    const edit = createScaffoldGalleryEdit(gallery);
    await edit.handleEditFramework();

    expect(gallery.detailMode).toBe('edit');
    expect(gallery.selectedArtifact.status).toBe('user-owned');
    expect(gallery.renderDetailHeader).toHaveBeenCalledOnce();
    expect(gallery.renderDetailModes).toHaveBeenCalledOnce();
    expect(gallery.renderDetailContent).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// closeDetail -- the editDirty guard the drawer's unsaved-guard mirrors
// ---------------------------------------------------------------------------

describe('closeDetail', () => {
  test('with no unsaved changes, closes immediately without prompting', () => {
    const confirmSpy = vi.fn(() => true);
    vi.stubGlobal('confirm', confirmSpy);

    const gallery = makeEditGallery({ editDirty: false, currentView: 'detail' });
    gallery.detailView.style.display = '';
    const edit = createScaffoldGalleryEdit(gallery);

    edit.closeDetail();

    expect(confirmSpy).not.toHaveBeenCalled();
    expect(gallery.currentView).toBe('gallery');
    expect(gallery.galleryView.style.display).toBe('');
    expect(gallery.detailView.style.display).toBe('none');
    expect(gallery.renderGallery).toHaveBeenCalledOnce();
  });

  test('with unsaved changes, cancelling the confirm dialog keeps the detail view open', () => {
    vi.stubGlobal('confirm', vi.fn(() => false));

    const gallery = makeEditGallery({ editDirty: true, currentView: 'detail' });
    const edit = createScaffoldGalleryEdit(gallery);

    edit.closeDetail();

    expect(gallery.currentView).toBe('detail');
    expect(gallery.editDirty).toBe(true);
    expect(gallery.renderGallery).not.toHaveBeenCalled();
  });

  test('with unsaved changes, confirming discards them and closes, firing onDetailClose', () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    const onDetailClose = vi.fn();

    const gallery = makeEditGallery({ editDirty: true, currentView: 'detail', onDetailClose });
    const edit = createScaffoldGalleryEdit(gallery);

    edit.closeDetail();

    expect(gallery.currentView).toBe('gallery');
    expect(gallery.editDirty).toBe(false);
    expect(gallery.selectedArtifact).toBeNull();
    expect(onDetailClose).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// reloadAndReopen
// ---------------------------------------------------------------------------

describe('reloadAndReopen', () => {
  test('reopens the same artifact by name after refetching, recomputing the summary', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        artifacts: [
          { name: 'a', category: 'agents', status: 'user-owned' },
          { name: 'b', category: 'rules', status: 'framework' },
        ],
      }),
    })));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    const edit = createScaffoldGalleryEdit(gallery);

    await edit.reloadAndReopen();

    expect(gallery.summary).toEqual({ total: 2, framework: 1, userOwned: 1 });
    expect(gallery.openDetail).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'a', status: 'user-owned' })
    );
    expect(gallery.renderGallery).not.toHaveBeenCalled();
  });

  test('falls back to the gallery grid when the artifact no longer exists (e.g. deleted)', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ artifacts: [] }),
    })));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'gone', status: 'user-owned' } });
    const edit = createScaffoldGalleryEdit(gallery);

    await edit.reloadAndReopen();

    expect(gallery.openDetail).not.toHaveBeenCalled();
    expect(gallery.renderGallery).toHaveBeenCalledOnce();
  });
});
