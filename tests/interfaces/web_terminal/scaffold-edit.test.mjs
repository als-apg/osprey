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
 * Covers: front-matter form field generation per type, the dirty-flag
 * transitions on edit (typing in any field/textarea), discard (clears dirty,
 * forces preview mode), and save (clears dirty only on a successful PUT,
 * reads the plain textarea or the front-matter form, and raises the
 * applies-on-restart notice); the write-side actions -- take/release
 * ownership with the reload and reopen that follows, the framework-edit
 * flow, and closeDetail's unsaved-changes guard (the same editDirty guard
 * the drawer's unsaved-changes prompt reads); and that every scaffold write
 * reaches the per-user-prefixed URL.
 *
 * NOTE: imported by RELATIVE path -- these modules live under web_terminal,
 * not design-system, so the `/design-system/js/*` alias does not apply.
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { createScaffoldGalleryEditForm } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit-form.js';
import { createScaffoldGalleryEdit } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit.js';
import { createScaffoldGalleryDetail } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/detail.js';
import { resetFetchCache, createScaffoldDataActions } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/data.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit-form.js').ScaffoldGalleryEditFormHost} ScaffoldGalleryEditFormHost
 * @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit-form.js').EditContentElement} EditFormContentElement
 * @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit.js').ScaffoldGalleryEditHost} ScaffoldGalleryEditHost
 * @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit.js').EditContentElement} EditContentElement
 */

/**
 * Test fixture variant of {@link ScaffoldGalleryEditFormHost}: makeEditFormGallery
 * always assigns a real element, never leaves detailContentEl null.
 * @typedef {ScaffoldGalleryEditFormHost & { detailContentEl: EditFormContentElement }} TestEditFormGallery
 */

/** @param {Partial<ScaffoldGalleryEditFormHost>} [overrides]
 * @returns {TestEditFormGallery}
 */
function makeEditFormGallery(overrides = {}) {
  return /** @type {TestEditFormGallery} */ ({
    selectedArtifact: null,
    editDirty: false,
    detailContentEl: document.createElement('div'),
    renderDetailModes: vi.fn(),
    ...overrides,
  });
}

/**
 * Test fixture variant of {@link ScaffoldGalleryEditHost}: makeEditGallery
 * always assigns real elements, never leaves the DOM refs null.
 * @typedef {ScaffoldGalleryEditHost & {
 *   detailContentEl: EditContentElement,
 *   errorEl: HTMLElement,
 *   galleryView: HTMLElement,
 *   detailView: HTMLElement,
 * }} TestEditGallery
 */

/** @param {Partial<ScaffoldGalleryEditHost>} [overrides]
 * @returns {TestEditGallery}
 */
function makeEditGallery(overrides = {}) {
  return /** @type {TestEditGallery} */ ({
    selectedArtifact: null,
    artifacts: [],
    reloadFull: vi.fn(async () => {}),
    currentView: 'detail',
    detailMode: 'edit',
    editDirty: false,
    detailContentEl: document.createElement('div'),
    errorEl: document.createElement('div'),
    galleryView: document.createElement('div'),
    detailView: document.createElement('div'),
    onDetailClose: null,
    openDetail: vi.fn(),
    renderDetailModes: vi.fn(),
    renderDetailContent: vi.fn(),
    renderGallery: vi.fn(),
    ...overrides,
  });
}

beforeEach(() => {
  resetFetchCache();
  vi.stubGlobal('confirm', vi.fn(() => true));
  vi.stubGlobal('alert', vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.__OSPREY_PREFIX__;
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

    const textarea = qs(gallery.detailContentEl, '.prompts-edit-textarea', HTMLTextAreaElement);
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
    const bodyTextarea = qs(gallery.detailContentEl, '.prompts-edit-textarea', HTMLTextAreaElement);
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
    const textarea = qs(gallery.detailContentEl, '.prompts-edit-textarea', HTMLTextAreaElement);
    expect(textarea.value).toBe('---\ndescription: no model here\n---\nBody text.');
  });
});

// ---------------------------------------------------------------------------
// Front-matter form field generation per type
// ---------------------------------------------------------------------------

describe('renderFrontMatterForm -- field generation per type', () => {
  test.each([
    ['name', 'a plain text input', { type: 'text', value: 'my-agent' }],
    ['model', 'a free text id with no fixed choices', { type: 'text', value: 'claude-sonnet-5' }],
    ['maxTurns', 'a bounded number input', { type: 'number', value: '5', min: '1', max: '100' }],
  ])('%s renders as %s carrying the front-matter value', async (key, _what, expected) => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({
        content: '---\nname: my-agent\nmodel: claude-sonnet-5\nmaxTurns: 5\n---\nBody.',
        language: 'markdown',
      }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'my-agent', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);
    await form.renderEdit();

    const field = [...gallery.detailContentEl.querySelectorAll('.prompts-fm-field')]
      .find((f) => qs(f, '.prompts-fm-field-label').textContent === key);
    if (field === undefined) throw new Error(`${key} field not found`);
    const input = qs(field, 'input', HTMLInputElement);
    expect(input).toMatchObject(expected);
    // Nothing served: no suggestion list, and never a closed select.
    expect(field.querySelector('select')).toBeNull();
    expect(field.querySelector('datalist')).toBeNull();
  });

  test('the served models are offered as suggestions with their display names', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ content: '---\nname: a\nmodel: claude-sonnet-5\n---\nBody.', language: 'markdown' }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    gallery.servedModels = [
      { id: 'claude-sonnet-5', name: 'Sonnet 5' },
      { id: 'claude-haiku-4-5', name: 'Haiku 4.5' },
    ];
    const form = createScaffoldGalleryEditForm(gallery);
    await form.renderEdit();

    const fields = gallery.detailContentEl.querySelectorAll('.prompts-fm-field');
    const modelField = [...fields].find((f) => (f.textContent ?? '').startsWith('model'));
    if (modelField === undefined) throw new Error('model field not found');
    const input = qs(modelField, 'input', HTMLInputElement);
    const list = modelField.querySelector('datalist');
    if (list === null) throw new Error('datalist not found');
    expect(input.getAttribute('list')).toBe(list.id);
    const options = [...list.querySelectorAll('option')];
    expect(options.map((o) => o.value)).toEqual(['claude-sonnet-5', 'claude-haiku-4-5']);
    expect(options.map((o) => o.label)).toEqual(['Sonnet 5', 'Haiku 4.5']);
  });

  test('choosing a suggested model (a change, not typed input) marks the gallery dirty', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ content: '---\nname: a\nmodel: claude-sonnet-5\n---\nBody.', language: 'markdown' }),
    })));

    const gallery = makeEditFormGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const form = createScaffoldGalleryEditForm(gallery);
    await form.renderEdit();

    const modelField2 = [...gallery.detailContentEl.querySelectorAll('.prompts-fm-field')]
      .find((f) => (f.textContent ?? '').startsWith('model'));
    if (modelField2 === undefined) throw new Error('model field not found');
    const input = qs(modelField2, 'input');

    input.dispatchEvent(new Event('change'));
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

});

// ---------------------------------------------------------------------------
// saveOverride -- content sources + dirty-flag transition
// ---------------------------------------------------------------------------

describe('saveOverride', () => {
  test('reads from the plain textarea, PUTs it, and clears the dirty flag on success', async () => {
    /** @type {{url: string, init: RequestInit}[]} */
    const putCalls = [];
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      putCalls.push({ url, init });
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [{ name: 'a', status: 'user-owned' }] }) });
    }));

    const gallery = makeEditGallery({
      selectedArtifact: { name: 'a', status: 'user-owned' },
      editDirty: true,
    });
    gallery.reloadFull = vi.fn(async () => {
      gallery.artifacts = [{ name: 'a', status: 'user-owned' }];
    });
    const textarea = document.createElement('textarea');
    textarea.className = 'prompts-edit-textarea';
    textarea.value = 'new content';
    gallery.detailContentEl.appendChild(textarea);

    const edit = createScaffoldGalleryEdit(gallery);
    await edit.saveOverride();

    expect(putCalls.some((c) => c.url === '/api/scaffold/a/override' && c.init.method === 'PUT'
      && JSON.parse(/** @type {string} */ (c.init.body)).content === 'new content')).toBe(true);
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


  describe('applies-on-restart notice', () => {
    // A save in a deployed container can land on the claude-config volume while
    // the read-only image tree refuses it; the server says so with
    // `applies_on_restart: true`, and the operator must be told the change is
    // not live yet.

    /** @param {object} body the PUT response */
    function stubSave(body) {
      vi.stubGlobal('fetch', vi.fn((/** @type {string} */ _url, /** @type {RequestInit|undefined} */ init) => {
        const isPut = init && init.method === 'PUT';
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(isPut ? body : { artifacts: [], untracked: [] }),
        });
      }));
    }

    function makeSavingGallery() {
      const gallery = makeEditGallery({
        selectedArtifact: { name: 'agents/channel-finder', status: 'user-owned' },
        editDirty: true,
      });
      const textarea = document.createElement('textarea');
      textarea.className = 'prompts-edit-textarea';
      textarea.value = 'edited body';
      gallery.detailContentEl.appendChild(textarea);
      return gallery;
    }

    test('renders the notice when the server says the save only lands on restart', async () => {
      stubSave({ status: 'saved', path: '.claude/agents/channel-finder.md', applies_on_restart: true });
      const gallery = makeSavingGallery();

      await createScaffoldGalleryEdit(gallery).saveOverride();

      expect(gallery.editDirty).toBe(false);
      expect(gallery.reloadFull).toHaveBeenCalled();
      expect(gallery.errorEl.style.display).toBe('flex');
      expect(gallery.errorEl.textContent).toContain('applies on container restart');
      expect(gallery.errorEl.classList.contains('prompts-error--notice')).toBe(true);
    });

    test.each([
      ['false', { status: 'saved', path: '.claude/agents/channel-finder.md', applies_on_restart: false }],
      ['absent', { status: 'saved', path: '.claude/agents/channel-finder.md' }],
    ])('says nothing when the flag is %s', async (_label, body) => {
      stubSave(body);
      const gallery = makeSavingGallery();

      await createScaffoldGalleryEdit(gallery).saveOverride();

      expect(gallery.errorEl.textContent).toBe('');
      expect(gallery.errorEl.classList.contains('prompts-error--notice')).toBe(false);
    });

    test('a failed save reads as a failure, clearing a notice left on the strip', async () => {
      vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
        ok: false,
        status: 403,
        json: () => Promise.resolve({ detail: "'rules/facility' belongs to the profile's `rules/` convention directory. NOTHING WAS WRITTEN." }),
      })));
      const gallery = makeSavingGallery();
      gallery.errorEl.classList.add('prompts-error--notice');

      await createScaffoldGalleryEdit(gallery).saveOverride();

      expect(gallery.errorEl.textContent.startsWith('Save failed: ')).toBe(true);
      expect(gallery.errorEl.classList.contains('prompts-error--notice')).toBe(false);
    });
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
    /** @type {{url: string, method: string|undefined}[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      calls.push({ url, method: init?.method });
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ name: 'a', category: 'agents', status: 'user-owned' }] }),
      });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    gallery.reloadFull = vi.fn(async () => {
      gallery.artifacts = [{ name: 'a', category: 'agents', status: 'user-owned' }];
    });
    const edit = createScaffoldGalleryEdit(gallery);
    await edit.takeOwnership();

    expect(calls.some((c) => c.url === '/api/scaffold/a/claim' && c.method === 'POST')).toBe(true);
    // The artifact reopens as the reload returned it, not as it was before the claim.
    expect(gallery.openDetail).toHaveBeenCalledOnce();
    expect(gallery.openDetail).toHaveBeenCalledWith(expect.objectContaining({ name: 'a', status: 'user-owned' }));
    expect(gallery.renderGallery).not.toHaveBeenCalled();
  });

  test('releaseToFramework DELETEs the override and falls back to the grid when the artifact is gone', async () => {
    /** @type {{url: string, method: string|undefined}[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      calls.push({ url, method: init?.method });
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    }));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
    const edit = createScaffoldGalleryEdit(gallery);
    await edit.releaseToFramework();

    expect(calls.some((c) => c.url === '/api/scaffold/a/override?delete_file=true' && c.method === 'DELETE')).toBe(true);
    // The reload no longer lists `a` (a released custom file is gone), so the
    // panel returns to the grid rather than reopening a stale detail view.
    expect(gallery.reloadFull).toHaveBeenCalledOnce();
    expect(gallery.openDetail).not.toHaveBeenCalled();
    expect(gallery.renderGallery).toHaveBeenCalledOnce();
  });

  test('handleEditFramework claims the file, reloads via the data pipeline, and reopens in edit mode', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
    ));

    const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
    // Mimic the real reloadFull's onLoaded side effect: the gallery view is
    // re-rendered (which flips visibility back to the grid) — so the action
    // must reopen the detail view itself via openDetail.
    gallery.reloadFull = vi.fn(async () => {
      gallery.artifacts = [{ name: 'a', category: 'agents', status: 'user-owned' }];
      gallery.renderGallery();
    });
    const edit = createScaffoldGalleryEdit(gallery);
    await edit.handleEditFramework();

    expect(gallery.reloadFull).toHaveBeenCalledOnce();
    // Edit mode is an argument to the open, not a flip afterwards: the flip
    // left a Preview render already in flight against the same pane.
    expect(gallery.openDetail).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'a', status: 'user-owned' }),
      'edit'
    );
    expect(gallery.renderDetailModes).not.toHaveBeenCalled();
    expect(gallery.renderDetailContent).not.toHaveBeenCalled();
  });

  test('handleEditFramework fetches the claimed file once, not once per render', async () => {
    // The real detail shell and edit-form renderer, wired the way
    // ArtifactGallery wires them: a stubbed openDetail issues no fetch, so it
    // cannot tell one render from the two this flow used to start.
    /** @type {string[]} */
    const calls = [];
    vi.stubGlobal('fetch', vi.fn((/** @type {string} */ url, /** @type {RequestInit} */ init) => {
      calls.push(`${(init && init.method) || 'GET'} ${url}`);
      return Promise.resolve({
        ok: true, json: () => Promise.resolve({ content: 'framework body', language: 'text' }),
      });
    }));

    const gallery = /** @type {any} */ ({
      selectedArtifact: { name: 'a', status: 'framework' },
      artifacts: [],
      currentView: 'detail',
      detailMode: 'preview',
      detailRenderSeq: 0,
      editDirty: false,
      galleryView: document.createElement('div'),
      detailView: document.createElement('div'),
      detailHeaderEl: document.createElement('div'),
      detailModesEl: document.createElement('div'),
      detailContentEl: document.createElement('div'),
      errorEl: document.createElement('div'),
      onDetailOpen: null,
      onDetailClose: null,
      load: () => Promise.resolve(),
      renderGallery: vi.fn(),
      handleEditFramework: vi.fn(),
      discardEdits: vi.fn(),
      saveOverride: vi.fn(),
      takeOwnership: vi.fn(),
      releaseToFramework: vi.fn(),
      closeDetail: vi.fn(),
    });
    gallery.reloadFull = vi.fn(async () => {
      gallery.artifacts = [{ name: 'a', category: 'agents', status: 'user-owned' }];
      gallery.renderGallery();
    });
    gallery.renderEdit = createScaffoldGalleryEditForm(gallery).renderEdit;

    const detail = createScaffoldGalleryDetail(gallery);
    gallery.openDetail = detail.openDetail;
    gallery.renderDetailHeader = detail.renderDetailHeader;
    gallery.renderDetailModes = detail.renderDetailModes;
    gallery.renderDetailContent = detail.renderDetailContent;

    const edit = createScaffoldGalleryEdit(gallery);
    await edit.handleEditFramework();
    // openDetail's render is started, not awaited.
    for (let i = 0; i < 3; i++) await new Promise((resolve) => setTimeout(resolve, 0));

    expect(gallery.detailMode).toBe('edit');
    expect(calls).toContain('POST /api/scaffold/a/claim');
    expect(calls.filter((c) => c === 'GET /api/scaffold/a')).toHaveLength(1);
    expect(qs(gallery.detailContentEl, '.prompts-edit-textarea')).toBeTruthy();
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
// Multi-user prefix
// ---------------------------------------------------------------------------

describe('every scaffold write reaches the per-user-prefixed URL', () => {
  // apiRequest owns the prefix (api.test.mjs pins the chokepoint). These rows
  // pin that no scaffold write goes around it: a direct fetch would write to
  // the unprefixed app, which in a multi-user deployment is not this user's.
  /**
   * Stand up a gallery with a plain textarea to save from and an artifact
   * named `a`, and drive one write action through its public entry point.
   * @type {Array<[string, string, string, () => Promise<unknown>]>}
   */
  const WRITES = [
    ['save', 'PUT', '/u/alice/api/scaffold/a/override', async () => {
      const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
      const textarea = document.createElement('textarea');
      textarea.className = 'prompts-edit-textarea';
      textarea.value = 'new content';
      gallery.detailContentEl.appendChild(textarea);
      await createScaffoldGalleryEdit(gallery).saveOverride();
    }],
    ['claim', 'POST', '/u/alice/api/scaffold/a/claim', async () => {
      const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'framework' } });
      await createScaffoldGalleryEdit(gallery).takeOwnership();
    }],
    ['release', 'DELETE', '/u/alice/api/scaffold/a/override?delete_file=true', async () => {
      const gallery = makeEditGallery({ selectedArtifact: { name: 'a', status: 'user-owned' } });
      await createScaffoldGalleryEdit(gallery).releaseToFramework();
    }],
    ['create', 'POST', '/u/alice/api/scaffold/create', async () => {
      vi.stubGlobal('prompt', vi.fn(() => 'my new agent'));
      const host = /** @type {any} */ ({ artifacts: [], load: () => Promise.resolve() });
      createScaffoldGalleryDetail(host).showCreateDialog('agents');
      // showCreateDialog's fetch chain is .then-based, not awaited internally.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }],
    ['register untracked', 'POST', '/u/alice/api/scaffold/untracked/register', async () => {
      const actions = createScaffoldDataActions(
        { categoryFilter: () => true, categoryOverrides: {}, categoryRemaps: {} },
        { onLoaded: vi.fn(), onLoadError: vi.fn() }
      );
      await actions.registerUntracked('my-hook');
    }],
    ['delete untracked', 'DELETE', `/u/alice/api/scaffold/untracked/${encodeURIComponent('my file')}`, async () => {
      const actions = createScaffoldDataActions(
        { categoryFilter: () => true, categoryOverrides: {}, categoryRemaps: {} },
        { onLoaded: vi.fn(), onLoadError: vi.fn() }
      );
      await actions.deleteUntracked('my file');
    }],
  ];

  test.each(WRITES)('%s: %s %s', async (_label, method, url, drive) => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = vi.fn(/** @type {(url: string, init?: RequestInit) => Promise<any>} */ (
      () => Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ artifacts: [], untracked: [], canonical_name: 'my-new-agent' }),
      })
    ));
    vi.stubGlobal('fetch', fetchMock);

    await drive();

    const write = fetchMock.mock.calls.find(([, init]) => init?.method === method);
    expect(write?.[0]).toBe(url);
  });
});
