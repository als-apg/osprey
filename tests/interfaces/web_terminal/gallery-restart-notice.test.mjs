/**
 * Unit tests for the scaffold gallery's "applies on container restart" notice
 * (scaffold/edit.js -- saveOverride's success branch).
 *
 * A save in a deployed container can succeed on the claude-config volume while
 * the read-only image tree refuses it. The server says so with
 * `applies_on_restart: true` in the PUT response, and this is the half that
 * puts it in front of the operator: without it a save that the agent will not
 * see until the next container start is indistinguishable from one that is
 * already live.
 *
 * Pure-logic + DOM guard, happy-dom environment, `fetch`/`confirm` mocked via
 * vi.stubGlobal -- mirrors the harness in scaffold-edit.test.mjs:
 *   npx vitest run tests/interfaces/web_terminal/gallery-restart-notice.test.mjs
 *
 * NOTE: imported by RELATIVE path -- these modules live under web_terminal,
 * not design-system, so the `/design-system/js/*` alias does not apply.
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import { createScaffoldGalleryEdit } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/edit.js';
import { resetFetchCache } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/data.js';

/**
 * A fake ArtifactGallery host, the same "pass `this`" shape the real class
 * uses, with a real error strip so the notice has somewhere to land.
 * @param {object} [overrides]
 */
function makeGallery(overrides = {}) {
  return /** @type {any} */ ({
    selectedArtifact: { name: 'agents/channel-finder', status: 'user-owned' },
    artifacts: [],
    reloadFull: vi.fn(async () => {}),
    currentView: 'detail',
    detailMode: 'edit',
    editDirty: true,
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

/**
 * Give the gallery a plain textarea to save from.
 * @param {{detailContentEl: HTMLElement}} gallery
 */
function withTextarea(gallery, value = 'edited body') {
  const textarea = document.createElement('textarea');
  textarea.className = 'prompts-edit-textarea';
  textarea.value = value;
  gallery.detailContentEl.appendChild(textarea);
  return textarea;
}

/**
 * Stub fetch so the PUT resolves with `body` and every other call (the reload)
 * resolves with an empty artifact list.
 * @param {object} body
 */
function stubSave(body) {
  vi.stubGlobal('fetch', vi.fn((url, init) => {
    const isPut = init && init.method === 'PUT';
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(isPut ? body : { artifacts: [], untracked: [] }),
    });
  }));
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

describe('applies-on-restart notice', () => {
  test('renders the notice when the server says the save only lands on restart', async () => {
    stubSave({ status: 'saved', path: '.claude/agents/channel-finder.md', applies_on_restart: true });
    const gallery = makeGallery();
    withTextarea(gallery);

    await createScaffoldGalleryEdit(gallery).saveOverride();

    expect(gallery.errorEl.style.display).toBe('flex');
    expect(gallery.errorEl.textContent).toContain('applies on container restart');
    expect(gallery.errorEl.classList.contains('prompts-error--notice')).toBe(true);
  });

  test('says nothing when the save reached the tree', async () => {
    stubSave({ status: 'saved', path: '.claude/agents/channel-finder.md', applies_on_restart: false });
    const gallery = makeGallery();
    withTextarea(gallery);

    await createScaffoldGalleryEdit(gallery).saveOverride();

    expect(gallery.errorEl.textContent).toBe('');
    expect(gallery.errorEl.classList.contains('prompts-error--notice')).toBe(false);
  });

  test('says nothing when the response carries no such field at all', async () => {
    // An older server, or any payload shape that predates the field: the
    // absence of a warning must not be read as a warning.
    stubSave({ status: 'saved', path: '.claude/agents/channel-finder.md' });
    const gallery = makeGallery();
    withTextarea(gallery);

    await createScaffoldGalleryEdit(gallery).saveOverride();

    expect(gallery.errorEl.textContent).toBe('');
  });

  test('the save still succeeds and clears the dirty flag alongside the notice', async () => {
    stubSave({ status: 'saved', path: '.claude/agents/channel-finder.md', applies_on_restart: true });
    const gallery = makeGallery();
    withTextarea(gallery);

    await createScaffoldGalleryEdit(gallery).saveOverride();

    expect(gallery.editDirty).toBe(false);
    expect(gallery.reloadFull).toHaveBeenCalled();
  });

  test('a failed save still reads as a failure, not a notice', async () => {
    vi.stubGlobal('fetch', vi.fn((url, init) => {
      if (init && init.method === 'PUT') {
        return Promise.resolve({
          ok: false,
          status: 403,
          json: () => Promise.resolve({ detail: "'rules/facility' belongs to the profile's `rules/` convention directory. NOTHING WAS WRITTEN." }),
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [], untracked: [] }) });
    }));
    const gallery = makeGallery();
    withTextarea(gallery);

    await createScaffoldGalleryEdit(gallery).saveOverride();

    expect(gallery.errorEl.textContent).toContain('Save failed');
    expect(gallery.errorEl.textContent).toContain('NOTHING WAS WRITTEN');
    expect(gallery.errorEl.classList.contains('prompts-error--notice')).toBe(false);
  });
});
