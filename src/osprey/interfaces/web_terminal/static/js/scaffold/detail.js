// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: detail-view shell
 *
 * The detail modal's "shell": opening it (openDetail), the create-artifact
 * dialog (showCreateDialog), the header (back button, name, ownership
 * badge/button), the mode tabs (Preview/Diff/Edit -- enabled/disabled and
 * highlighted based on the selected artifact's ownership), and the mode
 * dispatch (renderDetailContent). The detail modal is the gallery's core
 * UX.
 *
 * The two read-only content renderers (Preview, Diff) that
 * renderDetailContent dispatches to live in scaffold/detail-content.js --
 * kept separate (mirrors the view.js/cards.js seam) so both modules stay
 * comfortably under the 450-line cap; Preview/Diff is the natural
 * "content rendering" seam, distinct from this file's "shell" concern
 * (mode switching, header, dispatch).
 *
 * The Edit mode's own renderer (renderEdit) and everything the edit/save/
 * ownership workflow needs (discardEdits, saveOverride, takeOwnership,
 * releaseToFramework, handleEditFramework, closeDetail) live on the
 * ArtifactGallery instance as thin delegators into scaffold/edit.js and
 * scaffold/edit-form.js -- this module calls them as `gallery.<method>()`,
 * the same "pass `this`" factory pattern the rest of the scaffold modules
 * use (see scaffold/data.js's createScaffoldDataActions docstring).
 *
 * @module scaffold/detail
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { resetFetchCache, withPrefix } from './data.js';
import { createScaffoldGalleryDetailContent } from './detail-content.js';

/**
 * The subset of an ArtifactGallery instance this module reads, writes, or
 * calls into. The last seven properties (load through renderEdit) are the
 * edit/save/ownership workflow that lives on the gallery instance -- this
 * shell only ever calls them, never redefines them.
 * @typedef {object} ScaffoldGalleryDetailHost
 * @property {any} selectedArtifact
 * @property {string} currentView
 * @property {string} detailMode
 * @property {boolean} editDirty
 * @property {any[]} artifacts
 * @property {HTMLElement|null} galleryView
 * @property {HTMLElement|null} detailView
 * @property {HTMLElement|null} detailHeaderEl
 * @property {HTMLElement|null} detailModesEl
 * @property {HTMLElement|null} detailContentEl
 * @property {(() => void)|null} onDetailOpen
 * @property {() => Promise<any>} load
 * @property {() => void} renderDetailModes
 * @property {() => void} closeDetail
 * @property {() => Promise<any>} releaseToFramework
 * @property {() => Promise<any>} takeOwnership
 * @property {() => Promise<any>} handleEditFramework
 * @property {() => void} discardEdits
 * @property {() => Promise<any>} saveOverride
 * @property {() => Promise<any>} renderEdit
 */

/**
 * Create the scaffold gallery's detail-view shell functions, bound to a
 * fixed gallery host.
 *
 * @param {ScaffoldGalleryDetailHost} gallery
 */
export function createScaffoldGalleryDetail(gallery) {
  const { renderPreview, renderDiff } = createScaffoldGalleryDetailContent(gallery);

  /**
   * @param {any} artifact
   * @returns {void}
   */
  function openDetail(artifact) {
    gallery.selectedArtifact = artifact;
    gallery.currentView = 'detail';
    gallery.detailMode = 'preview';
    gallery.editDirty = false;

    if (gallery.galleryView) gallery.galleryView.style.display = 'none';
    if (gallery.detailView) gallery.detailView.style.display = '';

    if (gallery.onDetailOpen) gallery.onDetailOpen();

    renderDetailHeader();
    renderDetailModes();
    renderDetailContent();
  }

  /**
   * @param {string} category
   * @returns {void}
   */
  function showCreateDialog(category) {
    const name = prompt(`Name for new ${category.replace(/s$/, '')}:`);
    if (!name) return;

    const sanitized = name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
    if (!sanitized) {
      alert('Invalid name. Use letters, numbers, and hyphens.');
      return;
    }

    fetch(withPrefix('/api/scaffold/create'), { // prefix-aware
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category, name: sanitized }),
    })
      .then(async (res) => {
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `Create failed (HTTP ${res.status})`);
        }
        return res.json();
      })
      .then((result) => {
        resetFetchCache();
        gallery.load().then(() => {
          const newArt = gallery.artifacts.find((a) => a.name === result.canonical_name);
          if (newArt) {
            openDetail(newArt);
            // Switch to edit mode inline (no switchMode method exists)
            gallery.detailMode = 'edit';
            renderDetailModes();
            renderDetailContent();
          }
        });
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : String(err);
        alert(`Failed to create: ${message}`);
      });
  }

  /** @returns {void} */
  function renderDetailHeader() {
    if (!gallery.detailHeaderEl || !gallery.selectedArtifact) return;
    gallery.detailHeaderEl.innerHTML = '';

    // Row 1: [Back] name ... BADGE [Ownership Btn]
    const row1 = document.createElement('div');
    row1.className = 'prompts-header-row';

    const backBtn = document.createElement('button');
    backBtn.className = 'prompts-back-btn';
    backBtn.textContent = '← Back';
    backBtn.addEventListener('click', () => gallery.closeDetail());
    row1.appendChild(backBtn);

    const nameEl = document.createElement('span');
    nameEl.className = 'prompts-detail-name';
    nameEl.textContent = gallery.selectedArtifact.name;
    row1.appendChild(nameEl);

    const spacer = document.createElement('span');
    spacer.style.flex = '1';
    row1.appendChild(spacer);

    const isOwned = gallery.selectedArtifact.status === 'user-owned';

    const badge = document.createElement('span');
    badge.className = `prompts-badge ${isOwned ? 'user-owned' : 'framework'}`;
    badge.textContent = isOwned ? 'PROJECT-OWNED' : 'FRAMEWORK';
    row1.appendChild(badge);

    const ownerBtn = document.createElement('button');
    ownerBtn.className = 'prompts-ownership-btn';
    if (isOwned) {
      ownerBtn.textContent = 'Release to Framework';
      ownerBtn.addEventListener('click', () => gallery.releaseToFramework());
    } else {
      ownerBtn.textContent = 'Take Ownership';
      ownerBtn.addEventListener('click', () => gallery.takeOwnership());
    }
    row1.appendChild(ownerBtn);

    gallery.detailHeaderEl.appendChild(row1);

    // Row 2: path + language
    const row2 = document.createElement('div');
    row2.className = 'prompts-header-meta';

    if (gallery.selectedArtifact.output_path) {
      const pathEl = document.createElement('span');
      pathEl.className = 'prompts-detail-path';
      pathEl.textContent = gallery.selectedArtifact.output_path;
      row2.appendChild(pathEl);
    }

    if (gallery.selectedArtifact.language) {
      const langEl = document.createElement('span');
      langEl.className = 'prompts-detail-lang';
      langEl.textContent = gallery.selectedArtifact.language;
      row2.appendChild(langEl);
    }

    gallery.detailHeaderEl.appendChild(row2);
  }

  /** @returns {void} */
  function renderDetailModes() {
    if (!gallery.detailModesEl || !gallery.selectedArtifact) return;
    gallery.detailModesEl.innerHTML = '';

    // Left: mode buttons
    const left = document.createElement('div');
    left.className = 'prompts-modes-left';

    const modes = [{ key: 'preview', label: 'Preview' }];

    if (gallery.selectedArtifact.status === 'user-owned' && !gallery.selectedArtifact.custom) {
      modes.push({ key: 'diff', label: 'Diff' });
    }

    modes.push({ key: 'edit', label: 'Edit' });

    for (const mode of modes) {
      const btn = document.createElement('button');
      btn.className = 'prompts-mode-btn' + (gallery.detailMode === mode.key ? ' active' : '');
      btn.textContent = mode.label;
      btn.addEventListener('click', () => {
        if (gallery.detailMode === mode.key) return;

        if (mode.key === 'edit' && gallery.selectedArtifact.status === 'framework') {
          gallery.handleEditFramework();
          return;
        }

        if (gallery.editDirty) {
          if (!confirm('You have unsaved changes. Discard them?')) return;
          gallery.editDirty = false;
        }
        gallery.detailMode = mode.key;
        renderDetailModes();
        renderDetailContent();
      });
      left.appendChild(btn);
    }

    gallery.detailModesEl.appendChild(left);

    // Right: action buttons
    const right = document.createElement('div');
    right.className = 'prompts-modes-right';

    const isSettingsPreview = gallery.detailMode === 'preview'
      && gallery.selectedArtifact?.name === 'settings-json';

    if (gallery.detailMode === 'edit' || (isSettingsPreview && gallery.editDirty)) {
      const discardBtn = document.createElement('button');
      discardBtn.className = 'prompts-discard-btn';
      discardBtn.textContent = 'Discard';
      discardBtn.disabled = !gallery.editDirty;
      discardBtn.addEventListener('click', () => gallery.discardEdits());
      right.appendChild(discardBtn);

      const saveBtn = document.createElement('button');
      saveBtn.className = 'prompts-save-btn';
      saveBtn.textContent = 'Save';
      saveBtn.disabled = !gallery.editDirty;
      saveBtn.addEventListener('click', () => gallery.saveOverride());
      right.appendChild(saveBtn);
    }

    gallery.detailModesEl.appendChild(right);
  }

  /** @returns {Promise<void>} */
  async function renderDetailContent() {
    if (!gallery.detailContentEl || !gallery.selectedArtifact) return;

    gallery.detailContentEl.innerHTML = '<div class="prompts-loading-inline">Loading...</div>';

    try {
      if (gallery.detailMode === 'preview') {
        await renderPreview();
      } else if (gallery.detailMode === 'diff') {
        await renderDiff();
      } else if (gallery.detailMode === 'edit') {
        await gallery.renderEdit();
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (gallery.detailContentEl) {
        gallery.detailContentEl.innerHTML =
          `<div class="prompts-content-error">Error loading content: ${escapeHtml(message)}</div>`;
      }
    }
  }

  return { openDetail, showCreateDialog, renderDetailHeader, renderDetailModes, renderDetailContent };
}
