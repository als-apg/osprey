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
import { resetFetchCache, apiRequest } from './data.js';
import { createScaffoldGalleryDetailContent } from './detail-content.js';
import { READ_ONLY_REASON, createReadOnlyBadge } from './utils.js';
import { scaffoldWritesEnabled, WRITES_DISABLED_REASON } from './write-gate.js';

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

    apiRequest('/api/scaffold/create', {
      method: 'POST',
      json: { category, name: sanitized },
      errorPrefix: 'Create failed',
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
    const readOnly = Boolean(gallery.selectedArtifact.read_only);

    const badge = document.createElement('span');
    badge.className = `prompts-badge ${isOwned ? 'user-owned' : 'framework'}`;
    badge.textContent = isOwned ? 'PROJECT-OWNED' : 'FRAMEWORK';
    row1.appendChild(badge);

    if (readOnly) row1.appendChild(createReadOnlyBadge());

    // Claiming an artifact (POST .../claim) and releasing one (DELETE
    // .../override) are both writes, refused with 403 where the deployment
    // withdrew gallery writes -- so the button is not built at all. Absence
    // rather than a disabled button: unlike the mode tabs below, nothing about
    // the header reads as broken without it, and absence keeps it out of the
    // tab order. Everything else in the header is a fact about the artifact
    // and survives.
    if (scaffoldWritesEnabled()) {
      const ownerBtn = document.createElement('button');
      ownerBtn.className = 'prompts-ownership-btn';
      // Taking ownership of a reserved artifact is a claim the server refuses:
      // scaffold_override raises ScaffoldClaimError, which the app-level handler
      // renders as a 409. Offering the button live would be a promise the panel
      // cannot keep -- and the refusal, when it comes, is about the path rather
      // than anything the operator did wrong.
      if (readOnly) {
        ownerBtn.disabled = true;
        ownerBtn.title = READ_ONLY_REASON;
      }
      if (isOwned) {
        ownerBtn.textContent = 'Release to Framework';
        ownerBtn.addEventListener('click', () => gallery.releaseToFramework());
      } else {
        ownerBtn.textContent = 'Take Ownership';
        ownerBtn.addEventListener('click', () => gallery.takeOwnership());
      }
      row1.appendChild(ownerBtn);
    }

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

    // The badge states the fact; this line is where the panel's descriptive
    // copy says what to do about it, next to the path it applies to.
    if (readOnly) {
      const note = document.createElement('div');
      note.className = 'prompts-readonly-note';
      note.textContent = READ_ONLY_REASON;
      gallery.detailHeaderEl.appendChild(note);
    }
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

    const readOnly = Boolean(gallery.selectedArtifact.read_only);

    for (const mode of modes) {
      const btn = document.createElement('button');
      btn.className = 'prompts-mode-btn' + (gallery.detailMode === mode.key ? ' active' : '');
      btn.textContent = mode.label;
      if (!scaffoldWritesEnabled() && mode.key === 'edit') {
        // Rendered but disabled, unlike the withdrawn buttons: a Preview /
        // Diff / Edit strip missing its third entry reads as a broken panel,
        // and the title is where the operator learns it is the deployment's
        // posture rather than a fault. Preview and Diff stay live — reading
        // what the agent runs is not authoring it.
        btn.disabled = true;
        btn.title = WRITES_DISABLED_REASON;
      }
      if (readOnly && mode.key === 'edit') {
        // Preview and Diff stay open — reading a reserved file is fine, and
        // seeing what it holds is most of why the panel lists it at all.
        btn.disabled = true;
        btn.title = READ_ONLY_REASON;
      }
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

    if (gallery.detailMode === 'edit') {
      const discardBtn = document.createElement('button');
      discardBtn.className = 'prompts-discard-btn';
      discardBtn.textContent = 'Discard';
      discardBtn.disabled = !gallery.editDirty;
      discardBtn.addEventListener('click', () => gallery.discardEdits());
      right.appendChild(discardBtn);

      // PUT .../override answers 403 where writes are withdrawn, so Save is not
      // offered there. Discard above stays: throwing away local edits touches
      // nothing on the server, and an operator who typed into a still-open
      // editor needs a way out of it.
      if (scaffoldWritesEnabled()) {
        const saveBtn = document.createElement('button');
        saveBtn.className = 'prompts-save-btn';
        saveBtn.textContent = 'Save';
        // Disabled on `read_only` whatever the dirty flag says: the Edit tab is
        // already disabled for a reserved artifact, and this is the second gate
        // in case anything else puts the panel into edit mode.
        saveBtn.disabled = !gallery.editDirty || readOnly;
        if (readOnly) saveBtn.title = READ_ONLY_REASON;
        saveBtn.addEventListener('click', () => gallery.saveOverride());
        right.appendChild(saveBtn);
      }
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
