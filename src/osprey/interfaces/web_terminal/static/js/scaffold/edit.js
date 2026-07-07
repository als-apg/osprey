// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: edit-view write actions
 *
 * The write side of the scaffold gallery's edit workflow: taking/releasing
 * ownership of a framework artifact (takeOwnership, releaseToFramework,
 * handleEditFramework), discarding or saving in-progress edits
 * (discardEdits, saveOverride -- plus its settings.json ownership-warning
 * modal, _showOwnershipWarning), resetting to the framework default
 * (unoverrideArtifact), refetching + reopening the detail view after any of
 * the above (reloadAndReopen), and closing the detail view back to the
 * gallery grid (closeDetail). The "write-side actions" half of the edit
 * workflow; the edit forms themselves live in scaffold/edit-form.js.
 *
 * `editDirty` is the flag the settings drawer's unsaved-changes prompt
 * reads directly (see initScaffoldGallery's registerUnsavedGuard in
 * scaffold-gallery.js). closeDetail's own confirm-before-discard check here
 * is the same guard applied to the detail view's back button, so both
 * paths behave identically (pinned by the drawer/parity browser tests).
 *
 * @module scaffold/edit
 */

import { fetchArtifactsShared, resetFetchCache } from './data.js';

/**
 * `detailContentEl` grows a `_settingsEditor` property when the
 * settings.json structured editor is mounted, and `_frontMatterFields`/
 * `_bodyTextarea` when the front-matter form is mounted (see
 * scaffold/edit-form.js) -- both read here by saveOverride() to pull the
 * edited content back out.
 * @typedef {HTMLElement & {
 *   _settingsEditor?: { getData(): string, isDirty(): boolean } | null,
 *   _frontMatterFields?: Record<string, HTMLInputElement|HTMLSelectElement>,
 *   _bodyTextarea?: HTMLTextAreaElement,
 * }} EditContentElement
 */

/**
 * The subset of an ArtifactGallery instance these write-side actions read,
 * write, or call into.
 * @typedef {object} ScaffoldGalleryEditHost
 * @property {any} selectedArtifact
 * @property {any[]} artifacts
 * @property {(artifact: any) => boolean} categoryFilter
 * @property {Record<string, string>} categoryOverrides
 * @property {Record<string, string>} categoryRemaps
 * @property {{total: number, framework: number, userOwned: number}} summary
 * @property {string} currentView
 * @property {string} detailMode
 * @property {boolean} editDirty
 * @property {EditContentElement|null} detailContentEl
 * @property {HTMLElement|null} errorEl
 * @property {HTMLElement|null} galleryView
 * @property {HTMLElement|null} detailView
 * @property {(() => void)|null} onDetailClose
 * @property {(artifact: any) => void} openDetail
 * @property {() => void} renderDetailHeader
 * @property {() => void} renderDetailModes
 * @property {() => void} renderDetailContent
 * @property {() => void} renderGallery
 */

/**
 * Create the scaffold gallery's write-side edit actions, bound to a fixed
 * gallery host.
 *
 * @param {ScaffoldGalleryEditHost} gallery
 */
export function createScaffoldGalleryEdit(gallery) {
  /** @returns {Promise<void>} */
  async function takeOwnership() {
    if (!gallery.selectedArtifact) return;
    if (!confirm('By doing this you take responsibility for this file.')) return;

    try {
      const resp = await fetch(
        `/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/claim`,
        { method: 'POST' }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Scaffold failed (HTTP ${resp.status})`);
      }

      await reloadAndReopen();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (gallery.errorEl) {
        gallery.errorEl.style.display = 'flex';
        gallery.errorEl.textContent = `Scaffold failed: ${message}`;
      }
    }
  }

  /** @returns {Promise<void>} */
  async function releaseToFramework() {
    if (!gallery.selectedArtifact) return;
    if (!confirm('Your customizations will be removed.')) return;

    await unoverrideArtifact(true);
  }

  /** @returns {Promise<void>} */
  async function handleEditFramework() {
    if (!gallery.selectedArtifact) return;
    if (!confirm('This will create a project copy for editing.')) return;

    try {
      const resp = await fetch(
        `/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/claim`,
        { method: 'POST' }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Scaffold failed (HTTP ${resp.status})`);
      }

      // Reload from API (invalidate cache so we get fresh data)
      resetFetchCache();
      const data = await fetchArtifactsShared();
      const allArtifacts = data.artifacts || [];
      gallery.artifacts = allArtifacts
        .filter(gallery.categoryFilter)
        .map(/** @param {any} a */ (a) => ({
          ...a,
          displayCategory:
            gallery.categoryOverrides[a.name] ||
            gallery.categoryRemaps[a.category] ||
            a.category,
        }));

      const updated = gallery.artifacts.find((a) => a.name === gallery.selectedArtifact.name);
      if (updated) {
        gallery.selectedArtifact = updated;
        gallery.detailMode = 'edit';
        gallery.renderDetailHeader();
        gallery.renderDetailModes();
        gallery.renderDetailContent();
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (gallery.errorEl) {
        gallery.errorEl.style.display = 'flex';
        gallery.errorEl.textContent = `Scaffold failed: ${message}`;
      }
    }
  }

  /** @returns {void} */
  function discardEdits() {
    gallery.editDirty = false;
    if (gallery.detailMode !== 'preview') {
      gallery.detailMode = 'preview';
    }
    gallery.renderDetailModes();
    gallery.renderDetailContent();
  }

  /** @returns {Promise<void>} */
  async function saveOverride() {
    if (!gallery.selectedArtifact) return;

    const container = gallery.detailContentEl;
    if (!container) return;

    /** @type {string|undefined} */
    let content;

    if (container._settingsEditor) {
      // Settings.json structured editor
      content = container._settingsEditor.getData();
    } else if (container._frontMatterFields && container._bodyTextarea) {
      const fields = container._frontMatterFields;
      let yaml = '---\n';
      for (const [key, input] of Object.entries(fields)) {
        const val = input.value.trim();
        if (val) {
          if (val.includes(':') || val.includes('#') || val.includes(',')) {
            yaml += `${key}: "${val}"\n`;
          } else {
            yaml += `${key}: ${val}\n`;
          }
        }
      }
      yaml += '---\n';
      content = yaml + container._bodyTextarea.value;
    } else {
      const textarea = /** @type {HTMLTextAreaElement|null} */ (
        container.querySelector('.prompts-edit-textarea')
      );
      if (!textarea) return;
      content = textarea.value;
    }

    // Ownership warning + scaffold for framework-owned settings.json
    if (gallery.selectedArtifact.name === 'settings-json'
        && gallery.selectedArtifact.status === 'framework') {
      const confirmed = await _showOwnershipWarning();
      if (!confirmed) return;

      // Scaffold (claim) the file before writing the override
      const scaffoldResp = await fetch(
        `/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/claim`,
        { method: 'POST' }
      );
      if (!scaffoldResp.ok) {
        const detail = await scaffoldResp.json().catch(() => ({}));
        if (gallery.errorEl) {
          gallery.errorEl.style.display = 'flex';
          gallery.errorEl.textContent = `Scaffold failed: ${detail.detail || `HTTP ${scaffoldResp.status}`}`;
        }
        return;
      }
    }

    try {
      const resp = await fetch(
        `/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/override`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content }),
        }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Save failed (HTTP ${resp.status})`);
      }

      gallery.editDirty = false;
      await reloadAndReopen();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (gallery.errorEl) {
        gallery.errorEl.style.display = 'flex';
        gallery.errorEl.textContent = `Save failed: ${message}`;
      }
    }
  }

  /**
   * Show a modal warning that saving settings.json means taking ownership.
   * @returns {Promise<boolean>} resolves true (proceed) or false (cancel)
   */
  function _showOwnershipWarning() {
    return new Promise((resolve) => {
      const overlay = document.createElement('div');
      overlay.className = 'config-ownership-overlay';

      const dialog = document.createElement('div');
      dialog.className = 'config-ownership-dialog';

      const iconEl = document.createElement('div');
      iconEl.className = 'config-ownership-icon';
      iconEl.textContent = '⚠';
      dialog.appendChild(iconEl);

      const title = document.createElement('div');
      title.className = 'config-ownership-title';
      title.textContent = 'Taking Ownership';
      dialog.appendChild(title);

      const body = document.createElement('div');
      body.className = 'config-ownership-body';
      body.textContent =
        'You are about to take ownership of settings.json. ' +
        'OSPREY will no longer auto-manage this file during regeneration ' +
        '(osprey claude regen). Future framework updates to permissions, ' +
        'hooks, and model configuration will not be applied automatically. ' +
        'You can release ownership later to restore framework management.';
      dialog.appendChild(body);

      const actions = document.createElement('div');
      actions.className = 'config-ownership-actions';

      const cancelBtn = document.createElement('button');
      cancelBtn.className = 'config-ownership-cancel';
      cancelBtn.textContent = 'Cancel';
      cancelBtn.addEventListener('click', () => {
        overlay.remove();
        resolve(false);
      });
      actions.appendChild(cancelBtn);

      const confirmBtn = document.createElement('button');
      confirmBtn.className = 'config-ownership-confirm';
      confirmBtn.textContent = 'I Understand, Save';
      confirmBtn.addEventListener('click', () => {
        overlay.remove();
        resolve(true);
      });
      actions.appendChild(confirmBtn);

      dialog.appendChild(actions);
      overlay.appendChild(dialog);
      document.body.appendChild(overlay);

      // Animate in
      requestAnimationFrame(() => overlay.classList.add('visible'));
    });
  }

  /**
   * @param {boolean} [skipConfirm]
   * @returns {Promise<void>}
   */
  async function unoverrideArtifact(skipConfirm = false) {
    if (!gallery.selectedArtifact) return;

    if (!skipConfirm) {
      if (!confirm('Reset to framework default? This will remove your customizations.')) {
        return;
      }
    }

    try {
      const resp = await fetch(
        `/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/override?delete_file=true`,
        { method: 'DELETE' }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Reset failed (HTTP ${resp.status})`);
      }

      await reloadAndReopen();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (gallery.errorEl) {
        gallery.errorEl.style.display = 'flex';
        gallery.errorEl.textContent = `Reset failed: ${message}`;
      }
    }
  }

  /** @returns {Promise<void>} */
  async function reloadAndReopen() {
    const name = gallery.selectedArtifact ? gallery.selectedArtifact.name : null;

    // Invalidate cache and refetch
    resetFetchCache();
    const data = await fetchArtifactsShared();
    const allArtifacts = data.artifacts || [];
    gallery.artifacts = allArtifacts
      .filter(gallery.categoryFilter)
      .map(/** @param {any} a */ (a) => ({
        ...a,
        displayCategory:
          gallery.categoryOverrides[a.name] ||
          gallery.categoryRemaps[a.category] ||
          a.category,
      }));

    // Recompute summary
    const fw = gallery.artifacts.filter((a) => a.status === 'framework').length;
    const uo = gallery.artifacts.filter((a) => a.status === 'user-owned').length;
    gallery.summary = { total: gallery.artifacts.length, framework: fw, userOwned: uo };

    if (name) {
      const updated = gallery.artifacts.find((a) => a.name === name);
      if (updated) {
        gallery.openDetail(updated);
        return;
      }
    }

    gallery.renderGallery();
  }

  /** @returns {void} */
  function closeDetail() {
    if (gallery.editDirty) {
      if (!confirm('You have unsaved changes. Discard them?')) return;
    }

    gallery.currentView = 'gallery';
    gallery.selectedArtifact = null;
    gallery.editDirty = false;
    gallery.detailMode = 'preview';

    if (gallery.galleryView) gallery.galleryView.style.display = '';
    if (gallery.detailView) gallery.detailView.style.display = 'none';

    if (gallery.onDetailClose) gallery.onDetailClose();

    // Re-render gallery so cards reflect any ownership changes
    gallery.renderGallery();
  }

  return {
    takeOwnership,
    releaseToFramework,
    handleEditFramework,
    discardEdits,
    saveOverride,
    unoverrideArtifact,
    reloadAndReopen,
    closeDetail,
  };
}
