// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: edit-view write actions
 *
 * The write side of the scaffold gallery's edit workflow: taking/releasing
 * ownership of a framework artifact (takeOwnership, releaseToFramework,
 * handleEditFramework), discarding or saving in-progress edits
 * (discardEdits, saveOverride), resetting to the framework default
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

import { apiRequest } from './data.js';

/**
 * `detailContentEl` grows `_frontMatterFields`/`_bodyTextarea` when the
 * front-matter form is mounted (see scaffold/edit-form.js) -- both read here
 * by saveOverride() to pull the edited content back out.
 * @typedef {HTMLElement & {
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
 * @property {() => Promise<void>} reloadFull
 * @property {string} currentView
 * @property {string} detailMode
 * @property {boolean} editDirty
 * @property {EditContentElement|null} detailContentEl
 * @property {HTMLElement|null} errorEl
 * @property {HTMLElement|null} galleryView
 * @property {HTMLElement|null} detailView
 * @property {(() => void)|null} onDetailClose
 * @property {(artifact: any) => void} openDetail
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
  /**
   * Put a failed write in front of the operator, in the words the server used.
   *
   * The message matters more here than in most error paths: a protected-set
   * refusal names the channel that owns the change instead ("'settings-json'
   * belongs to the profile's `config:` keys (...). NOTHING WAS WRITTEN."), and
   * that sentence is the operator's only pointer to where the edit does belong.
   * `apiRequest` already carries the server's `detail` through as the Error
   * message, so this only has to keep it intact and say which action broke.
   *
   * @param {string} prefix   What was being attempted ("Save failed", ...).
   * @param {unknown} e       The caught rejection.
   * @returns {void}
   */
  function showWriteError(prefix, e) {
    const message = e instanceof Error ? e.message : String(e);
    if (!gallery.errorEl) return;
    gallery.errorEl.classList.remove('prompts-error--notice');
    gallery.errorEl.style.display = 'flex';
    gallery.errorEl.textContent = `${prefix}: ${message}`;
  }

  /**
   * Tell the operator a save landed somewhere other than where they think.
   *
   * The counterpart to {@link showWriteError} and deliberately the same strip:
   * there is one place in this panel that speaks to the operator about a write,
   * and a second one would be a second place to miss. The modifier class is
   * what separates "this failed" from "this worked, with a caveat" — without
   * it a successful save would be announced in the wording of a failure.
   *
   * @param {string} message  What the operator needs to know.
   * @returns {void}
   */
  function showWriteNotice(message) {
    if (!gallery.errorEl) return;
    gallery.errorEl.classList.add('prompts-error--notice');
    gallery.errorEl.style.display = 'flex';
    gallery.errorEl.textContent = message;
  }

  /** @returns {Promise<void>} */
  async function takeOwnership() {
    if (!gallery.selectedArtifact) return;
    if (!confirm(
      'Take ownership of this file? A project copy is written into .claude/, and '
      + 'OSPREY stops overwriting it when artifacts are regenerated — later framework '
      + 'updates to it will not reach your project. You can release ownership again.'
    )) return;

    try {
      await apiRequest(`/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/claim`, {
        method: 'POST',
        errorPrefix: 'Scaffold failed',
      });

      await reloadAndReopen();
    } catch (e) {
      showWriteError('Scaffold failed', e);
    }
  }

  /** @returns {Promise<void>} */
  async function releaseToFramework() {
    if (!gallery.selectedArtifact) return;
    if (!confirm(
      'Release this file back to the framework? Your project copy is deleted from '
      + 'disk and the framework version takes over again.'
    )) return;

    await unoverrideArtifact(true);
  }

  /** @returns {Promise<void>} */
  async function handleEditFramework() {
    if (!gallery.selectedArtifact) return;
    if (!confirm(
      'Editing this file takes ownership of it: a project copy is written into '
      + '.claude/, and OSPREY stops overwriting it when artifacts are regenerated.'
    )) return;

    try {
      await apiRequest(`/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/claim`, {
        method: 'POST',
        errorPrefix: 'Scaffold failed',
      });

      // Full reload (fresh cache) via the gallery's single data pipeline,
      // then reopen this artifact in edit mode.
      await gallery.reloadFull();

      const updated = gallery.artifacts.find((a) => a.name === gallery.selectedArtifact.name);
      if (updated) {
        // openDetail restores detail-view visibility (reloadFull's gallery
        // re-render flipped back to the grid); then switch to edit mode
        // inline — same pattern as detail.js's showCreateDialog.
        gallery.openDetail(updated);
        gallery.detailMode = 'edit';
        gallery.renderDetailModes();
        gallery.renderDetailContent();
      }
    } catch (e) {
      showWriteError('Scaffold failed', e);
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

    if (container._frontMatterFields && container._bodyTextarea) {
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

    try {
      const result = await apiRequest(`/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/override`, {
        method: 'PUT',
        json: { content },
        errorPrefix: 'Save failed',
      });

      gallery.editDirty = false;
      await reloadAndReopen();

      // After the reopen, not before: the reload re-renders this panel and
      // clears the strip, so a notice raised first would be wiped by the
      // refresh that follows it.
      if (result && result.applies_on_restart) {
        showWriteNotice(
          'Saved — applies on container restart. The project tree here is read-only, '
          + 'so your change is held on the claude-config volume and reaches the agent '
          + 'when the container next starts.'
        );
      }
    } catch (e) {
      showWriteError('Save failed', e);
    }
  }

  /**
   * @param {boolean} [skipConfirm]
   * @returns {Promise<void>}
   */
  async function unoverrideArtifact(skipConfirm = false) {
    if (!gallery.selectedArtifact) return;

    if (!skipConfirm) {
      if (!confirm('Reset to the framework default? Your project copy is deleted from disk.')) {
        return;
      }
    }

    try {
      await apiRequest(`/api/scaffold/${encodeURIComponent(gallery.selectedArtifact.name)}/override?delete_file=true`, {
        method: 'DELETE',
        errorPrefix: 'Reset failed',
      });

      await reloadAndReopen();
    } catch (e) {
      showWriteError('Reset failed', e);
    }
  }

  /** @returns {Promise<void>} */
  async function reloadAndReopen() {
    const name = gallery.selectedArtifact ? gallery.selectedArtifact.name : null;

    // Full reload (fresh cache + untracked-file refresh + summary + gallery
    // re-render) via the gallery's single data pipeline (scaffold/data.js).
    await gallery.reloadFull();

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
