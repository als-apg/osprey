// @ts-check
/**
 * ARIEL Entry Creation & Draft Module
 *
 * Entry creation form: submission, tags, file preview, and draft loading.
 */

import { entriesApi, draftsApi, ApiError } from './api.js';
import { escapeHtml } from './components.js';
import { formatFileSize } from './entries-helpers.js';
import { messageOf } from './utils.js';

// Draft metadata (populated when loading a draft)
/** @type {any} */
let draftMetadata = null;

/**
 * Handle entry creation form submission.
 * @param {Event} e - Submit event
 */
export async function handleCreateEntry(e) {
  e.preventDefault();

  const form = /** @type {HTMLFormElement} */ (e.target);
  const submitBtn = /** @type {HTMLButtonElement|null} */ (
    form.querySelector('button[type="submit"]')
  );
  const originalText = submitBtn ? submitBtn.textContent : null;

  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Saving...';
  }

  try {
    const formData = new FormData(form);
    const tagEls = /** @type {NodeListOf<HTMLElement>} */ (
      document.querySelectorAll('#entry-tags .tag')
    );
    const tags = Array.from(tagEls).map(t => t.dataset.value);

    const entryData = {
      subject: formData.get('subject'),
      details: formData.get('details'),
      author: formData.get('author'),
      logbook: formData.get('logbook'),
      shift: formData.get('shift'),
      tags,
      metadata: draftMetadata,
      auth_user: formData.get('olog_user') || null,
      auth_password: formData.get('olog_password') || null,
    };

    const fileInput = /** @type {HTMLInputElement|null} */ (
      document.getElementById('entry-files')
    );
    const files = Array.from(fileInput?.files || []);

    // Check for draft attachments (staged by Claude via artifact_ids)
    const preview = document.getElementById('file-preview');
    const draftAttachmentsJson = preview?.dataset?.draftAttachments;
    if (draftAttachmentsJson) {
      try {
        const draftAttachments = JSON.parse(draftAttachmentsJson);
        for (const att of draftAttachments) {
          const resp = await fetch(att.url);
          if (resp.ok) {
            const blob = await resp.blob();
            files.push(new File([blob], att.filename, { type: blob.type }));
          }
        }
      } catch (err) {
        console.warn('Failed to fetch draft attachments:', err);
      }
    }

    let result;
    if (files.length > 0) {
      result = await entriesApi.createWithAttachments(entryData, files);
    } else {
      result = await entriesApi.create(entryData);
    }

    // Show success message
    const attachMsg = result.attachment_count
      ? ` with ${result.attachment_count} attachment(s)`
      : '';
    alert(`Entry created: ${result.entry_id}${attachMsg}`);

    // Reset form, file preview, and draft state
    form.reset();
    const tagsContainer = document.getElementById('entry-tags');
    if (tagsContainer) tagsContainer.innerHTML = '';
    if (preview) preview.innerHTML = '';
    draftMetadata = null;
    const sessionPanel = document.getElementById('session-info-panel');
    if (sessionPanel) sessionPanel.remove();
    const draftBanner = document.getElementById('draft-banner');
    if (draftBanner) draftBanner.remove();

    // Navigate to entry
    /** @type {any} */ (window).app.showEntry(result.entry_id);

  } catch (error) {
    console.error('Failed to create entry:', error);
    if (error instanceof ApiError && error.code === 'auth_required') {
      // The logbook needs credentials to publish. Keep the form populated (it is
      // only reset on success) and focus the username field so the operator can
      // type and resubmit.
      alert('Logbook credentials required to publish. Please enter your username and password.');
      document.getElementById('entry-auth-user')?.focus();
    } else {
      alert(`Failed to create entry: ${messageOf(error)}`);
    }
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = originalText;
    }
  }
}

/**
 * Handle tag input keydown.
 * @param {KeyboardEvent} e - Keydown event
 */
export function handleTagInput(e) {
  if (e.key === 'Enter' || e.key === ',') {
    e.preventDefault();
    const input = /** @type {HTMLInputElement} */ (e.target);
    const value = input.value.trim();

    if (value) {
      addTag(value);
      input.value = '';
    }
  }
}

/**
 * Add a tag to the tags list.
 * @param {string} value - Tag value
 */
function addTag(value) {
  const container = document.getElementById('entry-tags');
  if (!container) return;

  // Check for duplicates
  const existing = container.querySelector(`[data-value="${value}"]`);
  if (existing) return;

  const tag = document.createElement('span');
  tag.className = 'tag tag-accent';
  tag.dataset.value = value;
  tag.innerHTML = `
    ${escapeHtml(value)}
    <button type="button" onclick="this.parentElement.remove()" style="background: none; border: none; cursor: pointer; color: inherit; margin-left: 4px;">&times;</button>
  `;
  container.appendChild(tag);
}

/**
 * Build the fixed-size icon shown in place of a thumbnail for non-image
 * attachments.
 * @returns {HTMLDivElement}
 */
function createAttachmentIcon() {
  const icon = document.createElement('div');
  icon.style.cssText = 'font-size: 32px; margin: 8px 0;';
  icon.textContent = '\u{1F4CE}';
  return icon;
}

/**
 * Build a preview card shell (thumbnail + filename, optionally a size line)
 * shared by the file-input preview and the draft-attachment preview — the
 * two only differ in how they obtain the thumbnail and whether a size is
 * known.
 * @param {string} filename - Display filename
 * @param {Node} mediaEl - Thumbnail element (an <img> or createAttachmentIcon())
 * @param {string} [sizeText] - Human-readable size, omitted when unknown
 * @returns {HTMLDivElement}
 */
function createPreviewCard(filename, mediaEl, sizeText) {
  const card = document.createElement('div');
  card.className = 'card';
  card.style.cssText = 'width: 120px; text-align: center; padding: 8px;';
  card.appendChild(mediaEl);

  const name = document.createElement('div');
  name.className = 'truncate text-xs';
  name.textContent = filename;
  card.appendChild(name);

  if (sizeText !== undefined) {
    const size = document.createElement('div');
    size.className = 'text-xs text-muted';
    size.textContent = sizeText;
    card.appendChild(size);
  }

  return card;
}

/**
 * Handle file input change to show preview thumbnails.
 * @param {Event} e - Change event
 */
export function handleFilePreview(e) {
  const container = document.getElementById('file-preview');
  if (!container) return;
  container.innerHTML = '';

  const files = /** @type {HTMLInputElement} */ (e.target).files;
  if (!files || files.length === 0) return;

  for (const file of files) {
    let media;
    if (file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.style.cssText = 'width: 100px; height: 80px; object-fit: cover; border-radius: 4px;';
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);
      media = img;
    } else {
      media = createAttachmentIcon();
    }

    container.appendChild(createPreviewCard(file.name, media, formatFileSize(file.size)));
  }
}

/**
 * Render a read-only panel showing session metadata from a draft.
 * @param {any} meta - session_metadata object
 */
function renderSessionInfoPanel(meta) {
  const existing = document.getElementById('session-info-panel');
  if (existing) existing.remove();

  const fields = [
    ['Operator', meta.operator],
    ['Branch', meta.git_branch],
    ['Session', meta.session_id],
    ['Model', meta.model],
    ['Source', meta.created_via],
  ].filter(([, v]) => v);

  if (fields.length === 0) return;

  const panel = document.createElement('div');
  panel.id = 'session-info-panel';
  panel.className = 'session-info-panel';
  panel.innerHTML = `
    <div class="session-info-header">Session Context</div>
    <div class="session-info-fields">
      ${fields.map(([label, value]) =>
        `<span><span class="session-info-label">${escapeHtml(label)}:</span>${escapeHtml(String(value))}</span>`
      ).join('')}
    </div>
  `;

  const banner = document.getElementById('draft-banner');
  if (banner) {
    banner.after(panel);
  }
}

/**
 * Load a draft into the entry creation form.
 * @param {string} draftId - Draft ID to load
 */
export async function loadDraft(draftId) {
  try {
    const draft = await draftsApi.get(draftId);

    // Populate form fields
    const subjectInput = /** @type {HTMLInputElement|null} */ (
      document.querySelector('#create-entry-form [name="subject"]')
    );
    const detailsInput = /** @type {HTMLTextAreaElement|null} */ (
      document.querySelector('#create-entry-form [name="details"]')
    );
    const authorInput = /** @type {HTMLInputElement|null} */ (
      document.querySelector('#create-entry-form [name="author"]')
    );
    const logbookSelect = /** @type {HTMLSelectElement|null} */ (
      document.querySelector('#create-entry-form [name="logbook"]')
    );
    const shiftSelect = /** @type {HTMLSelectElement|null} */ (
      document.querySelector('#create-entry-form [name="shift"]')
    );

    if (subjectInput) subjectInput.value = draft.subject || '';
    if (detailsInput) detailsInput.value = draft.details || '';
    if (authorInput) authorInput.value = draft.author || '';
    if (logbookSelect && draft.logbook) logbookSelect.value = draft.logbook;
    if (shiftSelect && draft.shift) shiftSelect.value = draft.shift;

    // Store draft metadata for forwarding on submit
    draftMetadata = draft.metadata || null;

    // Clear existing tags and add draft tags
    const tagsContainer = document.getElementById('entry-tags');
    if (tagsContainer) tagsContainer.innerHTML = '';
    if (draft.tags && draft.tags.length > 0) {
      draft.tags.forEach((/** @type {string} */ tag) => addTag(tag));
    }

    // Show draft attachments if present
    if (draft.attachment_paths && draft.attachment_paths.length > 0) {
      const preview = document.getElementById('file-preview');
      if (preview) {
        preview.innerHTML = '';
        const attachmentData = [];

        for (const fullPath of draft.attachment_paths) {
          const filename = fullPath.split('/').pop();
          const url = `/api/drafts/${draftId}/attachments/${encodeURIComponent(filename)}`;
          attachmentData.push({ filename, url });

          let media;
          const isImage = /\.(png|jpe?g|gif|webp|svg)$/i.test(filename);
          if (isImage) {
            const img = document.createElement('img');
            img.style.cssText = 'width: 100px; height: 80px; object-fit: cover; border-radius: 4px;';
            img.src = url;
            img.alt = filename;
            media = img;
          } else {
            media = createAttachmentIcon();
          }

          preview.appendChild(createPreviewCard(filename, media));
        }

        preview.dataset.draftAttachments = JSON.stringify(attachmentData);
        preview.dataset.draftId = draftId;
      }
    }

    // Show banner
    const form = document.getElementById('create-entry-form');
    if (form) {
      const existing = document.getElementById('draft-banner');
      if (existing) existing.remove();
      const banner = document.createElement('div');
      banner.id = 'draft-banner';
      banner.className = 'text-muted';
      banner.dataset.draftId = draftId;
      banner.style.cssText =
        'padding: 8px 12px; margin-bottom: 12px; border-left: 3px solid var(--color-amber); font-size: 0.85rem;';
      banner.textContent = 'Draft loaded from Claude — review and submit';
      form.prepend(banner);
    }

    // Show session info panel if metadata includes session context
    if (draftMetadata?.session_metadata) {
      renderSessionInfoPanel(draftMetadata.session_metadata);
    }
  } catch (error) {
    console.error('Failed to load draft:', error);
  }
}
