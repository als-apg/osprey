/**
 * ARIEL Entries Module
 *
 * Entry browsing, detail view, and creation.
 */

import { entriesApi, draftsApi } from './api.js';
import {
  formatTimestamp,
  renderEntryCard,
  renderLoading,
  renderEmptyState,
  renderTags,
  escapeHtml,
} from './components.js';

// Current entry detail
let currentEntry = null;

// Draft metadata (populated when loading a draft)
let draftMetadata = null;

/**
 * Check if an attachment is an image, inferring from filename when type is missing.
 * @param {Object} att - Attachment object with optional type and filename
 * @returns {boolean} True if the attachment is an image
 */
function isImageAttachment(att) {
  if (att.type && att.type.startsWith('image/')) return true;
  if (att.filename) {
    const ext = att.filename.split('.').pop().toLowerCase();
    return ['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'bmp'].includes(ext);
  }
  return false;
}

/**
 * Initialize entries module.
 */
export function initEntries() {
  // Entry creation form
  const createForm = document.getElementById('create-entry-form');
  createForm?.addEventListener('submit', handleCreateEntry);

  // Tag input
  const tagInput = document.getElementById('entry-tags-input');
  tagInput?.addEventListener('keydown', handleTagInput);

  // File input preview
  const fileInput = document.getElementById('entry-files');
  fileInput?.addEventListener('change', handleFilePreview);
}

/**
 * Load and display entry list.
 * @param {Object} params - List parameters
 */
export async function loadEntries(params = {}) {
  const container = document.getElementById('entries-list');
  if (!container) return;

  container.innerHTML = renderLoading('Loading entries...');

  try {
    const result = await entriesApi.list(params);
    renderEntriesList(container, result);
  } catch (error) {
    console.error('Failed to load entries:', error);
    container.innerHTML = `
      <div class="empty-state">
        <h3 class="empty-state-title text-error">Failed to Load Entries</h3>
        <p class="empty-state-text">${escapeHtml(error.message)}</p>
      </div>
    `;
  }
}

/**
 * Render entries list.
 * @param {HTMLElement} container - Container element
 * @param {Object} result - API result
 */
function renderEntriesList(container, result) {
  if (!result.entries?.length) {
    container.innerHTML = renderEmptyState(
      'No Entries',
      'No logbook entries found. Try adjusting your filters.'
    );
    return;
  }

  let html = `
    <div class="results-header">
      <span class="results-count">
        <strong>${result.total}</strong> total entries
        <span class="text-muted">(page ${result.page} of ${result.total_pages})</span>
      </span>
    </div>
    <div class="results-list">
  `;

  result.entries.forEach(entry => {
    html += renderEntryCard(entry);
  });

  html += '</div>';

  // Pagination
  if (result.total_pages > 1) {
    html += renderPagination(result.page, result.total_pages);
  }

  container.innerHTML = html;
}

/**
 * Render pagination controls.
 * @param {number} currentPage - Current page
 * @param {number} totalPages - Total pages
 * @returns {string} HTML string
 */
function renderPagination(currentPage, totalPages) {
  let html = '<div class="pagination" style="display: flex; justify-content: center; gap: 8px; margin-top: 24px;">';

  if (currentPage > 1) {
    html += `<button class="btn btn-secondary btn-sm" onclick="window.app.loadEntriesPage(${currentPage - 1})">Previous</button>`;
  }

  html += `<span class="text-muted" style="padding: 8px;">Page ${currentPage} of ${totalPages}</span>`;

  if (currentPage < totalPages) {
    html += `<button class="btn btn-secondary btn-sm" onclick="window.app.loadEntriesPage(${currentPage + 1})">Next</button>`;
  }

  html += '</div>';
  return html;
}

/**
 * Show entry detail view.
 * @param {string} entryId - Entry ID
 */
export async function showEntry(entryId) {
  const modal = document.getElementById('entry-modal');
  const modalBody = document.getElementById('entry-modal-body');

  if (!modal || !modalBody) return;

  // Show modal with loading state
  modal.classList.remove('hidden');
  modalBody.innerHTML = renderLoading('Loading entry...');

  try {
    const entry = await entriesApi.get(entryId);
    currentEntry = entry;
    renderEntryDetail(modalBody, entry);
  } catch (error) {
    console.error('Failed to load entry:', error);
    modalBody.innerHTML = `
      <div class="empty-state">
        <h3 class="empty-state-title text-error">Failed to Load Entry</h3>
        <p class="empty-state-text">${escapeHtml(error.message)}</p>
      </div>
    `;
  }
}

/**
 * Render entry detail view.
 * @param {HTMLElement} container - Container element
 * @param {Object} entry - Entry data
 */
function renderEntryDetail(container, entry) {
  const metadata = entry.metadata || {};
  const keywords = entry.keywords || [];
  const attachments = entry.attachments || [];

  // Parse raw_text for subject and details
  const rawText = entry.raw_text || '';
  const lines = rawText.split('\n');
  const subject = lines[0] || 'Untitled';
  const details = lines.slice(1).join('\n').trim() || rawText;

  container.innerHTML = `
    <div class="entry-detail">
      <div class="entry-detail-header">
        <h2 class="entry-detail-title">${escapeHtml(subject)}</h2>
        <div class="entry-detail-meta">
          <span class="entry-id font-mono text-amber">${escapeHtml(entry.entry_id)}</span>
          <span class="timestamp font-mono">${formatTimestamp(entry.timestamp)}</span>
          <span>${escapeHtml(entry.author || 'Unknown')}</span>
          <span class="text-muted">${escapeHtml(entry.source_system)}</span>
        </div>
      </div>

      <div class="entry-detail-grid">
        <div class="entry-detail-main">
          <div class="entry-detail-content">
            <h3>Content</h3>
            <div class="entry-detail-text">${escapeHtml(details)}</div>
          </div>

          ${attachments.length > 0 ? `
            <div class="entry-detail-content" style="margin-top: 24px;">
              <h3>Attachments (${attachments.length})</h3>
              <div style="display: flex; flex-wrap: wrap; gap: 16px;">
                ${attachments.map(att => {
                  const image = isImageAttachment(att);
                  const url = att.url || '#';
                  const escapedUrl = escapeHtml(url);
                  const escapedName = escapeHtml(att.filename || 'attachment');
                  if (image) {
                    return `
                    <div class="card" style="width: 150px; cursor: pointer;"
                         onclick="window.app.showImageLightbox('${escapedUrl}', '${escapedName}')">
                      <div class="card-body" style="padding: 12px; text-align: center;">
                        <img src="${escapedUrl}" alt="${escapedName}"
                             style="width: 126px; height: 100px; object-fit: cover; border-radius: 4px; margin-bottom: 8px;"
                             onerror="this.outerHTML='<div style=&quot;font-size: 32px; margin-bottom: 8px;&quot;>\u{1F4CE}</div>'">
                        <div class="truncate text-sm">${escapedName}</div>
                        <div class="text-xs text-muted">${escapeHtml(att.type || 'image')}</div>
                      </div>
                    </div>`;
                  }
                  return `
                  <a href="${escapedUrl}" target="_blank" rel="noopener"
                     class="card" style="width: 150px; text-decoration: none; color: inherit; cursor: pointer;">
                    <div class="card-body" style="padding: 12px; text-align: center;">
                      <div style="font-size: 32px; margin-bottom: 8px;">\u{1F4CE}</div>
                      <div class="truncate text-sm">${escapedName}</div>
                      <div class="text-xs text-muted">${escapeHtml(att.type || 'file')}</div>
                    </div>
                  </a>`;
                }).join('')}
              </div>
            </div>
          ` : ''}

          ${entry.summary ? `
            <div class="entry-detail-content" style="margin-top: 24px;">
              <h3>AI Summary</h3>
              <div class="text-secondary">${escapeHtml(entry.summary)}</div>
            </div>
          ` : ''}
        </div>

        <div class="entry-detail-sidebar">
          <div class="metadata-card">
            <h4>Metadata</h4>
            <div class="metadata-list">
              <div class="metadata-item">
                <span class="metadata-label">ID</span>
                <span class="metadata-value">${escapeHtml(entry.entry_id)}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Source</span>
                <span class="metadata-value">${escapeHtml(entry.source_system)}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Author</span>
                <span class="metadata-value">${escapeHtml(entry.author || 'Unknown')}</span>
              </div>
              <div class="metadata-item">
                <span class="metadata-label">Timestamp</span>
                <span class="metadata-value">${formatTimestamp(entry.timestamp)}</span>
              </div>
              ${metadata.logbook ? `
                <div class="metadata-item">
                  <span class="metadata-label">Logbook</span>
                  <span class="metadata-value">${escapeHtml(metadata.logbook)}</span>
                </div>
              ` : ''}
              ${metadata.shift ? `
                <div class="metadata-item">
                  <span class="metadata-label">Shift</span>
                  <span class="metadata-value">${escapeHtml(metadata.shift)}</span>
                </div>
              ` : ''}
            </div>
          </div>

          ${keywords.length > 0 ? `
            <div class="metadata-card">
              <h4>Keywords</h4>
              <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                ${renderTags(keywords, 'accent')}
              </div>
            </div>
          ` : ''}

          ${metadata.session_metadata ? (() => {
            const sm = metadata.session_metadata;
            const fields = [
              ['Operator', sm.operator],
              ['Session', sm.session_id, true],
              ['Branch', sm.git_branch, true],
              ['Model', sm.model_name || sm.model],
              ['Source', sm.created_via],
              ['Started', sm.session_start_time],
            ].filter(([, v]) => v);
            return fields.length > 0 ? `
              <div class="metadata-card">
                <h4>Session Context</h4>
                <div class="metadata-list">
                  ${fields.map(([label, value, mono]) => `
                    <div class="metadata-item">
                      <span class="metadata-label">${escapeHtml(label)}</span>
                      <span class="metadata-value${mono ? ' font-mono' : ''}">${escapeHtml(String(value))}</span>
                    </div>
                  `).join('')}
                </div>
              </div>
            ` : '';
          })() : ''}
        </div>
      </div>
    </div>
  `;
}

/**
 * Close entry detail modal.
 */
export function closeEntryModal() {
  const modal = document.getElementById('entry-modal');
  modal?.classList.add('hidden');
  currentEntry = null;
}

/**
 * Handle entry creation form submission.
 * @param {Event} e - Submit event
 */
async function handleCreateEntry(e) {
  e.preventDefault();

  const form = e.target;
  const submitBtn = form.querySelector('button[type="submit"]');
  const originalText = submitBtn?.textContent;

  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Saving...';
  }

  try {
    const formData = new FormData(form);
    const tags = Array.from(document.querySelectorAll('#entry-tags .tag'))
      .map(t => t.dataset.value);

    const entryData = {
      subject: formData.get('subject'),
      details: formData.get('details'),
      author: formData.get('author'),
      logbook: formData.get('logbook'),
      shift: formData.get('shift'),
      tags,
      metadata: draftMetadata,
    };

    const fileInput = document.getElementById('entry-files');
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
    document.getElementById('entry-tags').innerHTML = '';
    if (preview) preview.innerHTML = '';
    draftMetadata = null;
    const sessionPanel = document.getElementById('session-info-panel');
    if (sessionPanel) sessionPanel.remove();
    const draftBanner = document.getElementById('draft-banner');
    if (draftBanner) draftBanner.remove();

    // Navigate to entry
    window.app.showEntry(result.entry_id);

  } catch (error) {
    console.error('Failed to create entry:', error);
    alert(`Failed to create entry: ${error.message}`);
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
function handleTagInput(e) {
  if (e.key === 'Enter' || e.key === ',') {
    e.preventDefault();
    const input = e.target;
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
 * Handle file input change to show preview thumbnails.
 * @param {Event} e - Change event
 */
function handleFilePreview(e) {
  const container = document.getElementById('file-preview');
  if (!container) return;
  container.innerHTML = '';

  const files = e.target.files;
  if (!files || files.length === 0) return;

  for (const file of files) {
    const card = document.createElement('div');
    card.className = 'card';
    card.style.cssText = 'width: 120px; text-align: center; padding: 8px;';

    if (file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.style.cssText = 'width: 100px; height: 80px; object-fit: cover; border-radius: 4px;';
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);
      card.appendChild(img);
    } else {
      const icon = document.createElement('div');
      icon.style.cssText = 'font-size: 32px; margin: 8px 0;';
      icon.textContent = '\u{1F4CE}';
      card.appendChild(icon);
    }

    const name = document.createElement('div');
    name.className = 'truncate text-xs';
    name.textContent = file.name;
    card.appendChild(name);

    const size = document.createElement('div');
    size.className = 'text-xs text-muted';
    size.textContent = formatFileSize(file.size);
    card.appendChild(size);

    container.appendChild(card);
  }
}

/**
 * Format file size in human-readable format.
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted size
 */
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Show a lightbox overlay for an image attachment.
 * @param {string} url - Image URL
 * @param {string} filename - Display filename
 */
export function showImageLightbox(url, filename) {
  // Remove existing lightbox if any
  const existing = document.getElementById('image-lightbox');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'image-lightbox';
  overlay.style.cssText =
    'position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: flex; ' +
    'flex-direction: column; align-items: center; justify-content: center; z-index: 10000; ' +
    'cursor: pointer;';

  overlay.innerHTML = `
    <img src="${escapeHtml(url)}" alt="${escapeHtml(filename)}"
         style="max-width: 90vw; max-height: 80vh; object-fit: contain; border-radius: 8px; cursor: default;"
         onclick="event.stopPropagation()"
         onerror="this.outerHTML='<div style=&quot;color:#fff;font-size:1.2rem;&quot;>Failed to load image</div>'">
    <div style="margin-top: 16px; display: flex; align-items: center; gap: 16px;">
      <span style="color: #ccc; font-size: 0.9rem;">${escapeHtml(filename)}</span>
      <a href="${escapeHtml(url)}" target="_blank" rel="noopener"
         style="color: var(--amber, #f59e0b); font-size: 0.85rem; text-decoration: none;"
         onclick="event.stopPropagation()">Open in new tab &#x2197;</a>
    </div>
  `;

  overlay.addEventListener('click', () => overlay.remove());
  document.addEventListener('keydown', function onKey(e) {
    if (e.key === 'Escape') {
      overlay.remove();
      document.removeEventListener('keydown', onKey);
    }
  });

  document.body.appendChild(overlay);
}

/**
 * Get current entry.
 * @returns {Object|null} Current entry or null
 */
export function getCurrentEntry() {
  return currentEntry;
}

/**
 * Render a read-only panel showing session metadata from a draft.
 * @param {Object} meta - session_metadata object
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
    const subjectInput = document.querySelector('#create-entry-form [name="subject"]');
    const detailsInput = document.querySelector('#create-entry-form [name="details"]');
    const authorInput = document.querySelector('#create-entry-form [name="author"]');
    const logbookSelect = document.querySelector('#create-entry-form [name="logbook"]');
    const shiftSelect = document.querySelector('#create-entry-form [name="shift"]');

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
      draft.tags.forEach(tag => addTag(tag));
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

          const card = document.createElement('div');
          card.className = 'card';
          card.style.cssText = 'width: 120px; text-align: center; padding: 8px;';

          const isImage = /\.(png|jpe?g|gif|webp|svg)$/i.test(filename);
          if (isImage) {
            const img = document.createElement('img');
            img.style.cssText = 'width: 100px; height: 80px; object-fit: cover; border-radius: 4px;';
            img.src = url;
            img.alt = filename;
            card.appendChild(img);
          } else {
            const icon = document.createElement('div');
            icon.style.cssText = 'font-size: 32px; margin: 8px 0;';
            icon.textContent = '\u{1F4CE}';
            card.appendChild(icon);
          }

          const name = document.createElement('div');
          name.className = 'truncate text-xs';
          name.textContent = filename;
          card.appendChild(name);

          preview.appendChild(card);
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
        'padding: 8px 12px; margin-bottom: 12px; border-left: 3px solid var(--amber); font-size: 0.85rem;';
      banner.textContent = 'Draft loaded from Claude \u2014 review and submit';
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

export default {
  initEntries,
  loadEntries,
  showEntry,
  closeEntryModal,
  getCurrentEntry,
  loadDraft,
  showImageLightbox,
};
