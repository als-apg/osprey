// @ts-check
/**
 * ARIEL Entry Detail Module
 *
 * Entry detail view: loading, rendering, and the image lightbox.
 */

import { entriesApi } from './api.js';
import {
  formatTimestamp,
  renderLoading,
  renderTags,
  renderErrorState,
  escapeHtml,
} from './components.js';
import { isImageAttachment, parseEntryText } from './entries-helpers.js';

// Current entry detail
/** @type {any} */
let currentEntry = null;

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
    modalBody.innerHTML = renderErrorState('Failed to Load Entry', error);
  }
}

/**
 * Render entry detail view.
 * @param {HTMLElement} container - Container element
 * @param {any} entry - Entry data
 */
function renderEntryDetail(container, entry) {
  const metadata = entry.metadata || {};
  const keywords = entry.keywords || [];
  const attachments = entry.attachments || [];

  // Parse raw_text for subject and details
  const { subject, details } = parseEntryText(entry.raw_text);

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
                ${attachments.map((/** @type {any} */ att) => {
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
  // A lightbox scrim is conventionally dark regardless of the active site
  // theme (it exists to make the image itself readable), so its backdrop
  // and text use theme-invariant black/white composites rather than
  // themed tokens that would go light-on-light in light mode.
  overlay.style.cssText =
    'position: fixed; inset: 0; background: color-mix(in srgb, black 85%, transparent); display: flex; ' +
    'flex-direction: column; align-items: center; justify-content: center; z-index: 10000; ' +
    'cursor: pointer;';

  overlay.innerHTML = `
    <img src="${escapeHtml(url)}" alt="${escapeHtml(filename)}"
         style="max-width: 90vw; max-height: 80vh; object-fit: contain; border-radius: 8px; cursor: default;"
         onclick="event.stopPropagation()"
         onerror="this.outerHTML='<div style=&quot;color:white;font-size:1.2rem;&quot;>Failed to load image</div>'">
    <div style="margin-top: 16px; display: flex; align-items: center; gap: 16px;">
      <span style="color: silver; font-size: 0.9rem;">${escapeHtml(filename)}</span>
      <a href="${escapeHtml(url)}" target="_blank" rel="noopener"
         style="color: var(--color-amber); font-size: 0.85rem; text-decoration: none;"
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
 * @returns {any} Current entry or null
 */
export function getCurrentEntry() {
  return currentEntry;
}

export default {
  showEntry,
  closeEntryModal,
  showImageLightbox,
  getCurrentEntry,
};
