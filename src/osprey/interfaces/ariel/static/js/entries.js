/**
 * ARIEL Entries Module
 *
 * Entry browsing, detail view, and creation.
 */

import { entriesApi } from './api.js';
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
                ${attachments.map(att => `
                  <div class="card" style="width: 150px;">
                    <div class="card-body" style="padding: 12px; text-align: center;">
                      <div style="font-size: 32px; margin-bottom: 8px;">📎</div>
                      <div class="truncate text-sm">${escapeHtml(att.filename || 'attachment')}</div>
                      <div class="text-xs text-muted">${escapeHtml(att.type || 'file')}</div>
                    </div>
                  </div>
                `).join('')}
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

    const result = await entriesApi.create({
      subject: formData.get('subject'),
      details: formData.get('details'),
      author: formData.get('author'),
      logbook: formData.get('logbook'),
      shift: formData.get('shift'),
      tags,
    });

    // Show success message
    alert(`Entry created: ${result.entry_id}`);

    // Reset form
    form.reset();
    document.getElementById('entry-tags').innerHTML = '';

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
 * Get current entry.
 * @returns {Object|null} Current entry or null
 */
export function getCurrentEntry() {
  return currentEntry;
}

export default {
  initEntries,
  loadEntries,
  showEntry,
  closeEntryModal,
  getCurrentEntry,
};
