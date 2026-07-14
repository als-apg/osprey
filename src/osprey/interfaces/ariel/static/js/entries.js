// @ts-check
/**
 * ARIEL Entries Module
 *
 * Entry browsing, detail view, and creation.
 */

import { entriesApi } from './api.js';
import {
  renderEntryCard,
  renderLoading,
  renderEmptyState,
  renderErrorState,
} from './components.js';
import { showEntry, closeEntryModal, showImageLightbox, getCurrentEntry, initEntryDetail } from './entries-detail.js';
import { handleCreateEntry, handleTagInput, handleFilePreview, loadDraft, initEntryTags } from './entries-form.js';

// Re-export the detail-view and form public surface — app.js and window.app
// import these from entries.js, so this module stays their single point of entry.
export { showEntry, closeEntryModal, showImageLightbox, getCurrentEntry };
export { loadDraft };

/**
 * Wire up delegated click handling for entry cards and pagination.
 *
 * #entries-list's innerHTML is replaced wholesale on every loadEntries()
 * call, so the listener is delegated on the stable list container
 * (attached once, at init) instead of bound to child elements that get
 * discarded on the next render.
 */
export function initEntriesListDelegation() {
  const entriesList = document.getElementById('entries-list');
  entriesList?.addEventListener('click', (e) => {
    const target = /** @type {HTMLElement} */ (e.target);
    const pageBtn = target.closest('[data-page]');
    if (pageBtn) {
      const page = parseInt(/** @type {HTMLElement} */ (pageBtn).dataset.page ?? '', 10);
      if (!Number.isNaN(page)) loadEntries({ page });
      return;
    }
    const card = target.closest('[data-entry-id]');
    if (card) {
      const entryId = /** @type {HTMLElement} */ (card).dataset.entryId;
      if (entryId) showEntry(entryId);
    }
  });
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

  // Delegated handlers for the entries list, the detail modal, and the tags input.
  initEntriesListDelegation();
  initEntryDetail();
  initEntryTags();

  // Adapt the publishing section to the configured logbook adapter.
  adaptPublishingSection();
}

/**
 * Adapt the "Logbook Publishing" section to the configured adapter.
 *
 * The adapter declares whether publishing needs credentials, so the form shows
 * the credential fields only when they can actually be used and tells the
 * operator what leaving the form will do — instead of fixed, possibly-wrong text.
 */
async function adaptPublishingSection() {
  const helper = document.getElementById('publish-helper');
  const credentials = document.getElementById('publish-credentials');
  if (!helper) return;

  let info;
  try {
    info = await entriesApi.getPublishInfo();
  } catch {
    // Service/DB unavailable — keep the neutral default text.
    return;
  }

  const where = info.source_system ? ` to ${info.source_system}` : '';
  if (!info.supports_write) {
    helper.textContent = 'Entries are saved to ARIEL only — this logbook is read-only.';
    if (credentials) credentials.style.display = 'none';
  } else if (info.requires_auth) {
    helper.textContent = `Enter your logbook credentials to publish${where}.`;
    if (credentials) credentials.style.display = '';
  } else {
    helper.textContent = `Publishes${where} — no credentials required.`;
    if (credentials) credentials.style.display = 'none';
  }
}

/**
 * Load and display entry list.
 * @param {any} params - List parameters
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
    container.innerHTML = renderErrorState('Failed to Load Entries', error);
  }
}

/**
 * Render entries list.
 * @param {HTMLElement} container - Container element
 * @param {any} result - API result
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

  result.entries.forEach((/** @type {any} */ entry) => {
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
    html += `<button class="btn btn-secondary btn-sm" data-page="${currentPage - 1}">Previous</button>`;
  }

  html += `<span class="text-muted" style="padding: 8px;">Page ${currentPage} of ${totalPages}</span>`;

  if (currentPage < totalPages) {
    html += `<button class="btn btn-secondary btn-sm" data-page="${currentPage + 1}">Next</button>`;
  }

  html += '</div>';
  return html;
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
