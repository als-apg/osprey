/**
 * ARIEL Web Application
 *
 * Main application entry point and routing.
 */

import { capabilitiesApi } from './api.js';
import { initSearch, performSearch, clearSearch } from './search.js';
import { initEntries, loadEntries, showEntry, closeEntryModal, loadDraft, showImageLightbox } from './entries.js';
import { initDashboard, loadStatus, startAutoRefresh, stopAutoRefresh } from './dashboard.js';
import { initAdvancedOptions } from './advanced-options.js';
import { initDrawers } from './drawer.js';
import { initSettings, loadConfig } from './settings.js';
import { loadFileList } from './claude-setup.js';

// Current view
let currentView = 'search';

/**
 * Initialize the application.
 */
async function init() {
  // Embedded mode — hide logo when loaded inside web terminal iframe
  const params = new URLSearchParams(window.location.search);
  if (params.get('embedded') === 'true') {
    document.body.classList.add('embedded');
  }

  // Initialize modules — wrapped in try/catch so navigation always works
  // even if the backend is unavailable (degraded mode).
  try {
    let capabilities = null;
    try {
      capabilities = await capabilitiesApi.get();
    } catch (e) {
      console.warn('Failed to fetch capabilities, using fallback:', e);
    }

    initSearch();
    initEntries();
    initDashboard();
    initAdvancedOptions(capabilities);
    initDrawers();
    initSettings();
  } catch (e) {
    console.error('Module initialization failed (non-fatal):', e);
  }

  // Navigation and routing must always run
  setupNavigation();
  setupModals();

  const hash = window.location.hash.slice(1) || 'search';
  navigateTo(hash);

  // Expose app API to window for onclick handlers
  window.app = {
    navigateTo,
    performSearch,
    clearSearch,
    showEntry,
    closeEntryModal,
    showImageLightbox,
    loadEntriesPage: (page) => loadEntries({ page }),
    loadStatus,
  };

  console.log('ARIEL Web Interface initialized');
}

/**
 * Set up navigation handling.
 */
function setupNavigation() {
  // Handle nav link clicks
  document.querySelectorAll('.nav-link[data-view]').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const view = link.dataset.view;
      navigateTo(view);
    });
  });

  // Handle hash changes
  window.addEventListener('hashchange', () => {
    const hash = window.location.hash.slice(1) || 'search';
    navigateTo(hash);
  });
}

/**
 * Set up modal close handlers.
 */
function setupModals() {
  // Entry modal
  const entryModal = document.getElementById('entry-modal');
  const entryModalClose = document.getElementById('entry-modal-close');

  entryModalClose?.addEventListener('click', () => closeEntryModal());

  // Close on overlay click
  entryModal?.addEventListener('click', (e) => {
    if (e.target === entryModal) {
      closeEntryModal();
    }
  });

  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!entryModal?.classList.contains('hidden')) {
        closeEntryModal();
      }
    }
  });
}

/**
 * Parse a hash string into view name and query parameters.
 * E.g. "create?draft=abc" -> { viewName: "create", params: URLSearchParams }
 * @param {string} hash - Hash string (without leading #)
 * @returns {{ viewName: string, params: URLSearchParams }}
 */
function parseHash(hash) {
  const qIdx = hash.indexOf('?');
  if (qIdx === -1) {
    return { viewName: hash, params: new URLSearchParams() };
  }
  return {
    viewName: hash.substring(0, qIdx),
    params: new URLSearchParams(hash.substring(qIdx + 1)),
  };
}

/**
 * Navigate to a view.
 * @param {string} hash - Full hash string (view name, optionally with query params)
 */
function navigateTo(hash) {
  const { viewName, params } = parseHash(hash);

  // Update URL hash
  window.location.hash = hash;

  // Update nav links (match on view name only, not query params)
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.view === viewName);
  });

  // Hide all views
  document.querySelectorAll('.view').forEach(v => {
    v.classList.remove('active');
  });

  // Show target view
  const viewEl = document.getElementById(`view-${viewName}`);
  if (viewEl) {
    viewEl.classList.add('active');
  }

  // Handle view-specific initialization
  const viewChanged = viewName !== currentView;

  if (viewChanged) {
    // Cleanup previous view
    if (currentView === 'status') {
      stopAutoRefresh();
    }
  }

  // Initialize new view (or reload draft for same-view navigation)
  switch (viewName) {
    case 'browse':
      if (viewChanged) loadEntries();
      break;
    case 'create': {
      const draftId = params.get('draft');
      if (draftId) {
        loadDraft(draftId);
      }
      break;
    }
    case 'status':
      if (viewChanged) {
        loadStatus();
        startAutoRefresh();
      }
      break;
  }

  currentView = viewName;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

export { navigateTo, currentView };
