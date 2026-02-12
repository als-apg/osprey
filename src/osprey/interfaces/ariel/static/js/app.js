/**
 * ARIEL Web Application
 *
 * Main application entry point and routing.
 */

import { capabilitiesApi } from './api.js';
import { initSearch, performSearch, clearSearch } from './search.js';
import { initEntries, loadEntries, showEntry, closeEntryModal } from './entries.js';
import { initDashboard, loadStatus, startAutoRefresh, stopAutoRefresh } from './dashboard.js';
import { initAdvancedOptions } from './advanced-options.js';

// Current view
let currentView = 'search';

/**
 * Initialize the application.
 */
async function init() {
  // Fetch capabilities from backend (modes + parameters)
  let capabilities = null;
  try {
    capabilities = await capabilitiesApi.get();
  } catch (e) {
    console.warn('Failed to fetch capabilities, using fallback:', e);
  }

  // Initialize modules
  initSearch();
  initEntries();
  initDashboard();
  initAdvancedOptions(capabilities);

  // Set up navigation
  setupNavigation();

  // Set up modal close handlers
  setupModals();

  // Show initial view
  const hash = window.location.hash.slice(1) || 'search';
  navigateTo(hash);

  // Expose app API to window for onclick handlers
  window.app = {
    navigateTo,
    performSearch,
    clearSearch,
    showEntry,
    closeEntryModal,
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
    if (hash !== currentView) {
      navigateTo(hash);
    }
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
 * Navigate to a view.
 * @param {string} view - View name
 */
function navigateTo(view) {
  // Update URL hash
  window.location.hash = view;

  // Update nav links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.view === view);
  });

  // Hide all views
  document.querySelectorAll('.view').forEach(v => {
    v.classList.remove('active');
  });

  // Show target view
  const viewEl = document.getElementById(`view-${view}`);
  if (viewEl) {
    viewEl.classList.add('active');
  }

  // Handle view-specific initialization
  if (view !== currentView) {
    // Cleanup previous view
    if (currentView === 'status') {
      stopAutoRefresh();
    }

    // Initialize new view
    switch (view) {
      case 'browse':
        loadEntries();
        break;
      case 'status':
        loadStatus();
        startAutoRefresh();
        break;
    }

    currentView = view;
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

export { navigateTo, currentView };
