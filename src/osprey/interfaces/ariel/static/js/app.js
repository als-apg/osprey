// @ts-check
/**
 * ARIEL Web Application
 *
 * Main application entry point and routing.
 */

import { initTheme } from '/design-system/js/theme-manager.js';
import { applyEmbedded, onModeChange } from '/design-system/js/frame-params.js';
import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { capabilitiesApi } from './api.js';
import { initSearch, performSearch, clearSearch, onUiModeChange } from './search.js';
import { initEntries, loadEntries, showEntry, closeEntryModal, loadDraft, showImageLightbox } from './entries.js';
import { initDashboard, loadStatus, startAutoRefresh, stopAutoRefresh } from './dashboard.js';
import { initAdvancedOptions } from './advanced-options.js';
import '/design-system/js/components/osprey-drawer.js';
import '/design-system/js/components/osprey-display-menu.js';
import { initSettings } from './settings.js';

// Embedded mode — drops the panel's own header entirely (the host tile bar is
// the one header) when loaded inside the web terminal iframe. Applied before
// initTheme() below, which needs to know which surface this is.
applyEmbedded();

// Standalone ARIEL owns theming: the display menu is the operator's control
// and its pick must persist (hub role). Embedded, the hub is that control and
// this panel only follows its broadcasts. theme-boot.js already applied
// data-theme pre-paint either way.
initTheme({ role: isEmbedded() ? 'follower' : 'hub' });

// Live Expert<->Simple switch — broadcast by the hub's header toggle when
// embedded, posted by the display menu's View row when standalone. The shared
// receive-side helper stamps data-ui-mode (CSS deltas key off it); the
// callback repaints the current results so plain cards <-> scored cards swap
// without re-running the query. Mirrors the artifacts panel's gallery.js.
onModeChange(() => onUiModeChange());

// Current view
let currentView = 'search';

/**
 * Whether the app runs embedded in the web terminal iframe.
 * Reads the `embedded` body class set by the applyEmbedded() call at module
 * scope above — which is why the theme role can already read it there.
 * @returns {boolean}
 */
function isEmbedded() {
  return document.body.classList.contains('embedded');
}

/**
 * Default view when no hash is present. Search is standalone-only: when
 * embedded in the web terminal, logbook search goes through the agent, so
 * the panel opens on Browse instead.
 * @returns {string}
 */
function defaultView() {
  return isEmbedded() ? 'browse' : 'search';
}

/**
 * Views the tile bar's nav offers when embedded. Search and Status are
 * standalone-only (navigateTo redirects both to browse), so the bar carries
 * the two views an operator actually works in.
 * @type {{ id: string, label: string }[]}
 */
const HEADER_VIEWS = [
  { id: 'browse', label: 'Browse' },
  { id: 'create', label: 'New Entry' },
];

/**
 * Publish this panel's tile-bar contribution for the current view. The hub
 * renders only what the last contribution says, so the WHOLE nav goes out
 * again on every view change. No-op standalone.
 */
function publishHeaderContribution() {
  contributeHeader([
    {
      kind: 'nav',
      id: 'view',
      items: HEADER_VIEWS.map(v => ({ id: v.id, label: v.label, active: v.id === currentView })),
    },
  ]);
}

/**
 * Initialize the application.
 */
async function init() {
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
    initSettings();
  } catch (e) {
    console.error('Module initialization failed (non-fatal):', e);
  }

  // Navigation and routing must always run
  setupNavigation();
  setupModals();
  setupCreateFormReset();

  const hash = window.location.hash.slice(1) || defaultView();
  navigateTo(hash);

  // Expose app API to window for onclick handlers
  /** @type {any} */ (window).app = {
    navigateTo,
    performSearch,
    clearSearch,
    showEntry,
    closeEntryModal,
    showImageLightbox,
    loadEntriesPage: (/** @type {number} */ page) => loadEntries({ page }),
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
      const view = /** @type {HTMLElement} */ (link).dataset.view;
      if (view) navigateTo(view);
    });
  });

  // Handle hash changes
  window.addEventListener('hashchange', () => {
    const hash = window.location.hash.slice(1) || defaultView();
    navigateTo(hash);
  });

  // Embedded: the same nav lives in the host tile bar. The hub posts the
  // clicked entry's id back here and navigateTo re-publishes the active state.
  onHeaderAction((id, value) => {
    if (id === 'view' && value) navigateTo(value);
  });
}

/**
 * Wire the New Entry form's Cancel button: clear the fields and the staged
 * file thumbnails (a native form reset leaves the previews behind, since it
 * fires no change event for handleFilePreview to see).
 */
function setupCreateFormReset() {
  const cancelBtn = document.getElementById('create-cancel-btn');
  cancelBtn?.addEventListener('click', () => {
    /** @type {HTMLFormElement|null} */ (
      document.getElementById('create-entry-form')
    )?.reset();
    const preview = document.getElementById('file-preview');
    if (preview) preview.innerHTML = '';
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

  // Search and Status are standalone-only: embedded users search the logbook
  // through the agent, and the hub's own ARIEL dot reports service health.
  // Redirect stray hashes so neither view can activate (which also keeps the
  // status auto-refresh interval — started only on that activation — off).
  if (isEmbedded() && (viewName === 'search' || viewName === 'status')) {
    navigateTo('browse');
    return;
  }

  // Update URL hash
  window.location.hash = hash;

  // Update nav links (match on view name only, not query params)
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', /** @type {HTMLElement} */ (link).dataset.view === viewName);
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

  publishHeaderContribution();
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

export { navigateTo, currentView };
