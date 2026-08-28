// @ts-check
/**
 * OSPREY Channel Finder — Application Entry Point
 *
 * Hash routing, module initialization, nav tab management.
 */

import { initTheme } from '/design-system/js/theme-manager.js';
import { applyEmbedded, isEmbedded } from '/design-system/js/frame-params.js';
import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import '/design-system/js/components/osprey-display-menu.js';
import { fetchJSON, putJSON } from './api.js';
import { esc, messageOf } from './utils.js';
import { state } from './state.js';
import { mountExplore, unmountExplore } from './explore.js';
import { mountFeedback, unmountFeedback } from './feedback.js';
import { refreshStatsBadges } from './stats-badges.js';

// Standalone, this page owns its own theme chrome (the header
// <osprey-display-menu>), so it runs theme-manager.js in the hub role:
// persistence, OS auto-follow and ?theme= handling all come with it, and
// broadcast is a structural no-op on a page with no iframes. Embedded in the
// Web Terminal hub it is a follower instead: theme-boot.js already applied
// data-theme pre-paint, and this attaches the postMessage listener for the
// hub's live broadcasts.
initTheme({ role: isEmbedded() ? 'follower' : 'hub' });

applyEmbedded();

// Live Expert<->Simple switch broadcast by the hub's header toggle. The
// pre-paint rung (mode-boot.js) already set the initial data-ui-mode; this is
// the runtime flip. Coerce anything non-"simple" to "expert", re-stamp the
// attribute (CSS deltas key off it), then re-mount the active view so the
// in-context renderer repaints its plain cards <-> dense table. Mirrors the
// ARIEL panel's app.js listener.
window.addEventListener('message', (e) => {
  if (e.origin !== window.location.origin) return;
  if (e.data && e.data.type === 'osprey-mode-change' && e.data.mode) {
    const mode = e.data.mode === 'simple' ? 'simple' : 'expert';
    document.documentElement.setAttribute('data-ui-mode', mode);
    remountCurrentView();
  }
});

/** @typedef {{ mount: (container: HTMLElement) => void, unmount: () => void }} ViewModule */

/** @type {Record<string, ViewModule>} */
const VIEWS = {
  explore:  { mount: mountExplore,  unmount: unmountExplore },
  feedback: { mount: mountFeedback, unmount: unmountFeedback },
};

/** @type {string|null} */
let currentView = null;
let _initialized = false;

// ---- Initialization ----

async function init() {
  if (_initialized) return;
  _initialized = true;
  // Load pipeline info
  try {
    const info = await fetchJSON('/api/info');
    state.setPipelineInfo(info.pipeline_type, info.metadata);
    state.availablePipelines = info.available_pipelines || [info.pipeline_type];
    state.dbPath = info.db_path || null;
    state.setGraphInfo(info.tools, info.graph_store);
    updatePipelineBadge(info.pipeline_type);
    buildPipelineDropdown();
  } catch (e) {
    console.error('Failed to load pipeline info:', e);
    showToast('Failed to connect to API', 'error');
  }

  // Load stats badges
  refreshStatsBadges();

  // Set up navigation
  setupNav();
  onHeaderAction((id, value) => {
    if (id === 'view' && value && VIEWS[value]) location.hash = value;
  });

  // Route to initial view
  routeFromHash();
  window.addEventListener('hashchange', routeFromHash);
}

// ---- Routing ----

function routeFromHash() {
  const hash = location.hash.replace('#', '') || 'explore';
  const view = VIEWS[hash] ? hash : 'explore';
  activateView(view);
}

/**
 * @param {string} viewName
 */
function activateView(viewName) {
  if (currentView === viewName) return;

  const container = document.getElementById('view-container');
  if (!container) return;

  // Unmount previous view
  if (currentView) {
    const prev = VIEWS[currentView];
    if (prev) prev.unmount();
  }

  // Clear container
  container.innerHTML = '';

  // Update nav links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', /** @type {HTMLElement} */ (link).dataset.view === viewName);
  });

  // Mount new view
  currentView = viewName;
  state.setActiveView(viewName);
  VIEWS[viewName].mount(container);
  contributeViewNav();
}

/** Force a re-mount of the active view (e.g. after a live UI-mode flip). */
function remountCurrentView() {
  if (!currentView) return;
  const view = currentView;
  currentView = null;
  activateView(view);
}

// ---- Navigation ----

function setupNav() {
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const view = /** @type {HTMLElement} */ (link).dataset.view;
      if (view) location.hash = view;
    });
  });
}

/**
 * Publish the Explore/Feedback tabs into the hub's tile bar, marking the
 * current view active. Sent whole on every view change — the hub renders
 * only the latest contribution. No-op standalone, where `.header-nav`
 * carries the same tabs; Simple mode hides those tabs there, so it
 * contributes nothing here either (a mode flip re-mounts, so this re-runs).
 */
function contributeViewNav() {
  if (document.documentElement.getAttribute('data-ui-mode') === 'simple') {
    contributeHeader([]);
    return;
  }
  contributeHeader([
    {
      kind: 'nav',
      id: 'view',
      items: [
        { id: 'explore', label: 'Explore', active: currentView === 'explore' },
        { id: 'feedback', label: 'Feedback', active: currentView === 'feedback' },
      ],
    },
  ]);
}

// ---- Pipeline Switcher ----

/**
 * Header badge / switcher labels, one entry per pipeline type the panel serves.
 * A type with no entry falls back to its raw key upper-cased, so every
 * paradigm the Explore view mounts belongs here. Exported for the dispatch
 * contract test (explore-graph.test.mjs).
 * @type {Record<string, string>}
 */
export const PIPELINE_LABELS = {
  hierarchical: 'HIERARCHICAL',
  in_context: 'IN-CONTEXT',
  middle_layer: 'MIDDLE LAYER',
  graph: 'GRAPH',
};

/**
 * @param {any} type
 */
function updatePipelineBadge(type) {
  const badge = document.getElementById('pipeline-badge');
  if (!badge) return;
  // A null type is the unconfigured project; the badge must still carry text,
  // or the switcher collapses to a zero-size box and vanishes from the layout.
  badge.textContent = (PIPELINE_LABELS[type] || type?.toUpperCase() || 'NOT CONFIGURED') + ' ▾';
}

function buildPipelineDropdown() {
  const dropdown = document.getElementById('pipeline-dropdown');
  const badge = document.getElementById('pipeline-badge');
  if (!dropdown || !badge) return;

  const available = state.availablePipelines || [];
  if (available.length <= 1) {
    // Only one pipeline — no switcher needed, remove caret
    badge.textContent = (badge.textContent ?? '').replace(' ▾', '');
    badge.style.cursor = 'default';
    return;
  }

  dropdown.innerHTML = available.map(pt => {
    const active = pt === state.pipelineType ? ' active' : '';
    return `<button class="pipeline-dropdown-item${active}" data-pipeline="${esc(pt)}">${esc(PIPELINE_LABELS[pt] || pt)}</button>`;
  }).join('');

  // Toggle dropdown on badge click
  badge.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('open');
  });

  // Close on outside click
  document.addEventListener('click', () => dropdown.classList.remove('open'));

  // Switch pipeline on item click
  dropdown.querySelectorAll('.pipeline-dropdown-item').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      dropdown.classList.remove('open');
      const pt = /** @type {HTMLElement} */ (btn).dataset.pipeline;
      if (!pt || pt === state.pipelineType) return;
      await switchPipeline(pt);
    });
  });
}

/**
 * @param {string} newType
 */
async function switchPipeline(newType) {
  try {
    await putJSON('/api/pipeline', { pipeline_type: newType });

    // Re-fetch pipeline info to get fresh metadata
    const info = await fetchJSON('/api/info');
    state.setPipelineInfo(info.pipeline_type, info.metadata);
    state.dbPath = info.db_path || null;
    updatePipelineBadge(info.pipeline_type);

    // Update dropdown active state
    document.querySelectorAll('.pipeline-dropdown-item').forEach(btn => {
      btn.classList.toggle('active', /** @type {HTMLElement} */ (btn).dataset.pipeline === info.pipeline_type);
    });

    // Refresh stats and re-mount current view
    refreshStatsBadges();
    currentView = null;  // Force re-mount
    routeFromHash();

    showToast(`Switched to ${PIPELINE_LABELS[newType] || newType}`, 'success');
  } catch (e) {
    showToast(`Failed to switch pipeline: ${messageOf(e)}`, 'error');
  }
}

// ---- Toast ----

/**
 * @param {string} message
 * @param {string} [type='info']
 */
export function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Re-export for use by explore renderers after CRUD mutations
export { refreshStatsBadges } from './stats-badges.js';

// ---- Boot ----

document.addEventListener('DOMContentLoaded', init);
// Also handle embedded mode where DOMContentLoaded may have already fired
if (document.readyState !== 'loading') init();
