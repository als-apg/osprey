/**
 * OSPREY Channel Finder — Application Entry Point
 *
 * Hash routing, module initialization, nav tab management.
 */

import { fetchJSON, putJSON } from './api.js';
import { state } from './state.js';
import { mountExplore, unmountExplore } from './explore.js';
import { mountFeedback, unmountFeedback } from './feedback.js';
import { refreshStatsBadges } from './stats-badges.js';

const VIEWS = {
  explore:  { mount: mountExplore,  unmount: unmountExplore },
  feedback: { mount: mountFeedback, unmount: unmountFeedback },
};

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

function activateView(viewName) {
  if (currentView === viewName) return;

  const container = document.getElementById('view-container');
  if (!container) return;

  // Unmount previous view
  if (currentView && VIEWS[currentView]?.unmount) {
    VIEWS[currentView].unmount();
  }

  // Clear container
  container.innerHTML = '';

  // Update nav links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.toggle('active', link.dataset.view === viewName);
  });

  // Mount new view
  currentView = viewName;
  state.setActiveView(viewName);
  VIEWS[viewName].mount(container);
}

// ---- Navigation ----

function setupNav() {
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const view = link.dataset.view;
      location.hash = view;
    });
  });
}

// ---- Pipeline Switcher ----

const PIPELINE_LABELS = {
  hierarchical: 'HIERARCHICAL',
  in_context: 'IN-CONTEXT',
  middle_layer: 'MIDDLE LAYER',
};

function updatePipelineBadge(type) {
  const badge = document.getElementById('pipeline-badge');
  if (!badge) return;
  badge.textContent = (PIPELINE_LABELS[type] || type?.toUpperCase() || '') + ' \u25BE';
}

function buildPipelineDropdown() {
  const dropdown = document.getElementById('pipeline-dropdown');
  const badge = document.getElementById('pipeline-badge');
  if (!dropdown || !badge) return;

  const available = state.availablePipelines || [];
  if (available.length <= 1) {
    // Only one pipeline — no switcher needed, remove caret
    badge.textContent = badge.textContent.replace(' \u25BE', '');
    badge.style.cursor = 'default';
    return;
  }

  dropdown.innerHTML = available.map(pt => {
    const active = pt === state.pipelineType ? ' active' : '';
    return `<button class="pipeline-dropdown-item${active}" data-pipeline="${pt}">${PIPELINE_LABELS[pt] || pt}</button>`;
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
      const pt = btn.dataset.pipeline;
      if (pt === state.pipelineType) return;
      await switchPipeline(pt);
    });
  });
}

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
      btn.classList.toggle('active', btn.dataset.pipeline === info.pipeline_type);
    });

    // Refresh stats and re-mount current view
    refreshStatsBadges();
    currentView = null;  // Force re-mount
    routeFromHash();

    showToast(`Switched to ${PIPELINE_LABELS[newType] || newType}`, 'success');
  } catch (e) {
    showToast(`Failed to switch pipeline: ${e.message}`, 'error');
  }
}

// ---- Toast ----

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
