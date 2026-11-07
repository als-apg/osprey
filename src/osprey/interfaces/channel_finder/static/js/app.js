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
import { mountAllPvs, unmountAllPvs } from './allpvs.js';
import { mountPvPage, unmountPvPage } from './pv-page.js';
import { mountIocPage, unmountIocPage } from './ioc-page.js';
import { refreshStatsBadges, renderProvenance } from './stats-badges.js';

// Standalone, this page owns its own theme chrome (the header
// <osprey-display-menu>), so it runs theme-manager.js in the hub role:
// persistence, OS auto-follow and ?theme= handling all come with it, and
// broadcast is a structural no-op on a page with no iframes. Embedded in the
// Web Terminal hub it is a follower instead: theme-boot.js already applied
// data-theme pre-paint, and this attaches the postMessage listener for the
// hub's live broadcasts.
initTheme({ role: isEmbedded() ? 'follower' : 'hub' });

applyEmbedded();

/**
 * Route parameters: `name` for pv and ioc, `q` for allpvs when the route names
 * words.
 * @typedef {{ name?: string, q?: string }} RouteParams
 */

/** @typedef {{ view: string, params: RouteParams }} Route */

/** @typedef {{ mount: (container: HTMLElement, params: RouteParams) => void, unmount: () => void }} ViewModule */

/** @type {Record<string, ViewModule>} */
const VIEWS = {
  explore: { mount: mountExplore, unmount: unmountExplore },
  feedback: { mount: mountFeedback, unmount: unmountFeedback },
  allpvs: { mount: mountAllPvs, unmount: unmountAllPvs },
  pv: { mount: mountPvPage, unmount: unmountPvPage },
  ioc: { mount: mountIocPage, unmount: unmountIocPage },
};

// Views that exist only while /api/info carries the directory descriptor.
const DIRECTORY_VIEWS = new Set(['allpvs', 'pv', 'ioc']);

/** @type {string|null} */
let currentView = null;
// The view and params mounted now; a route equal to it is not re-mounted.
/** @type {string|null} */
let currentRouteKey = null;
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
    state.directory = info.directory ?? null;
    renderProvenance();
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
    if (id === 'view') {
      if (value && viewAvailable(value)) location.hash = value;
      return;
    }
    state.viewHeader?.onAction(id, value);
  });
  state.on('viewHeaderChanged', () => contributeViewNav());
  state.on('directoryLabel', () => {
    renderHeaderNav();
    contributeViewNav();
  });

  // Route to initial view
  routeFromHash();
  window.addEventListener('hashchange', routeFromHash);
}

// ---- Routing ----

/**
 * Parse a location hash: a bare view name (`#explore`), or a route of the Deep
 * links form (`#/pv/<name>`, `#/ioc/<name>`, `#/allpvs?q=<words>`), where `<name>`
 * is one percent-encoded segment. The words are decoded the same way and not by
 * URLSearchParams, because a `+` in them is a plus. Whether the view exists is
 * not judged here: only a malformed route answers null.
 * @param {string} hash
 * @returns {Route|null}
 */
export function parseRoute(hash) {
  const route = hash.replace(/^#\/?/, '');
  const queryAt = route.indexOf('?');
  const path = queryAt < 0 ? route : route.slice(0, queryAt);
  const query = queryAt < 0 ? '' : route.slice(queryAt + 1);
  const [view, ...rest] = path.split('/');
  try {
    if (view === 'pv' || view === 'ioc') {
      if (rest.length !== 1 || !rest[0]) return null;
      return { view, params: { name: decodeURIComponent(rest[0]) } };
    }
    if (rest.length) return null;
    const words = query.split('&').find((part) => part.startsWith('q='));
    if (view === 'allpvs' && words !== undefined) {
      return { view, params: { q: decodeURIComponent(words.slice(2)) } };
    }
    return { view: view || 'explore', params: {} };
  } catch {
    return null; // malformed percent-encoding
  }
}

/**
 * @param {string} view
 * @returns {boolean}
 */
function viewAvailable(view) {
  if (!Object.hasOwn(VIEWS, view)) return false;
  return !DIRECTORY_VIEWS.has(view) || state.directory !== null;
}

function routeFromHash() {
  const route = parseRoute(location.hash);
  if (route && viewAvailable(route.view)) activateView(route.view, route.params);
  else activateView('explore', {});
}

/**
 * @param {string} viewName
 * @param {RouteParams} params
 */
function activateView(viewName, params) {
  const routeKey = JSON.stringify([viewName, params]);
  if (currentRouteKey === routeKey) return;

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
  document.querySelectorAll('.nav-link').forEach((link) => {
    link.classList.toggle('active', /** @type {HTMLElement} */ (link).dataset.view === viewName);
  });

  // Mount new view
  currentView = viewName;
  currentRouteKey = routeKey;
  state.setActiveView(viewName);
  VIEWS[viewName].mount(container, params);
  contributeViewNav();
}

// ---- Navigation ----

function setupNav() {
  // On the nav and not on its links, which renderHeaderNav rewrites.
  document.querySelector('.header-nav')?.addEventListener('click', (e) => {
    const link = /** @type {HTMLElement} */ (e.target).closest('.nav-link');
    if (!link) return;
    e.preventDefault();
    const view = /** @type {HTMLElement} */ (link).dataset.view;
    if (view) location.hash = view;
  });
  renderHeaderNav();
}

/**
 * The panel's tabs: the scope switch when /api/info carries the directory
 * descriptor, with Curated being whichever paradigm runs, and today's tabs
 * otherwise.
 * @returns {{ id: string, label: string }[]}
 */
function viewTabs() {
  if (!state.directory) {
    return [
      { id: 'explore', label: 'Explore' },
      { id: 'feedback', label: 'Feedback' },
    ];
  }
  return [
    { id: 'explore', label: 'Curated' },
    { id: 'allpvs', label: state.directory.label },
    { id: 'feedback', label: 'Feedback' },
  ];
}

/**
 * Write the tabs into the standalone `.header-nav`. Unchanged tabs are left
 * alone, so an All PVs page repeating its label keeps the focus on the tab.
 */
function renderHeaderNav() {
  const nav = document.querySelector('.header-nav');
  if (!nav) return;
  const html = viewTabs()
    .map(({ id, label }) => {
      const active = id === currentView ? ' active' : '';
      return `<a href="#${id}" class="nav-link${active}" data-view="${id}">${esc(label)}</a>`;
    })
    .join('');
  if (nav.innerHTML !== html) nav.innerHTML = html;
}

/**
 * Publish the tabs into the hub's tile bar, marking the current view active,
 * behind whatever the active view adds of its own (the
 * graph finder's search box and ⋯ menu). The view's items come first: they
 * exist only in one view, and the header contract puts conditional items
 * before the persistent switcher so the switcher never slides. Sent whole on
 * every change — the hub renders only the latest contribution. No-op standalone, where `.header-nav` carries the same tabs.
 */
function contributeViewNav() {
  contributeHeader([
    ...(state.viewHeader?.items ?? []),
    {
      kind: 'nav',
      id: 'view',
      items: viewTabs().map((tab) => ({ ...tab, active: currentView === tab.id })),
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

  dropdown.innerHTML = available
    .map((pt) => {
      const active = pt === state.pipelineType ? ' active' : '';
      return `<button class="pipeline-dropdown-item${active}" data-pipeline="${esc(pt)}">${esc(PIPELINE_LABELS[pt] || pt)}</button>`;
    })
    .join('');

  // Toggle dropdown on badge click
  badge.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('open');
  });

  // Close on outside click
  document.addEventListener('click', () => dropdown.classList.remove('open'));

  // Switch pipeline on item click
  dropdown.querySelectorAll('.pipeline-dropdown-item').forEach((btn) => {
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
    state.setGraphInfo(info.tools, info.graph_store);
    state.directory = info.directory ?? null;
    renderHeaderNav();
    renderProvenance();
    updatePipelineBadge(info.pipeline_type);

    // Update dropdown active state
    document.querySelectorAll('.pipeline-dropdown-item').forEach((btn) => {
      btn.classList.toggle(
        'active',
        /** @type {HTMLElement} */ (btn).dataset.pipeline === info.pipeline_type
      );
    });

    // Refresh stats and re-mount current view
    refreshStatsBadges();
    currentRouteKey = null; // Force re-mount
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
