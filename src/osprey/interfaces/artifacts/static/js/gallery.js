// @ts-check
/**
 * OSPREY Artifact Gallery — the page entry.
 *
 * Standalone, the page is the browse gallery in every UI mode: the browser
 * column (type tree or activity timeline, filter, ⋯ menu) beside the preview
 * pane and its header.
 *
 * Embedded in the Web Terminal hub (`?embedded=true`), the page is a viewer of
 * ONE artifact. The hub's sidebar owns the artifact list and the hub renders
 * the pane header, so this page wires no browser column (gallery.css hides
 * it), renders the preview without its header, and shows the artifact the hub
 * selects (viewer.js). It opens no event stream: every stream holds one of the
 * browser's six connections to the hub's origin, and the hub already follows
 * the panel's stream, so it forwards the shown artifact's updates and deletes
 * (viewer.js) and acts on the agent's focus itself.
 *
 * Both modes hold the session-scoped artifact list, because the logbook
 * composer's artifact picker reads it. Standalone, the event stream keeps it
 * current; embedded, it is fetched again before the composer opens.
 */
import { getTheme, initTheme, subscribe } from '/design-system/js/theme-manager.js';
import { applyEmbedded, isEmbedded } from '/design-system/js/frame-params.js';
import '/design-system/js/components/osprey-display-menu.js';
import { debounce } from '/design-system/js/dom.js';
import {
  getArtifacts,
  setArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getFocusedArtifact,
  setFocusedArtifact,
  setCurrentSessionId,
  getShowAllSessions,
  setShowAllSessions,
  addArtifact,
  removeArtifact,
  hasMoreArtifacts,
  inCurrentScope,
  fetchArtifacts as fetchArtifactsData,
  fetchMoreArtifacts as fetchMoreArtifactsData,
  fetchFocus,
} from './state.js';
import { initTypeRegistry } from './types.js';
import { createSidebarRenderer, isNearListEnd } from './render.js';
import { initBrowseLayout } from './browse-layout.js';
import { initSidebarMenu } from './sidebar-menu.js';
import { createPreviewRenderer } from './preview.js';
import { mountPickBar } from './pick-bar.js';
import { createViewer } from './viewer.js';
import { printArtifact } from './print.js';
import { openComposeModal } from './logbook.js';
import { renderTimeseriesView, restyleMountedCharts } from './timeseries.js';

// Embedded, the page is the hub's single-artifact viewer (see above).
const viewerOnly = isEmbedded();

// ---- DOM Refs ----

const healthDot = document.getElementById('health-indicator');
const refreshBtn = /** @type {HTMLElement} */ (document.getElementById('refresh-btn'));
const searchInput = /** @type {HTMLInputElement} */ (document.getElementById('search'));
const sidebarBody = /** @type {HTMLElement} */ (document.getElementById('sidebar-body'));
const sidebar = document.getElementById('browse-sidebar');
const resizeHandle = document.getElementById('resize-handle');
const scopePill = /** @type {HTMLElement|null} */ (document.getElementById('scope-pill'));
const orientToggleBtn = document.getElementById('orient-toggle-btn');

// ---- State ----
// artifacts/selectedArtifact/focusedArtifact/currentSessionId/
// showAllSessions live in state.js behind explicit accessors, and
// typeRegistry lives in types.js (behind getTypeRegistry()) — see the
// imports above. browseMode/sidebarLayout live behind sidebarRenderer's
// own accessors (below) — see render.js. isFullscreen/
// newArtifactsSinceFullscreen live behind previewRenderer's own accessors
// (below) — see preview.js. No closure vars here.

// ---- Preview Renderer / Sidebar Renderer ----
// previewRenderer (preview.js) owns renderPreview and the pin/fullscreen/
// focus state; sidebarRenderer (render.js) owns the sidebar
// rendering. Each needs an effect the other one owns (previewRenderer
// triggers a sidebar re-render on delete/pin/fullscreen-exit;
// sidebarRenderer triggers a preview render/focus/fullscreen-enter on
// selection), so they're wired together via injected callbacks in both
// directions. None of those callbacks runs until well after both are
// constructed, so previewRenderer's callbacks may reach sidebarRenderer
// (through renderSidebar) although sidebarRenderer is declared after them.

const previewRenderer = createPreviewRenderer(
  {
    onArtifactDeleted: () => renderSidebar(),
    onPinToggled: () => renderSidebar(),
    onFullscreenExit: () => renderSidebar(),
    onTimeseriesNeeded: (container, artifact) => renderTimeseriesView(container, artifact),
  },
  // Embedded, the hub renders this pane's header (title and actions) itself.
  { header: !viewerOnly }
);

const sidebarRenderer = createSidebarRenderer({
  onSelect: (a) => previewRenderer.setAsFocus(a),
  onPreviewNeeded: () => previewRenderer.renderPreview(),
  onEnterFullscreen: (a) => previewRenderer.enterFullscreen(a),
  onPicksChanged: () => pickBar.render(),
});

/**
 * Re-render the browser column. Every data change goes through here, so the
 * embedded viewer, which never shows that column, skips them all in one place.
 */
function renderSidebar() {
  if (!viewerOnly) sidebarRenderer.renderSidebar();
}

// The bar above the list that deletes the picked rows together. Rendering the
// sidebar re-renders it, so it always follows the picks.
const pickBar = mountPickBar({
  onDeleted: () => {
    renderSidebar();
    previewRenderer.renderPreview();
  },
  onCleared: () => renderSidebar(),
});

// ---- Health / Scope UI ----
// escapeHtml/formatSize/formatTime/formatFullTime/formatDate/openUrl/
// isNewThisSession/requestColorPass/typeBadge/typeColor now
// live in types.js, consumed directly by render.js/preview.js instead of
// through this module; updateHealth/updateScopeUi stay here — they touch
// this module's own top-level DOM refs, not stateless.

/** @param {boolean} ok */
function updateHealth(ok) {
  if (healthDot) healthDot.className = ok ? 'health-dot healthy' : 'health-dot';
}

/**
 * Reflect the all-sessions scope in its indicators: the ⋯-menu checkbox
 * item and the scope pill above the list (visible only while the
 * non-default all-sessions scope is on).
 */
function updateScopeUi() {
  const on = getShowAllSessions();
  const btn = document.getElementById('all-sessions-btn');
  if (btn) btn.setAttribute('aria-checked', String(on));
  if (scopePill) scopePill.hidden = !on;
}

/** Flip the all-sessions scope (the ⋯-menu checkbox item). */
function toggleAllSessions() {
  setShowAllSessions(!getShowAllSessions());
  updateScopeUi();
  fetchArtifacts();
}

// ---- API ----
// state.js owns showErrorBanner/hideErrorBanner/fetchArtifacts/fetchFocus, and its
// fetchArtifacts() renders nothing: the wrapper below supplies the render effects.

/**
 * Fetch the next page while the drawn list is too short to scroll to its
 * end, so a tall sidebar still reaches older artifacts. A hidden sidebar
 * (fullscreen, or the embedded viewer, which never shows it) has no height
 * and fetches nothing.
 */
function fillSidebar() {
  if (sidebarBody.clientHeight > 0) loadMoreIfNearEnd();
}

const listCallbacks = {
  onHealthChange: updateHealth,
  onArtifactsUpdated: () => {
    renderSidebar();
    fillSidebar();
  },
};

/** Fetch the first page for the filter text and the session scope. */
function fetchArtifacts() {
  return fetchArtifactsData({ ...listCallbacks, search: searchInput.value.trim().toLowerCase() });
}

function loadMoreIfNearEnd() {
  if (hasMoreArtifacts() && isNearListEnd(sidebarBody)) fetchMoreArtifactsData(listCallbacks);
}

/**
 * Fetch one artifact the held pages do not include. An entry inside the
 * current session scope is held from then on; one outside it is returned but never held, because the
 * held list is what the scope admits.
 * @param {string} id
 * @returns {Promise<any|null>} the held copy, or the entry, or null when the server has none
 */
async function fetchEntry(id) {
  const resp = await fetch(`/api/artifacts/${encodeURIComponent(id)}`);
  if (!resp.ok) return null;
  const entry = await resp.json();
  if (!inCurrentScope(entry)) return entry;
  addArtifact(entry);
  return getArtifacts().find((a) => a.id === id) || entry;
}

// ---- Browser Column (standalone only) ----

/**
 * Wire the browser column: the filter, the list's scroll paging, the
 * Types/Activity and list/gallery pairs, the keyboard shortcuts, the browse
 * layout and its splitters, the ⋯ menu with the items it holds, and the
 * scope pill. The embedded viewer has none of it: the hub's sidebar is its
 * list.
 */
function wireBrowserColumn() {
  if (searchInput) {
    const onSearch = () => {
      renderSidebar();
      fetchArtifacts();
    };
    searchInput.addEventListener('input', debounce(onSearch, 200));
  }

  // The listener sits on the container, which renderSidebar() never replaces.
  // No throttle: fetchMoreArtifacts() is a no-op while a page is in flight.
  sidebarBody.addEventListener('scroll', loadMoreIfNearEnd);

  // Mode toggle (tree/activity)
  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = /** @type {HTMLElement} */ (btn).dataset.mode;
      if (!mode || mode === sidebarRenderer.getBrowseMode()) return;
      sidebarRenderer.setBrowseMode(/** @type {'tree'|'activity'} */ (mode));
      document.querySelectorAll('.mode-btn').forEach((b) => {
        b.classList.toggle('active', /** @type {HTMLElement} */ (b).dataset.mode === mode);
      });
      renderSidebar();
    });
  });

  // Layout toggle (list/gallery)
  document.querySelectorAll('.layout-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const layout = /** @type {HTMLElement} */ (btn).dataset.layout;
      if (layout === sidebarRenderer.getSidebarLayout()) return;
      sidebarRenderer.setSidebarLayout(/** @type {'list'|'gallery'} */ (layout));
      document.querySelectorAll('.layout-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      renderSidebar();
    });
  });

  // Keyboard shortcuts (priority order)
  document.addEventListener('keydown', (e) => {
    const tag = document.activeElement?.tagName;
    const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

    // 1. Escape + fullscreen → exit fullscreen (highest priority)
    if (e.key === 'Escape' && previewRenderer.isFullscreen()) {
      e.preventDefault();
      previewRenderer.exitFullscreen();
      return;
    }

    // 2. Escape + search focused → clear search
    if (e.key === 'Escape' && document.activeElement === searchInput) {
      searchInput.blur();
      searchInput.value = '';
      renderSidebar();
      fetchArtifacts();
      return;
    }

    // 2b. Escape with rows picked (no input focused) → clear the picks
    if (e.key === 'Escape' && !isInput && pickBar.clear()) return;

    // 3. "/" (no input focused) → exit fullscreen first, then focus search
    if (e.key === '/' && !isInput) {
      e.preventDefault();
      if (previewRenderer.isFullscreen()) previewRenderer.exitFullscreen();
      searchInput.focus();
      return;
    }

    // 4. "F" (no input focused, no modifiers) → toggle fullscreen
    if (e.key === 'f' && !isInput && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      if (previewRenderer.isFullscreen()) {
        previewRenderer.exitFullscreen();
      } else {
        previewRenderer.enterFullscreen();
      }
    }
  });

  initBrowseLayout({
    handle: resizeHandle,
    handleY: document.getElementById('resize-handle-y'),
    sidebar,
    toggle: orientToggleBtn,
  });
  initSidebarMenu({
    button: document.getElementById('sidebar-menu-btn'),
    menu: document.getElementById('sidebar-menu'),
  });
  if (refreshBtn) refreshBtn.addEventListener('click', doRefresh);

  const allSessionsBtn = document.getElementById('all-sessions-btn');
  if (allSessionsBtn) allSessionsBtn.addEventListener('click', toggleAllSessions);

  // Scope pill ✕ — one-click way back to the default this-session scope.
  const scopePillClear = document.getElementById('scope-pill-clear');
  if (scopePillClear) {
    scopePillClear.addEventListener('click', () => {
      setShowAllSessions(false);
      updateScopeUi();
      fetchArtifacts();
    });
  }
}

// ---- Embedded Viewer ----

/**
 * Show what the hub selected: an entry, or the empty state for null. A
 * repeat select of the entry already on screen leaves the pane alone, because
 * a re-render would reload the preview's frame for nothing.
 * @param {any|null} entry
 */
function showInViewer(entry) {
  if (entry !== null && entry === getSelectedArtifact()) return;
  setSelectedArtifact(entry);
  previewRenderer.renderPreview();
}

// ---- Artifact changes (the stream standalone, the hub embedded) ----

/**
 * Take the server's new copy of an artifact: merge it into the held entry,
 * and re-render the preview when it is the artifact shown.
 * @param {any} fields
 */
function applyUpdated(fields) {
  const held = getArtifacts();
  const idx = held.findIndex((a) => a.id === fields.id);
  if (idx >= 0) {
    held[idx] = { ...held[idx], ...fields };
    setArtifacts(held);
  }
  const selected = getSelectedArtifact();
  if (selected?.id === fields.id) {
    setSelectedArtifact(idx >= 0 ? held[idx] : { ...selected, ...fields });
    previewRenderer.renderPreview();
  }
  if (idx >= 0) renderSidebar();
}

/**
 * Forget a deleted artifact. Embedded, deleting the shown one empties the
 * viewer; the hub then selects another.
 * @param {string} id
 */
function applyDeleted(id) {
  removeArtifact(id);
  if (getFocusedArtifact()?.id === id) setFocusedArtifact(null);
  if (getSelectedArtifact()?.id === id) {
    setSelectedArtifact(null);
    previewRenderer.renderPreview();
  }
  renderSidebar();
}

// ---- SSE (Server-Sent Events; standalone only) ----

/** @type {any} */
let sseSource = null;

/**
 * Act on an agent focus: the artifact becomes both the on-screen selection
 * and the agent-focus target, the pane re-renders, and it goes fullscreen
 * when the agent asked. Shared by the SSE "focus" handler's two branches
 * (artifact already held vs. fetched) so the sequence has one definition.
 * @param {any} artifact
 * @param {boolean} wantFullscreen
 */
function applyFocus(artifact, wantFullscreen) {
  setFocusedArtifact(artifact);
  setSelectedArtifact(artifact);
  renderSidebar();
  previewRenderer.renderPreview();
  if (wantFullscreen) previewRenderer.enterFullscreen(artifact);
}

/**
 * Follow the panel's event stream. Standalone only: embedded, the hub follows
 * it and tells the viewer what it needs (see the module docstring).
 */
function connectSSE() {
  if (sseSource) {
    sseSource.close();
    sseSource = null;
  }
  const source = new EventSource('/api/events');
  sseSource = source;
  source.onopen = () => updateHealth(true);
  source.onmessage = (event) => {
    updateHealth(true);
    let eventData = null;
    try {
      eventData = JSON.parse(event.data);
    } catch {
      return;
    }
    const eventType = eventData && eventData.type;

    if (eventType === 'focus') {
      // The agent called artifact_focus.
      const focusId = eventData.id;
      if (!focusId) return;
      const wantFullscreen = !!eventData.fullscreen;
      const held = getArtifacts().find((x) => x.id === focusId);
      if (held) {
        applyFocus(held, wantFullscreen);
        // Scroll the selected item into view
        requestAnimationFrame(() => {
          const sel = sidebarBody.querySelector(`[data-id="${focusId}"]`);
          if (sel) sel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
        return;
      }
      // Not in the loaded pages — fetch that one artifact. An entry outside
      // the current scope is not focused; one inside it is focused even
      // while a search keeps it out of the list.
      fetchEntry(focusId)
        .then((entry) => {
          if (entry && inCurrentScope(entry)) applyFocus(entry, wantFullscreen);
        })
        .catch(() => {});
      return;
    }

    if (eventType === 'artifact_deleted') {
      applyDeleted(eventData.id);
      return;
    }

    if (eventType === 'artifact_updated') {
      const fields = { ...eventData };
      delete fields.type;
      applyUpdated(fields);
      return;
    }

    if (eventType === 'artifact' || !eventType) {
      if (previewRenderer.isFullscreen()) previewRenderer.noteNewArtifactArrival();
      const entry = { ...eventData };
      delete entry.type;
      if (eventType && entry.id && addArtifact(entry)) {
        renderSidebar();
        if (previewRenderer.isFullscreen()) previewRenderer.updateNewArtifactBadge();
        return;
      }
      fetchArtifacts()
        .then(() => {
          if (previewRenderer.isFullscreen()) previewRenderer.updateNewArtifactBadge();
        })
        .catch(() => {});
    }
  };
  source.onerror = () => updateHealth(false);
}

function doRefresh() {
  refreshBtn.classList.add('refreshing');
  fetchArtifacts().finally(() => {
    refreshBtn.classList.remove('refreshing');
  });
  connectSSE();
}

// ---- Theme: forward to nested previews + re-style plots ----
//
// initTheme() replaces the old hand-rolled 'osprey-theme-change' listener
// and data-theme MutationObserver: the theme-manager runtime already applies
// broadcasts from the hub and whatever ?theme=/localStorage/data-theme
// theme-boot.js resolved pre-paint. subscribe() below is the one thing still
// gallery-specific: re-forwarding to nested preview iframes (Plotly HTML
// artifacts) and re-styling the visible timeseries chart. It fires on every
// apply, even one that re-applies an unchanged id (the hidden-iframe repair
// path), which is exactly what a hidden preview iframe needs on tab
// activation.
// Standalone, this page owns its own theme chrome (the header
// <osprey-display-menu>), so it runs theme-manager.js in the hub role:
// persistence, OS auto-follow and ?theme= handling all come with it, and
// broadcast is a structural no-op on a page with no iframes. Embedded in the
// Web Terminal hub it is a follower instead: theme-boot.js already applied
// data-theme pre-paint, and this attaches the postMessage listener for the
// hub's live broadcasts.
initTheme({ role: isEmbedded() ? 'follower' : 'hub' });

// Embedded mode (contract-version 1, see frame-params.js): hides the app
// header with its logo and the browser column (gallery.css's `body.embedded`
// rules) and, via the component's own injected rule, the
// <osprey-display-menu> in the header -- all of them defer to the hub's
// chrome when this page is loaded inside a web_terminal panel.
applyEmbedded();

/**
 * Send a theme to one embedded artifact page.
 * @param {Element} iframe
 * @param {string} theme
 */
function _sendThemeToFrame(iframe, theme) {
  // Intentional '*' (same-origin contract exception): nested preview iframe may be null/cross-origin.
  try {
    /** @type {any} */ (iframe).contentWindow.postMessage(
      { type: 'osprey-theme-change', theme },
      '*'
    );
  } catch {
    // PostMessage to a stale/cross-origin frame is best-effort.
  }
}

/**
 * The "Open in new tab" links carry `?theme=` (see types.js `openUrl`),
 * computed when their pane was rendered; a theme change in between would
 * otherwise open the new tab in the stale id -- and `?theme=` is the top
 * rung of theme-boot.js's ladder, so it would win. Re-stamp them in place.
 * @param {string} theme
 */
function _refreshThemeLinks(theme) {
  document.querySelectorAll('a[data-theme-link]').forEach((a) => {
    const url = new URL(/** @type {HTMLAnchorElement} */ (a).href, window.location.origin);
    url.searchParams.set('theme', theme);
    /** @type {HTMLAnchorElement} */ (a).href = `${url.pathname}${url.search}${url.hash}`;
  });
}

subscribe((theme) => {
  // Every iframe the gallery mounts is a served artifact page (the preview
  // pane's viewport, each sidebar card thumbnail) running the same follower
  // bridge, so there is no selector here to keep in step with the markup.
  document.querySelectorAll('iframe').forEach((iframe) => _sendThemeToFrame(iframe, theme));
  _refreshThemeLinks(theme);
  restyleMountedCharts();
});

// The load-time half of the same contract: an artifact page that finishes
// loading after the last apply (a newly selected preview, a lazily loaded
// card thumbnail scrolled into view) gets the theme in force right away
// instead of waiting for the next change. `load` doesn't bubble, so it is
// caught in the capture phase at the document.
document.addEventListener(
  'load',
  (e) => {
    const theme = getTheme();
    if (theme && e.target instanceof HTMLIFrameElement) _sendThemeToFrame(e.target, theme);
  },
  true
);

// Session changes are unrelated to theming and stay a plain message
// listener (theme-manager owns the 'osprey-theme-change' type now).
window.addEventListener('message', (e) => {
  if (e.origin !== window.location.origin) return;
  if (e.data && e.data.type === 'osprey-session-change' && e.data.session_id) {
    setCurrentSessionId(e.data.session_id);
    setShowAllSessions(false);
    updateScopeUi();
    fetchArtifacts();
  }
});

// ---- Init ----

if (viewerOnly) {
  // The hub says what to show, what to do with it and when it changed
  // (viewer.js). Each action runs the code path the standalone preview
  // header's button runs.
  createViewer({
    getHeld: getArtifacts,
    fetchEntry,
    show: showInViewer,
    update: applyUpdated,
    remove: applyDeleted,
    actions: {
      pin: (a) => previewRenderer.setPinned(a, true),
      unpin: (a) => previewRenderer.setPinned(a, false),
      print: printArtifact,
      // No stream keeps the held list current here, and the composer's
      // picker offers it, so it is fetched again first.
      logbook: (a) => {
        fetchArtifacts().finally(() => openComposeModal({ artifact_id: a.id }));
      },
      delete: (a) => previewRenderer.deleteArtifact(a),
    },
  });
} else {
  wireBrowserColumn();
}

initTypeRegistry().then(() => {
  fetchArtifacts();
  // Embedded, the hub acts on the focus and follows the stream (see above).
  if (viewerOnly) return;
  fetchFocus();
  connectSSE();
});
