// @ts-check
/**
 * OSPREY Artifact Gallery — Unified Browse View
 *
 * Single gallery for all artifacts with type filtering, pin flag,
 * and inline timeseries rendering.
 */
import { initTheme, subscribe, chartSeries } from "/design-system/js/theme-manager.js";
import { applyEmbedded } from "/design-system/js/frame-params.js";
import "/design-system/js/components/osprey-theme-switcher.js";
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
  getRecentArtifacts,
  fileUrl,
  fetchArtifacts as fetchArtifactsData,
  fetchFocus,
} from "./state.js";
import {
  initTypeRegistry,
  thumbnailHtml,
  typeIcon,
  formatTime,
  formatFullTime,
  isNewThisSession,
  openUrl,
  escapeHtml,
} from "./types.js";
import { initSplitPaneResize, createSidebarRenderer } from "./render.js";
import { createPreviewRenderer } from "./preview.js";
import { renderTimeseriesView, _tsChartTheme } from "./timeseries.js";

// ---- DOM Refs ----

const headerCount = document.getElementById("header-count");
const healthDot = document.getElementById("health-indicator");
const refreshBtn = /** @type {HTMLElement} */ (document.getElementById("refresh-btn"));
const searchInput = /** @type {HTMLInputElement} */ (document.getElementById("search"));
const sidebarBody = /** @type {HTMLElement} */ (document.getElementById("sidebar-body"));
const sidebar = document.getElementById("browse-sidebar");
const resizeHandle = document.getElementById("resize-handle");

// ---- Simple-mode DOM refs (frame 2b) ----
const simpleEmpty = document.getElementById("simple-empty");
const simpleResult = document.getElementById("simple-result");
const simpleResultTitle = document.getElementById("simple-result-title");
const simpleResultBadge = /** @type {HTMLElement} */ (document.getElementById("simple-result-badge"));
const simpleOpenFull = /** @type {HTMLAnchorElement} */ (document.getElementById("simple-open-full"));
const simpleSave = /** @type {HTMLAnchorElement} */ (document.getElementById("simple-save"));
const simpleResultPreview = document.getElementById("simple-result-preview");
const simpleResultCaption = document.getElementById("simple-result-caption");
const simpleListCount = document.getElementById("simple-list-count");
const simpleListBody = /** @type {HTMLElement} */ (document.getElementById("simple-list-body"));
const simpleShowAll = /** @type {HTMLElement} */ (document.getElementById("simple-show-all"));

// Page-load timestamp for the "NEW" badge (an artifact created this session).
// Independent of render.js's own _sessionStart; both are just page-load time.
const _sessionStart = new Date().toISOString();
// Simple mode's session list truncates to the most recent few until the user
// clicks "Show all"; latched here so re-renders (SSE, fetch) keep it expanded.
let simpleShowAllResults = false;
const SIMPLE_LIST_LIMIT = 6;

// ---- State ----
// artifacts/selectedArtifact/focusedArtifact/activeFilter/currentSessionId/
// showAllSessions live in state.js behind explicit accessors, and
// typeRegistry lives in types.js (behind getTypeRegistry()) — see the
// imports above. browseMode/sidebarLayout live behind sidebarRenderer's
// own accessors (below) — see render.js. isFullscreen/
// newArtifactsSinceFullscreen live behind previewRenderer's own accessors
// (below) — see preview.js. No closure vars here.

// ---- Preview Renderer / Sidebar Renderer ----
// previewRenderer (preview.js) owns renderPreview and the pin/fullscreen/
// focus state; sidebarRenderer (render.js) owns the sidebar/filter-bar
// rendering. Each needs an effect the other one owns (previewRenderer
// triggers a sidebar re-render on delete/pin/fullscreen-exit;
// sidebarRenderer triggers a preview render/focus/fullscreen-enter on
// selection), so they're wired together via injected callbacks in both
// directions. `sidebarRenderer` is declared with `let` and assigned after
// `previewRenderer` so previewRenderer's callbacks — none of which run
// until well after both are constructed — can close over it.

/** @type {any} */
// eslint-disable-next-line prefer-const -- intentional forward reference: assigned later after previewRenderer is constructed
let sidebarRenderer;

const previewRenderer = createPreviewRenderer({
  onArtifactDeleted: () => {
    updateHeaderCount();
    sidebarRenderer.renderSidebar();
  },
  onPinToggled: () => sidebarRenderer.renderSidebar(),
  onFullscreenExit: () => sidebarRenderer.renderSidebar(),
  onTimeseriesNeeded: (container, artifact) => renderTimeseriesView(container, artifact),
});

sidebarRenderer = createSidebarRenderer({
  onSelect: (a) => previewRenderer.setAsFocus(a),
  onPreviewNeeded: () => previewRenderer.renderPreview(),
  onEnterFullscreen: (a) => previewRenderer.enterFullscreen(a),
});

// ---- Health / Header UI ----
// escapeHtml/formatSize/formatTime/formatFullTime/formatDate/openUrl/
// isNewThisSession/sendToTerminal/requestColorPass/typeBadge/typeColor now
// live in types.js, consumed directly by render.js/preview.js instead of
// through this module; updateHealth/updateHeaderCount stay here — they
// touch this module's own top-level DOM refs (healthDot/headerCount), not
// stateless.

/** @param {boolean} ok */
function updateHealth(ok) {
  if (healthDot) healthDot.className = ok ? "health-dot healthy" : "health-dot";
}

function updateHeaderCount() {
  if (headerCount) /** @type {any} */ (headerCount).textContent = getArtifacts().length;
}

// ---- Simple Mode (frame 2b) ----
// The Simple layout renders from the same artifact list + selection/focus
// state as Expert, into its own #view-artifacts-simple section (shown only
// under html[data-ui-mode="simple"]). renderSimple() is called alongside the
// sidebar re-render on every data change, so switching modes shows fresh
// content instantly. It writes into hidden DOM in Expert mode, which is cheap.

/**
 * The artifact Simple mode shows in the big latest-result card: the
 * user-selected one if it's still in the list, else the agent-focused one,
 * else the newest.
 * @param {any[]} recent - newest-first artifact list
 * @returns {any|null}
 */
function simpleResultArtifact(recent) {
  const sel = getSelectedArtifact();
  if (sel && recent.some((a) => a.id === sel.id)) return sel;
  const foc = getFocusedArtifact();
  if (foc && recent.some((a) => a.id === foc.id)) return foc;
  return recent[0] || null;
}

/**
 * Simple mode's inline preview. Reuses types.js's thumbnailHtml (same
 * <img>/<iframe> markup the Expert preview builds) — the result card's CSS
 * sizes it full-bleed rather than as a thumbnail.
 * @param {any} a
 * @returns {string}
 */
function simplePreviewHtml(a) {
  return thumbnailHtml(a);
}

/** @returns {void} */
function renderSimple() {
  if (!simpleListBody) return;
  // Only the active Simple layout needs rebuilding: in Expert mode this DOM is
  // hidden, so skip the sort + innerHTML churn on every SSE/fetch event. The
  // osprey-mode-change handler re-renders on the switch into Simple, so the
  // view is always fresh when shown.
  if (document.documentElement.dataset.uiMode !== "simple") return;
  const recent = getRecentArtifacts();
  const latest = simpleResultArtifact(recent);

  if (!latest) {
    simpleEmpty?.classList.remove("hidden");
    simpleResult?.classList.add("hidden");
  } else {
    simpleEmpty?.classList.add("hidden");
    simpleResult?.classList.remove("hidden");
    if (simpleResultTitle) simpleResultTitle.textContent = latest.title;
    if (simpleResultBadge) simpleResultBadge.hidden = !isNewThisSession(latest, _sessionStart);
    if (simpleOpenFull) simpleOpenFull.href = openUrl(latest);
    if (simpleSave) { simpleSave.href = fileUrl(latest); simpleSave.setAttribute("download", latest.filename); }
    if (simpleResultPreview) simpleResultPreview.innerHTML = simplePreviewHtml(latest);
    if (simpleResultCaption) {
      simpleResultCaption.textContent =
        latest.description || `${latest.title} · ${formatFullTime(latest.timestamp)}`;
    }
  }

  if (simpleListCount) simpleListCount.textContent = String(recent.length);
  if (simpleShowAll) simpleShowAll.hidden = recent.length <= SIMPLE_LIST_LIMIT;
  const shown = simpleShowAllResults ? recent : recent.slice(0, SIMPLE_LIST_LIMIT);
  const selId = latest?.id;
  simpleListBody.innerHTML = shown
    .map(
      (a) => `
    <div class="simple-list-item ${a.id === selId ? "selected" : ""}" data-id="${escapeHtml(a.id)}">
      <span class="simple-list-item-icon">${typeIcon(a.artifact_type)}</span>
      <span class="simple-list-item-name" title="${escapeHtml(a.title)}">${escapeHtml(a.title)}</span>
      ${isNewThisSession(a, _sessionStart) ? '<span class="simple-badge-new">NEW</span>' : ""}
      <span class="simple-list-item-time">${escapeHtml(formatTime(a.timestamp))}</span>
    </div>`
    )
    .join("");
}

// ---- API ----
// showErrorBanner/hideErrorBanner/fetchArtifacts/fetchFocus now live in
// state.js. fetchArtifacts() no longer triggers render effects itself
// (state.js has no access to this module's DOM-rendering functions) — this
// wrapper supplies them via the callbacks state.js's fetchArtifacts()
// accepts, so every call site below can keep calling one local function.

function fetchArtifacts() {
  return fetchArtifactsData({
    onHealthChange: updateHealth,
    onArtifactsUpdated: () => {
      updateHeaderCount();
      sidebarRenderer.rebuildTypeChips();
      sidebarRenderer.renderSidebar();
      renderSimple();
    },
  });
}

// ---- Events ----

if (searchInput) {
  searchInput.addEventListener("input", debounce(() => sidebarRenderer.renderSidebar(), 200));
}

// Mode toggle (tree/activity)
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = /** @type {HTMLElement} */ (btn).dataset.mode;
    if (mode === sidebarRenderer.getBrowseMode()) return;
    sidebarRenderer.setBrowseMode(mode);
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    sidebarRenderer.renderSidebar();
  });
});

// Layout toggle (list/gallery)
document.querySelectorAll(".layout-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const layout = /** @type {HTMLElement} */ (btn).dataset.layout;
    if (layout === sidebarRenderer.getSidebarLayout()) return;
    sidebarRenderer.setSidebarLayout(layout);
    document.querySelectorAll(".layout-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    sidebarRenderer.renderSidebar();
  });
});

// Keyboard shortcuts (priority order)
document.addEventListener("keydown", (e) => {
  const tag = document.activeElement?.tagName;
  const isInput = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";

  // 1. Escape + fullscreen → exit fullscreen (highest priority)
  if (e.key === "Escape" && previewRenderer.isFullscreen()) {
    e.preventDefault();
    previewRenderer.exitFullscreen();
    return;
  }

  // 2. Escape + search focused → clear search
  if (e.key === "Escape" && document.activeElement === searchInput) {
    searchInput.blur();
    searchInput.value = "";
    sidebarRenderer.renderSidebar();
    return;
  }

  // 3. "/" (no input focused) → exit fullscreen first, then focus search
  if (e.key === "/" && !isInput) {
    e.preventDefault();
    if (previewRenderer.isFullscreen()) previewRenderer.exitFullscreen();
    searchInput.focus();
    return;
  }

  // 4. "F" (no input focused, no modifiers) → toggle fullscreen
  if (e.key === "f" && !isInput && !e.ctrlKey && !e.metaKey && !e.altKey) {
    e.preventDefault();
    if (previewRenderer.isFullscreen()) {
      previewRenderer.exitFullscreen();
    } else {
      previewRenderer.enterFullscreen();
    }
    return;
  }
});

/**
 * @template {(...args: any[]) => any} F
 * @param {F} fn
 * @param {number} ms
 * @returns {(...args: Parameters<F>) => void}
 */
function debounce(fn, ms) {
  /** @type {ReturnType<typeof setTimeout>|undefined} */
  let timer;
  /**
   * @this {any}
   * @param {Parameters<F>} args
   * @returns {void}
   */
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

// ---- SSE (Server-Sent Events) ----

/** @type {any} */
let sseSource = null;

/**
 * Select an artifact as both the on-screen selection and the agent-focus
 * target, re-render, and optionally enter fullscreen. Shared by the SSE
 * "focus" handler's two branches (artifact already local vs. fetched on
 * retry) so the select→focus→render→fullscreen sequence has one definition.
 * @param {any} artifact
 * @param {boolean} wantFullscreen
 */
function applyFocus(artifact, wantFullscreen) {
  setSelectedArtifact(artifact);
  setFocusedArtifact(artifact);
  sidebarRenderer.renderSidebar();
  previewRenderer.renderPreview();
  renderSimple();
  if (wantFullscreen) previewRenderer.enterFullscreen(artifact);
}

function connectSSE() {
  if (sseSource) { sseSource.close(); sseSource = null; }
  const source = new EventSource("/api/events");
  sseSource = source;
  source.onopen = () => updateHealth(true);
  source.onmessage = (event) => {
    updateHealth(true);
    let eventData = null;
    try { eventData = JSON.parse(event.data); } catch { return; }
    const eventType = eventData && eventData.type;

    if (eventType === "focus") {
      // Agent called artifact_focus — select that artifact in the gallery
      const focusId = eventData.id;
      const wantFullscreen = !!eventData.fullscreen;
      if (focusId) {
        const a = getArtifacts().find((x) => x.id === focusId);
        if (a) {
          applyFocus(a, wantFullscreen);
          // Scroll the selected item into view
          requestAnimationFrame(() => {
            const sel = sidebarBody.querySelector(`[data-id="${focusId}"]`);
            if (sel) sel.scrollIntoView({ behavior: "smooth", block: "nearest" });
          });
        } else {
          // Artifact not yet in local list — refresh and retry
          fetchArtifacts().then(() => {
            const retry = getArtifacts().find((x) => x.id === focusId);
            if (retry) applyFocus(retry, wantFullscreen);
          });
        }
      }
      return;
    }

    if (eventType === "artifact_deleted") {
      setArtifacts(getArtifacts().filter((a) => a.id !== eventData.id));
      if (getFocusedArtifact()?.id === eventData.id) setFocusedArtifact(null);
      if (getSelectedArtifact()?.id === eventData.id) { setSelectedArtifact(null); previewRenderer.renderPreview(); }
      updateHeaderCount();
      sidebarRenderer.rebuildTypeChips();
      sidebarRenderer.renderSidebar();
      renderSimple();
      return;
    }

    if (eventType === "artifact_updated") {
      // Update the artifact in-place
      const idx = getArtifacts().findIndex((a) => a.id === eventData.id);
      if (idx >= 0) {
        const updated = getArtifacts();
        updated[idx] = { ...updated[idx], ...eventData };
        setArtifacts(updated);
        if (getSelectedArtifact()?.id === eventData.id) {
          setSelectedArtifact(updated[idx]);
          previewRenderer.renderPreview();
        }
        sidebarRenderer.renderSidebar();
        renderSimple();
      }
      return;
    }

    if (eventType === "artifact" || !eventType) {
      if (previewRenderer.isFullscreen()) previewRenderer.noteNewArtifactArrival();
      fetchArtifacts().then(() => {
        if (previewRenderer.isFullscreen()) previewRenderer.updateNewArtifactBadge();
      }).catch(() => {});
    }
  };
  source.onerror = () => updateHealth(false);
}

function doRefresh() {
  refreshBtn.classList.add("refreshing");
  fetchArtifacts().finally(() => {
    refreshBtn.classList.remove("refreshing");
  });
  connectSSE();
}

// ---- Theme: follower role; forward to nested previews + re-style plots ----
//
// initTheme({role:'follower'}) replaces the old hand-rolled
// 'osprey-theme-change' listener and data-theme MutationObserver: the
// theme-manager runtime already applies broadcasts from the hub and
// whatever ?theme=/localStorage/data-theme theme-boot.js resolved
// pre-paint. subscribe() below is the one thing still gallery-specific:
// re-forwarding to nested preview iframes (Plotly HTML artifacts) and
// re-styling the visible timeseries chart. It fires on every apply, even
// one that re-applies an unchanged id (the hidden-iframe repair path),
// which is exactly what a hidden preview iframe needs on tab activation.

initTheme({ role: "follower" });

// Embedded mode (contract-version 1, see frame-params.js): hides the
// logo (via gallery.css's `body.embedded .logo` rule) and, via the
// theme-switcher component's own
// injected rule, the <osprey-theme-switcher> in the header -- both defer
// to the hub's chrome when this page is loaded inside a web_terminal panel.
applyEmbedded();

/** @param {string} theme */
function _forwardThemeToPreviewFrames(theme) {
  document.querySelectorAll(".preview-viewport iframe, .browse-preview-pane iframe").forEach((iframe) => {
    // Intentional '*' (same-origin contract exception): nested preview iframe may be null/cross-origin.
    // eslint-disable-next-line no-empty -- intentional empty catch: postMessage to a stale/cross-origin frame is best-effort
    try { /** @type {any} */ (iframe).contentWindow.postMessage({ type: "osprey-theme-change", theme }, "*"); } catch {}
  });
}

function _restyleTimeseriesChart() {
  // Target the actual Plotly graph div inside the container, not the
  // outer #ts-viewport wrapper.
  const tsChart = document.querySelector("#ts-viewport [data-ts-chart]");
  if (!tsChart || typeof Plotly === "undefined") return;
  const t = _tsChartTheme();
  try {
    Plotly.relayout(tsChart, {
      paper_bgcolor: t.paper_bgcolor, plot_bgcolor: t.plot_bgcolor,
      "font.color": t.font.color,
      "xaxis.gridcolor": t.xaxis.gridcolor, "xaxis.linecolor": t.line,
      "yaxis.gridcolor": t.yaxis.gridcolor, "yaxis.linecolor": t.line,
      "legend.bgcolor": t.legendBg, "legend.bordercolor": t.legendBorder,
    });
    // relayout doesn't touch trace colors, so the data lines and their legend
    // dots keep the prior theme's palette until reload. Restyle each trace's
    // line+marker to the current series palette so they re-theme live too.
    const series = chartSeries();
    const traces = /** @type {any} */ (tsChart).data || [];
    if (series.length && traces.length) {
      const colors = traces.map((/** @type {any} */ _t, /** @type {number} */ i) => series[i % series.length]);
      Plotly.restyle(tsChart, { "line.color": colors, "marker.color": colors });
    }
  // eslint-disable-next-line no-empty -- intentional empty catch: Plotly relayout is best-effort restyle
  } catch {}
}

subscribe((theme) => {
  _forwardThemeToPreviewFrames(theme);
  _restyleTimeseriesChart();
});

// Session changes are unrelated to theming and stay a plain message
// listener (theme-manager owns the 'osprey-theme-change' type now).
window.addEventListener("message", (e) => {
  if (e.origin !== window.location.origin) return;
  if (e.data && e.data.type === "osprey-session-change" && e.data.session_id) {
    setCurrentSessionId(e.data.session_id);
    const btn = document.getElementById("all-sessions-btn");
    if (btn) btn.classList.remove("active");
    setShowAllSessions(false);
    fetchArtifacts();
  }
  // Live Expert<->Simple switch broadcast by the hub (mode-toggle task). The
  // pre-paint rung (mode-boot.js) already set the initial data-ui-mode; this
  // is the runtime flip. Re-render Simple so its content is fresh on arrival.
  if (e.data && e.data.type === "osprey-mode-change" && e.data.mode) {
    const mode = e.data.mode === "simple" ? "simple" : "expert";
    document.documentElement.setAttribute("data-ui-mode", mode);
    renderSimple();
  }
});

// ---- Init ----

initSplitPaneResize(resizeHandle, sidebar);
refreshBtn.addEventListener("click", doRefresh);

const allSessionsBtn = document.getElementById("all-sessions-btn");
if (allSessionsBtn) {
  allSessionsBtn.addEventListener("click", () => {
    setShowAllSessions(!getShowAllSessions());
    allSessionsBtn.classList.toggle("active", getShowAllSessions());
    fetchArtifacts();
  });
}

// Simple mode: clicking a session-list row promotes it to the shown result.
if (simpleListBody) {
  simpleListBody.addEventListener("click", (e) => {
    const row = /** @type {HTMLElement} */ (e.target).closest(".simple-list-item");
    if (!row) return;
    const id = row.getAttribute("data-id");
    const a = getArtifacts().find((x) => x.id === id);
    if (a) { setSelectedArtifact(a); renderSimple(); }
  });
}
if (simpleShowAll) {
  simpleShowAll.addEventListener("click", () => {
    simpleShowAllResults = true;
    renderSimple();
  });
}
initTypeRegistry().then(() => {
  sidebarRenderer.initFilterBar();
  fetchArtifacts();
  fetchFocus();
  connectSSE();
});
