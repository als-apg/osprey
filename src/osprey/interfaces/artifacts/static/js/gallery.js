// @ts-check
/**
 * OSPREY Artifact Gallery — Unified Browse View
 *
 * Single gallery for all artifacts with type filtering, pin flag,
 * and inline timeseries rendering.
 */
import { initTheme, subscribe } from "/design-system/js/theme-manager.js";
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
  fetchArtifacts as fetchArtifactsData,
  fetchFocus,
} from "./state.js";
import { initTypeRegistry } from "./types.js";
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
initTypeRegistry().then(() => {
  sidebarRenderer.initFilterBar();
  fetchArtifacts();
  fetchFocus();
  connectSSE();
});
