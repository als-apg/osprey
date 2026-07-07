// @ts-check
/**
 * OSPREY Artifact Gallery — preview pane shell, pin/fullscreen/focus.
 *
 * Owns the biggest single view in the gallery: `renderPreview` (the preview
 * pane's header/meta/viewport markup and its button wiring), pin toggling,
 * fullscreen enter/exit (+ the "N new" badge while fullscreen), and
 * agent-focus (`setAsFocus`).
 *
 * The markdown+KaTeX render pipeline and the JSON viewer live in the sibling
 * preview-content.js module — kept separate purely to hold both files under
 * the 450-line cap. That module is stateless aside from its own
 * one-time marked-config flag, so the split is a clean seam: `renderPreview`
 * below just calls `renderMarkdownView`/`renderJsonView` with a container and
 * an artifact, the same way it already calls `escapeHtml`/`typeBadge` from
 * types.js.
 *
 * `createPreviewRenderer(callbacks)` follows render.js's
 * `createSidebarRenderer(callbacks)` precedent: the fullscreen flag and the
 * "new artifacts since fullscreen" counter are real mutable state, so they
 * live in this factory's closure behind explicit accessors, while a handful
 * of effects this module doesn't own (re-rendering the sidebar, updating the
 * header count, and rendering the timeseries chart/table — timeseries.js's
 * job) are supplied as injected callbacks.
 *
 * `injectLogbookButtons()`/`injectPrintButton()` are called directly at the
 * end of `renderPreview`: logbook.js/print.js are ES modules that import
 * state.js directly — there is no global-object bridge.
 *
 * @module preview
 */

import {
  getArtifacts,
  setArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getFocusedArtifact,
  setFocusedArtifact,
  fileUrl,
} from "./state.js";
import {
  typeBadge,
  escapeHtml,
  formatSize,
  formatFullTime,
  openUrl,
  requestColorPass,
} from "./types.js";
import { renderMarkdownView, renderJsonView } from "./preview-content.js";
import { injectLogbookButtons } from "./logbook.js";
import { injectPrintButton } from "./print.js";

// ---- Preview Renderer (factory: fullscreen state + effects this module doesn't own) ---- //

/**
 * @typedef {object} PreviewRenderCallbacks
 * @property {() => void} onArtifactDeleted - fired after a successful DELETE (local artifact/selection/focus state is already updated by this module); caller updates the header count and re-renders the sidebar
 * @property {() => void} onPinToggled - fired after a successful pin/unpin; caller re-renders the sidebar
 * @property {() => void} onFullscreenExit - fired on exiting fullscreen, before the resize/scroll-into-view pass; caller re-renders the sidebar
 * @property {(container: HTMLElement, artifact: any) => void} onTimeseriesNeeded - fired once the `#ts-viewport` container exists for a timeseries artifact; caller (gallery.js, via timeseries.js's renderTimeseriesView) renders the chart/table into it
 */

/**
 * Create the gallery's preview renderer: the preview pane (`renderPreview`),
 * pin toggling, fullscreen enter/exit (+ its "N new" badge), and setting
 * agent focus. Bound to a small set of injected callbacks for the effects
 * owned elsewhere (sidebar re-render, header count, timeseries rendering).
 * @param {PreviewRenderCallbacks} callbacks
 */
export function createPreviewRenderer(callbacks) {
  let isFullscreen = false;
  let newArtifactsSinceFullscreen = 0;

  // ---- Preview Pane ----

  /** @returns {void} */
  function renderPreview() {
    const previewEmpty = document.getElementById("preview-empty");
    const previewContent = document.getElementById("preview-content");
    if (!previewEmpty || !previewContent) return;

    if (!getSelectedArtifact()) {
      previewEmpty.classList.remove("hidden");
      previewContent.classList.add("hidden");
      return;
    }

    previewEmpty.classList.add("hidden");
    previewContent.classList.remove("hidden");

    const a = getSelectedArtifact();
    const url = fileUrl(a);

    let viewportHtml = "";

    // Check for timeseries data
    if ((a.metadata && a.metadata.data_type === "timeseries" || a.category === "archiver_data") && (a.data_file || (a.metadata && a.metadata.data_file))) {
      viewportHtml = `<div id="ts-viewport" class="ts-viewport-container"></div>`;
    } else {
      switch (a.artifact_type) {
        case "plot_html":
        case "table_html":
        case "dashboard_html":
        case "html":
          viewportHtml = `<iframe src="${url}" class="preview-iframe-light" sandbox="allow-scripts allow-same-origin"></iframe>`;
          break;
        case "notebook":
          viewportHtml = `<iframe src="/api/notebooks/${a.id}/rendered" class="preview-iframe-light" sandbox="allow-scripts allow-same-origin"></iframe>`;
          break;
        case "plot_png":
        case "image":
          viewportHtml = `<img src="${url}" alt="${escapeHtml(a.title)}" />`;
          break;
        case "markdown":
          viewportHtml = `<div id="md-viewport" class="md-preview-container"></div>`;
          break;
        case "json":
          viewportHtml = `<div id="json-viewport" class="json-viewer"></div>`;
          break;
        case "text":
          viewportHtml = `<iframe src="${url}" class="preview-iframe-dark"></iframe>`;
          break;
        default:
          if (a.mime_type === "application/pdf") {
            viewportHtml = `<iframe src="${url}" class="preview-iframe-light"></iframe>`;
          } else {
            viewportHtml = `<div class="preview-download">
              <a href="${url}" target="_blank" class="btn btn-secondary">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
                </svg>
                Download ${escapeHtml(a.filename)}
              </a>
            </div>`;
          }
      }
    }

    previewContent.innerHTML = `
      <div class="preview-header">
        <div class="preview-header-left">
          <button class="fullscreen-back-btn" id="fullscreen-back" title="Exit fullscreen (Esc)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
            Back
          </button>
          <span class="badge badge-${a.category || a.artifact_type}">${typeBadge(a.category || a.artifact_type)}</span>
          <span class="preview-header-title">${escapeHtml(a.title)}</span>
        </div>
        <div class="preview-header-actions">
          <span class="fullscreen-new-badge" id="fullscreen-new-badge" data-count="0"></span>
          <button class="btn-action-icon ${a.pinned ? "btn-action-pinned" : ""}" id="preview-toggle-pin" title="${a.pinned ? "Unpin" : "Pin"}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 17v5"/><path d="M5 9l2-1V4a1 1 0 011-1h8a1 1 0 011 1v4l2 1a1 1 0 01.4 1.6l-1.4 2a1 1 0 01-.8.4H6.8a1 1 0 01-.8-.4l-1.4-2A1 1 0 015 9z"/></svg>
          </button>
          <button class="btn-action-icon fullscreen-enter-btn" id="fullscreen-enter" title="Maximize (F)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H5a2 2 0 00-2 2v3M21 8V5a2 2 0 00-2-2h-3M16 21h3a2 2 0 002-2v-3M3 16v3a2 2 0 002 2h3"/></svg>
          </button>
          <a href="${openUrl(a)}" target="_blank" class="btn-action-icon" title="Open in new tab">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/></svg>
          </a>
          <button class="btn-action-icon btn-action-danger" id="preview-delete" title="Delete artifact">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/></svg>
          </button>
        </div>
      </div>
      ${a.description ? `<div class="preview-desc">${escapeHtml(a.description)}</div>` : ""}
      <div class="preview-meta-strip">
        <span class="preview-meta-item">
          <span class="preview-meta-label">Size</span>
          <span class="preview-meta-value">${formatSize(a.size_bytes)}</span>
        </span>
        <span class="preview-meta-item">
          <span class="preview-meta-label">Created</span>
          <span class="preview-meta-value">${formatFullTime(a.timestamp)}</span>
        </span>
        ${a.tool_source ? `
        <span class="preview-meta-item">
          <span class="preview-meta-label">Source</span>
          <span class="preview-meta-value">${escapeHtml(a.tool_source)}</span>
        </span>` : ""}
        <span class="preview-meta-path" id="preview-copy-path" title="Click to copy path">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
          <span class="preview-path-text">_agent_data/artifacts/${escapeHtml(a.filename)}</span>
        </span>
      </div>
      <div class="preview-viewport">
        ${viewportHtml}
      </div>
    `;

    // Wire action buttons
    /** @type {HTMLElement} */ (document.getElementById("preview-toggle-pin")).addEventListener("click", () => togglePin(a));
    /** @type {HTMLElement} */ (document.getElementById("preview-delete")).addEventListener("click", () => {
      if (!confirm(`Delete "${a.title}"? This cannot be undone.`)) return;
      fetch(`/api/artifacts/${a.id}`, { method: "DELETE" })
        .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
        .then(() => {
          setArtifacts(getArtifacts().filter((x) => x.id !== a.id));
          if (getSelectedArtifact()?.id === a.id) setSelectedArtifact(null);
          if (getFocusedArtifact()?.id === a.id) setFocusedArtifact(null);
          callbacks.onArtifactDeleted();
          renderPreview();
        })
        .catch((err) => console.error("Delete failed:", err));
    });

    // Copy path to clipboard
    const copyPathBtn = document.getElementById("preview-copy-path");
    if (copyPathBtn) {
      copyPathBtn.addEventListener("click", () => {
        const path = `_agent_data/artifacts/${a.filename}`;
        navigator.clipboard.writeText(path).then(() => {
          copyPathBtn.classList.add("copied");
          setTimeout(() => copyPathBtn.classList.remove("copied"), 1500);
        });
      });
    }

    // Fullscreen buttons
    const enterBtn = document.getElementById("fullscreen-enter");
    if (enterBtn) enterBtn.addEventListener("click", () => enterFullscreen());

    const backBtn = document.getElementById("fullscreen-back");
    if (backBtn) backBtn.addEventListener("click", () => exitFullscreen());

    const newBadge = document.getElementById("fullscreen-new-badge");
    if (newBadge) {
      newBadge.addEventListener("click", () => {
        // Navigate to newest artifact, stay in fullscreen
        if (getArtifacts().length > 0) {
          const sorted = [...getArtifacts()].sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));
          setSelectedArtifact(sorted[0]);
          newArtifactsSinceFullscreen = 0;
          renderPreview();
        }
      });
    }

    // Update badge if in fullscreen
    if (isFullscreen) updateNewArtifactBadge();

    // Render timeseries if applicable
    if ((a.metadata && a.metadata.data_type === "timeseries" || a.category === "archiver_data") && (a.data_file || (a.metadata && a.metadata.data_file))) {
      const tsEl = document.getElementById("ts-viewport");
      if (tsEl) callbacks.onTimeseriesNeeded(tsEl, a);
    }

    // Render markdown inline
    if (a.artifact_type === "markdown") {
      const mdEl = document.getElementById("md-viewport");
      if (mdEl) renderMarkdownView(mdEl, a);
    }

    // Render JSON inline
    if (a.artifact_type === "json") {
      const jsonEl = document.getElementById("json-viewport");
      if (jsonEl) renderJsonView(jsonEl, a);
    }

    requestColorPass();
    injectLogbookButtons();
    injectPrintButton();
  }

  // ---- Pin / Focus actions ----

  /** @param {any} artifact */
  async function togglePin(artifact) {
    const newPinned = !artifact.pinned;
    try {
      const resp = await fetch(`/api/artifacts/${artifact.id}/pin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned: newPinned }),
      });
      if (resp.ok) {
        artifact.pinned = newPinned;
        callbacks.onPinToggled();
        renderPreview();
      }
    } catch (err) {
      console.error("Failed to toggle pin:", err);
    }
  }

  // ---- Fullscreen Mode ----

  /** @param {any} [artifact] */
  function enterFullscreen(artifact) {
    if (isFullscreen) return;
    if (artifact) {
      setSelectedArtifact(artifact);
    }
    if (!getSelectedArtifact()) return;

    isFullscreen = true;
    newArtifactsSinceFullscreen = 0;
    document.body.classList.add('fullscreen-mode');
    document.body.classList.add('fullscreen-entering');
    setTimeout(() => document.body.classList.remove('fullscreen-entering'), 700);

    updateNewArtifactBadge();
    renderPreview();

    // Resize iframes / Plotly charts
    requestAnimationFrame(() => {
      const previewContent = document.getElementById("preview-content");
      const iframe = previewContent?.querySelector('iframe');
      if (iframe?.contentWindow) {
        try { iframe.contentWindow.dispatchEvent(new Event('resize')); } catch {}
      }
      const tsChart = document.getElementById('ts-viewport');
      if (tsChart && typeof Plotly !== 'undefined') {
        try { Plotly.Plots.resize(tsChart); } catch {}
      }
    });
  }

  /** @returns {void} */
  function exitFullscreen() {
    if (!isFullscreen) return;
    isFullscreen = false;
    newArtifactsSinceFullscreen = 0;
    document.body.classList.remove('fullscreen-mode');
    document.body.classList.remove('fullscreen-entering');

    renderPreview();
    callbacks.onFullscreenExit();

    // Resize iframes / Plotly charts
    requestAnimationFrame(() => {
      const previewContent = document.getElementById("preview-content");
      const iframe = previewContent?.querySelector('iframe');
      if (iframe?.contentWindow) {
        try { iframe.contentWindow.dispatchEvent(new Event('resize')); } catch {}
      }
      const tsChart = document.getElementById('ts-viewport');
      if (tsChart && typeof Plotly !== 'undefined') {
        try { Plotly.Plots.resize(tsChart); } catch {}
      }
      // Scroll selected item into view in sidebar
      if (getSelectedArtifact()) {
        const sidebarBody = document.getElementById("sidebar-body");
        const sel = sidebarBody?.querySelector(`[data-id="${getSelectedArtifact().id}"]`);
        if (sel) sel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }

  /** @returns {void} */
  function updateNewArtifactBadge() {
    const badge = document.getElementById('fullscreen-new-badge');
    if (!badge) return;
    const count = newArtifactsSinceFullscreen;
    badge.dataset.count = String(count);
    badge.textContent = count > 0 ? `${count} new` : '';
  }

  /** @param {any} artifact */
  async function setAsFocus(artifact) {
    try {
      const resp = await fetch("/api/focus", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ artifact_id: artifact.id }),
      });
      if (resp.ok) {
        setFocusedArtifact(artifact);
      }
    } catch (err) {
      console.error("Failed to set focus:", err);
    }
  }

  return {
    renderPreview,
    enterFullscreen,
    exitFullscreen,
    setAsFocus,
    isFullscreen: () => isFullscreen,
    noteNewArtifactArrival: () => { newArtifactsSinceFullscreen++; },
    updateNewArtifactBadge,
  };
}
