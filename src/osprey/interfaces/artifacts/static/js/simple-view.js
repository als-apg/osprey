// @ts-check
/**
 * OSPREY Artifact Gallery — the Simple layout.
 *
 * The Simple layout renders from the same artifact list + selection/focus
 * state as Expert, into its own #view-artifacts-simple section (shown only
 * under html[data-ui-mode="simple"]). `render()` is called alongside the
 * sidebar re-render on every data change, so switching modes shows fresh
 * content instantly. It writes into hidden DOM in Expert mode, which is cheap.
 *
 * What Simple owns is the chrome around the result — a friendlier header
 * (title, NEW badge, Open full size / Save) and the session list beneath it.
 * The result *content* is artifact-viewport.js's shared dispatch, exactly as
 * Expert's preview pane renders it: Simple has no renderer of its own, so no
 * artifact type can render in one mode and not the other.
 *
 * `createSimpleView(callbacks)` follows preview.js's
 * `createPreviewRenderer(callbacks)` precedent: it reads state.js directly,
 * formats via types.js, binds to the page's `#simple-*` elements once, and
 * takes the one effect it does not own — rendering a timeseries chart/table,
 * timeseries.js's job — as an injected callback. The "Show all" latch is its
 * only state and lives in the factory's closure.
 *
 * @module simple-view
 */

import { getTheme } from "/design-system/js/theme-manager.js";
import {
  getArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getFocusedArtifact,
  getRecentArtifacts,
  getArtifactTotal,
  fileUrl,
} from "./state.js";
import {
  typeIcon,
  formatTime,
  formatFullTime,
  isNewThisSession,
  openUrl,
  escapeHtml,
} from "./types.js";
import { artifactViewportHtml, mountArtifactViewport } from "./artifact-viewport.js";

// Page-load timestamp for the "NEW" badge (an artifact created this session).
// Independent of render.js's own _sessionStart; both are just page-load time.
const _sessionStart = new Date().toISOString();
const SIMPLE_LIST_LIMIT = 6;

/**
 * @typedef {object} SimpleViewCallbacks
 * @property {(container: HTMLElement, artifact: any) => void} onTimeseriesNeeded - forwarded to artifact-viewport.js's `mountArtifactViewport`; fired with the mounted `.ts-viewport-container` for a timeseries artifact, which the caller (gallery.js, via timeseries.js's renderTimeseriesView) renders the chart/table into
 */

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
  // Newest real result first; the shipped example only when nothing else
  // exists yet (that is what it is there for).
  return recent.find((a) => a.origin !== "demo") || recent[0] || null;
}

/**
 * Create the Simple layout's renderer: the latest-result card and the
 * "Results from this session" list, bound to the page's `#simple-*` elements.
 * @param {SimpleViewCallbacks} callbacks
 * @returns {{render: () => void}}
 */
export function createSimpleView(callbacks) {
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

  // Simple mode's session list truncates to the most recent few until the user
  // clicks "Show all"; latched here so re-renders (SSE, fetch) keep it expanded.
  let simpleShowAllResults = false;

  /** @returns {void} */
  function render() {
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
      if (simpleOpenFull) {
        simpleOpenFull.href = openUrl(latest, getTheme());
        simpleOpenFull.setAttribute("data-theme-link", "");
      }
      if (simpleSave) { simpleSave.href = fileUrl(latest); simpleSave.setAttribute("download", latest.filename); }
      if (simpleResultPreview) {
        // Same dispatch the Expert preview pane renders through — Simple has no
        // renderer of its own, so every type Expert can show, Simple shows too.
        simpleResultPreview.innerHTML = artifactViewportHtml(latest);
        mountArtifactViewport(simpleResultPreview, latest, {
          onTimeseriesNeeded: callbacks.onTimeseriesNeeded,
        });
      }
      if (simpleResultCaption) {
        simpleResultCaption.textContent =
          latest.description || `${latest.title} · ${formatFullTime(latest.timestamp)}`;
      }
    }

    const resultCount = getArtifactTotal();
    if (simpleListCount) simpleListCount.textContent = String(resultCount);
    if (simpleShowAll) simpleShowAll.hidden = resultCount <= SIMPLE_LIST_LIMIT;
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

  // Clicking a session-list row promotes it to the shown result.
  if (simpleListBody) {
    simpleListBody.addEventListener("click", (e) => {
      const row = /** @type {HTMLElement} */ (e.target).closest(".simple-list-item");
      if (!row) return;
      const id = row.getAttribute("data-id");
      const a = getArtifacts().find((x) => x.id === id);
      if (a) { setSelectedArtifact(a); render(); }
    });
  }
  if (simpleShowAll) {
    simpleShowAll.addEventListener("click", () => {
      simpleShowAllResults = true;
      render();
    });
  }

  return { render };
}
