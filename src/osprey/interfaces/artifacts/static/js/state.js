// @ts-check
/**
 * OSPREY Artifact Gallery — shared state, fetch layer, and filtering.
 *
 * Owns the gallery's core mutable state (the artifact list, the selected/
 * focused artifact, the active type filter, and the current-session
 * scoping) behind explicit get/set accessors, not raw exported `let`
 * bindings — ES modules only give importers a read-only live view of an
 * exported binding, so reassignment has to go through a function here.
 * gallery.js and its sibling render/preview/timeseries modules call these
 * accessors instead of holding local closure copies, so everyone reads and
 * writes the same source of truth.
 *
 * Also owns the artifact-fetch API calls (fetchArtifacts/fetchFocus), the
 * error banner they drive, and getFilteredArtifacts() (filter+search+sort).
 * Render effects (health indicator, header count, sidebar re-render) are
 * NOT triggered here — DOM rendering belongs to gallery.js and the renderer
 * modules, so fetchArtifacts() takes an optional callbacks object instead,
 * mirroring scaffold/data.js's createScaffoldDataActions callback-injection
 * pattern.
 *
 * logbook.js/print.js import `getSelectedArtifact`/`fileUrl` and
 * preview.js/gallery.js import `getFocusedArtifact` directly from here —
 * there is no global-object bridge.
 *
 * @module state
 */

// ---- Gallery State (module-private) ---- //

/** @type {any[]} */
let artifacts = [];
/** @type {any|null} */
let selectedArtifact = null;
/** @type {any|null} */
let focusedArtifact = null;
/** @type {string} "all" | "pinned" | a type/category string */
let activeFilter = "all";
/** @type {string|null} */
let currentSessionId = null;
let showAllSessions = false;

// ---- Accessors ---- //

/** @returns {any[]} */
export function getArtifacts() { return artifacts; }
/** @param {any[]} list */
export function setArtifacts(list) { artifacts = list; }

/** @returns {any|null} */
export function getSelectedArtifact() { return selectedArtifact; }
/** @param {any|null} a */
export function setSelectedArtifact(a) { selectedArtifact = a; }

/** @returns {any|null} */
export function getFocusedArtifact() { return focusedArtifact; }
/** @param {any|null} a */
export function setFocusedArtifact(a) { focusedArtifact = a; }

/** @returns {string} */
export function getActiveFilter() { return activeFilter; }
/** @param {string} f */
export function setActiveFilter(f) { activeFilter = f; }

/** @returns {string|null} */
export function getCurrentSessionId() { return currentSessionId; }
/** @param {string|null} id */
export function setCurrentSessionId(id) { currentSessionId = id; }

/** @returns {boolean} */
export function getShowAllSessions() { return showAllSessions; }
/** @param {boolean} v */
export function setShowAllSessions(v) { showAllSessions = v; }

// ---- File URL ---- //

/** @param {{id: string, filename: string}} a */
export function fileUrl(a) {
  return `/files/${encodeURIComponent(a.id)}/${encodeURIComponent(a.filename)}`;
}

// ---- Error Banner ---- //

/** @param {string} msg */
export function showErrorBanner(msg) {
  let banner = document.getElementById("error-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "error-banner";
    banner.style.cssText =
      "position:fixed;top:0;left:0;right:0;z-index:9999;padding:12px 20px;" +
      "background:var(--color-error);color:#fff;font-size:14px;text-align:center;"; // hygiene-allow-color: fixed white-on-error banner text, theme-invariant by design
    document.body.prepend(banner);
  }
  banner.textContent = msg;
  banner.style.display = "block";
}

/** @returns {void} */
export function hideErrorBanner() {
  const banner = document.getElementById("error-banner");
  if (banner) banner.style.display = "none";
}

// ---- API ---- //

/**
 * @typedef {object} FetchArtifactsCallbacks
 * @property {(ok: boolean) => void} [onHealthChange] - fired with the health status after every attempt
 * @property {() => void} [onArtifactsUpdated] - fired after a successful fetch, once `artifacts` holds the fresh list
 */

/**
 * Extract a display message from a caught value of unknown type (TS strict
 * mode types `catch` bindings as `unknown`).
 * @param {unknown} e
 * @returns {string}
 */
function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Fetch the artifact list, scoped to the current session unless
 * `showAllSessions` is set. Updates the shared `artifacts` state and drives
 * the error banner directly; render effects (header count, filter chips,
 * sidebar) are the caller's job, via `callbacks`.
 * @param {FetchArtifactsCallbacks} [callbacks]
 * @returns {Promise<void>}
 */
export async function fetchArtifacts(callbacks = {}) {
  try {
    let url = "/api/artifacts";
    if (currentSessionId && !showAllSessions) {
      url += "?session_id=" + encodeURIComponent(currentSessionId);
    }
    const resp = await fetch(url);
    if (!resp.ok) {
      const errText = await resp.text();
      showErrorBanner("API error (" + resp.status + "): " + errText);
      callbacks.onHealthChange?.(false);
      return;
    }
    hideErrorBanner();
    const data = await resp.json();
    artifacts = data.artifacts || [];
    callbacks.onHealthChange?.(true);
    callbacks.onArtifactsUpdated?.();
  } catch (err) {
    console.error("Failed to fetch artifacts:", err);
    showErrorBanner("Failed to fetch artifacts: " + messageOf(err));
    callbacks.onHealthChange?.(false);
  }
}

/**
 * Fetch the current agent focus target, if any, and store it as
 * `focusedArtifact`. Silent on failure (matches the original: console-only).
 * @returns {Promise<void>}
 */
export async function fetchFocus() {
  try {
    const resp = await fetch("/api/focus");
    const data = await resp.json();
    if (data.artifact) {
      focusedArtifact = data.artifact;
    }
  } catch (err) {
    console.error("Failed to fetch focus:", err);
  }
}

// ---- Filtering ---- //

/**
 * Filter/search/sort the current artifact list for display: applies the
 * active type filter, then the (already-normalized) search query, then
 * sorts pinned-first / newest-first.
 * @param {string} [searchQuery] - already-trimmed, lowercased search text (the caller reads it from the DOM)
 * @returns {any[]}
 */
export function getFilteredArtifacts(searchQuery = "") {
  let filtered = [...artifacts];

  if (activeFilter === "pinned") {
    filtered = filtered.filter((a) => a.pinned);
  } else if (activeFilter !== "all") {
    filtered = filtered.filter((a) => a.category === activeFilter || a.artifact_type === activeFilter);
  }

  if (searchQuery) {
    filtered = filtered.filter(
      (a) =>
        a.title.toLowerCase().includes(searchQuery) ||
        a.filename.toLowerCase().includes(searchQuery) ||
        (a.description && a.description.toLowerCase().includes(searchQuery)) ||
        a.artifact_type.toLowerCase().includes(searchQuery)
    );
  }

  filtered.sort((a, b) => {
    if (a.pinned && !b.pinned) return -1;
    if (!a.pinned && b.pinned) return 1;
    return (b.timestamp || "").localeCompare(a.timestamp || "");
  });

  return filtered;
}

/**
 * The full artifact list sorted newest-first, independent of the active
 * type filter and the pinned-first ordering getFilteredArtifacts() applies.
 * Simple mode's "latest result" + "Results from this session" list read this.
 *
 * Session scoping is unchanged: `artifacts` already holds exactly what the
 * last fetch returned — the current session's artifacts when a session scope
 * has been received, or (when none has) the most-recent set across sessions,
 * because fetchArtifacts() adds no `session_id` filter without a
 * currentSessionId. So this is never empty when any artifacts exist.
 * @returns {any[]}
 */
export function getRecentArtifacts() {
  return [...artifacts].sort((a, b) => (b.timestamp || "").localeCompare(a.timestamp || ""));
}
