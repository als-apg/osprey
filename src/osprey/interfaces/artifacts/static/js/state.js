// @ts-check
/**
 * OSPREY Artifact Gallery — shared state, fetch layer, and filtering.
 *
 * Owns the gallery's core mutable state (the artifact list, the selected/
 * focused artifact, and the current-session scoping) behind explicit
 * get/set accessors, not raw exported `let`
 * bindings — ES modules only give importers a read-only live view of an
 * exported binding, so reassignment has to go through a function here.
 * gallery.js and its sibling render/preview/timeseries modules call these
 * accessors instead of holding local closure copies, so everyone reads and
 * writes the same source of truth.
 *
 * Also owns the artifact-fetch API calls (fetchArtifacts/fetchMoreArtifacts/
 * fetchFocus), the page cursor they follow, the error banner they drive, and
 * getFilteredArtifacts() (search+sort). The list it holds is the pages
 * fetched so far, newest first on the server, not the whole store.
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
/** @type {string|null} */
let currentSessionId = null;
let showAllSessions = false;
/** @type {string|null} */
let nextCursor = null;          // the cursor the next page needs; null when none is left
let total = 0;                  // how many entries the current filters admit
let lastSearch = "";            // the search the current page sequence was fetched with
/** @type {Set<string>} */
let heldUncounted = new Set();  // entries held on their own that the held total may not count yet
/** @type {object|null} */
let loadingMore = null;         // the next-page call in flight; only the call that set it clears it
let fetchGeneration = 0;        // bumped by every first-page fetch; a stale page is dropped
let settledGeneration = 0;      // the first-page fetch whose outcome the held list reflects; behind fetchGeneration while one is in flight

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

/** @returns {string|null} */
export function getCurrentSessionId() { return currentSessionId; }
/** @param {string|null} id */
export function setCurrentSessionId(id) { currentSessionId = id; }

/** @returns {boolean} */
export function getShowAllSessions() { return showAllSessions; }
/** @param {boolean} v */
export function setShowAllSessions(v) { showAllSessions = v; }

/** @returns {number} how many entries the current filters admit, fetched or not */
export function getArtifactTotal() { return total; }
/** @returns {string} the search the held list was fetched with */
export function getListSearch() { return lastSearch; }
/** @returns {boolean} whether the server holds a page not yet fetched */
export function hasMoreArtifacts() { return nextCursor !== null; }

/**
 * Whether an artifact belongs in the list the current session scope fetches.
 * A locally added artifact has to pass the same session test the fetch
 * applied, and that test admits an untagged artifact.
 * @param {any} entry
 * @returns {boolean}
 */
export function inCurrentScope(entry) {
  if (showAllSessions || !currentSessionId) return true;
  return !entry.session_id || entry.session_id === currentSessionId;
}

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
 * @typedef {FetchArtifactsCallbacks & {search?: string}} FetchArtifactsOptions
 */

/**
 * The listing URL for a search and the current session scope. No `limit` is
 * sent, so the server's configured page size governs.
 * @param {string} search
 * @param {string|null} [cursor]
 * @returns {string}
 */
function listingUrl(search, cursor = null) {
  const params = [];
  if (search) params.push("search=" + encodeURIComponent(search));
  if (currentSessionId && !showAllSessions) {
    params.push("session_id=" + encodeURIComponent(currentSessionId));
  }
  if (cursor) params.push("cursor=" + encodeURIComponent(cursor));
  return "/api/artifacts" + (params.length ? "?" + params.join("&") : "");
}

/**
 * Fetch the first page of the artifact list, filtered by `search` and scoped
 * to the current session unless `showAllSessions` is set. Replaces the shared
 * `artifacts` state with that page, records the total and the next-page
 * cursor, and drives the error banner directly; render effects (header count,
 * filter chips, sidebar) are the caller's job, via `callbacks`.
 * @param {FetchArtifactsOptions} [callbacks]
 * @returns {Promise<void>}
 */
export async function fetchArtifacts(callbacks = {}) {
  const search = callbacks.search || "";
  const generation = ++fetchGeneration;
  try {
    const resp = await fetch(listingUrl(search));
    if (!resp.ok) {
      const errText = await resp.text();
      showErrorBanner("API error (" + resp.status + "): " + errText);
      callbacks.onHealthChange?.(false);
      return;
    }
    hideErrorBanner();
    const data = await resp.json();
    if (generation !== fetchGeneration) return;
    artifacts = data.artifacts || [];
    total = data.total ?? artifacts.length;
    nextCursor = data.next_cursor ?? null;
    lastSearch = search;
    heldUncounted = new Set();
    settledGeneration = generation;
    callbacks.onHealthChange?.(true);
    callbacks.onArtifactsUpdated?.();
  } catch (err) {
    console.error("Failed to fetch artifacts:", err);
    showErrorBanner("Failed to fetch artifacts: " + messageOf(err));
    callbacks.onHealthChange?.(false);
  } finally {
    if (generation === fetchGeneration) settledGeneration = generation;
  }
}

/**
 * Fetch the page after the ones held and append it. A no-op when no cursor is
 * held, a page is already in flight, or a first page is in flight, because the
 * held cursor and search belong to the sequence that page replaces. A scroll
 * handler may call it freely.
 * On failure the cursor is kept, so a later call retries.
 * @param {FetchArtifactsCallbacks} [callbacks]
 * @returns {Promise<void>}
 */
export async function fetchMoreArtifacts(callbacks = {}) {
  if (nextCursor === null || loadingMore !== null || settledGeneration !== fetchGeneration) return;
  const token = {};
  loadingMore = token;
  const generation = fetchGeneration;
  try {
    const resp = await fetch(listingUrl(lastSearch, nextCursor));
    if (!resp.ok) {
      const errText = await resp.text();
      showErrorBanner("API error (" + resp.status + "): " + errText);
      callbacks.onHealthChange?.(false);
      return;
    }
    hideErrorBanner();
    const data = await resp.json();
    if (generation !== fetchGeneration) return;
    // A save between two requests can repeat an entry across pages.
    const held = new Set(artifacts.map((a) => a.id));
    const fresh = (data.artifacts || []).filter((/** @type {any} */ a) => !held.has(a.id));
    artifacts = [...artifacts, ...fresh];
    total = data.total ?? total;
    heldUncounted = new Set();
    nextCursor = data.next_cursor ?? null;
    // Cleared before the callbacks, so one of them may request the next page.
    loadingMore = null;
    callbacks.onHealthChange?.(true);
    callbacks.onArtifactsUpdated?.();
  } catch (err) {
    console.error("Failed to fetch more artifacts:", err);
    showErrorBanner("Failed to fetch artifacts: " + messageOf(err));
    callbacks.onHealthChange?.(false);
  } finally {
    // A page chained from the callbacks owns the guard now; only this call's own token is cleared.
    if (loadingMore === token) loadingMore = null;
  }
}

/**
 * Add an artifact the server announced, without a request.
 * @param {any} entry
 * @param {{counted?: boolean}} [options] - `counted`: the entry was fetched
 *   on its own and the held total may already include it (an older
 *   artifact), so it is held without raising the total. If its save event
 *   arrives while it is held that way, the artifact is new, and that event
 *   raises the total once.
 * @returns {boolean} whether the held list now reflects that artifact: false
 *   while a search is active, because only the server can say whether the
 *   artifact matches it; true without adding when it is outside the session
 *   scope, because it is correctly absent.
 */
export function addArtifact(entry, { counted = false } = {}) {
  if (lastSearch) return false;
  if (!inCurrentScope(entry)) return true;
  if (artifacts.some((a) => a.id === entry.id)) {
    if (!counted && heldUncounted.delete(entry.id)) total += 1;
    return true;
  }
  artifacts = [...artifacts, entry];
  if (counted) heldUncounted.add(entry.id);
  else total += 1;
  return true;
}

/**
 * Drop an artifact from the held list and from the total. An id the list does
 * not hold leaves both alone: it may be out of scope as much as on a page not yet fetched.
 * @param {string} id
 * @returns {void}
 */
export function removeArtifact(id) {
  const kept = artifacts.filter((a) => a.id !== id);
  if (kept.length === artifacts.length) return;
  artifacts = kept;
  if (!heldUncounted.delete(id)) total = Math.max(0, total - 1);
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
 * Search/sort the current artifact list for display: applies the
 * (already-normalized) search query, then sorts pinned-first /
 * newest-first. (Type narrowing is the tree's job — its sections group by
 * type — so there is no separate type filter.)
 * @param {string} [searchQuery] - already-trimmed, lowercased search text (the caller reads it from the DOM)
 * @returns {any[]}
 */
export function getFilteredArtifacts(searchQuery = "") {
  let filtered = [...artifacts];

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
 * The pages fetched so far, sorted newest-first, independent of the
 * pinned-first ordering getFilteredArtifacts() applies.
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
