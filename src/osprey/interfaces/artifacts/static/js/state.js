// @ts-check
/**
 * OSPREY Artifact Gallery — shared state and the list fetch.
 *
 * Owns the viewer's core mutable state (the artifact list, the selected artifact,
 * the current-session scoping and the transcript the session view resolved its
 * key to) behind explicit get/set accessors, not raw exported `let` bindings — ES
 * modules only give importers a read-only live view of an exported binding, so
 * reassignment has to go through a function here. gallery.js and its sibling
 * preview/timeseries modules call these accessors instead of holding local
 * closure copies, so everyone reads and writes the same source of truth.
 *
 * Also owns the artifact-list request (fetchArtifacts) and the error banner it
 * drives. The list it holds is the server's first page, newest first, not the
 * whole store; the logbook composer's picker reads it.
 *
 * logbook.js/print.js import `getSelectedArtifact`/`fileUrl` directly from here —
 * there is no global-object bridge.
 *
 * @module state
 */

// ---- Gallery State (module-private) ---- //

/** @type {any[]} */
let artifacts = [];
/** @type {any|null} */
let selectedArtifact = null;
/** @type {string|null} */
let currentSessionId = null; // the session key the host reports; null outside a session
/** @type {string|null} */
let heldTranscript = null; // the transcript the session view's first page resolved the key to; null otherwise
let fetchGeneration = 0; // bumped by every fetch; a stale response is dropped

// ---- Accessors ---- //

/** @returns {any[]} */
export function getArtifacts() {
  return artifacts;
}
/** @param {any[]} list */
export function setArtifacts(list) {
  artifacts = list;
}

/** @returns {any|null} */
export function getSelectedArtifact() {
  return selectedArtifact;
}
/** @param {any|null} a */
export function setSelectedArtifact(a) {
  selectedArtifact = a;
}

/** @returns {string|null} */
export function getCurrentSessionId() {
  return currentSessionId;
}
/** @param {string|null} id */
export function setCurrentSessionId(id) {
  currentSessionId = id;
}

/**
 * The transcript the session view's listing resolved the key to (contract
 * C6), held from its first page. Null outside a session and for a key the
 * ledger has no row for.
 * @returns {string|null}
 */
export function getHeldTranscript() {
  return heldTranscript;
}

/**
 * Whether the gallery shows one session (the sidebar's row for its key): any time
 * the host has reported a session.
 * @returns {boolean}
 */
function inSessionView() {
  return currentSessionId !== null && currentSessionId !== '';
}

/**
 * Whether an artifact belongs in the list the current view fetches. The
 * session view lists what its held transcript owns and the operator sees, so
 * a held null transcript admits nothing, an unowned artifact included.
 * Every other view admits everything.
 * @param {any} entry
 * @returns {boolean}
 */
export function inCurrentScope(entry) {
  if (!inSessionView()) return true;
  return (
    heldTranscript !== null && entry.owner === heldTranscript && entry.operator_visible !== false
  );
}

// ---- File URL ---- //

/** @param {{id: string, filename: string}} a */
export function fileUrl(a) {
  return `/files/${encodeURIComponent(a.id)}/${encodeURIComponent(a.filename)}`;
}

// ---- Error Banner ---- //

/** @param {string} msg */
export function showErrorBanner(msg) {
  let banner = document.getElementById('error-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'error-banner';
    banner.style.cssText =
      'position:fixed;top:0;left:0;right:0;z-index:9999;padding:12px 20px;' +
      'background:var(--color-error);color:#fff;font-size:14px;text-align:center;'; // hygiene-allow-color: fixed white-on-error banner text, theme-invariant by design
    document.body.prepend(banner);
  }
  banner.textContent = msg;
  banner.style.display = 'block';
}

/** @returns {void} */
export function hideErrorBanner() {
  const banner = document.getElementById('error-banner');
  if (banner) banner.style.display = 'none';
}

// ---- API ---- //

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
 * Request the listing's first page. The session view posts the sidebar's body
 * to `POST /api/artifacts/session-list` (contract C6), so it lists exactly the
 * sidebar's row and a run once at its head; every other view reads the unscoped
 * `GET /api/artifacts`. No `limit` is sent, so the server's configured page size
 * governs.
 * @param {boolean} sessionView
 * @returns {Promise<Response>}
 */
function requestListing(sessionView) {
  if (sessionView) {
    return fetch('/api/artifacts/session-list', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: currentSessionId, search: '', cursor: null, runs: 'grouped' }),
    });
  }
  return fetch('/api/artifacts');
}

/**
 * Fetch the artifact list, scoped to the current session. Replaces the shared
 * `artifacts` state with that page and drives the error banner directly.
 * @returns {Promise<void>}
 */
export async function fetchArtifacts() {
  const generation = ++fetchGeneration;
  const sessionView = inSessionView();
  try {
    const resp = await requestListing(sessionView);
    if (!resp.ok) {
      const errText = await resp.text();
      showErrorBanner('API error (' + resp.status + '): ' + errText);
      return;
    }
    hideErrorBanner();
    const data = await resp.json();
    if (generation !== fetchGeneration) return;
    artifacts = data.artifacts || [];
    heldTranscript = sessionView ? (data.transcript ?? null) : null;
  } catch (err) {
    console.error('Failed to fetch artifacts:', err);
    showErrorBanner('Failed to fetch artifacts: ' + messageOf(err));
  }
}

/**
 * Hold an artifact fetched on its own. One outside the session scope is left out,
 * because the held list is what the scope admits.
 * @param {any} entry
 * @returns {void}
 */
export function addArtifact(entry) {
  if (!inCurrentScope(entry)) return;
  if (artifacts.some((a) => a.id === entry.id)) return;
  artifacts = [...artifacts, entry];
}

/**
 * Drop an artifact from the held list. An id the list does not hold leaves the
 * list alone: it may be out of scope as much as on a page not fetched.
 * @param {string} id
 * @returns {void}
 */
export function removeArtifact(id) {
  const kept = artifacts.filter((a) => a.id !== id);
  if (kept.length === artifacts.length) return;
  artifacts = kept;
}
