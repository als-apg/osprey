// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: data layer
 *
 * The shared artifact-fetch cache (single-flight across gallery instances)
 * plus the data actions that populate an ArtifactGallery's
 * {artifacts, untrackedFiles, summary} state: loading, registering an
 * untracked file, and deleting one.
 *
 * Mirrors the factory/injection pattern in lattice_dashboard/net.js: the
 * network/filtering logic is pure standalone functions with no DOM access or
 * `this`, unit-testable in isolation. {@link createScaffoldDataActions} is the
 * factory that binds that pure logic to a gallery's fixed domain (its
 * categoryFilter/categoryOverrides/categoryRemaps) and to DOM/render effects
 * injected as named callbacks (onLoaded, onLoadError) — so this module has no
 * dependency on the DOM-rendering code that still lives in scaffold-gallery.js.
 *
 * @module scaffold/data
 */

import { fetchJSON } from '../api.js';

// ---- Shared Fetch Cache ---- //

/** @type {Promise<any>|null} */
let _fetchPromise = null;

/**
 * Fetch artifacts from the API, caching the promise so multiple
 * gallery instances don't duplicate the request.
 * @returns {Promise<any>}
 */
export async function fetchArtifactsShared() {
  if (!_fetchPromise) _fetchPromise = fetchJSON('/api/scaffold');
  return _fetchPromise;
}

/** Reset the shared cache (called when the drawer closes, or to force a fresh fetch). */
export function resetFetchCache() {
  _fetchPromise = null;
}

// ---- Data Actions ---- //

/**
 * The subset of an ArtifactGallery instance that {@link loadArtifacts} needs:
 * the domain filter and the two category-remapping tables.
 * @typedef {object} ArtifactFilterState
 * @property {(artifact: {category: string, name?: string}) => boolean} categoryFilter
 * @property {Record<string, string>} categoryOverrides
 * @property {Record<string, string>} categoryRemaps
 */

/**
 * @typedef {object} ArtifactLoadResult
 * @property {object[]} artifacts
 * @property {object[]} untrackedFiles
 * @property {{total: number, framework: number, userOwned: number}} summary
 */

/**
 * Fetch artifacts + untracked files and filter/map them to a gallery's
 * domain, computing the summary counts. Shared by the gallery's `load()`
 * (initial load, uses the shared cache) and `reloadFull()` (post-action
 * refresh, `skipCache: true` to invalidate the shared cache first).
 *
 * @param {ArtifactFilterState} state
 * @param {{skipCache?: boolean}} [opts]
 * @returns {Promise<ArtifactLoadResult>}
 */
export async function loadArtifacts(state, opts = {}) {
  if (opts.skipCache) resetFetchCache();

  const [data, untrackedData] = await Promise.all([
    fetchArtifactsShared(),
    fetchJSON('/api/scaffold/untracked').catch(() => ({ untracked: [] })),
  ]);

  const allArtifacts = data.artifacts || [];
  const artifacts = allArtifacts
    .filter(state.categoryFilter)
    .map(/** @param {any} a */ (a) => ({
      ...a,
      displayCategory:
        state.categoryOverrides[a.name] ||
        state.categoryRemaps[a.category] ||
        a.category,
    }));

  const allUntracked = untrackedData.untracked || [];
  const untrackedFiles = allUntracked.filter(/** @param {any} u */ (u) => {
    const mapped = state.categoryRemaps[u.category] || u.category;
    return state.categoryFilter({ category: u.category, name: u.canonical_name })
      || state.categoryFilter({ category: mapped, name: u.canonical_name });
  });

  const framework = artifacts.filter(/** @param {any} a */ (a) => a.status === 'framework').length;
  const userOwned = artifacts.filter(/** @param {any} a */ (a) => a.status === 'user-owned').length;
  const summary = { total: artifacts.length, framework, userOwned };

  return { artifacts, untrackedFiles, summary };
}

/**
 * Register an untracked file into the project config via the API.
 * @param {string} canonicalName
 * @returns {Promise<void>}
 */
export async function registerUntrackedFile(canonicalName) {
  const resp = await fetch('/api/scaffold/untracked/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: canonicalName }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `Register failed (HTTP ${resp.status})`);
  }
}

/**
 * Delete an untracked file from disk via the API, after confirming with the
 * operator. Makes no request (and resolves `false`) if the operator declines.
 * @param {string} canonicalName
 * @returns {Promise<boolean>} true if the file was deleted
 */
export async function deleteUntrackedFile(canonicalName) {
  if (!confirm(`Delete "${canonicalName}"? This file will be removed from disk.`)) return false;

  const resp = await fetch(
    `/api/scaffold/untracked/${encodeURIComponent(canonicalName)}`,
    { method: 'DELETE' }
  );
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `Delete failed (HTTP ${resp.status})`);
  }
  return true;
}

// ---- Bound Actions Factory ---- //

/**
 * Extract a display message from a caught value of unknown type (TS strict
 * mode types `catch` bindings as `unknown`, and the fetch helpers above may
 * throw a plain string in edge cases, not just an Error).
 * @param {unknown} e
 * @returns {string}
 */
function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}

/**
 * @typedef {object} ScaffoldDataCallbacks
 * @property {() => void} [onLoadStart] - fired synchronously when `load()` begins (initial load only; `reloadFull()` doesn't toggle the loading indicator, matching the original behavior)
 * @property {(result: ArtifactLoadResult) => void} onLoaded - fired after a successful load/reload with the fresh {artifacts, untrackedFiles, summary}
 * @property {(message: string) => void} onLoadError - fired with a user-facing message when the initial load, register, or delete action fails
 */

/**
 * Create the scaffold gallery's data actions, bound to a fixed domain
 * (categoryFilter/categoryOverrides/categoryRemaps) and a set of effect
 * callbacks. Mirrors {@link createNetClient} in lattice_dashboard/net.js:
 * this factory owns no DOM state itself, it just wires the pure functions
 * above to the caller's render effects.
 *
 * @param {ArtifactFilterState} domain
 * @param {ScaffoldDataCallbacks} callbacks
 */
export function createScaffoldDataActions(domain, callbacks) {
  /** Initial load. Reuses the shared fetch cache. */
  async function load() {
    callbacks.onLoadStart?.();
    try {
      const result = await loadArtifacts(domain);
      callbacks.onLoaded(result);
    } catch (e) {
      callbacks.onLoadError(`Failed to load prompts: ${messageOf(e)}`);
    }
  }

  /**
   * Full reload after a mutating action: invalidates the shared cache first.
   * Does not catch its own errors — callers (registerUntracked/deleteUntracked
   * below) are the only callers and already wrap this in error handling.
   */
  async function reloadFull() {
    const result = await loadArtifacts(domain, { skipCache: true });
    callbacks.onLoaded(result);
  }

  /** @param {string} canonicalName */
  async function registerUntracked(canonicalName) {
    try {
      await registerUntrackedFile(canonicalName);
      await reloadFull();
    } catch (e) {
      callbacks.onLoadError(`Register failed: ${messageOf(e)}`);
    }
  }

  /** @param {string} canonicalName */
  async function deleteUntracked(canonicalName) {
    try {
      const deleted = await deleteUntrackedFile(canonicalName);
      if (deleted) await reloadFull();
    } catch (e) {
      callbacks.onLoadError(`Delete failed: ${messageOf(e)}`);
    }
  }

  return { load, reloadFull, registerUntracked, deleteUntracked };
}
