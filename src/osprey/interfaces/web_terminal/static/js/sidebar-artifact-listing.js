// @ts-check
/* OSPREY Web Terminal — one artifact listing of the sidebar (layout F)
 *
 * The sidebar shows two of the artifacts panel's listings: the Artifacts list
 * (`run=false`) and the Script runs list (`run=true`). The server splits them
 * with its own run-record rule, so each list's total and paging are exact.
 * This module is one such listing: the pages fetched so far through the hub's
 * proxy, the server's total and cursor, and the edits the panel's event stream
 * announces. It draws nothing; its owner draws from `entries` whenever
 * `onChange` fires.
 *
 * The page sequence is the gallery's own (artifacts state.js): a first page
 * replaces the list and supersedes any page in flight; the next page is
 * appended on request, one at a time, never while a first page is in flight;
 * a failed request keeps the entries and the cursor, so a later one retries.
 * No `limit` is sent, so the panel's configured page size governs. The session
 * scope is the server's: a session's artifacts plus the untagged ones.
 *
 * @module sidebar-artifact-listing
 */

import { withPrefix } from './api.js';

/** Where the hub's proxy mounts the artifacts panel (routes/panels.py `/api/artifact-server`). */
export const ARTIFACTS_BASE = '/panel/artifacts';

/**
 * One artifact as the artifacts panel lists it. Only the fields the sidebar
 * reads are named.
 * @typedef {object} ArtifactEntry
 * @property {string} id
 * @property {string} [title]
 * @property {string} [filename]
 * @property {string} [artifact_type]
 * @property {string} [timestamp]
 * @property {string} [session_id]
 * @property {string} [origin]
 * @property {boolean} [pinned]
 * @property {boolean} [run_record]  the server's verdict (artifact_store.is_run_record)
 */

/**
 * @typedef {object} ListingOptions
 * @property {boolean} run  the `run` filter: true lists the run records, false the rest
 * @property {() => string | null} sessionId  the session to scope to, or null for every session
 * @property {() => void} onChange  called after a page lands or fails
 */

/** @typedef {ReturnType<typeof createListing>} Listing */

/**
 * @param {string} url
 * @returns {Promise<any>}
 */
async function getJSON(url) {
  const resp = await fetch(url, { cache: 'no-store' });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

/**
 * One artifact by id, or null when the panel does not have it.
 * @param {string} id
 * @returns {Promise<ArtifactEntry | null>}
 */
export async function fetchArtifact(id) {
  try {
    return await getJSON(withPrefix(`${ARTIFACTS_BASE}/api/artifacts/${encodeURIComponent(id)}`));
  } catch {
    return null;
  }
}

/**
 * The artifact the agent last focused, as the panel keeps it across reloads
 * and tabs; null when nothing is focused (the panel then names its newest
 * artifact with `focused: false`, which is no focus) or the panel is down.
 * @returns {Promise<string | null>}
 */
export async function fetchFocusedId() {
  try {
    const data = await getJSON(withPrefix(`${ARTIFACTS_BASE}/api/focus`));
    const id = data?.focused === true ? data.artifact?.id : null;
    return typeof id === 'string' ? id : null;
  } catch {
    return null;
  }
}

/**
 * Newest first, the listing's own order: by timestamp, the id breaking a tie.
 * @param {ArtifactEntry[]} entries
 * @returns {ArtifactEntry[]}
 */
export function newestFirst(entries) {
  return [...entries].sort(
    (a, b) =>
      (b.timestamp ?? '').localeCompare(a.timestamp ?? '') || (b.id ?? '').localeCompare(a.id ?? '')
  );
}

/**
 * @param {ListingOptions} options
 */
export function createListing({ run, sessionId, onChange }) {
  /** @type {ArtifactEntry[]} */
  let entries = [];
  let total = 0;
  /** @type {string | null} */
  let cursor = null;
  let loaded = false;
  let failed = false;
  let stopped = false;
  // Bumped by every first-page request: a page from an older sequence is dropped.
  let generation = 0;
  let loading = false;
  /** @type {object | null} */
  let paging = null;
  // Entries held on their own (an older artifact the agent focused) that the
  // total may not count yet; their save event, if it comes, counts them.
  /** @type {Set<string>} */
  const uncounted = new Set();

  /**
   * The listing URL for the current scope.
   * @param {string | null} [nextCursor]
   * @returns {string}
   */
  function url(nextCursor = null) {
    const params = new URLSearchParams({ run: String(run) });
    const session = sessionId();
    if (session) params.set('session_id', session);
    if (nextCursor) params.set('cursor', nextCursor);
    return withPrefix(`${ARTIFACTS_BASE}/api/artifacts?${params}`);
  }

  /** Replace the entries with the first page of the current scope. */
  async function loadFirst() {
    if (stopped) return;
    const mine = ++generation;
    loading = true;
    try {
      const data = await getJSON(url());
      if (mine !== generation) return;
      entries = newestFirst(data.artifacts ?? []);
      total = data.total ?? entries.length;
      cursor = data.next_cursor ?? null;
      uncounted.clear();
      failed = false;
    } catch {
      if (mine !== generation) return;
      failed = true;
    } finally {
      if (mine === generation) {
        loading = false;
        loaded = true;
      }
    }
    onChange();
  }

  /** Append the next page, when there is one and no request is in flight. */
  async function loadNext() {
    if (stopped || cursor === null || paging || loading) return;
    const token = {};
    paging = token;
    const mine = generation;
    try {
      const data = await getJSON(url(cursor));
      if (mine !== generation) return;
      // A save between two requests can repeat an entry across pages.
      const held = new Set(entries.map((a) => a.id));
      /** @type {ArtifactEntry[]} */
      const page = data.artifacts ?? [];
      // An entry held on its own is counted by the server's total once its
      // page arrives.
      for (const a of page) uncounted.delete(a.id);
      const fresh = page.filter((a) => !held.has(a.id));
      entries = newestFirst([...entries, ...fresh]);
      total = data.total ?? total;
      cursor = data.next_cursor ?? null;
    } catch {
      return;
    } finally {
      if (paging === token) paging = null;
    }
    onChange();
  }

  /**
   * @param {string} id
   * @returns {ArtifactEntry | null}
   */
  function find(id) {
    return entries.find((a) => a.id === id) ?? null;
  }

  /**
   * Hold an artifact the stream announced (`counted`: it is new, so the total
   * grows) or one fetched on its own (not counted: the total may hold it
   * already). A held one is not added twice; its save event counts it once.
   * @param {ArtifactEntry} entry
   * @param {{counted?: boolean}} [options]
   */
  function add(entry, { counted = true } = {}) {
    if (find(entry.id)) {
      if (counted && uncounted.delete(entry.id)) total += 1;
      return;
    }
    entries = newestFirst([...entries, entry]);
    if (counted) total += 1;
    else uncounted.add(entry.id);
  }

  /**
   * Merge changed fields into a held artifact.
   * @param {Partial<ArtifactEntry> & {id: string}} fields
   * @returns {boolean} whether the listing holds that artifact
   */
  function update(fields) {
    const index = entries.findIndex((a) => a.id === fields.id);
    if (index < 0) return false;
    const next = [...entries];
    next[index] = { ...next[index], ...fields };
    entries = newestFirst(next);
    return true;
  }

  /**
   * Drop an artifact. An id the listing does not hold changes no count: it may
   * be out of scope as much as on a page not fetched yet.
   * @param {string} id
   * @returns {boolean} whether the listing held it
   */
  function remove(id) {
    const kept = entries.filter((a) => a.id !== id);
    if (kept.length === entries.length) return false;
    entries = kept;
    if (!uncounted.delete(id)) total = Math.max(0, total - 1);
    return true;
  }

  /** Stop for good: requests in flight land nowhere and new ones are not made. */
  function stop() {
    stopped = true;
    generation += 1;
  }

  return {
    run,
    get entries() {
      return entries;
    },
    get total() {
      return total;
    },
    get loaded() {
      return loaded;
    },
    get failed() {
      return failed;
    },
    get hasMore() {
      return cursor !== null;
    },
    url,
    loadFirst,
    loadNext,
    find,
    add,
    update,
    remove,
    stop,
  };
}
