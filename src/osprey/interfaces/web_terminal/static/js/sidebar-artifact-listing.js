// @ts-check
/* OSPREY Web Terminal — the session tree's artifact listings
 *
 * One listing is one paged `POST /api/artifacts/session-list` of the artifacts
 * panel: a session's artifacts (`{session}`) or the active key's (`{key}`),
 * optionally one form's (`form`), always with `runs: "grouped"` so a run lists once, at its head. Pages walk
 * the server's opaque `next_cursor`.
 *
 * A first page replaces what the listing holds and bumps a generation, so a
 * late answer from an older request is dropped; a next page appends
 * de-duplicated, one in flight at a time and never during a first page; a
 * failed request keeps the entries and the cursor.
 *
 * The module also reads one artifact by id, the agent's persisted focus, and
 * the session summary (`POST /api/artifacts/session-summary`).
 *
 * @module sidebar-artifact-listing
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { withPrefix } from './api.js';

/** The artifacts panel's mount on the hub. */
export const ARTIFACTS_BASE = '/panel/artifacts';

/** The session-list route. */
export const SESSION_LIST_PATH = `${ARTIFACTS_BASE}/api/artifacts/session-list`;

/** The session-summary route. */
export const SESSION_SUMMARY_PATH = `${ARTIFACTS_BASE}/api/artifacts/session-summary`;

/**
 * An artifact as the panel lists it (the fields the sidebar reads).
 * @typedef {object} ArtifactEntry
 * @property {string} id
 * @property {string} [title]
 * @property {string} [filename]
 * @property {string} [artifact_type]
 * @property {string} [timestamp]
 * @property {string} [session_id]
 * @property {string} [origin]
 * @property {boolean} [pinned]
 * @property {boolean} [run_record]
 * @property {string|null} [owner]
 * @property {boolean} [operator_visible]
 * @property {string|null} [form]
 * @property {string} [exec_id]
 * @property {string} [run_part]
 * @property {'succeeded' | 'failed' | 'not_run' | null} [run_status]
 */

/**
 * What a listing lists: exactly one of `session` and `key`, and an optional
 * form.
 * @typedef {{session?: string, key?: string, form?: string}} ListingScope
 */

/**
 * @typedef {ReturnType<typeof createListing>} Listing
 */

/**
 * GET (or *init*) one artifacts-panel route as JSON; throws on a non-OK answer.
 * @param {string} url
 * @param {RequestInit} [init]
 */
export async function requestJSON(url, init) {
  const resp = await fetch(withPrefix(url), { cache: 'no-store', ...init });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

/** @param {string} url @param {any} body */
export function postJSON(url, body) {
  return requestJSON(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/**
 * One artifact by id, or null when it cannot be read.
 * @param {string} id
 * @returns {Promise<ArtifactEntry | null>}
 */
export async function fetchArtifact(id) {
  try {
    return await requestJSON(`${ARTIFACTS_BASE}/api/artifacts/${encodeURIComponent(id)}`);
  } catch {
    return null;
  }
}

/**
 * The id of the artifact the agent focused, or null (none, or unreadable).
 * @returns {Promise<string | null>}
 */
export async function fetchFocusedId() {
  try {
    const data = await requestJSON(`${ARTIFACTS_BASE}/api/focus`);
    const id = data?.focused === true ? data.artifact?.id : null;
    return typeof id === 'string' ? id : null;
  } catch {
    return null;
  }
}

/**
 * The session summary for one page of transcript ids, or null on failure.
 * @param {string[]} sessions
 * @returns {Promise<{counts: Record<string, {total: number, by_form: Record<string, number>}>,
 *   owned_total: number, ledger: string} | null>}
 */
export async function fetchSummary(sessions) {
  try {
    return await postJSON(SESSION_SUMMARY_PATH, { sessions });
  } catch {
    return null;
  }
}

/**
 * The type registry (base forms and extension types), or null.
 * @returns {Promise<any>}
 */
export async function fetchTypeRegistry() {
  try {
    return await requestJSON(`${ARTIFACTS_BASE}/api/type-registry`);
  } catch {
    return null;
  }
}

/**
 * Newest first, ties by id, so a re-sort is stable.
 * @param {ArtifactEntry[]} entries
 * @returns {ArtifactEntry[]}
 */
export function newestFirst(entries) {
  return [...entries].sort(
    (a, b) =>
      (b.timestamp ?? '').localeCompare(a.timestamp ?? '') || (b.id ?? '').localeCompare(a.id ?? '')
  );
}

/** The key a run is de-duplicated by: its `exec_id`, else its id. @param {ArtifactEntry} a */
export function countKey(a) {
  return a.exec_id || a.id;
}

/**
 * One paged session listing.
 * @param {{scope: ListingScope, onChange: () => void, limit?: number}} options
 */
export function createListing({ scope, onChange, limit }) {
  /** @type {ArtifactEntry[]} */
  let entries = [];
  let total = 0;
  /** @type {string | null} */
  let cursor = null;
  let loaded = false;
  let failed = false;
  let stopped = false;
  let generation = 0;
  let loading = false;
  /** @type {object | null} */
  let paging = null;
  /** Entries held on their own (a reveal) that the total does not count. */
  const uncounted = new Set();

  /**
   * The request body of a page.
   * @param {string | null} [nextCursor]
   */
  function body(nextCursor = null) {
    /** @type {Record<string, any>} */
    const out = { ...scope, runs: 'grouped' };
    if (nextCursor) out.cursor = nextCursor;
    if (limit) out.limit = limit;
    return out;
  }

  async function loadFirst() {
    if (stopped) return;
    const mine = ++generation;
    loading = true;
    try {
      const data = await postJSON(SESSION_LIST_PATH, body());
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

  /** @returns {Promise<boolean>} whether a page was appended */
  async function loadNext() {
    if (stopped || cursor === null || paging || loading) return false;
    const token = {};
    paging = token;
    const mine = generation;
    try {
      const data = await postJSON(SESSION_LIST_PATH, body(cursor));
      if (mine !== generation) return false;
      const held = new Set(entries.map((a) => a.id));
      /** @type {ArtifactEntry[]} */
      const page = data.artifacts ?? [];
      for (const a of page) uncounted.delete(a.id);
      entries = newestFirst([...entries, ...page.filter((a) => !held.has(a.id))]);
      total = data.total ?? total;
      cursor = data.next_cursor ?? null;
    } catch {
      return false;
    } finally {
      if (paging === token) paging = null;
    }
    onChange();
    return true;
  }

  /** @param {string} id */
  function find(id) {
    return entries.find((a) => a.id === id) ?? null;
  }

  /**
   * Hold an entry. A run already held (same `exec_id`) is replaced in place.
   * @param {ArtifactEntry} entry
   * @param {{counted?: boolean}} [options]
   */
  function add(entry, { counted = true } = {}) {
    const existing = entries.find(
      (a) => a.id === entry.id || (entry.exec_id && a.exec_id === entry.exec_id)
    );
    if (existing) {
      if (counted && uncounted.delete(existing.id)) total += 1;
      if (existing.id !== entry.id)
        entries = newestFirst([...entries.filter((a) => a !== existing), entry]);
      return;
    }
    entries = newestFirst([...entries, entry]);
    if (counted) total += 1;
    else uncounted.add(entry.id);
  }

  /** @param {Partial<ArtifactEntry> & {id: string}} fields */
  function update(fields) {
    const index = entries.findIndex((a) => a.id === fields.id);
    if (index < 0) return false;
    const next = [...entries];
    next[index] = { ...next[index], ...fields };
    entries = newestFirst(next);
    return true;
  }

  /** @param {string} id */
  function remove(id) {
    const kept = entries.filter((a) => a.id !== id);
    if (kept.length === entries.length) return false;
    entries = kept;
    if (!uncounted.delete(id)) total = Math.max(0, total - 1);
    return true;
  }

  function stop() {
    stopped = true;
    generation += 1;
  }

  return {
    scope,
    get entries() {
      return entries;
    },
    get total() {
      return total;
    },
    get loaded() {
      return loaded;
    },
    get loading() {
      return loading;
    },
    get failed() {
      return failed;
    },
    get hasMore() {
      return cursor !== null;
    },
    body,
    loadFirst,
    loadNext,
    find,
    add,
    update,
    remove,
    stop,
  };
}

/* ---- The tree's data: listings, summary counts, live frames ---------- */

/** Sessions summarized per request: one page of `GET /api/sessions`. */
const SUMMARY_PAGE = 200;

/** How many pages a reveal walks to find a run's head. */
const REVEAL_MAX_PAGES = 20;

/** Base key of the persisted tree mode, scoped per user. */
export const VIEW_STORAGE_KEY = 'osprey-sidebar-artifacts-view';

/** @returns {'kind' | 'activity'} the persisted tree mode, Activity by default */
export function readTreeView() {
  try {
    return localStorage.getItem(scopedStorageKey(VIEW_STORAGE_KEY)) === 'kind'
      ? 'kind'
      : 'activity';
  } catch {
    return 'activity';
  }
}

/** @param {'kind' | 'activity'} view */
export function writeTreeView(view) {
  try {
    localStorage.setItem(scopedStorageKey(VIEW_STORAGE_KEY), view);
  } catch {
    /* storage blocked: the mode holds for this page */
  }
}

/**
 * What a tree id lists: a transcript, or a key (`key:<K>`, a session the
 * listing has not caught up with).
 * @param {string} treeId
 * @param {string} [form]
 * @returns {ListingScope}
 */
export function scopeOf(treeId, form = '') {
  /** @type {ListingScope} */
  const scope = treeId.startsWith('key:') ? { key: treeId.slice(4) } : { session: treeId };
  if (form) scope.form = form;
  return scope;
}

/** @typedef {{total: number, by_form: Record<string, number>}} FormCounts */

/**
 * The session tree's data: one listing per session (`${treeId}|`) and per
 * open group (`${treeId}|${form}`), every entry seen by id, the summary
 * counts, and the run keys already counted. Live frames count once per run
 * (`exec_id`, else id), only when `operator_visible` and owned, by the listed
 * form.
 * @param {{onChange: () => void}} options
 */
export function createTreeData({ onChange }) {
  /** @type {Map<string, Listing>} */
  let listings = new Map();
  /** @type {Map<string, ArtifactEntry>} */
  const held = new Map();
  /** @type {Set<string>} */
  const counted = new Set();
  /** @type {Set<string>} */
  let summarized = new Set();
  const totals = {
    /** @type {Record<string, FormCounts>} */
    counts: {},
    ownedTotal: 0,
  };

  /** @param {string} treeId @param {string} [form] */
  function listingOf(treeId, form = '') {
    const key = `${treeId}|${form}`;
    let listing = listings.get(key);
    if (!listing) {
      const made = createListing({
        scope: scopeOf(treeId, form),
        onChange: () => {
          for (const entry of made.entries) {
            held.set(entry.id, entry);
            counted.add(countKey(entry));
          }
          onChange();
        },
      });
      listing = made;
      listings.set(key, listing);
    }
    return listing;
  }

  /** @param {Listing} listing */
  function ensureLoaded(listing) {
    if (!listing.loaded && !listing.loading) void listing.loadFirst();
  }

  /**
   * Summarize the sessions not summarized yet, or all of them (`all`: a
   * delete, a reconnect), one request per page. Resolves whether it asked.
   * @param {string[]} ids
   * @param {boolean} [all]
   */
  async function summarize(ids, all = false) {
    const wanted = all ? ids : ids.filter((id) => !summarized.has(id));
    if (!wanted.length) return false;
    for (const id of wanted) summarized.add(id);
    const pages = [];
    for (let i = 0; i < wanted.length; i += SUMMARY_PAGE)
      pages.push(wanted.slice(i, i + SUMMARY_PAGE));
    const answers = await Promise.all(pages.map((page) => fetchSummary(page)));
    for (const summary of answers) {
      if (!summary) continue;
      Object.assign(totals.counts, summary.counts ?? {});
      totals.ownedTotal = summary.owned_total ?? totals.ownedTotal;
    }
    return true;
  }

  /** An unowned entry counts nowhere. @param {ArtifactEntry} entry @param {number} delta */
  function bump(entry, delta) {
    const form = entry.form;
    if (!form || entry.owner == null) return;
    const bucket = (totals.counts[entry.owner] ??= { total: 0, by_form: {} });
    bucket.total = Math.max(0, bucket.total + delta);
    bucket.by_form[form] = Math.max(0, (bucket.by_form[form] ?? 0) + delta);
    totals.ownedTotal = Math.max(0, totals.ownedTotal + delta);
  }

  /**
   * Put an entry into the loaded listings of its session (and `extra`, the
   * active key's listing when the entry is the active session's). An unowned
   * entry has no session's.
   * @param {ArtifactEntry} entry
   * @param {string[]} extra
   */
  function place(entry, extra) {
    const treeId = entry.owner;
    const own =
      treeId == null ? [] : [`${treeId}|`, ...(entry.form ? [`${treeId}|${entry.form}`] : [])];
    const keys = [...own, ...extra];
    for (const key of keys) {
      const listing = listings.get(key);
      if (listing?.loaded) listing.add(entry);
    }
  }

  /** @param {string} id */
  function unplace(id) {
    for (const listing of listings.values()) listing.remove(id);
  }

  /** A counted run's later frame only takes the new status. @param {ArtifactEntry} entry */
  function setStatus(entry) {
    for (const listing of listings.values()) {
      const head = listing.entries.find(
        (a) => a.id === entry.id || (entry.exec_id && a.exec_id === entry.exec_id)
      );
      if (head) listing.update({ id: head.id, run_status: entry.run_status ?? null });
    }
  }

  /**
   * An `artifact` frame. Resolves false for one the tree never shows.
   * @param {ArtifactEntry} entry
   * @param {string[]} extra  see {@link place}
   */
  function added(entry, extra) {
    if (entry.operator_visible === false) return false;
    const key = countKey(entry);
    held.set(entry.id, entry);
    if (counted.has(key)) {
      setStatus(entry);
    } else {
      counted.add(key);
      bump(entry, 1);
      place(entry, extra);
    }
    return true;
  }

  /**
   * An `artifact_updated` frame: the status follows, and an entry whose
   * owner changed is re-placed with its count. Resolves whether it moved.
   * @param {ArtifactEntry} entry
   * @param {string[]} extra  see {@link place}
   */
  function updated(entry, extra) {
    const before = held.get(entry.id) ?? null;
    held.set(entry.id, { ...(before ?? {}), ...entry });
    if (before && before.owner !== entry.owner && entry.operator_visible !== false) {
      bump(before, -1);
      unplace(entry.id);
      bump(entry, 1);
      place(entry, extra);
      return true;
    }
    for (const listing of listings.values()) listing.update(entry);
    return false;
  }

  /** An `artifact_deleted` frame. @param {string} id */
  function deleted(id) {
    unplace(id);
    held.delete(id);
  }

  /**
   * What an open session's box lists: the whole session (Activity mode) or
   * its open groups (Kind mode), with the counts its group heads show (a
   * group that holds rows shows even before the summary counts them).
   * @param {string} treeId
   * @param {'kind' | 'activity'} view
   * @param {Set<string>} openForms
   * @returns {{entries: import('./sidebar-artifact-rows.js').TreeEntry[],
   *   counts: Record<string, number> | null}}
   */
  function boxEntries(treeId, view, openForms) {
    if (view === 'activity') {
      const listing = listingOf(treeId);
      ensureLoaded(listing);
      return { entries: listing.entries, counts: null };
    }
    const bucket = totals.counts[treeId];
    /** @type {Record<string, number> | null} */
    const counts = bucket ? { ...bucket.by_form } : null;
    /** @type {ArtifactEntry[]} */
    const entries = [];
    for (const form of openForms) {
      const listing = listingOf(treeId, form);
      ensureLoaded(listing);
      entries.push(...listing.entries);
      if (counts) counts[form] = Math.max(counts[form] ?? 0, listing.entries.length);
    }
    return { entries, counts };
  }

  /**
   * The next page of what a box lists.
   * @param {string} treeId @param {'kind' | 'activity'} view @param {Set<string>} openForms
   */
  function loadMoreOf(treeId, view, openForms) {
    if (view === 'activity') void listingOf(treeId).loadNext();
    else for (const form of openForms) void listingOf(treeId, form).loadNext();
  }

  /**
   * The whole-session listing of the active row: by key once first paint
   * listed it so (before the session rows landed), else by its tree id.
   * @param {{session_key: string | null, transcript_id: string | null}} row
   */
  function activeListing(row) {
    return (
      listings.get(`key:${row.session_key}|`) ??
      listingOf(row.transcript_id ?? `key:${row.session_key}`)
    );
  }

  /**
   * Where `id` sits in a listing, for the pager.
   * @param {Listing} listing @param {string} id
   */
  function position(listing, id) {
    ensureLoaded(listing);
    const order = listing.entries;
    const index = order.findIndex((a) => a.id === id);
    if (index < 0) return null;
    return {
      index,
      count: Math.max(listing.total, order.length),
      last: index === order.length - 1 && !listing.hasMore,
    };
  }

  /**
   * The entry `delta` steps from `id`, fetching the next page past the end.
   * @param {Listing} listing @param {string} id @param {1 | -1} delta
   */
  async function stepFrom(listing, id, delta) {
    if (delta > 0 && listing.entries.at(-1)?.id === id) await listing.loadNext();
    const index = listing.entries.findIndex((a) => a.id === id);
    return index < 0 ? null : (listing.entries[index + delta] ?? null);
  }

  /**
   * Put an entry into its Kind-mode group for a reveal: a run's part waits
   * for its head's page (walking the cursor), any other entry is inserted.
   * Resolves the group's form and, for a run part, its run key.
   * @param {string} treeId @param {ArtifactEntry} entry
   */
  async function revealIn(treeId, entry) {
    const runKey = entry.exec_id && entry.form !== 'runs' ? entry.exec_id : null;
    const form = runKey ? 'runs' : (entry.form ?? null);
    if (!form) return { form, runKey };
    const listing = listingOf(treeId, form);
    if (!listing.loaded) await listing.loadFirst();
    if (runKey) {
      const has = () => listing.entries.some((a) => a.exec_id === runKey);
      for (let page = 0; !has() && listing.hasMore && page < REVEAL_MAX_PAGES; page++) {
        if (!(await listing.loadNext())) break;
      }
    } else if (!listing.find(entry.id)) {
      listing.add(entry, { counted: false });
    }
    return { form, runKey };
  }

  function stop() {
    for (const listing of listings.values()) listing.stop();
    listings = new Map();
    summarized = new Set();
    held.clear();
    counted.clear();
    totals.counts = {};
    totals.ownedTotal = 0;
  }

  return {
    totals,
    listingOf,
    /** @param {string} key */
    listing: (key) => listings.get(key) ?? null,
    ensureLoaded,
    summarize,
    /** @param {ArtifactEntry} entry */
    hold: (entry) => void held.set(entry.id, entry),
    /** @param {string | null} id */
    held: (id) => (id ? (held.get(id) ?? null) : null),
    added,
    updated,
    deleted,
    boxEntries,
    loadMoreOf,
    activeListing,
    position,
    stepFrom,
    revealIn,
    stop,
  };
}
