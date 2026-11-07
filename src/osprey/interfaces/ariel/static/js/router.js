// @ts-check
/**
 * ARIEL panel router: maps the location hash onto the store and keeps the
 * hash canonical.
 *
 * Canonical form: `#q=<query>&mode=<mode>&box=<all|drafts|mine|saved:<name>>&f=<filter>&sel=<id>&att=<id>&compose=draft:<id>`,
 * URL-encoded, keys omitted when empty. `f` is the store's C-FILTER `filter`
 * as compact JSON with sorted keys. Under a `saved:` box `q` and `f` are always
 * written, even empty (`q=&f={}`), so a reload never brings back a query or
 * filter the user cleared: a saved box restores, from its record, only the
 * fields whose hash key is absent (a hash `f` always wins). `compose` is written
 * only for an open draft; a new entry or follow-up never reaches the hash, so a
 * canonical hash without `compose` leaves the sheet alone, except that hash
 * navigation (Back/Forward) away from a `compose=draft:` hash closes it. A
 * `compose` value without the `draft:` prefix or with an empty id is ignored;
 * the id's format is the server's to check. Every parsed filter
 * goes through `sanitizeFilter` against the deployment's dimensions; the names
 * it drops become the store's `notice`. Legacy aliases (`#entry?id=X&attachment=Y`,
 * `#create`, `#create?draft=ID`, `#search`, `#browse`, `#status`) are an
 * external contract — MCP tools, the artifacts logbook and the docs link to
 * them — and are applied with `force`, so following the same link twice always
 * takes effect. After any hash is applied the canonical hash is written back
 * with `history.replaceState`, which fires no `hashchange`; a repeated alias
 * therefore always arrives as a real fragment change.
 *
 * Embedded, the router also takes the host's `canopy-host-navigate` (main's
 * panel-navigate contract): it says `canopy-host-navigate-ready` once its
 * listener is installed and again on every probe, and accepts navigate and
 * probe messages only from `window.parent`, same origin, for urls under the
 * page's own mount. A url with the current path and a sel-only hash (`sel`,
 * optional `att`) is an intent for `selectEntry`, which keeps the search; an
 * alias keeps its alias meaning; any other canonical hash applies its full
 * state; a different path under the mount reloads there. All are forced, so
 * the same message twice takes effect twice. `hashchange`, `parseHash` and
 * fresh loads do not take this path: a sel-only hash there is still the full
 * canonical state.
 */

import { getState, setState, subscribe, mailboxKey, parseMailboxKey, savedName } from './store.js';
import { normalizeFilter, sanitizeFilter, mergeFilter } from './filter-state.js';
import { findSaved } from './saved-searches.js';
import { showToast } from '/design-system/js/toast.js';
import { flashElement } from '/design-system/js/highlight.js';

/** @typedef {import('./store.js').State} State */

/**
 * The deployment's filter dimensions (`capabilities.filter_dimensions`), set
 * by `initRouter`; null while unknown, which keeps every dimension.
 * @type {Array<{name: string}>|null}
 */
let dimensions = null;

/**
 * The `f` value of a hash as a filter; unreadable JSON is no filter.
 * @param {string} raw
 * @returns {unknown}
 */
function readFilterParam(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

const DRAFT_PREFIX = 'draft:';

/**
 * The draft id a `compose` value names, or null when it is absent or malformed
 * (no `draft:` prefix, or an empty id).
 * @param {string|null} raw
 * @returns {string|null}
 */
function composeDraft(raw) {
  if (!raw || !raw.startsWith(DRAFT_PREFIX)) return null;
  return raw.slice(DRAFT_PREFIX.length) || null;
}

/**
 * The notice for dimensions a restored filter lost, or null for none.
 * @param {string[]} dropped
 * @returns {string|null}
 */
function droppedNotice(dropped) {
  if (dropped.length === 0) return null;
  const noun = dropped.length === 1 ? 'filter' : 'filters';
  return `Removed ${noun} this logbook does not offer: ${dropped.join(', ')}`;
}

/**
 * The patch for "show the list": empty query, nothing selected or open.
 * @returns {Partial<State>}
 */
function listPatch() {
  return {
    query: '',
    mailbox: 'all',
    sel: null,
    att: null,
    compose: null,
    drawer: null,
    narrowView: 'list',
    rerank: 'now',
  };
}

/**
 * Split a hash into its alias name (null for a canonical hash) and params.
 * @param {string} hash
 * @returns {{alias: string|null, params: URLSearchParams}}
 */
function classify(hash) {
  const body = hash.startsWith('#') ? hash.slice(1) : hash;
  const qIdx = body.indexOf('?');
  const head = qIdx === -1 ? body : body.slice(0, qIdx);
  if (head.includes('=')) return { alias: null, params: new URLSearchParams(body) };
  return { alias: head, params: new URLSearchParams(qIdx === -1 ? '' : body.slice(qIdx + 1)) };
}

/**
 * Translate a hash into the store patch it stands for. An empty hash is no
 * patch. Omitted canonical keys come back empty, so a canonical hash fully
 * describes the query, filter, mailbox and selection; under `box=saved:<name>`
 * an omitted `q`, `mode` or `f` comes from that saved search instead, and its
 * knobs (`filters`) are always restored, all in this one patch.
 * @param {string} hash with or without the leading `#`
 * @returns {Partial<State>}
 */
export function parseHash(hash) {
  const { alias, params } = classify(hash);
  if (alias === '') return {};
  if (alias === null) {
    const sel = params.get('sel') || null;
    const mailbox = parseMailboxKey(params.get('box'));
    const name = savedName(mailbox);
    const saved = name === null ? undefined : findSaved(name);
    /** @type {unknown} */
    let rawFilter = {};
    if (params.has('f')) rawFilter = readFilterParam(/** @type {string} */ (params.get('f')));
    else if (saved) rawFilter = saved.filter;
    const sanitized = sanitizeFilter(rawFilter, dimensions);
    let filter = normalizeFilter(sanitized.filter);
    // Mine scopes the author itself (requestFilter), so its own selection goes.
    if (mailbox === 'mine') filter = mergeFilter(filter, { author: null });
    /** @type {Partial<State>} */
    const patch = {
      query: params.has('q') ? params.get('q') || '' : (saved?.query ?? ''),
      mode: params.has('mode') ? params.get('mode') || null : (saved?.mode ?? null),
      mailbox,
      filter,
      notice: droppedNotice(sanitized.dropped),
      sel,
      att: params.get('att') || null,
      rerank: 'now',
    };
    // The knobs have no hash key: a saved box always restores them.
    if (saved) patch.filters = structuredClone(saved.filters);
    const draft = composeDraft(params.get('compose'));
    if (draft !== null) patch.compose = { draft };
    if (sel) patch.narrowView = 'reader';
    return patch;
  }
  switch (alias) {
    case 'entry': {
      const id = params.get('id');
      if (!id) return listPatch();
      return {
        sel: id,
        att: params.get('attachment') || null,
        narrowView: 'reader',
        rerank: 'now',
      };
    }
    case 'create':
      return { compose: { draft: params.get('draft') || null }, rerank: 'now' };
    default:
      // `search`, `browse`, `status`, and any unknown name.
      return listPatch();
  }
}

/**
 * The canonical hash for a state, with its leading `#`.
 * @param {State} state
 * @returns {string}
 */
export function serializeState(state) {
  const saved = savedName(state.mailbox) !== null;
  const filter = normalizeFilter(state.filter);
  /** @type {Array<[string, string|null|undefined, boolean]>} [key, value, written even when empty] */
  const pairs = [
    ['q', state.query ?? '', saved],
    ['mode', state.mode, false],
    ['box', mailboxKey(state.mailbox), false],
    ['f', Object.keys(filter).length ? JSON.stringify(filter) : saved ? '{}' : '', saved],
    ['sel', state.sel, false],
    ['att', state.att, false],
    ['compose', state.compose?.draft ? DRAFT_PREFIX + state.compose.draft : null, false],
  ];
  const parts = pairs
    .filter(
      ([, value, always]) => always || (value !== null && value !== undefined && value !== '')
    )
    .map(([key, value]) => `${key}=${encodeURIComponent(/** @type {string} */ (value ?? ''))}`);
  return `#${parts.join('&')}`;
}

let applying = false;
let started = false;
/**
 * The hash as the router last left it, i.e. the one a `hashchange` navigates
 * away from; null before the first apply.
 * @type {string|null}
 */
let lastHash = null;

/** Write the canonical hash for the current state, when it differs. */
function writeHash() {
  const next = serializeState(getState());
  if (window.location.hash !== next) history.replaceState(history.state, '', next);
  lastHash = window.location.hash;
}

/**
 * Apply a patch as the router, so the store subscriber does not write the
 * hash mid-patch, then write the canonical hash once.
 * @param {Partial<State>} patch
 * @param {boolean} force
 */
function applyPatch(patch, force) {
  applying = true;
  try {
    if (Object.keys(patch).length > 0) setState(patch, { force });
  } finally {
    applying = false;
  }
  writeHash();
}

/** Apply the current location hash to the store, then write the canonical hash once. */
function applyLocationHash() {
  const hash = window.location.hash;
  const patch = parseHash(hash);
  // Back/Forward away from an open draft closes it; nothing else does.
  if (!('compose' in patch) && lastHash !== null) {
    const left = classify(lastHash);
    if (left.alias === null && composeDraft(left.params.get('compose')) !== null) {
      patch.compose = null;
    }
  }
  applyPatch(patch, classify(hash).alias !== null);
}

const NAVIGATE = 'canopy-host-navigate';
const NAVIGATE_PROBE = 'canopy-host-navigate-probe';
const NAVIGATE_READY = 'canopy-host-navigate-ready';
const ROW_SELECTOR = '#ariel-list [data-entry-id]';

/**
 * A path without its trailing slashes, so `/panel/ariel/` and `/panel/ariel`
 * compare equal; the root is the empty string.
 * @param {string} path
 * @returns {string}
 */
function trimPath(path) {
  return path.replace(/\/+$/, '');
}

/**
 * Show one entry without leaving the current search: `sel` and `att` are
 * applied forced (the same entry twice re-selects), `q/mode/box/f`, the list
 * and its scroll stay, and the painted row for `sel` flashes. When the list
 * paints no row for it the reader still opens, and a toast says so.
 * @param {{sel: string, att?: string|null}} intent
 */
export function selectEntry({ sel, att = null }) {
  applyPatch({ sel, att: att || null, narrowView: 'reader' }, true);
  const rows = /** @type {HTMLElement[]} */ (Array.from(document.querySelectorAll(ROW_SELECTOR)));
  const row = rows.find((el) => el.dataset.entryId === sel);
  if (row) {
    row.scrollIntoView?.({ block: 'nearest' });
    flashElement(row);
  } else {
    showToast('This entry is not shown in the list.');
  }
}

/**
 * Whether a parsed hash is an intent: canonical, with `sel` and at most `att`.
 * @param {{alias: string|null, params: URLSearchParams}} parsed
 * @returns {boolean}
 */
function isIntent({ alias, params }) {
  if (alias !== null || !params.get('sel')) return false;
  return [...params.keys()].every((key) => key === 'sel' || key === 'att');
}

/**
 * Handle one `canopy-host-navigate` url: ignored unless it is same origin and
 * under this page's mount, a reload when its path differs from the current
 * one (keeping `location.search`), else its hash classified and applied.
 * @param {unknown} raw
 */
function navigateTo(raw) {
  if (typeof raw !== 'string') return;
  /** @type {URL} */
  let url;
  try {
    url = new URL(raw, window.location.href);
  } catch {
    return;
  }
  if (url.origin !== window.location.origin) return;
  const mount = trimPath(window.location.pathname);
  const path = trimPath(url.pathname);
  if (path !== mount && !path.startsWith(`${mount}/`)) return;
  if (path !== mount) {
    window.location.assign(url.pathname + window.location.search + url.hash);
    return;
  }
  const parsed = classify(url.hash);
  if (isIntent(parsed)) {
    selectEntry({
      sel: /** @type {string} */ (parsed.params.get('sel')),
      att: parsed.params.get('att'),
    });
    return;
  }
  applyPatch(parseHash(url.hash), true);
}

/** Tell the host this page handles `canopy-host-navigate`. */
function sayReady() {
  window.parent.postMessage({ type: NAVIGATE_READY }, window.location.origin);
}

/**
 * The embedded navigate listener: probes and navigates from the parent,
 * same origin, only.
 * @param {MessageEvent} event
 */
function onMessage(event) {
  if (event.source !== window.parent || event.origin !== window.location.origin) return;
  const type = event.data?.type;
  if (type === NAVIGATE_PROBE) sayReady();
  else if (type === NAVIGATE) navigateTo(event.data.url);
}

/**
 * Apply the current hash, follow every `hashchange`, and keep the hash
 * canonical as the store changes. Call once, after every store subscriber
 * exists and capabilities are known, so the initial hash reaches them and its
 * filter is checked against the deployment's dimensions. Further calls do
 * nothing.
 * @param {{dimensions?: Array<{name: string}>|null}} [opts] - `capabilities.filter_dimensions`;
 *   null or absent keeps every dimension
 */
export function initRouter(opts = {}) {
  if (started) return;
  started = true;
  dimensions = opts.dimensions ?? null;
  subscribe(() => {
    if (!applying) writeHash();
  });
  window.addEventListener('hashchange', applyLocationHash);
  applyLocationHash();
  // After the initial hash, so a navigate never races the fresh load. A probe
  // the host sent before this point is answered by the unsolicited ready.
  window.addEventListener('message', onMessage);
  sayReady();
}
