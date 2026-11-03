// @ts-check
/**
 * ARIEL panel store: the single state object behind everything the screen
 * shows as chosen, open or selected. No DOM, no imports.
 *
 * `setState` shallow-merges a patch; a key counts as changed when its value
 * differs structurally (object-valued keys are compared by content, not by
 * reference). Subscribers are told only when some key changed, or always when
 * the write carries `force`, in which case every key of the patch counts as
 * changed. Subscribers key off the `changed` set.
 */

/**
 * @typedef {'all'|'drafts'|'mine'|{saved: string}} Mailbox
 */

/**
 * @typedef {Object} State
 * @property {string} query
 * @property {string|null} mode
 * @property {Record<string, any>} filters
 * @property {'now'|'when-stable'} rerank
 * @property {Mailbox} mailbox
 * @property {string|null} sel
 * @property {string|null} att
 * @property {null|{draft: string|null}} compose
 * @property {null|'status'|'settings'} drawer
 * @property {boolean} railCollapsed
 * @property {boolean} stacked
 * @property {'list'|'reader'} narrowView
 * @property {string|null} mineAuthor
 */

/**
 * @callback Subscriber
 * @param {State} state
 * @param {State} prev
 * @param {Set<string>} changed
 * @returns {void}
 */

/** @type {State} */
let state = {
  query: '',
  mode: null,
  filters: {},
  rerank: 'now',
  mailbox: 'all',
  sel: null,
  att: null,
  compose: null,
  drawer: null,
  railCollapsed: false,
  stacked: false,
  narrowView: 'list',
  mineAuthor: null,
};

/** @type {Subscriber[]} */
let subscribers = [];

/**
 * Structural equality for the JSON-like values the store holds.
 * @param {unknown} a
 * @param {unknown} b
 * @returns {boolean}
 */
function sameValue(a, b) {
  if (Object.is(a, b)) return true;
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ra = /** @type {Record<string, unknown>} */ (a);
  const rb = /** @type {Record<string, unknown>} */ (b);
  const keys = Object.keys(ra);
  if (keys.length !== Object.keys(rb).length) return false;
  return keys.every((k) => Object.prototype.hasOwnProperty.call(rb, k) && sameValue(ra[k], rb[k]));
}

const SAVED_PREFIX = 'saved:';

/**
 * The name of a saved-search mailbox, or null for any other mailbox.
 * @param {Mailbox} mailbox
 * @returns {string|null}
 */
export function savedName(mailbox) {
  return typeof mailbox === 'object' && mailbox !== null ? mailbox.saved : null;
}

/**
 * A mailbox as one string: the hash's `box` value and the rail row's `data-box`.
 * @param {Mailbox} mailbox
 * @returns {string}
 */
export function mailboxKey(mailbox) {
  const saved = savedName(mailbox);
  return saved === null ? /** @type {string} */ (mailbox) : SAVED_PREFIX + saved;
}

/**
 * The mailbox a key names; `mailboxKey`'s inverse. Anything else is All entries.
 * @param {string|null} key
 * @returns {Mailbox}
 */
export function parseMailboxKey(key) {
  if (key === 'drafts' || key === 'mine') return key;
  if (key && key.startsWith(SAVED_PREFIX)) return { saved: key.slice(SAVED_PREFIX.length) };
  return 'all';
}

/** @returns {State} the current state; treat it as read-only. */
export function getState() {
  return state;
}

/**
 * Merge `patch` into the state and notify subscribers.
 * @param {Partial<State>} patch
 * @param {{force?: boolean}} [opts]
 */
export function setState(patch, opts = {}) {
  const force = opts.force === true;
  const p = /** @type {Record<string, unknown>} */ (patch);
  const current = /** @type {Record<string, unknown>} */ (/** @type {unknown} */ (state));
  /** @type {Set<string>} */
  const changed = new Set();
  for (const key of Object.keys(p)) {
    if (force || !sameValue(current[key], p[key])) changed.add(key);
  }
  if (!force && changed.size === 0) return;
  const prev = state;
  state = { ...state, ...patch };
  const next = state;
  for (const fn of subscribers.slice()) fn(next, prev, changed);
}

/**
 * Be told after every write that changed something (or was forced).
 * @param {Subscriber} fn
 * @returns {() => void} unsubscribe
 */
export function subscribe(fn) {
  subscribers.push(fn);
  return () => {
    subscribers = subscribers.filter((other) => other !== fn);
  };
}
