// @ts-check
/**
 * ARIEL panel router: maps the location hash onto the store and keeps the
 * hash canonical.
 *
 * Canonical form: `#q=<query>&mode=<mode>&box=<all|drafts|mine|saved:<name>>&sel=<id>&att=<id>`,
 * URL-encoded, keys omitted when empty. Legacy aliases (`#entry?id=X&attachment=Y`,
 * `#create`, `#create?draft=ID`, `#search`, `#browse`, `#status`) are an
 * external contract — MCP tools, the artifacts logbook and the docs link to
 * them — and are applied with `force`, so following the same link twice always
 * takes effect. After any hash is applied the canonical hash is written back
 * with `history.replaceState`, which fires no `hashchange`; a repeated alias
 * therefore always arrives as a real fragment change.
 */

import { getState, setState, subscribe, mailboxKey, parseMailboxKey } from './store.js';
import { isEmbedded } from '/design-system/js/frame-params.js';

/** @typedef {import('./store.js').State} State */

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
 * describes the query, mailbox and selection.
 * @param {string} hash with or without the leading `#`
 * @returns {Partial<State>}
 */
export function parseHash(hash) {
  const { alias, params } = classify(hash);
  if (alias === '') return {};
  if (alias === null) {
    const sel = params.get('sel') || null;
    /** @type {Partial<State>} */
    const patch = {
      query: params.get('q') || '',
      mode: params.get('mode') || null,
      mailbox: parseMailboxKey(params.get('box')),
      sel,
      att: params.get('att') || null,
      rerank: 'now',
    };
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
      return { compose: { draft: params.get('draft') || null }, narrowView: 'list', rerank: 'now' };
    case 'status':
      // Status is hidden in an embedded frame, so the link shows the list there.
      return isEmbedded() ? listPatch() : { drawer: 'status', rerank: 'now' };
    default:
      // `search`, `browse`, and any unknown name.
      return listPatch();
  }
}

/**
 * The canonical hash for a state, with its leading `#`.
 * @param {State} state
 * @returns {string}
 */
export function serializeState(state) {
  /** @type {Array<[string, string|null|undefined]>} */
  const pairs = [
    ['q', state.query],
    ['mode', state.mode],
    ['box', mailboxKey(state.mailbox)],
    ['sel', state.sel],
    ['att', state.att],
  ];
  const parts = pairs
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .map(([key, value]) => `${key}=${encodeURIComponent(/** @type {string} */ (value))}`);
  return `#${parts.join('&')}`;
}

let applying = false;
let started = false;

/** Write the canonical hash for the current state, when it differs. */
function writeHash() {
  const next = serializeState(getState());
  if (window.location.hash !== next) history.replaceState(history.state, '', next);
}

/** Apply the current location hash to the store, then write the canonical hash once. */
function applyLocationHash() {
  const hash = window.location.hash;
  const patch = parseHash(hash);
  applying = true;
  try {
    if (Object.keys(patch).length > 0) {
      setState(patch, { force: classify(hash).alias !== null });
    }
  } finally {
    applying = false;
  }
  writeHash();
}

/**
 * Apply the current hash, follow every `hashchange`, and keep the hash
 * canonical as the store changes. Call once, after every store subscriber
 * exists, so the initial hash reaches them. Further calls do nothing.
 */
export function initRouter() {
  if (started) return;
  started = true;
  subscribe(() => {
    if (!applying) writeHash();
  });
  window.addEventListener('hashchange', applyLocationHash);
  applyLocationHash();
}
