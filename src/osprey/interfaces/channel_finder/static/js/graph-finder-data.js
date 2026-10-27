// @ts-check
/**
 * OSPREY Channel Finder — graph finder reads and payload shapes.
 *
 * The two pieces of the graph finder that touch what the server sends rather
 * than what the operator sees: reading one graph endpoint while keeping the
 * remedy fields a failure carries, and building the class forest the rail
 * draws from the flat taxonomy the ontology endpoint answers with.
 *
 * The endpoints are read with `fetch` rather than `api.js`'s `fetchJSON`: the
 * graph routes carry `error_type` and `suggestions` beside `detail`, and
 * `fetchJSON` keeps only `detail`.
 */

import { messageOf } from './utils.js';

/**
 * One class of the store's taxonomy while the forest is being built. `parents`
 * is what the store declared; `parent` is the one it is drawn under.
 * @typedef {object} ClassNode
 * @property {string} uri
 * @property {string} name
 * @property {string[]} parents
 * @property {string|null} parent
 * @property {boolean} abstract
 * @property {ClassNode[]} children
 */

/**
 * A settled request: the payload, a failure carrying the store's own remedy, or
 * `null` when the request was abandoned because the view moved on.
 * @typedef {{ok: true, data: any}
 *   | {ok: false, status: number, detail: string, errorType: string, suggestions: string[]}
 *   | null} Reply
 */

/**
 * Read one graph endpoint, keeping the remedy fields a failure carries.
 *
 * @param {string} url - The url to read.
 * @param {AbortSignal} signal - Abort signal for the request.
 * @returns {Promise<Reply>} The payload, a failure, or null if aborted.
 */
export async function fetchJson(url, signal) {
  try {
    const resp = await fetch(url, { signal });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      return {
        ok: false,
        status: Number(resp.status) || 0,
        detail: String(body.detail || resp.statusText || 'The graph store did not answer.'),
        errorType: String(body.error_type || ''),
        suggestions: stringList(body.suggestions),
      };
    }
    return { ok: true, data: body };
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') return null;
    return { ok: false, status: 0, detail: messageOf(e), errorType: '', suggestions: [] };
  }
}

/**
 * Coerce an untyped payload field into a list of display strings.
 * @param {unknown} value - The field as the server sent it.
 * @returns {string[]} The strings it held, or an empty list.
 */
export function stringList(value) {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

/**
 * Build the class forest the rail draws from the flat taxonomy the ontology
 * endpoint answers with.
 *
 * The endpoint sends one row per class naming its parents by URI; the rail
 * needs those rows nested. A class with several known parents is attached under
 * the parent whose URI sorts first, parents absent from the payload are
 * ignored, and a `parents` cycle is broken by dropping the link that closes it,
 * so a corpus with a cyclic taxonomy still draws rather than recursing forever.
 * Siblings are ordered by name, URI breaking a tie, so the rail is stable
 * across reloads.
 *
 * @param {unknown} classes - The `classes` block of an ontology payload.
 * @returns {ClassNode[]} The roots, each carrying its children.
 */
export function buildForest(classes) {
  /** @type {Map<string, ClassNode>} */
  const nodes = new Map();
  for (const cls of Array.isArray(classes) ? classes : []) {
    const uri = cls && cls.uri != null ? String(cls.uri) : '';
    if (!uri || nodes.has(uri)) continue;
    const parents = Array.isArray(cls.parents) ? cls.parents.map(String) : [];
    const name = cls.name ? String(cls.name) : uri;
    const abstract = Boolean(cls.abstract);
    nodes.set(uri, { uri, name, parents, parent: null, abstract, children: [] });
  }

  for (const node of nodes.values()) {
    const known = node.parents.filter((uri) => uri !== node.uri && nodes.has(uri)).sort();
    node.parent = known.length > 0 ? known[0] : null;
  }

  // Break a cycle at a class that is ON it: the walk is only allowed to demote
  // a node it returns to, so a class that merely descends from a cycle keeps
  // its parent. Every cycle still breaks, because each of its members closes on
  // itself, and once the first one is demoted the rest walk out to a root.
  for (const node of nodes.values()) {
    const seen = new Set([node.uri]);
    let cursor = node.parent;
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      cursor = nodes.get(cursor)?.parent ?? null;
    }
    if (cursor === node.uri) node.parent = null;
  }

  /** @type {ClassNode[]} */
  const roots = [];
  for (const node of nodes.values()) {
    const parent = node.parent ? nodes.get(node.parent) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node);
  }

  /** @param {ClassNode[]} siblings @returns {void} */
  const order = (siblings) => {
    siblings.sort((a, b) => (a.name === b.name ? cmp(a.uri, b.uri) : cmp(a.name, b.name)));
    for (const node of siblings) order(node.children);
  };
  order(roots);
  return roots;
}

/**
 * The name the rail draws for a class URI, or the URI when the forest does not
 * hold it.
 * @param {ClassNode[]} forest
 * @param {string} uri
 * @returns {string}
 */
export function classNameIn(forest, uri) {
  /** @param {ClassNode[]} nodes @returns {string|null} */
  const find = (nodes) => {
    for (const node of nodes) {
      if (node.uri === uri) return node.name;
      const below = find(node.children);
      if (below) return below;
    }
    return null;
  };
  return find(forest) ?? uri;
}

/**
 * Compare two strings for a stable sort.
 * @param {string} a - First string.
 * @param {string} b - Second string.
 * @returns {number} Negative, zero or positive per `Array.prototype.sort`.
 */
function cmp(a, b) {
  return a === b ? 0 : a < b ? -1 : 1;
}
