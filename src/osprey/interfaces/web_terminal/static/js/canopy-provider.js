// @ts-check
/* OSPREY Web Terminal — CANOPY search provider for the command palette
 *
 * Feeds the palette's CANOPY group: the CANOPY screens and collections the
 * operator can open, ranked for the typed query. The palette renders the rows
 * in the order this module returns them and applies no matcher of its own.
 *
 * Data is read on every palette open — the CANOPY manifest and the collection
 * list — with the same newest-read-wins sequence the palette's own config read
 * uses: a read that settles after a newer one started is dropped. While the
 * reads are in flight `search` answers null (the palette shows "Loading…");
 * a read that fails contributes no rows.
 *
 * Ranking is a port of CANOPY's screen matcher (the viewer's Quick open and
 * sidebar filter share it). It is pinned by a byte copy of CANOPY's
 * tests/canopy_mcp/fixtures/collections/match-vectors.json, held here at
 * tests/integrations/canopy/fixtures/match-vectors.json and run in full by
 * canopy-provider.test.mjs.
 *
 *   Words: a label or keyword is split on non-alphanumerics, then at
 *   lower->Upper, Upper->Upper+lower and letter<->digit boundaries, and each
 *   part is lower-cased (GTLView -> gtl, view; SR01C -> sr, 01, c).
 *
 *   A query is lower-cased and split on whitespace into tokens; no tokens match
 *   nothing. An entry matches when every token is a prefix of some word of its
 *   label or a keyword, or of the joined words of that label or keyword.
 *   Matches are ordered by: all tokens matched in the label first; then the
 *   first token prefixing the first label word; then more label words prefixed
 *   by some token; then list order.
 *
 * Picking a row opens it in the CANOPY panel through the injected `navigate`
 * (the app passes navigateAndActivatePanel), never through a message of its
 * own: a screen at `/panel/canopy/embed/<category>/<id>`, a collection at
 * `/panel/canopy/embed/?_canopyOpen=<source>:<id>`.
 */

import { fetchJSON, withPrefix } from './api.js';

/** The panel id CANOPY is mounted under. */
export const CANOPY_PANEL_ID = 'canopy';

/** At most this many CANOPY rows are offered for one query. */
export const CANOPY_CAP = 8;

/** Where the screens manifest and the collection list are read. */
export const MANIFEST_PATH = '/panel/canopy/drafts/manifest.json';
export const COLLECTIONS_PATH = '/panel/canopy/embed/collections';

/**
 * @typedef {{ label: string, keywords?: readonly string[] }} Matchable
 * @typedef {readonly [number, number, number]} MatchRank
 * @typedef {{ label: string, detail?: string, run: () => void }} CanopyItem
 * @typedef {{ id: string, category: string, kind: string, title?: string }} ManifestScreen
 * @typedef {{ ref: string, name: string, source?: string, members: unknown[] }} ListedCollection
 */

const BOUNDARY =
  /(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])/;

/**
 * The lower-case words of a label or keyword.
 * @param {string} text
 * @returns {string[]}
 */
export function words(text) {
  /** @type {string[]} */
  const out = [];
  for (const chunk of text.split(/[^A-Za-z0-9]+/)) {
    if (chunk === '') continue;
    for (const part of chunk.split(BOUNDARY)) {
      if (part !== '') out.push(part.toLowerCase());
    }
  }
  return out;
}

/** @param {string} query @returns {string[]} */
function tokens(query) {
  return query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t !== '');
}

/**
 * True when `token` prefixes one of `parts` or their joined string.
 * @param {string} token
 * @param {readonly string[]} parts
 */
function hits(token, parts) {
  return parts.some((w) => w.startsWith(token)) || parts.join('').startsWith(token);
}

/**
 * The rank of `entry` for `query` (lower is better, compared element by
 * element), or null when it does not match.
 * @param {string} query
 * @param {Matchable} entry
 * @returns {MatchRank | null}
 */
export function matchEntry(query, entry) {
  const ts = tokens(query);
  if (ts.length === 0) return null;
  const label = words(entry.label);
  const keywords = (entry.keywords ?? []).map(words);
  let allInLabel = true;
  for (const t of ts) {
    if (hits(t, label)) continue;
    allInLabel = false;
    if (!keywords.some((k) => hits(t, k))) return null;
  }
  const first = label.length > 0 && label[0].startsWith(ts[0]);
  const prefixed = label.filter((w) => ts.some((t) => w.startsWith(t))).length;
  return [allInLabel ? 0 : 1, first ? 0 : 1, -prefixed];
}

/** @param {MatchRank} a @param {MatchRank} b */
function compare(a, b) {
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    if (d !== 0) return d;
  }
  return 0;
}

/**
 * The entries matching `query`, best first; ties keep list order. A blank
 * query matches nothing.
 * @template {Matchable} E
 * @param {readonly E[]} entries
 * @param {string} query
 * @returns {E[]}
 */
export function rankEntries(entries, query) {
  /** @type {Array<{ entry: E, rank: MatchRank, index: number }>} */
  const ranked = [];
  entries.forEach((entry, index) => {
    const rank = matchEntry(query, entry);
    if (rank) ranked.push({ entry, rank, index });
  });
  ranked.sort((a, b) => compare(a.rank, b.rank) || a.index - b.index);
  return ranked.map((r) => r.entry);
}

/**
 * A sub-screen id has a path segment `embedded` or a basename starting `_`.
 * @param {string} id
 */
export function isSubScreen(id) {
  const segments = id.split('/');
  const basename = segments[segments.length - 1] ?? '';
  return segments.includes('embedded') || basename.startsWith('_');
}

/**
 * Whether a manifest screen is searchable: an `ir` screen outside the `@`
 * categories, and not a sub-screen unless sub-screens are shown.
 * @param {{ id: string, category: string, kind: string }} entry
 * @param {{ showSubScreens: boolean }} options
 */
export function searchable(entry, options) {
  if (entry.kind !== 'ir' || entry.category.startsWith('@')) return false;
  return options.showSubScreens || !isSubScreen(entry.id);
}

/**
 * The CANOPY panel URL that shows `screen`.
 * @param {{ id: string, category: string }} screen
 */
export function screenUrl(screen) {
  const id = screen.id.split('/').map(encodeURIComponent).join('/');
  return withPrefix(`/panel/canopy/embed/${encodeURIComponent(screen.category)}/${id}`);
}

/**
 * The CANOPY panel URL that opens the collection `ref` ('<source>:<id>').
 * @param {string} ref
 */
export function collectionUrl(ref) {
  return withPrefix(`/panel/canopy/embed/?_canopyOpen=${encodeURIComponent(ref)}`);
}

/** @param {unknown} v @returns {v is Record<string, unknown>} */
function isRecord(v) {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * The searchable screens of a manifest body (anything malformed is skipped).
 * @param {unknown} body
 * @returns {ManifestScreen[]}
 */
function screensOf(body) {
  if (!isRecord(body) || !Array.isArray(body.screens)) return [];
  /** @type {ManifestScreen[]} */
  const out = [];
  for (const s of body.screens) {
    if (!isRecord(s)) continue;
    const { id, category, kind, title } = s;
    if (typeof id !== 'string' || typeof category !== 'string' || typeof kind !== 'string')
      continue;
    const screen = { id, category, kind, title: typeof title === 'string' ? title : undefined };
    if (searchable(screen, { showSubScreens: false })) out.push(screen);
  }
  return out;
}

/**
 * The listed collections of a collections body (anything malformed is skipped).
 * @param {unknown} body
 * @returns {ListedCollection[]}
 */
function collectionsOf(body) {
  if (!isRecord(body) || !Array.isArray(body.collections)) return [];
  /** @type {ListedCollection[]} */
  const out = [];
  for (const c of body.collections) {
    if (!isRecord(c)) continue;
    const { ref, name, source, members } = c;
    if (typeof ref !== 'string' || ref === '' || typeof name !== 'string') continue;
    if (!Array.isArray(members)) continue;
    out.push({ ref, name, source: typeof source === 'string' ? source : undefined, members });
  }
  return out;
}

/**
 * @typedef {{
 *   load?: (path: string) => Promise<unknown>,
 *   navigate: (panelId: string, url: string) => void,
 * }} ProviderDeps
 */

/**
 * Create the CANOPY provider. `start(onChange)` begins a fresh read of both
 * sources (call it on every palette open); `onChange` fires once the read
 * settles, unless a newer `start` superseded it. `search(q)` answers null while
 * the newest read is in flight, else at most CANOPY_CAP ranked rows (none for a
 * blank query).
 * @param {ProviderDeps} deps
 */
export function createCanopyProvider(deps) {
  const load = deps.load ?? fetchJSON;
  const navigate = deps.navigate;
  let fetchSeq = 0;
  let loading = false;
  /** @type {Array<Matchable & { run: () => void, text: string, detail?: string }>} */
  let entries = [];

  /** @param {() => void} [onChange] */
  function start(onChange) {
    const token = ++fetchSeq;
    loading = true;
    entries = [];
    Promise.allSettled([
      Promise.resolve().then(() => load(MANIFEST_PATH)),
      Promise.resolve().then(() => load(COLLECTIONS_PATH)),
    ]).then(([manifest, collections]) => {
      if (token !== fetchSeq) return;
      /** @type {typeof entries} */
      const next = [];
      if (manifest.status === 'fulfilled') {
        for (const screen of screensOf(manifest.value)) {
          const title = screen.title || screen.id;
          const url = screenUrl(screen);
          next.push({
            label: title,
            keywords: [`${screen.category}/${screen.id}`],
            text: `Open screen ${title}`,
            detail: `${screen.category}/${screen.id}`,
            run: () => navigate(CANOPY_PANEL_ID, url),
          });
        }
      }
      if (collections.status === 'fulfilled') {
        for (const c of collectionsOf(collections.value)) {
          const url = collectionUrl(c.ref);
          next.push({
            label: c.name,
            keywords: [c.ref],
            text: `Open collection ${c.name} (${c.members.length})`,
            detail: c.source,
            run: () => navigate(CANOPY_PANEL_ID, url),
          });
        }
      }
      entries = next;
      loading = false;
      if (onChange) onChange();
    });
  }

  /**
   * @param {string} q
   * @returns {CanopyItem[] | null}
   */
  function search(q) {
    if (loading) return null;
    return rankEntries(entries, q)
      .slice(0, CANOPY_CAP)
      .map((e) => {
        /** @type {CanopyItem} */
        const item = { label: e.text, run: e.run };
        if (e.detail) item.detail = e.detail;
        return item;
      });
  }

  return { start, search };
}
