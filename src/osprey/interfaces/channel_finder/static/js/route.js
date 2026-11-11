// @ts-check
/**
 * OSPREY Channel Finder — the panel's route grammar.
 *
 * One statement of what a panel hash looks like (Deep links, Panel view
 * contract): `#/pv/<name>`, `#/ioc/<name>`, `#/allpvs?q=<words>&ioc=<name>`
 * and `#/<view>`, where `<name>` is one percent-encoded segment.
 * {@link parseRoute} reads a hash and {@link formatRoute} writes one; they are
 * inverses. Values are encoded with encodeURIComponent and never through
 * URLSearchParams, so a `+` is a plus. No imports and no side effects, so any
 * view module may import it.
 */

/** The repeatable All PVs filters a route carries beside `q`, in hash order. */
export const FACET_KEYS = /** @type {const} */ (['origin', 'ioc', 'record_type']);

/**
 * The family a view belongs to: the pv and ioc routes are pages of All PVs;
 * any other view is its own.
 * @param {string} view
 * @returns {string}
 */
export function familyOf(view) {
  return view === 'pv' || view === 'ioc' ? 'allpvs' : view;
}

/** @typedef {typeof FACET_KEYS[number]} FacetKey */

/**
 * Route parameters: `name` for pv and ioc; for allpvs, the view's words `q`
 * and the filters, one list per {@link FACET_KEYS} key.
 * @typedef {{ name?: string, q?: string } & Partial<Record<FacetKey, string[]>>} RouteParams
 */

/** @typedef {{ view: string, params: RouteParams }} Route */

/**
 * @param {string} key
 * @returns {key is FacetKey}
 */
function isFacetKey(key) {
  return /** @type {readonly string[]} */ (FACET_KEYS).includes(key);
}

/**
 * The All PVs params of a route's query: `q` once, the filters as lists, any
 * other key ignored. Values are decoded as a `<name>` is and not by
 * URLSearchParams, because a `+` in them is a plus; a malformed one throws.
 * @param {string} query
 * @returns {RouteParams}
 */
function allPvsParams(query) {
  /** @type {RouteParams} */
  const params = {};
  for (const part of query.split('&')) {
    const at = part.indexOf('=');
    if (at < 0) continue;
    const key = part.slice(0, at);
    const value = part.slice(at + 1);
    if (key === 'q') {
      params.q ??= decodeURIComponent(value);
    } else if (isFacetKey(key) && value) {
      (params[key] ??= []).push(decodeURIComponent(value));
    }
  }
  return params;
}

/**
 * Parse a location hash: a bare view name (`#explore`), or a route of the Deep
 * links form (`#/pv/<name>`, `#/ioc/<name>`, `#/allpvs?q=<words>&ioc=<name>`).
 * Whether the view exists is not judged here: only a malformed route answers
 * null.
 * @param {string} hash
 * @returns {Route|null}
 */
export function parseRoute(hash) {
  const route = hash.replace(/^#\/?/, '');
  const queryAt = route.indexOf('?');
  const path = queryAt < 0 ? route : route.slice(0, queryAt);
  const query = queryAt < 0 ? '' : route.slice(queryAt + 1);
  const [view, ...rest] = path.split('/');
  try {
    if (view === 'pv' || view === 'ioc') {
      if (rest.length !== 1 || !rest[0]) return null;
      return { view, params: { name: decodeURIComponent(rest[0]) } };
    }
    if (rest.length) return null;
    if (view === 'allpvs') return { view, params: allPvsParams(query) };
    return { view: view || 'explore', params: {} };
  } catch {
    return null; // malformed percent-encoding
  }
}

/**
 * The hash of a route, the inverse of {@link parseRoute}. On allpvs `q` is
 * written whenever it is set, empty included: `#/allpvs?q=` clears the shared
 * words where `#/allpvs` keeps them.
 * @param {Route} route
 * @returns {string}
 * @throws {TypeError} a pv or ioc route without a name
 */
export function formatRoute({ view, params }) {
  if (view === 'pv' || view === 'ioc') {
    if (params.name === undefined) throw new TypeError(`a ${view} route needs a name`);
    return `#/${view}/${encodeURIComponent(params.name)}`;
  }
  if (view !== 'allpvs') return `#/${view}`;
  const parts = params.q === undefined ? [] : [`q=${encodeURIComponent(params.q)}`];
  for (const key of FACET_KEYS) {
    for (const value of params[key] ?? []) parts.push(`${key}=${encodeURIComponent(value)}`);
  }
  return parts.length ? `#/allpvs?${parts.join('&')}` : '#/allpvs';
}

/**
 * The All PVs route for `words`: the trimmed words as `q`, none when blank.
 * @param {string} words
 * @returns {string}
 */
export function allPvsHash(words) {
  const text = words.trim();
  return formatRoute({ view: 'allpvs', params: text ? { q: text } : {} });
}
