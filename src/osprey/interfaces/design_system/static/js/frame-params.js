// @ts-check
/**
 * Micro-frontend query-param contract for the design-system front-end.
 *
 * Documents and applies the WELL-KNOWN query params a host page may pass to
 * an embedded design-system frame:
 *
 * - `embedded` — `"true"` marks the page as running inside a host frame; see
 *   {@link applyEmbedded}, which adds the `embedded` class to `document.body`
 *   when set.
 * - `theme` — owned and read pre-paint by theme-boot.js / theme-manager.js.
 *   It is deliberately NOT read here: theme-boot.js is a non-module inline
 *   script that resolves and applies `data-theme` before first paint, so
 *   re-reading `theme` in this (deferred) ES module would just duplicate that
 *   read after the fact and risks a visible theme flash. Consult
 *   theme-boot.js / theme-manager.js for the theme contract.
 *
 * `CONTRACT_VERSION` identifies the version of this query-param contract so
 * host pages and embedded frames can detect a mismatch.
 *
 * Note (decision OC-1): a generic `frameParam()` / `frameParams()` getter is
 * intentionally NOT provided here — that surface is deferred until a second
 * consumer actually needs it. What this module does export are predicates and
 * helpers for one named param each ({@link isEmbedded}, {@link applyEmbedded}),
 * and the redirect of a page that is not framed ({@link redirectTopLevelVisit}).
 *
 * @module frame-params
 */

/**
 * Version of the micro-frontend query-param contract described in this
 * module's JSDoc.
 *
 * @type {string}
 */
export const CONTRACT_VERSION = '2';

/**
 * Whether this page was loaded as an embedded panel: the `embedded` query
 * param is exactly `"true"` (any other value -- `"false"`, `"1"`, absent --
 * reads as standalone). The one predicate behind {@link applyEmbedded},
 * exposed so a page can branch on it before `<body>` carries the class -- a
 * standalone page runs theme-manager.js in the `hub` role (persisting picks
 * from its own `<osprey-display-menu>`), an embedded one as a `follower`.
 *
 * @returns {boolean}
 */
export function isEmbedded() {
  return new URLSearchParams(window.location.search).get('embedded') === 'true';
}

/**
 * A panel page's address, as the mount it is under and what follows it. The
 * mount is `/panel/<id>/` or `/panel-static/<id>/`, under the per-user prefix a
 * multi-user deployment serves at (`/u/<user>`, compute_url_prefix), and a
 * capability segment (`~c/<cap>/`) may sit right after it. Anchored, so a user
 * called `panel` is still a user.
 */
const PANEL_PAGE_RE = /^((?:\/u\/[^/]+)?)(\/panel(?:-static)?\/([^/]+)\/)(?:~c\/[^/]+\/)?(.*)$/;

/**
 * The web terminal's address for a direct visit to a panel page, or null off
 * every panel mount: `<prefix>/?open=<id>&at=<at>`.
 *
 * `at` is the one contract both sides of the redirect keep (the bundle route
 * in `routes/panels.py` is the other): the prefix, the path with a capability
 * segment removed, the query and the hash. The prefix stays because the hub
 * admits only a url inside the panel's prefixed mount (`isUrlWithinMount`), and
 * the capability goes because that mount is the plain base.
 *
 * @param {{pathname: string, search: string, hash: string}} location
 * @returns {string | null}
 */
export function directVisitHref({ pathname, search, hash }) {
  const page = PANEL_PAGE_RE.exec(pathname);
  if (!page) return null;
  const [, prefix, mount, rawId, rest] = page;
  let id = rawId;
  try {
    id = decodeURIComponent(rawId);
  } catch {
    // A malformed escape names no panel; the hub says so about the raw text.
  }
  const at = `${prefix}${mount}${rest}${search}${hash}`;
  return `${prefix}/?open=${encodeURIComponent(id)}&at=${encodeURIComponent(at)}`;
}

/**
 * Send a panel page opened on its own to the web terminal, with that panel on
 * that page. A panel has no frontend of its own: it runs in a tile or in a
 * window the terminal drives, and in both it is framed. An explicit
 * `embedded=true` visit is left alone: it renders the bare page the frames load.
 *
 * The hash is why this is on the client: a panel's deep links are hash routes,
 * and a fragment never reaches a server.
 *
 * @param {Window} [win] the window to ask; the global one by default
 * @returns {boolean} whether the page is leaving
 */
export function redirectTopLevelVisit(win = window) {
  if (win.top !== win) return false;
  if (new URLSearchParams(win.location.search).get('embedded') === 'true') return false;
  const href = directVisitHref(win.location);
  if (href === null) return false;
  win.location.replace(href);
  return true;
}

/**
 * Read the `embedded` query param and, when it is exactly `"true"`, add the
 * `embedded` class to `document.body`. No-op otherwise (including when the
 * param is absent, or set to any other value such as `"false"` or `"1"`).
 *
 * Every panel page calls this on load. First, a page opened on its own leaves
 * for the web terminal: see {@link redirectTopLevelVisit}.
 *
 * @returns {void}
 */
export function applyEmbedded() {
  if (redirectTopLevelVisit()) return;
  if (isEmbedded()) {
    document.body.classList.add('embedded');
  }
}

/**
 * Whether this page offers the terminal on the Agent tile. False only when the
 * server stamped `data-terminal="off"` on `<html>` (the web terminal, from
 * `web.terminal: false`); every other page, and every deployment that never set
 * the key, offers it.
 *
 * @returns {boolean}
 */
export function terminalOffered() {
  return document.documentElement.getAttribute('data-terminal') !== 'off';
}
