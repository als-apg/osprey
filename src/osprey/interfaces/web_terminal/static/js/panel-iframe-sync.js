// @ts-check
/* OSPREY Web Terminal — Panel Iframe State Sync
 *
 * The per-iframe "push host state into a panel" seam: theme, active session,
 * header actions, plus the artifacts viewer's three messages
 * (which artifact to show, a tile-menu action on it, and a server-side change
 * to it), each serialized to a panel
 * iframe via postMessage. Extracted from panel-manager so that module stays
 * focused on panel lifecycle, rail rendering, and health — these helpers
 * share one shape (take an iframe, guard its contentWindow, swallow the
 * cross-origin throw) and one concern (host → follower messaging), and depend
 * on none of panel-manager's private panel state, so they live cleanly on
 * their own.
 *
 * A panel popped out into its own window (popout.js) is a follower too, with
 * a WindowProxy where an iframe has a contentWindow: the three state messages
 * have a window-addressed twin each, and the pop-out URL is built beside the
 * embed URL so the two query contracts cannot drift.
 *
 * The hub-role fan-outs that iterate every panel (theme-manager's own theme
 * broadcast; panel-manager's broadcastMode) stay with their owners — this
 * module is only the single-target sends those fan-outs and their callers
 * use.
 */

import { getTheme } from '/design-system/js/theme-manager.js';
import { getCurrentSessionId } from './terminal.js';

/**
 * The shared single-window send: swallow the cross-origin postMessage throw.
 * Every host → follower message below funnels through here, by way of
 * {@link postToIframe} for a frame.
 *
 * The target origin is the hub's own, so an opaque panel (origin "null") never
 * receives the session id or a header action: the browser drops a
 * message whose target origin does not match. Only the theme message to a
 * frame passes `targetOrigin: '*'` (see sendThemeToIframe).
 * @param {Window | null} win
 * @param {object} message
 * @param {string} [targetOrigin]
 */
function postToWindow(win, message, targetOrigin = window.location.origin) {
  if (!win) return;
  try {
    win.postMessage(message, targetOrigin);
  } catch {
    /* cross-origin */
  }
}

/**
 * The single-iframe send: guard a missing contentWindow (a not-yet-loaded or
 * detached iframe), then {@link postToWindow}.
 * @param {HTMLIFrameElement | null} iframe
 * @param {object} message
 * @param {string} [targetOrigin]
 */
function postToIframe(iframe, message, targetOrigin = window.location.origin) {
  postToWindow(iframe?.contentWindow ?? null, message, targetOrigin);
}

/**
 * Send the hub's current theme to a popped-out window. A pop-out is always a
 * first-party page (an opaque panel has no standalone URL to pop out to), so
 * unlike the frame send this one is addressed to the hub's own origin.
 * @param {Window | null} win
 */
export function sendThemeToWindow(win) {
  postToWindow(win, { type: 'osprey-theme-change', theme: getTheme() });
}

/**
 * Send the hub's active session id to a popped-out window — the window twin
 * of {@link sendSessionToIframe}, with the same no-session guard.
 * @param {Window | null} win
 */
export function sendSessionToWindow(win) {
  const sid = getCurrentSessionId();
  if (!sid) return;
  postToWindow(win, { type: 'osprey-session-change', session_id: sid });
}

/**
 * Send the hub's current theme to one iframe. Always sends — never
 * conditioned on whether the id differs from the last send — because a
 * hidden iframe can read empty custom properties on Firefox and needs a
 * fresh, unconditional resend once it's visible again (theme-manager's
 * hidden-iframe protocol; see that module's docstring). Broadcasting to
 * every iframe on every hub theme change is theme-manager's own job (hub
 * role); this is only the two per-iframe trigger points panel-manager
 * owns: iframe creation and tab activation.
 *
 * 'osprey-theme-change' is the one message type theme-manager's follower
 * role (and every embedded interface) actually listens for.
 *
 * Sent with target origin "*", matching theme-manager's own hub broadcast: an
 * opaque panel (an agent-writable bundle or runtime panel, origin "null") is
 * unreachable by a named origin and must still follow the theme. The message
 * carries a public theme id and nothing else — nothing secret may ever be
 * added to it.
 * @param {HTMLIFrameElement | null} iframe
 */
export function sendThemeToIframe(iframe) {
  postToIframe(iframe, { type: 'osprey-theme-change', theme: getTheme() }, '*');
}

/**
 * Send the hub's active session id to one iframe — the twin of
 * sendThemeToIframe(), owned by the same two per-iframe trigger points
 * (iframe creation and tab activation). Guards on a missing contentWindow
 * (a not-yet-loaded or detached iframe) and on there being no active
 * session yet, and swallows the cross-origin postMessage throw the same
 * way its theme twin does.
 *
 * 'osprey-session-change' is the message type embedded interfaces listen
 * for to scope their view to the hub's active session.
 * @param {HTMLIFrameElement | null} iframe
 */
export function sendSessionToIframe(iframe) {
  const sid = getCurrentSessionId();
  if (!sid) return;
  postToIframe(iframe, { type: 'osprey-session-change', session_id: sid });
}

/**
 * Send a tile-bar header action to one iframe — the hub→panel half of the
 * header-contribution round-trip (tile-header-contrib.js owns the receive
 * side and the bar rendering; design_system/js/header-contrib.js documents
 * the contract). Funnels through the same guarded send as the theme and session
 * twins above.
 * @param {HTMLIFrameElement | null} iframe
 * @param {string} id      the contributed item's id
 * @param {string} [value] nav items: the clicked entry's id
 */
export function sendHeaderActionToIframe(iframe, id, value) {
  postToIframe(iframe, { type: 'osprey-header-action', id, ...(value != null ? { value } : {}) });
}

/**
 * Show one artifact in the artifacts panel's viewer. The hub's sidebar owns
 * the artifact list (sidebar-artifacts.js) and the embedded gallery shows the
 * one artifact named here (artifacts viewer.js). Showing is not focusing: the
 * agent's focus is never touched by this message.
 * @param {HTMLIFrameElement | null} iframe  the artifacts panel's iframe
 * @param {string | null} artifactId  null shows the viewer's empty state
 */
export function sendArtifactSelectToIframe(iframe, artifactId) {
  postToIframe(iframe, { type: 'osprey-artifact-select', artifact_id: artifactId });
}

/**
 * Ask the artifacts viewer to run a tile-menu action on an artifact, through
 * the gallery's own code path for it (its confirm, its request, its popup).
 * @param {HTMLIFrameElement | null} iframe  the artifacts panel's iframe
 * @param {string} artifactId
 * @param {'pin' | 'unpin' | 'print' | 'logbook' | 'delete'} action
 */
export function sendArtifactActionToIframe(iframe, artifactId, action) {
  postToIframe(iframe, { type: 'osprey-artifact-action', artifact_id: artifactId, action });
}

/**
 * Tell the artifacts viewer that the artifact it shows changed on the server.
 * The embedded viewer opens no event stream of its own (each stream holds one
 * of the browser's few connections to the hub's origin), so the hub forwards
 * the shown artifact's own frames from its stream: an update, with the
 * server's copy, re-renders it; a delete empties the viewer until the hub
 * selects another.
 * @param {HTMLIFrameElement | null} iframe  the artifacts panel's iframe
 * @param {string} artifactId  the artifact the viewer shows
 * @param {'updated' | 'deleted'} change
 * @param {object} [artifact]  updated: the server's copy of the artifact
 */
export function sendArtifactChangedToIframe(iframe, artifactId, change, artifact) {
  postToIframe(iframe, {
    type: 'osprey-artifact-changed',
    artifact_id: artifactId,
    change,
    ...(artifact ? { artifact } : {}),
  });
}

// ---- Navigation Guard ----
//
// The client half of the panel navigation guard; the server half is
// interfaces/panel_urls.py's normalise_navigation_url / navigation_url_refusal, and
// both apply one rule. A panel url is read the way the WHATWG URL parser reads
// it (leading and trailing C0 controls and spaces stripped, every tab and
// newline removed, `\` read as `/`) and must then be a root-relative path or
// an http(s) URL with a host. Anything else (`javascript:`, `data:`,
// `vbscript:`, `blob:`, scheme-relative `//host`) never becomes an iframe src
// or a pop-out's location: a `javascript:` src would run in THIS (the host's)
// origin.

/** The src a refused url is replaced with: an empty document, no script. */
export const REFUSED_EMBED_SRC = 'about:blank';

// eslint-disable-next-line no-control-regex -- the URL spec strips C0 controls and space from both ends
const EDGE_C0_OR_SPACE = /^[\u0000- ]+|[\u0000- ]+$/g;
const TAB_OR_NEWLINE = /[\t\n\r]/g;
const URL_SCHEME = /^([A-Za-z][A-Za-z0-9+.-]*):/;

/**
 * Read `raw` as the browser's URL parser will, the twin of the server's
 * normalise_navigation_url: strip edge C0/space, drop tabs and newlines, read
 * `\` as `/`, lower-case the scheme.
 * @param {string} raw
 * @returns {string}
 */
export function normaliseNavigationUrl(raw) {
  const url = String(raw)
    .replace(EDGE_C0_OR_SPACE, '')
    .replace(TAB_OR_NEWLINE, '')
    .replace(/\\/g, '/');
  const m = URL_SCHEME.exec(url);
  return m ? m[1].toLowerCase() + url.slice(m[1].length) : url;
}

/**
 * True iff `raw`, once normalised, is a root-relative path (never `//host`)
 * or an http(s) URL with a host — the server's navigation_url_refusal rule.
 * @param {unknown} raw
 * @returns {boolean}
 */
export function isNavigableUrl(raw) {
  if (typeof raw !== 'string') return false;
  const url = normaliseNavigationUrl(raw);
  if (url.startsWith('//')) return false;
  if (url.startsWith('/')) return true;
  // Spelled out with its authority: the parser would read 'http:///host' or
  // 'http:host' as a host too, but the server's rule does not, and the two
  // halves agree on every url.
  if (!/^https?:\/\/[^/?#]/.test(url)) return false;
  try {
    return new URL(url).hostname !== '';
  } catch {
    return false;
  }
}

//: How many percent-decoding rounds a path may need before it is stable; the
//: server's MAX_DECODE_ROUNDS.
const MAX_DECODE_ROUNDS = 8;

/**
 * The path a server would act on for `pathname`: `\` read as `/`, then
 * percent-escapes unwrapped until the text stops changing. Null when it is
 * still changing after MAX_DECODE_ROUNDS rounds or an escape is malformed.
 * @param {string} pathname
 * @returns {string | null}
 */
function decodedPath(pathname) {
  let current = pathname;
  for (let round = 0; round < MAX_DECODE_ROUNDS; round += 1) {
    const slashed = current.replace(/\\/g, '/');
    let decoded;
    try {
      decoded = decodeURIComponent(slashed);
    } catch {
      return null;
    }
    if (decoded === slashed) return decoded;
    current = decoded;
  }
  return null;
}

/**
 * True iff `raw` stays within the panel whose mount is `mountUrl`: the twin of
 * the server's member_navigation_url path rule. The url must be navigable
 * (isNavigableUrl), resolve against this page to this page's origin (so an
 * http(s) url is admitted only as the server's rewritten mount form), and its
 * path, read as a server reads it (decodedPath), must name the mount's
 * segments or extend them. `..`, `//`, `\` and their escapes fall out of that
 * reading: an empty or dot segment anywhere is refused.
 * @param {unknown} raw
 * @param {unknown} mountUrl  the panel's root-relative mount, already prefixed
 * @returns {boolean}
 */
export function isUrlWithinMount(raw, mountUrl) {
  if (typeof raw !== 'string' || typeof mountUrl !== 'string' || !mountUrl) return false;
  if (!isNavigableUrl(raw)) return false;
  let target;
  let mount;
  try {
    target = new URL(normaliseNavigationUrl(raw), window.location.origin);
    mount = new URL(mountUrl, window.location.origin);
  } catch {
    return false;
  }
  if (target.origin !== window.location.origin || mount.origin !== window.location.origin) {
    return false;
  }
  const path = decodedPath(target.pathname);
  const base = decodedPath(mount.pathname);
  if (path === null || base === null) return false;
  const segments = path.split('/').slice(1);
  if (segments.length && segments[segments.length - 1] === '') segments.pop();
  if (segments.some((s) => s === '' || s === '.' || s === '..' || s.includes(';'))) return false;
  const baseSegments = base.split('/').filter(Boolean);
  if (baseSegments.some((s) => s === '.' || s === '..')) return false;
  return baseSegments.every((segment, i) => segments[i] === segment);
}

/**
 * Build the src for a panel embed — the load-time half of the host→follower
 * contract whose live half is the postMessage senders above: the target URL
 * stamped with `embedded` plus the current theme, so the panel first-
 * paints in the right state before any message arrives.
 *
 * `targetUrl` (a panel's server-reported url, or a panel_focus SSE payload) is
 * root-relative and ALREADY server-prefixed. `new URL(path, origin)` keeps a
 * leading-slash path verbatim — it does not resolve against the origin's own
 * path — so a prefixed path stays prefixed and an unprefixed one stays
 * unprefixed. Never strip or re-add window.__OSPREY_PREFIX__ here.
 *
 * This is the one choke point every panel iframe src passes (panel-manager,
 * panel-iframe-factory, panel-lifecycle), so the navigation guard lives here:
 * a url that is not navigable (see isNavigableUrl) yields REFUSED_EMBED_SRC
 * rather than a throw, so no caller's flow is cut short by a hostile url.
 * @param {string} targetUrl
 * @returns {string}
 */
export function buildEmbedSrc(targetUrl) {
  return stampedUrl(targetUrl, 'embedded');
}

/**
 * Build the URL a panel pops out to: the same load-time state as
 * {@link buildEmbedSrc} under `popout=true` instead of `embedded=true`. The
 * page keeps its own chrome — there is no tile bar around a window — and the
 * `popout` flag is what makes it report to the hub (frame-params.js). Never
 * a token or a cap: a pop-out is a plain first-party page under the
 * operator's own session.
 * @param {string} targetUrl
 * @returns {string}
 */
export function buildPopoutSrc(targetUrl) {
  return stampedUrl(targetUrl, 'popout');
}

/**
 * @param {string} targetUrl
 * @param {'embedded' | 'popout'} flag
 * @returns {string}
 */
function stampedUrl(targetUrl, flag) {
  if (!isNavigableUrl(targetUrl)) {
    console.warn('[panels] refused a panel url that is not a root-relative path or an http(s) URL');
    return REFUSED_EMBED_SRC;
  }
  const url = new URL(normaliseNavigationUrl(targetUrl), window.location.origin);
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return REFUSED_EMBED_SRC;
  url.searchParams.set(flag, 'true');
  url.searchParams.set('theme', /** @type {string} */ (getTheme()));
  return url.toString();
}
