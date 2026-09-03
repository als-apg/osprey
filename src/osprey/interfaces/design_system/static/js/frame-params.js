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
 * - `mode` — a one-shot Expert/Simple hint, resolved pre-paint by
 *   mode-boot.js. It is not re-read here, but {@link stripQueryMode} drops it
 *   from the URL once the operator makes an explicit choice, so a leftover
 *   `?mode=` can't out-rank that choice on the next reload.
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
 * consumer actually needs it. What this module does export are helpers for
 * one named param each ({@link applyEmbedded}, {@link stripQueryMode}).
 *
 * Beyond query params, this module also owns both sides of the host's runtime
 * `osprey-mode-change` postMessage broadcast — {@link pickUiMode} sends it,
 * {@link onModeChange} receives it.
 *
 * @module frame-params
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/**
 * Base Expert/Simple key — the one mode-boot.js resolves from. Never used
 * bare: it goes through `scopedStorageKey()` so that on a multi-user mount the
 * pick lands in this persona's own slot instead of the shared origin-wide one
 * every persona would otherwise overwrite in turn.
 */
const MODE_STORAGE_KEY = 'osprey-ui-mode';

/**
 * Version of the micro-frontend query-param contract described in this
 * module's JSDoc.
 *
 * @type {string}
 */
export const CONTRACT_VERSION = '1';

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
 * Read the `embedded` query param and, when it is exactly `"true"`, add the
 * `embedded` class to `document.body`. No-op otherwise (including when the
 * param is absent, or set to any other value such as `"false"` or `"1"`).
 *
 * @returns {void}
 */
export function applyEmbedded() {
  if (isEmbedded()) {
    document.body.classList.add('embedded');
  }
}

/**
 * Strip a one-shot `mode` param from the URL's query string, if present,
 * without adding a history entry — the mode-axis twin of theme-manager's
 * _stripQueryTheme(). Once the user makes an explicit choice, a leftover
 * `?mode=` must not out-rank it (or localStorage) on the next reload. Other
 * params and the hash are preserved.
 *
 * @returns {void}
 */
export function stripQueryMode() {
  try {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('mode')) return;
    params.delete('mode');
    const query = params.toString();
    const url = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
    window.history.replaceState(window.history.state, '', url);
  } catch { /* non-browser environment or a blocked history API — non-fatal */ }
}

/**
 * Pick the Expert/Simple mode: persist the explicit choice, drop a leftover
 * one-shot `?mode=`, then broadcast it to this window for the page's
 * same-origin listener ({@link onModeChange}, every `<osprey-display-menu>`
 * included) to apply. The display menu's View row calls this; so does the
 * web terminal's command palette, which must flip the mode with no display
 * menu on the page, since that menu is a bar item the operator may remove.
 *
 * A pick of the mode already stamped on `<html>` persists and strips but does
 * not broadcast: the listeners treat every broadcast as a flip (the web
 * terminal's dock stashes the on-screen arrangement under the view being
 * left), so re-broadcasting the current mode would file one view's layout
 * under the other's key. The guard sits here, on the one sender, because the
 * receivers share `data-ui-mode` as their state and a receive-side compare
 * would silence every listener after the first.
 *
 * @param {'expert'|'simple'} mode
 * @returns {void}
 */
export function pickUiMode(mode) {
  try {
    // Key resolved at write time, not at module load: the scope attribute is
    // a property of the document this call ended up in.
    window.localStorage.setItem(scopedStorageKey(MODE_STORAGE_KEY), mode);
  } catch { /* storage blocked — the mode still applies for this session */ }
  stripQueryMode();
  if (document.documentElement.getAttribute('data-ui-mode') === mode) return;
  window.postMessage({ type: 'osprey-mode-change', mode }, window.location.origin);
}

/**
 * Subscribe to the host's live Expert/Simple UI-mode broadcasts.
 *
 * The runtime half of the mode contract (the pre-paint half is
 * mode-boot.js): the web-terminal hub posts
 * `{type: 'osprey-mode-change', mode}` to every embedded frame when the
 * operator flips the header toggle. This helper owns the receive side
 * once — it checks the message origin, normalizes the mode (`'simple'`,
 * anything else → `'expert'`), stamps `data-ui-mode` on `<html>`, then
 * invokes `callback(mode)` for the page's own follow-up (re-render, tab
 * fixup, ...). Pages whose Simple/Expert deltas are pure CSS pass no
 * callback.
 *
 * @param {(mode: 'expert'|'simple') => void} [callback]
 * @returns {void}
 */
export function onModeChange(callback) {
  window.addEventListener('message', (e) => {
    if (e.origin !== window.location.origin) return;
    if (!e.data || e.data.type !== 'osprey-mode-change' || !e.data.mode) return;
    const mode = e.data.mode === 'simple' ? 'simple' : 'expert';
    document.documentElement.setAttribute('data-ui-mode', mode);
    if (callback) callback(mode);
  });
}
