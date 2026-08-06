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
 * consumer actually needs it.
 *
 * Beyond query params, this module also owns the receive side of the host's
 * runtime `osprey-mode-change` postMessage broadcast — see
 * {@link onModeChange}.
 *
 * @module frame-params
 */

/**
 * Version of the micro-frontend query-param contract described in this
 * module's JSDoc.
 *
 * @type {string}
 */
export const CONTRACT_VERSION = '1';

/**
 * Read the `embedded` query param and, when it is exactly `"true"`, add the
 * `embedded` class to `document.body`. No-op otherwise (including when the
 * param is absent, or set to any other value such as `"false"` or `"1"`).
 *
 * @returns {void}
 */
export function applyEmbedded() {
  const embedded = new URLSearchParams(window.location.search).get('embedded') === 'true';
  if (embedded) {
    document.body.classList.add('embedded');
  }
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
