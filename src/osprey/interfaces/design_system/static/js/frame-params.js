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
 * - `popout` — `"true"` marks the page as a panel the web terminal popped out
 *   into its own window. {@link applyEmbedded} stamps `data-popout="true"` on
 *   `<html>` and starts the pop-out client (popout-client.js), which reports to
 *   the hub that opened the window. The page keeps its own chrome: unlike an
 *   embedded frame it has no tile bar around it.
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
 * helpers for one named param each ({@link isEmbedded}, {@link isPopout},
 * {@link applyEmbedded}).
 *
 * @module frame-params
 */

import { initPopoutClient } from '/design-system/js/popout-client.js';

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
 * Whether this page was opened as a pop-out window by the web terminal: the
 * `popout` query param is exactly `"true"`, the rule {@link isEmbedded} applies
 * to `embedded`. A pop-out is a standalone page that follows one hub, so
 * theme-manager.js runs it in the `follower` role whatever role the page asks
 * for.
 *
 * @returns {boolean}
 */
export function isPopout() {
  return new URLSearchParams(window.location.search).get('popout') === 'true';
}

/**
 * Read the `embedded` query param and, when it is exactly `"true"`, add the
 * `embedded` class to `document.body`. No-op otherwise (including when the
 * param is absent, or set to any other value such as `"false"` or `"1"`).
 *
 * On a pop-out page ({@link isPopout}) this also stamps `data-popout="true"`
 * on `<html>` and starts the pop-out client, which reports the window to the
 * hub. Every panel page calls this on load, which is what makes one call enough
 * for both cases.
 *
 * @returns {void}
 */
export function applyEmbedded() {
  if (isEmbedded()) {
    document.body.classList.add('embedded');
  }
  if (isPopout()) {
    document.documentElement.setAttribute('data-popout', 'true');
    initPopoutClient();
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
