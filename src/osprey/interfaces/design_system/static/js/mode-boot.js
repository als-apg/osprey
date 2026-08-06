// @ts-check
/* OSPREY Design System — UI Mode Boot
 *
 * Hand-written (not generated) sibling of theme-manager.js, mirroring the
 * generated theme-boot.js's job for the UI-mode axis: it stamps
 * data-ui-mode onto <html> before first paint, so the mode-specific layout
 * (expert vs simple) never flashes the wrong shell on load.
 *
 * Deliberately NOT an ES module — module scripts are deferred, which would
 * let a pre-mode flash slip through — and dependency-free (imports nothing)
 * for the same reason: it must run synchronously in <head>, ahead of every
 * stylesheet. It intentionally duplicates the small mode vocabulary
 * (expert|simple) as inline literals rather than importing it.
 *
 * Resolution ladder — the first rung that yields a valid mode wins; an
 * invalid value at any rung is ignored and resolution falls through to the
 * next:
 *   1. the `?mode=` URL query param
 *   2. localStorage['osprey-ui-mode']
 *   3. the data-ui-mode attribute the server already rendered on <html>
 *      (web_terminal stamps it from config; artifacts renders none, so this
 *      rung is simply absent there and resolution falls through to 4)
 *   4. 'expert' — the default
 * The resolved mode is stamped as data-ui-mode on <html>. theme-manager.js
 * is unaffected: the theme axis (data-theme) and the mode axis
 * (data-ui-mode) are independent.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "osprey-ui-mode";
  const VALID_MODES = ["expert", "simple"];
  const DEFAULT_MODE = "expert";

  /** @param {string|null} value @returns {value is string} */
  function isValidMode(value) {
    return value !== null && VALID_MODES.indexOf(value) !== -1;
  }

  function readQueryMode() {
    try {
      return new URLSearchParams(window.location.search).get("mode");
    } catch {
      return null;
    }
  }

  function readStoredMode() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  }

  // The server-rendered rung: whatever data-ui-mode already sits on <html>
  // when this script runs (web_terminal stamps it from config; artifacts
  // renders none). Read once so both the resolution candidate below and the
  // no-clobber check at the end use the exact same value.
  function readServerMode() {
    try {
      return document.documentElement.getAttribute("data-ui-mode");
    } catch {
      return null;
    }
  }

  const queryMode = readQueryMode();
  const storedMode = readStoredMode();
  const serverMode = readServerMode();

  let resolved = DEFAULT_MODE;
  if (isValidMode(queryMode)) {
    resolved = queryMode;
  } else if (isValidMode(storedMode)) {
    resolved = storedMode;
  } else if (isValidMode(serverMode)) {
    resolved = serverMode;
  }

  // No-clobber: only touch the DOM when the resolved mode actually differs
  // from what's already there, so a correct server-rendered attribute
  // causes neither a flash nor a redundant write.
  if (resolved !== serverMode) {
    document.documentElement.setAttribute("data-ui-mode", resolved);
  }
})();
