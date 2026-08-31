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
 *   2. localStorage['osprey-ui-mode'], per-persona scoped — see below
 *   3. the data-ui-mode attribute the server already rendered on <html>
 *      (web_terminal stamps it from config; artifacts renders none, so this
 *      rung is simply absent there and resolution falls through to 4)
 *   4. 'expert' — the default
 * The resolved mode is stamped as data-ui-mode on <html>. theme-manager.js
 * is unaffected: the theme axis (data-theme) and the mode axis
 * (data-ui-mode) are independent.
 *
 * Storage rung, scoped: localStorage is origin-scoped, so on a multi-user
 * deployment (every persona served from one origin under `/u/<user>/`) a bare
 * key is a single shared slot and the last picker decides what everyone else
 * boots into. When the server stamps data-osprey-storage-scope on <html>, rung
 * 2 reads `osprey-ui-mode--<scope>` instead — and does NOT fall back to the
 * bare key, since that polluted slot is the very thing being escaped; a scoped
 * page with no scoped value simply falls through to rung 3. With the attribute
 * absent (single-user serving, and every non-web_terminal interface that loads
 * this script) the legacy bare key is used unchanged.
 *
 * This duplicates storage-scope.js's `scopedStorageKey()` inline: as a
 * pre-paint IIFE this file imports nothing, so it mirrors the rule rather than
 * calling it. storage-scope.js is the written-down definition — keep them in
 * step.
 */
(function () {
  "use strict";

  const STORAGE_KEY_BASE = "osprey-ui-mode";
  const SCOPE_ATTRIBUTE = "data-osprey-storage-scope";
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

  // Inline mirror of storage-scope.js's `scopedStorageKey()`. An empty
  // attribute value counts as unscoped: the server omits the attribute rather
  // than rendering `=""`, so this only guards against a key ending in a bare
  // "--" that would belong to no persona.
  function storageKey() {
    try {
      const scope = document.documentElement.getAttribute(SCOPE_ATTRIBUTE);
      return scope ? STORAGE_KEY_BASE + "--" + scope : STORAGE_KEY_BASE;
    } catch {
      return STORAGE_KEY_BASE;
    }
  }

  function readStoredMode() {
    try {
      return window.localStorage.getItem(storageKey());
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
