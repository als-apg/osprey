// @ts-check
/* OSPREY Design System — Rail Position Boot
 *
 * Hand-written (not generated) sibling of mode-boot.js for the
 * rail-position axis: it stamps data-rail-position onto <html> before first
 * paint, so the shell never flashes the wrong rail orientation (left column
 * vs top strip) on load.
 *
 * Deliberately NOT an ES module — module scripts are deferred, which would
 * let a pre-position flash slip through — and dependency-free (imports
 * nothing) for the same reason: it must run synchronously in <head>, ahead
 * of every stylesheet. It intentionally duplicates the small position
 * vocabulary (left|top) as inline literals rather than importing it.
 *
 * Chrome-only axis: unlike ui-mode, rail position is never forwarded to
 * panel iframes — embedded panels don't care where the host rail sits.
 *
 * Resolution ladder — the first rung that yields a valid position wins; an
 * invalid value at any rung is ignored and resolution falls through to the
 * next:
 *   1. the `?rail=` URL query param
 *   2. localStorage['osprey-rail-position'], per-persona scoped — see below
 *   3. the data-rail-position attribute the server already rendered on
 *      <html> (web_terminal stamps it from web.rail_position config)
 *   4. 'left' — the default
 * The resolved position is stamped as data-rail-position on <html>. The
 * theme (data-theme) and mode (data-ui-mode) axes are independent.
 *
 * Storage rung, scoped: localStorage is origin-scoped, so on a multi-user
 * deployment (every persona served from one origin under `/u/<user>/`) a bare
 * key is a single shared slot and the last persona to flip the rail decides
 * where everyone else's sits. When the server stamps
 * data-osprey-storage-scope on <html>, rung 2 reads
 * `osprey-rail-position--<scope>` instead — and does NOT fall back to the bare
 * key, since that polluted slot is the very thing being escaped; a scoped page
 * with no scoped value simply falls through to rung 3. With the attribute
 * absent (single-user serving, and every non-web_terminal interface that loads
 * this script) the legacy bare key is used unchanged.
 *
 * This duplicates storage-scope.js's `scopedStorageKey()` inline: as a
 * pre-paint IIFE this file imports nothing, so it mirrors the rule rather than
 * calling it. storage-scope.js is the written-down definition — keep them in
 * step. rail-position.js (web_terminal) is the WRITER of this same key and
 * derives it through that module, so the two must stay byte-identical.
 */
(function () {
  "use strict";

  const STORAGE_KEY_BASE = "osprey-rail-position";
  const SCOPE_ATTRIBUTE = "data-osprey-storage-scope";
  const VALID_POSITIONS = ["left", "top"];
  const DEFAULT_POSITION = "left";

  /** @param {string|null} value @returns {value is string} */
  function isValidPosition(value) {
    return value !== null && VALID_POSITIONS.indexOf(value) !== -1;
  }

  function readQueryPosition() {
    try {
      return new URLSearchParams(window.location.search).get("rail");
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

  function readStoredPosition() {
    try {
      return window.localStorage.getItem(storageKey());
    } catch {
      return null;
    }
  }

  // The server-rendered rung: whatever data-rail-position already sits on
  // <html> when this script runs. Read once so both the resolution candidate
  // below and the no-clobber check at the end use the exact same value.
  function readServerPosition() {
    try {
      return document.documentElement.getAttribute("data-rail-position");
    } catch {
      return null;
    }
  }

  const queryPosition = readQueryPosition();
  const storedPosition = readStoredPosition();
  const serverPosition = readServerPosition();

  let resolved = DEFAULT_POSITION;
  if (isValidPosition(queryPosition)) {
    resolved = queryPosition;
  } else if (isValidPosition(storedPosition)) {
    resolved = storedPosition;
  } else if (isValidPosition(serverPosition)) {
    resolved = serverPosition;
  }

  // No-clobber: only touch the DOM when the resolved position actually
  // differs from what's already there, so a correct server-rendered
  // attribute causes neither a flash nor a redundant write.
  if (resolved !== serverPosition) {
    document.documentElement.setAttribute("data-rail-position", resolved);
  }
})();
