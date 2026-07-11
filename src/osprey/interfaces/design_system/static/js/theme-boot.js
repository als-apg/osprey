// @ts-check
// AUTO-GENERATED — DO NOT EDIT.
// Source: src/osprey/interfaces/design_system/tokens/
// Regenerate with: python -m osprey.interfaces.design_system.generator.build

// Applies data-theme before first paint. Deliberately NOT an ES module —
// module scripts are deferred, which would let a pre-theme flash slip
// through. Duplicates THEMES/DEFAULTS identity from tokens.js as inline
// literals for the same reason: this script must not import anything.
(function () {
  "use strict";

  const STORAGE_KEY = "osprey-theme";
  const VALID_IDS = ["dark", "high-contrast-dark", "high-contrast-light", "light"];
  // Per-family {mode: id} map: DEFAULTS[family][mode]. Typed as a
  // Record (not the narrower literal shape object-literal inference would
  // give it) because resolveAuto() below indexes it with a general
  // `string` family, not just the exact DEFAULT_FAMILY literal.
  /** @type {Record<string, {dark?: string, light?: string}>} */
  const DEFAULTS = {
    "osprey": {
      "dark": "dark",
      "light": "light"
    },
    "high-contrast": {
      "dark": "high-contrast-dark",
      "light": "high-contrast-light"
    }
  };
  // id -> family, so a valid server-rendered data-theme id can supply the
  // family 'auto' resolves within instead of DEFAULT_FAMILY. See the
  // render_theme_boot_js docstring in generator/emit_js.py.
  /** @type {Record<string, string>} */
  const FAMILY_BY_ID = {
    "dark": "osprey",
    "high-contrast-dark": "high-contrast",
    "high-contrast-light": "high-contrast",
    "light": "osprey"
  };
  // Fallback family for 'auto' when no server data-theme attribute is
  // present/valid: the first family declared in the manifest.
  const DEFAULT_FAMILY = "osprey";

  /** @param {string|null} value @returns {value is string} */
  function isValidId(value) {
    return value !== null && VALID_IDS.indexOf(value) !== -1;
  }

  /** @param {string|null} value @returns {value is string} */
  function isKnownId(value) {
    return value !== null && (value === "auto" || isValidId(value));
  }

  /** @param {string} family */
  function resolveAuto(family) {
    let prefersDark = true;
    try {
      prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    } catch {
      prefersDark = true;
    }
    const familyDefaults = DEFAULTS[family] || {};
    return prefersDark ? familyDefaults.dark : familyDefaults.light;
  }

  function readQueryTheme() {
    try {
      return new URLSearchParams(window.location.search).get("theme");
    } catch {
      return null;
    }
  }

  function readStoredTheme() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  }

  // The server-rendered rung (finding I4): whatever data-theme already
  // sits on <html> when this script runs, e.g. stamped by the web server
  // from config (Task 1.10). Read once so both the resolution candidate
  // below and the no-clobber check at the end use the exact same value.
  function readServerTheme() {
    try {
      return document.documentElement.getAttribute("data-theme");
    } catch {
      return null;
    }
  }

  const queryTheme = readQueryTheme();
  const storedTheme = readStoredTheme();
  const serverTheme = readServerTheme();
  // auto's family: the valid server theme's declared family wins over
  // DEFAULT_FAMILY, even if the final candidate below turns out to be a
  // literal "auto" from ?theme=/localStorage rather than serverTheme
  // itself — see docstring. (isValidId is called inline, not via a
  // stored boolean, so its type-predicate narrows serverTheme for the
  // FAMILY_BY_ID lookup.)
  const familyForAuto = isValidId(serverTheme) ? FAMILY_BY_ID[serverTheme] : DEFAULT_FAMILY;

  let candidate = "auto";
  if (isKnownId(queryTheme)) {
    candidate = queryTheme;
  } else if (isKnownId(storedTheme)) {
    candidate = storedTheme;
  } else if (isValidId(serverTheme)) {
    candidate = serverTheme;
  }

  let resolved = candidate === "auto" ? resolveAuto(familyForAuto) : candidate;
  if (!resolved && VALID_IDS.length > 0) {
    resolved = VALID_IDS[0];
  }
  // No-clobber: only touch the DOM when the resolved id actually differs
  // from what's already there, so a correct server-rendered attribute
  // causes neither a flash nor a redundant write.
  if (resolved && resolved !== serverTheme) {
    document.documentElement.setAttribute("data-theme", resolved);
  }
})();
