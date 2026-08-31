// @ts-check
// AUTO-GENERATED — DO NOT EDIT.
// Source: src/osprey/interfaces/design_system/tokens/
// Regenerate with: python -m osprey.interfaces.design_system.generator.build

// Applies data-theme before first paint. Deliberately NOT an ES module —
// module scripts are deferred, which would let a pre-theme flash slip
// through. Duplicates THEMES/DEFAULTS identity from tokens.js as inline
// literals for the same reason: this script must not import anything.
//
// Storage rungs, scoped: localStorage is origin-scoped, so on a multi-user
// deployment (every persona served from one origin under `/u/<user>/`) a bare
// key is a single shared slot and the last picker decides what everyone else
// boots into. When the server stamps data-osprey-storage-scope on <html>, the
// storage rungs read `osprey-theme--<scope>` instead — and do NOT fall back to
// the bare key, since that polluted slot is the very thing being escaped; a
// scoped page with no scoped value simply falls through to the server rung.
// With the attribute absent (single-user serving, and every non-web_terminal
// interface that loads this script) the legacy bare key is used unchanged,
// legacy bare-token format included.
(function () {
  "use strict";

  const STORAGE_KEY = "osprey-theme";
  const SCOPE_ATTRIBUTE = "data-osprey-storage-scope";
  const VALID_IDS = ["dark", "desy-dark", "desy-light", "high-contrast-dark", "high-contrast-light", "light", "retro-dark", "retro-light"];
  // Per-family {mode: id} map: DEFAULTS[family][mode]. Typed as a
  // Record (not the narrower literal shape object-literal inference would
  // give it) because resolveAuto() below indexes it with a general
  // `string` family, not just the exact DEFAULT_FAMILY literal.
  /** @type {Record<string, {dark?: string, light?: string}>} */
  const DEFAULTS = {
    "main": {
      "dark": "dark",
      "light": "light"
    },
    "desy": {
      "dark": "desy-dark",
      "light": "desy-light"
    },
    "high-contrast": {
      "dark": "high-contrast-dark",
      "light": "high-contrast-light"
    },
    "retro": {
      "dark": "retro-dark",
      "light": "retro-light"
    }
  };
  // id -> family, so a valid server-rendered data-theme id can supply the
  // family 'auto' resolves within instead of DEFAULT_FAMILY. See the
  // render_theme_boot_js docstring in generator/emit_js.py.
  /** @type {Record<string, string>} */
  const FAMILY_BY_ID = {
    "dark": "main",
    "desy-dark": "desy",
    "desy-light": "desy",
    "high-contrast-dark": "high-contrast",
    "high-contrast-light": "high-contrast",
    "light": "main",
    "retro-dark": "retro",
    "retro-light": "retro"
  };
  // Fallback family for 'auto' when no server data-theme attribute is
  // present/valid: the first family declared in the manifest.
  const DEFAULT_FAMILY = "main";

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

  // Inline mirror of storage-scope.js's `scopedStorageKey()`. An empty
  // attribute value counts as unscoped: the server omits the attribute rather
  // than rendering `=""`, so this only guards against a key ending in a bare
  // "--" that would belong to no persona.
  function storageKey() {
    try {
      const scope = document.documentElement.getAttribute(SCOPE_ATTRIBUTE);
      return scope ? STORAGE_KEY + "--" + scope : STORAGE_KEY;
    } catch {
      return STORAGE_KEY;
    }
  }

  function readStoredTheme() {
    try {
      return window.localStorage.getItem(storageKey());
    } catch {
      return null;
    }
  }

  // The structured storage format: the {family, mode} pair theme-manager
  // persists under the same key (see its _readStoredPreference). Resolves
  // to the concrete id that pair names, or null when the stored string
  // isn't that format -- a legacy bare token, an unknown family or mode,
  // or a family that declares no theme for the requested mode -- so
  // resolution falls through to the legacy bare-token rung below.
  /** @param {string|null} raw @returns {string|null} */
  function resolveStoredPreference(raw) {
    if (raw === null) return null;
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return null;
    }
    if (!parsed || typeof parsed !== "object") return null;
    // Typed as unknown, not left as JSON.parse's any: the checks below then
    // genuinely narrow both fields instead of being erased by any.
    /** @type {unknown} */
    const family = parsed.family;
    /** @type {unknown} */
    const mode = parsed.mode;
    if (typeof family !== "string" || !Object.prototype.hasOwnProperty.call(DEFAULTS, family)) {
      return null;
    }
    if (mode === "auto") return resolveAuto(family) || null;
    if (mode === "dark" || mode === "light") return DEFAULTS[family][mode] || null;
    return null;
  }

  // The server-rendered rung (finding I4): whatever data-theme already
  // sits on <html> when this script runs, e.g. stamped by the web server
  // from config. Read once so both the resolution candidate
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
  const storedPreferenceId = resolveStoredPreference(storedTheme);
  // auto's family for the bare-token rungs: the valid server theme's
  // declared family wins over DEFAULT_FAMILY, even if the final candidate
  // below turns out to be a literal "auto" from ?theme=/legacy storage
  // rather than serverTheme itself — see docstring. (A structured stored
  // preference names its own family and never comes through here; it is
  // already resolved by resolveStoredPreference.) isValidId is called
  // inline, not via a stored boolean, so its type-predicate narrows
  // serverTheme for the FAMILY_BY_ID lookup.
  const familyForAuto = isValidId(serverTheme) ? FAMILY_BY_ID[serverTheme] : DEFAULT_FAMILY;

  // Storage contributes two rungs: the structured {family, mode} pair
  // first (already resolved to a concrete id above), then the legacy bare
  // token for preferences written before the family model existed.
  let candidate = "auto";
  if (isKnownId(queryTheme)) {
    candidate = queryTheme;
  } else if (isValidId(storedPreferenceId)) {
    candidate = storedPreferenceId;
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
