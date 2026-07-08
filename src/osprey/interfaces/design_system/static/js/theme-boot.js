// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
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
  const VALID_IDS = ["dark", "light"];
  const DEFAULTS = {
    "dark": "dark",
    "light": "light"
  };

  function isKnownId(value) {
    return value === "auto" || VALID_IDS.indexOf(value) !== -1;
  }

  function resolveAuto() {
    let prefersDark = true;
    try {
      prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    } catch (error) {
      prefersDark = true;
    }
    return prefersDark ? DEFAULTS.dark : DEFAULTS.light;
  }

  function readQueryTheme() {
    try {
      return new URLSearchParams(window.location.search).get("theme");
    } catch (error) {
      return null;
    }
  }

  function readStoredTheme() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (error) {
      return null;
    }
  }

  let candidate = "auto";
  const queryTheme = readQueryTheme();
  const storedTheme = readStoredTheme();
  if (isKnownId(queryTheme)) {
    candidate = queryTheme;
  } else if (isKnownId(storedTheme)) {
    candidate = storedTheme;
  }

  let resolved = candidate === "auto" ? resolveAuto() : candidate;
  if (!resolved && VALID_IDS.length > 0) {
    resolved = VALID_IDS[0];
  }
  if (resolved) {
    document.documentElement.setAttribute("data-theme", resolved);
  }
})();
