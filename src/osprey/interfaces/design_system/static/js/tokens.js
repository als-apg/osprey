// AUTO-GENERATED — DO NOT EDIT.
// Source: src/osprey/interfaces/design_system/tokens/
// Regenerate with: python -m osprey.interfaces.design_system.generator.build

// Theme registry only: no color palettes here (see module docstring
// in generator/emit_js.py for why). Consumers read colors from
// tokens.css via theme-manager.js's computed-style bridges.
export const THEMES = [
  {
    "id": "apex-dark",
    "label": "Apex Dark",
    "mode": "dark",
    "family": "apex"
  },
  {
    "id": "apex-light",
    "label": "Apex Light",
    "mode": "light",
    "family": "apex"
  },
  {
    "id": "dark",
    "label": "Dark",
    "mode": "dark",
    "family": "osprey"
  },
  {
    "id": "high-contrast-dark",
    "label": "High Contrast Dark",
    "mode": "dark",
    "family": "high-contrast"
  },
  {
    "id": "high-contrast-light",
    "label": "High Contrast Light",
    "mode": "light",
    "family": "high-contrast"
  },
  {
    "id": "light",
    "label": "Light",
    "mode": "light",
    "family": "osprey"
  }
];

export const DEFAULTS = {
  "apex": {
    "dark": "apex-dark",
    "light": "apex-light"
  },
  "osprey": {
    "dark": "dark",
    "light": "light"
  },
  "high-contrast": {
    "dark": "high-contrast-dark",
    "light": "high-contrast-light"
  }
};

// The explicit-default family ($extensions.default), else the first
// declared -- the single fallback
// theme-manager.js reads instead of re-deriving it from DEFAULTS.
export const DEFAULT_FAMILY = "osprey";
