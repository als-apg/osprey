// AUTO-GENERATED — DO NOT EDIT.
// Source: src/osprey/interfaces/design_system/tokens/
// Regenerate with: python -m osprey.interfaces.design_system.generator.build

// Theme registry only: no color palettes here (see module docstring
// in generator/emit_js.py for why). Consumers read colors from
// tokens.css via theme-manager.js's computed-style bridges.
export const THEMES = [
  {
    "id": "dark",
    "label": "Dark",
    "mode": "dark",
    "family": "main"
  },
  {
    "id": "desy-dark",
    "label": "DESY Dark",
    "mode": "dark",
    "family": "desy"
  },
  {
    "id": "desy-light",
    "label": "DESY Light",
    "mode": "light",
    "family": "desy"
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
    "family": "main"
  },
  {
    "id": "retro-dark",
    "label": "Retro Dark",
    "mode": "dark",
    "family": "retro"
  },
  {
    "id": "retro-light",
    "label": "Retro Light",
    "mode": "light",
    "family": "retro"
  }
];

export const DEFAULTS = {
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

// The explicit-default family ($extensions.default), else the first
// declared -- the single fallback
// theme-manager.js reads instead of re-deriving it from DEFAULTS.
export const DEFAULT_FAMILY = "main";

// Display names for families whose id does not title-case correctly
// ('desy' -> 'DESY'). Sparse BY DESIGN: a family that declares no
// $extensions.family_label is absent here, and consumers derive its
// label from the family id instead.
export const FAMILY_LABELS = {
  "desy": "DESY"
};
