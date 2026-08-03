/**
 * Drift alarm for the accent-token family: keeps `theme-lab.js`'s
 * `deriveAccentVars` and the design system's committed `tokens.css` from
 * silently diverging.
 *
 * This is a tripwire, not a formality. If someone adds a new accent-named
 * custom property to `tokens.css` -- or renames/removes one -- this test must
 * fail until the theme lab is taught about it (or the name is added to
 * `EXCLUSIONS` below with a reason). It asserts two-sided set equality:
 *
 *   (a) every `--*accent*` property in tokens.css is either produced by
 *       `deriveAccentVars` or explicitly excluded, and
 *   (b) every property `deriveAccentVars` produces actually exists in
 *       tokens.css.
 *
 *   npx vitest run tests/interfaces/design_system/js/theme-lab-coverage.test.mjs
 */

/* global process */

import { readFileSync } from 'node:fs';
import path from 'node:path';

import { test, expect } from 'vitest';

import { deriveAccentVars } from '/design-system/js/theme-lab.js';

/**
 * Names that match the accent regex below but are deliberately NOT part of
 * the derived accent family, each with the reason it is excluded. Keep this
 * list as short as possible -- an exclusion is a claim that a name only
 * *looks* like an accent token, and each one needs a real justification.
 */
const EXCLUSIONS = {
  // Despite the name, this is a near-background terminal cursor color (e.g.
  // #050a10 in dark, #fafbfd in light) -- not derived from the accent.
  '--ansi-cursor-accent': 'near-background terminal cursor color, not derived from the accent',
};

// `import.meta.url` is not a `file:` URL under this repo's vitest setup
// (environment: happy-dom), so `fileURLToPath(new URL(...))` throws. Vitest
// runs from the repo root, so resolve off `process.cwd()` instead.
const cssPath = path.join(
  process.cwd(),
  'src/osprey/interfaces/design_system/static/css/tokens.css'
);
const css = readFileSync(cssPath, 'utf8');

/** Every distinct accent-named custom property declared in tokens.css. */
const tokensAccentNames = new Set(css.match(/--[a-z0-9-]*accent[a-z0-9-]*/g));

/** A plausible lab state/scope, exercised only for the KEY SET it derives. */
const STATE = {
  dark: { hue: 178, saturation: 51, lightness: 39, emphasisLightness: 60 },
  light: { hue: 178, saturation: 51, lightness: 39, emphasisLightness: 29 },
};
const SCOPE_COLORS = { bgPrimary: '#0a0f1a', textPrimary: '#f1f5f9' };

const derivedNames = new Set(Object.keys(deriveAccentVars(STATE, 'dark', SCOPE_COLORS)));

test('every exclusion actually matches a token in tokens.css', () => {
  // Self-check: an exclusion for a name that no longer exists would let this
  // guard pass vacuously instead of catching real drift.
  const staleExclusions = Object.keys(EXCLUSIONS).filter((name) => !tokensAccentNames.has(name));
  expect(
    staleExclusions,
    `EXCLUSIONS names no accent token that exists in tokens.css: ${staleExclusions.join(', ')}`
  ).toEqual([]);
});

test('every accent-named token in tokens.css is derived or explicitly excluded', () => {
  const undocumented = [...tokensAccentNames].filter(
    (name) => !derivedNames.has(name) && !(name in EXCLUSIONS)
  );
  expect(
    undocumented,
    `tokens.css has accent token(s) deriveAccentVars does not produce and that are not in ` +
      `EXCLUSIONS: ${undocumented.join(', ')}. Teach deriveAccentVars about them, or add an ` +
      `exclusion with a reason.`
  ).toEqual([]);
});

test('every token deriveAccentVars produces exists in tokens.css', () => {
  const orphaned = [...derivedNames].filter((name) => !tokensAccentNames.has(name));
  expect(
    orphaned,
    `deriveAccentVars produces propert(y/ies) not found in tokens.css: ${orphaned.join(', ')}`
  ).toEqual([]);
});
