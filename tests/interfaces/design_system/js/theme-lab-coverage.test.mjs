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

import {
  BORDER_ALPHA,
  SYSTEM_TINT_ALPHA,
  TINT_LEVELS,
  deriveAccentVars,
  hexToRgb,
  rgbaCss,
} from '/design-system/js/theme-lab.js';

/**
 * Names that match the accent regex below but are deliberately NOT part of
 * the derived accent family, each with the reason it is excluded. Keep this
 * list as short as possible -- an exclusion is a claim that a name only
 * *looks* like an accent token, and each one needs a real justification.
 */
const SECONDARY_ACCENT_REASON =
  'second, independent accent slot picked per theme family and gated on its own ' +
  'contrast rules, not derived from the accent the lab proposes';

const EXCLUSIONS = {
  // Despite the name, this is a near-background terminal cursor color (e.g.
  // #050a10 in dark, #fafbfd in light) -- not derived from the accent.
  '--ansi-cursor-accent': 'near-background terminal cursor color, not derived from the accent',

  // The `accent-secondary` family is a role, not a shade of the accent: tan on
  // `main`, orange on `desy`, grey on high-contrast. Each family chooses it by
  // hand, and `accent-secondary.light` carries its own contrast gate (4.5:1
  // against bg.primary, 7:1 for high-contrast). Deriving it from the single
  // accent the lab takes would defeat the point of a second, independent slot.
  '--color-accent-secondary': SECONDARY_ACCENT_REASON,
  '--color-accent-secondary-light': SECONDARY_ACCENT_REASON,
  '--color-accent-secondary-hover': SECONDARY_ACCENT_REASON,
  '--accent-secondary-tint-04': SECONDARY_ACCENT_REASON,
  '--accent-secondary-tint-06': SECONDARY_ACCENT_REASON,
  '--accent-secondary-tint-08': SECONDARY_ACCENT_REASON,
  '--accent-secondary-tint-12': SECONDARY_ACCENT_REASON,
  '--accent-secondary-tint-15': SECONDARY_ACCENT_REASON,
  '--accent-secondary-tint-25': SECONDARY_ACCENT_REASON,
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

// ---------------------------------------------------------------------------
// Value-level guard: the derivation RULES, not just the names
// ---------------------------------------------------------------------------
//
// The three tests above compare NAMES. They stay green if someone changes what
// a name is worth -- re-basing `tint.accent.*` on accent.base instead of
// accent.light, or moving `border.accent` off 0.15/0.25. The math spec doesn't
// close that either: it builds its expectations with the same helpers it is
// checking, so it is tautological with respect to the alphas.
//
// This test closes it from the other side. It takes each shipped theme's own
// declared `--color-accent` / `--color-accent-light` straight out of the
// committed CSS, and requires the rest of that block to be exactly what the
// lab's constants say it should be. Nothing here is re-derived from the lab's
// state, so it fails if EITHER side moves: change `BORDER_ALPHA` in the lab and
// it breaks; re-base the tints in `dark.json` and regenerate, and it breaks.
//
// It deliberately does NOT assert `--color-accent-light` itself. The lab moves
// lightness only, while the shipped emphasis colors also shift hue/saturation
// (#319795 h179/s51 -> #4fd1c5 h175/s59), so the lab cannot reproduce them --
// a documented simplification, not a drift. `--color-on-accent` is likewise
// excluded: it is the lab's own selection rule, hand-picked in every theme.

/**
 * Extract one theme block's declarations from tokens.css.
 *
 * @param {string} selector the exact selector text, e.g. `[data-theme="light"]`.
 * @returns {Record<string, string>} custom-property name to declared value.
 */
function themeBlock(selector) {
  const start = css.indexOf(selector);
  expect(start, `tokens.css has no ${selector} block`).toBeGreaterThan(-1);
  const body = css.slice(css.indexOf('{', start) + 1, css.indexOf('}', start));
  /** @type {Record<string, string>} */
  const declarations = {};
  for (const [, name, value] of body.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/g)) {
    declarations[name] = value.trim();
  }
  return declarations;
}

/** @type {ReadonlyArray<['dark' | 'light', string]>} */
const THEME_BLOCKS = [
  ['dark', ':root, [data-theme="dark"]'],
  ['light', '[data-theme="light"]'],
];

for (const [mode, selector] of THEME_BLOCKS) {
  test(`the shipped ${mode} theme follows the alpha rules the lab derives by`, () => {
    const block = themeBlock(selector);
    const base = hexToRgb(block['--color-accent']);
    const emphasis = hexToRgb(block['--color-accent-light']);
    expect(base, `${mode}: --color-accent is not a hex color`).not.toBeNull();
    expect(emphasis, `${mode}: --color-accent-light is not a hex color`).not.toBeNull();

    /** @type {string[]} */
    const mismatches = [];
    /** @param {string} name @param {string} expected */
    const check = (name, expected) => {
      if (block[name] !== expected) {
        mismatches.push(`${name}: tokens.css ${block[name]} but the lab derives ${expected}`);
      }
    };

    // Tints and the accent border come from the EMPHASIS color.
    for (const level of TINT_LEVELS) {
      const key = `--accent-tint-${String(level).padStart(2, '0')}`;
      check(key, rgbaCss(/** @type {{r:number,g:number,b:number}} */ (emphasis), level / 100));
    }
    check(
      '--border-accent',
      rgbaCss(/** @type {{r:number,g:number,b:number}} */ (emphasis), BORDER_ALPHA[mode])
    );
    // The one member of the family that tints from the BASE instead.
    check(
      '--wt-accent-system-tint-04',
      rgbaCss(/** @type {{r:number,g:number,b:number}} */ (base), SYSTEM_TINT_ALPHA)
    );

    expect(
      mismatches,
      `the lab's derivation rules no longer describe how this design system authors the ` +
        `${mode} accent family, so the values it exports would be wrong:\n  ` +
        mismatches.join('\n  ')
    ).toEqual([]);
  });
}

test('deriveAccentVars applies those same rules to a fresh accent', () => {
  // Guards the direction the test above cannot: that deriveAccentVars actually
  // USES the constants it is checked against, for an accent that ships nowhere.
  const state = {
    dark: { hue: 262, saturation: 89, lightness: 66, emphasisLightness: 82 },
    light: { hue: 262, saturation: 89, lightness: 56, emphasisLightness: 48 },
  };
  const derived = deriveAccentVars(state, 'dark', SCOPE_COLORS);
  const emphasis = hexToRgb(derived['--color-accent-light']);
  const base = hexToRgb(derived['--color-accent']);

  for (const level of TINT_LEVELS) {
    expect(derived[`--accent-tint-${String(level).padStart(2, '0')}`]).toBe(
      rgbaCss(/** @type {{r:number,g:number,b:number}} */ (emphasis), level / 100)
    );
  }
  expect(derived['--border-accent']).toBe(
    rgbaCss(/** @type {{r:number,g:number,b:number}} */ (emphasis), BORDER_ALPHA.dark)
  );
  expect(derived['--wt-accent-system-tint-04']).toBe(
    rgbaCss(/** @type {{r:number,g:number,b:number}} */ (base), SYSTEM_TINT_ALPHA)
  );
});
