/**
 * Unit tests for theme-lab.js's pure layer -- the color math, the accent
 * derivation rules, and the name/collision helpers. Nothing here touches the
 * DOM: these functions are side-effect free by contract, and this spec is what
 * keeps them that way as the UI layer lands in the same module.
 *
 * The three things worth stating explicitly, because they are the rules the
 * lab's output is judged against:
 *
 *   - `contrastRatio`/`relativeLuminance` must agree with
 *     `generator/validate.py` bit for bit, so a badge that says "pass" and a
 *     build-time gate that says "fail" can never coexist. Checked against the
 *     WCAG reference pair (pure black vs pure white = 21:1) and against
 *     #767676-on-white, the canonical AA boundary gray.
 *   - the derived name set is exactly the accent family in the generated
 *     tokens.css MINUS `--ansi-cursor-accent`, which is a near-background
 *     color rather than an accent member despite its name.
 *   - `--color-on-accent` follows the target scope, not the accent: the same
 *     accent picks the background in a dark scope and the text color in a
 *     light one.
 *
 *   npx vitest run tests/interfaces/design_system/js/theme-lab-math.test.mjs
 */

import { test, expect, describe } from 'vitest';

import {
  hexToRgb,
  parseColor,
  rgbToHex,
  rgbToHsl,
  hslToRgb,
  relativeLuminance,
  contrastRatio,
  hslCss,
  rgbaCss,
  deriveAccentVars,
  deriveThemeVars,
  evaluateGates,
  slugifyThemeName,
  checkCollision,
} from '/design-system/js/theme-lab.js';

/** The osprey family's two scopes, verbatim from the generated tokens.css. */
const DARK_SCOPE = { bgPrimary: '#0a0f1a', textPrimary: '#f1f5f9' };
const LIGHT_SCOPE = { bgPrimary: '#f7f9fc', textPrimary: '#0c1322' };

/** The shipped registry, as the DOM layer will pass it in from tokens.js. */
const MANIFEST = {
  ids: ['apex-dark', 'apex-light', 'dark', 'high-contrast-dark', 'high-contrast-light', 'light'],
  families: ['osprey', 'apex', 'high-contrast'],
};

/**
 * Build a lab state whose two modes carry the given base/emphasis hex pairs.
 *
 * @param {[string, string]} dark base and emphasis hex for dark mode.
 * @param {[string, string]} light base and emphasis hex for light mode.
 */
function stateFrom(dark, light) {
  /** @param {[string, string]} pair */
  const mode = ([base, emphasis]) => {
    const b = rgbToHsl(/** @type {{r:number,g:number,b:number}} */ (hexToRgb(base)));
    const e = rgbToHsl(/** @type {{r:number,g:number,b:number}} */ (hexToRgb(emphasis)));
    return { hue: b.h, saturation: b.s, lightness: b.l, emphasisLightness: e.l };
  };
  return { dark: mode(dark), light: mode(light) };
}

/** The osprey family's own accents, as a lab state. */
const OSPREY_STATE = stateFrom(['#319795', '#4fd1c5'], ['#0a8f8c', '#0d7377']);

/** The same family's own second accent, as a lab state. */
const OSPREY_SECONDARY = stateFrom(['#d4a574', '#e8c9a0'], ['#a07040', '#7a5530']);

describe('hex/rgb conversion', () => {
  test('parses the six-digit form', () => {
    expect(hexToRgb('#319795')).toEqual({ r: 49, g: 151, b: 149 });
  });

  test('expands the three-digit shorthand', () => {
    expect(hexToRgb('#abc')).toEqual({ r: 170, g: 187, b: 204 });
  });

  test('parses the alpha-bearing forms, discarding alpha', () => {
    expect(hexToRgb('#31979580')).toEqual({ r: 49, g: 151, b: 149 });
    expect(hexToRgb('#abcd')).toEqual({ r: 170, g: 187, b: 204 });
  });

  test('rejects anything that is not a hex color', () => {
    expect(hexToRgb('319795')).toBeNull();
    expect(hexToRgb('#12345')).toBeNull();
    expect(hexToRgb('rgb(49, 151, 149)')).toBeNull();
    expect(hexToRgb('')).toBeNull();
  });

  test('round-trips hex -> rgb -> hex', () => {
    for (const hex of ['#000000', '#ffffff', '#319795', '#4fd1c5', '#0a0f1a', '#f7f9fc']) {
      expect(rgbToHex(/** @type {any} */ (hexToRgb(hex)))).toBe(hex);
    }
  });

  test('clamps out-of-range channels when serializing', () => {
    expect(rgbToHex({ r: -20, g: 300, b: 128 })).toBe('#00ff80');
  });
});

describe('parseColor', () => {
  test('accepts the legacy comma form getComputedStyle returns', () => {
    expect(parseColor('rgb(49, 151, 149)')).toEqual({ r: 49, g: 151, b: 149 });
    expect(parseColor('rgba(49, 151, 149, 0.15)')).toEqual({ r: 49, g: 151, b: 149 });
  });

  test('accepts hex too, and rejects out-of-range channels', () => {
    expect(parseColor('#319795')).toEqual({ r: 49, g: 151, b: 149 });
    expect(parseColor('rgb(300, 0, 0)')).toBeNull();
    expect(parseColor('hsl(180, 50%, 40%)')).toBeNull();
  });
});

describe('hsl conversion', () => {
  test('round-trips rgb -> hsl -> rgb exactly', () => {
    for (const hex of ['#319795', '#4fd1c5', '#0a8f8c', '#0d7377', '#0a0f1a', '#c07a00']) {
      const rgb = /** @type {any} */ (hexToRgb(hex));
      expect(hslToRgb(rgbToHsl(rgb))).toEqual(rgb);
    }
  });

  test('handles achromatic colors (zero saturation, undefined hue)', () => {
    expect(rgbToHsl({ r: 128, g: 128, b: 128 })).toEqual({ h: 0, s: 0, l: (128 / 255) * 100 });
    expect(hslToRgb({ h: 0, s: 0, l: 100 })).toEqual({ r: 255, g: 255, b: 255 });
    expect(hslToRgb({ h: 0, s: 0, l: 0 })).toEqual({ r: 0, g: 0, b: 0 });
  });

  test('reports hue per sector', () => {
    expect(rgbToHsl({ r: 255, g: 0, b: 0 }).h).toBeCloseTo(0, 6);
    expect(rgbToHsl({ r: 0, g: 255, b: 0 }).h).toBeCloseTo(120, 6);
    expect(rgbToHsl({ r: 0, g: 0, b: 255 }).h).toBeCloseTo(240, 6);
  });

  test('wraps out-of-range hue instead of clipping it', () => {
    expect(hslToRgb({ h: 480, s: 100, l: 50 })).toEqual(hslToRgb({ h: 120, s: 100, l: 50 }));
    expect(hslToRgb({ h: -120, s: 100, l: 50 })).toEqual(hslToRgb({ h: 240, s: 100, l: 50 }));
  });
});

describe('WCAG luminance and contrast', () => {
  test('luminance spans the full range at the extremes', () => {
    expect(relativeLuminance({ r: 0, g: 0, b: 0 })).toBe(0);
    expect(relativeLuminance({ r: 255, g: 255, b: 255 })).toBeCloseTo(1, 10);
  });

  test('black on white is the reference 21:1', () => {
    expect(contrastRatio({ r: 0, g: 0, b: 0 }, { r: 255, g: 255, b: 255 })).toBeCloseTo(21, 10);
  });

  test('#767676 on white is the canonical AA boundary gray', () => {
    // 0.2126+0.7152+0.0722 == 1, so a gray's luminance is just the channel
    // transfer of 118/255 -> ((0.46275+0.055)/1.055)^2.4 == 0.181164...,
    // giving 1.05/0.231164 == 4.5422:1 -- the value the WCAG reference
    // implementations report for this gray, and just over the 4.5 body-text
    // threshold that made it the canonical "darkest passing gray".
    const gray = /** @type {any} */ (hexToRgb('#767676'));
    expect(relativeLuminance(gray)).toBeCloseTo(0.181164, 6);
    expect(contrastRatio(gray, { r: 255, g: 255, b: 255 })).toBeCloseTo(4.5422, 4);
  });

  test('is symmetric in its arguments', () => {
    const a = /** @type {any} */ (hexToRgb('#319795'));
    const b = /** @type {any} */ (hexToRgb('#0a0f1a'));
    expect(contrastRatio(a, b)).toBe(contrastRatio(b, a));
  });

  test('a color against itself is 1:1', () => {
    const a = /** @type {any} */ (hexToRgb('#319795'));
    expect(contrastRatio(a, a)).toBe(1);
  });
});

describe('color serializers', () => {
  test('rgbaCss emits the legacy comma form with two-decimal alpha', () => {
    expect(rgbaCss({ r: 49, g: 151, b: 149 }, 0.04)).toBe('rgba(49, 151, 149, 0.04)');
    expect(rgbaCss({ r: 79, g: 209, b: 197 }, 0.1)).toBe('rgba(79, 209, 197, 0.10)');
    expect(rgbaCss({ r: 13, g: 115, b: 119 }, 0.25)).toBe('rgba(13, 115, 119, 0.25)');
  });

  test('hslCss emits degrees and percentages', () => {
    expect(hslCss({ h: 178.8, s: 51, l: 39.2 })).toBe('hsl(178.8, 51%, 39.2%)');
    expect(hslCss({ h: 480, s: 100, l: 50 })).toBe('hsl(120, 100%, 50%)');
  });
});

describe('deriveAccentVars', () => {
  const derived = deriveAccentVars(OSPREY_STATE, 'dark', DARK_SCOPE);

  test('emits exactly the accent family, minus --ansi-cursor-accent', () => {
    expect(Object.keys(derived).sort()).toEqual(
      [
        '--accent-tint-04',
        '--accent-tint-06',
        '--accent-tint-08',
        '--accent-tint-10',
        '--accent-tint-12',
        '--accent-tint-20',
        '--accent-tint-25',
        '--accent-tint-30',
        '--border-accent',
        '--color-accent',
        '--color-accent-light',
        '--color-on-accent',
        '--wt-accent-system-tint-04',
      ].sort()
    );
    // Named explicitly: --ansi-cursor-accent is a near-background color, not
    // an accent member, and must never be derived from the accent.
    expect(derived).not.toHaveProperty('--ansi-cursor-accent');
  });

  test('reproduces the accent it was built from', () => {
    expect(derived['--color-accent']).toBe('#319795');
  });

  test('tints derive from the emphasis variant, not the base', () => {
    const emphasis = /** @type {any} */ (hexToRgb(derived['--color-accent-light']));
    expect(derived['--accent-tint-04']).toBe(rgbaCss(emphasis, 0.04));
    expect(derived['--accent-tint-12']).toBe(rgbaCss(emphasis, 0.12));
    expect(derived['--accent-tint-30']).toBe(rgbaCss(emphasis, 0.3));
    expect(derived['--border-accent']).toBe(rgbaCss(emphasis, 0.15));
  });

  test('--wt-accent-system-tint-04 alone derives from the accent base', () => {
    expect(derived['--wt-accent-system-tint-04']).toBe('rgba(49, 151, 149, 0.04)');
    expect(derived['--wt-accent-system-tint-04']).not.toBe(derived['--accent-tint-04']);
  });

  test('--border-accent alpha is per-mode: 0.15 dark, 0.25 light', () => {
    const light = deriveAccentVars(OSPREY_STATE, 'light', LIGHT_SCOPE);
    expect(derived['--border-accent']).toContain('0.15');
    expect(light['--border-accent']).toContain('0.25');
  });

  test('emphasis is lighter than base in dark mode and darker in light mode', () => {
    const light = deriveAccentVars(OSPREY_STATE, 'light', LIGHT_SCOPE);
    /** @param {string} hex */
    const lightnessOf = (hex) => rgbToHsl(/** @type {any} */ (hexToRgb(hex))).l;
    expect(lightnessOf(derived['--color-accent-light'])).toBeGreaterThan(
      lightnessOf(derived['--color-accent'])
    );
    expect(lightnessOf(light['--color-accent-light'])).toBeLessThan(
      lightnessOf(light['--color-accent'])
    );
  });

  test('reads the mode it is asked for, not the other one', () => {
    expect(deriveAccentVars(OSPREY_STATE, 'light', LIGHT_SCOPE)['--color-accent']).toBe('#0a8f8c');
  });
});

describe('--color-on-accent picks the most readable candidate', () => {
  /** One accent, evaluated against both scopes. */
  const state = stateFrom(['#319795', '#4fd1c5'], ['#319795', '#4fd1c5']);

  /** @param {string} accentHex @param {{bgPrimary: string, textPrimary: string}} scope */
  const bestOf = (accentHex, scope) => {
    const accent = /** @type {any} */ (hexToRgb(accentHex));
    const candidates = [scope.bgPrimary, scope.textPrimary, '#000000', '#ffffff'];
    let best = candidates[0];
    let bestRatio = -1;
    for (const candidate of candidates) {
      const ratio = contrastRatio(accent, /** @type {any} */ (hexToRgb(candidate)));
      if (ratio > bestRatio) {
        bestRatio = ratio;
        best = candidate;
      }
    }
    return { best, bestRatio };
  };

  test('yields the highest-contrast candidate in a dark scope', () => {
    const derived = deriveAccentVars(state, 'dark', DARK_SCOPE);
    expect(derived['--color-on-accent']).toBe(bestOf(derived['--color-accent'], DARK_SCOPE).best);
  });

  test('yields the highest-contrast candidate in a light scope', () => {
    const derived = deriveAccentVars(state, 'light', LIGHT_SCOPE);
    expect(derived['--color-on-accent']).toBe(bestOf(derived['--color-accent'], LIGHT_SCOPE).best);
  });

  test('the pick flips when the scope background flips', () => {
    const flipped = { bgPrimary: LIGHT_SCOPE.bgPrimary, textPrimary: DARK_SCOPE.textPrimary };
    const derived = deriveAccentVars(state, 'dark', flipped);
    expect(derived['--color-on-accent']).toBe(bestOf(derived['--color-accent'], flipped).best);
  });

  test('never returns a candidate a rival beats', () => {
    const derived = deriveAccentVars(state, 'dark', DARK_SCOPE);
    const accent = /** @type {any} */ (hexToRgb(derived['--color-accent']));
    const chosen = contrastRatio(accent, /** @type {any} */ (hexToRgb(derived['--color-on-accent'])));
    for (const rival of [DARK_SCOPE.bgPrimary, DARK_SCOPE.textPrimary, '#000000', '#ffffff']) {
      expect(chosen).toBeGreaterThanOrEqual(
        contrastRatio(accent, /** @type {any} */ (hexToRgb(rival)))
      );
    }
  });

  test('considers black and white, not only the scope primaries', () => {
    // Regression: restricting the candidates to the scope's own two colors made
    // the lab reject accents that ship fine. A mid grey reaches only 4.22:1
    // against dark's background and 4.15:1 against its text -- both under the
    // 4.5 gate -- but 4.62:1 against black, which is exactly what
    // high-contrast-dark ships. Without black in the set the lab tells a
    // colleague this accent cannot ship, and it can.
    const grey = stateFrom(['#767676', '#767676'], ['#767676', '#767676']);
    const derived = deriveAccentVars(grey, 'dark', DARK_SCOPE);

    expect(derived['--color-on-accent']).toBe('#000000');
    const onAccentGate = evaluateGates(derived, DARK_SCOPE)[1];
    expect(onAccentGate.name).toBe('accent.on vs accent.base');
    expect(onAccentGate.pass).toBe(true);
    expect(onAccentGate.ratio).toBeGreaterThan(4.5);
  });
});

describe('evaluateGates', () => {
  test('mirrors every accent gate in validate.py, in its order and thresholds', () => {
    const derived = deriveThemeVars(OSPREY_STATE, OSPREY_SECONDARY, 'dark', DARK_SCOPE);
    expect(evaluateGates(derived, DARK_SCOPE).map((gate) => [gate.name, gate.threshold])).toEqual([
      ['accent.base vs bg.primary', 3.0],
      ['accent.on vs accent.base', 4.5],
      ['accent-secondary.light vs bg.primary', 4.5],
    ]);
  });

  test('the shipped osprey accents clear every gate, in both modes', () => {
    /** @type {ReadonlyArray<['dark' | 'light', {bgPrimary: string, textPrimary: string}]>} */
    const cases = [
      ['dark', DARK_SCOPE],
      ['light', LIGHT_SCOPE],
    ];
    for (const [mode, scope] of cases) {
      const derived = deriveThemeVars(OSPREY_STATE, OSPREY_SECONDARY, mode, scope);
      const gates = evaluateGates(derived, scope);
      expect(gates.every((gate) => gate.pass), `${mode}: ${JSON.stringify(gates)}`).toBe(true);
    }
  });

  test('scores the second accent independently of the accent', () => {
    // A second accent too pale for the body-text tier fails its own gate while
    // the accent's two gates are untouched -- the point of a separate role.
    const pale = stateFrom(['#f3e6d5', '#f7efe4'], ['#f3e6d5', '#f7efe4']);
    const gates = evaluateGates(
      deriveThemeVars(OSPREY_STATE, pale, 'light', LIGHT_SCOPE),
      LIGHT_SCOPE
    );
    expect(gates.slice(0, 2).every((gate) => gate.pass)).toBe(true);
    expect(gates[2].name).toBe('accent-secondary.light vs bg.primary');
    expect(gates[2].pass).toBe(false);
  });

  test('the 3.0 gate is decided at the boundary, not near it', () => {
    // Against white, #949494 is 3.033:1 and #959595 is 2.995:1 -- one step of
    // gray on either side of the non-text-UI threshold.
    const white = { bgPrimary: '#ffffff', textPrimary: '#000000' };
    const passing = evaluateGates({ '--color-accent': '#949494', '--color-on-accent': '#000000' }, white);
    const failing = evaluateGates({ '--color-accent': '#959595', '--color-on-accent': '#000000' }, white);
    expect(passing[0].ratio).toBeCloseTo(3.033, 3);
    expect(passing[0].pass).toBe(true);
    expect(failing[0].ratio).toBeCloseTo(2.995, 3);
    expect(failing[0].pass).toBe(false);
  });

  test('the 4.5 gate is decided at the boundary, not near it', () => {
    // On-accent against the accent fill: #767676 under white is 4.542:1,
    // #777777 is 4.478:1 -- either side of the body-text threshold.
    const scope = { bgPrimary: '#000000', textPrimary: '#ffffff' };
    const passing = evaluateGates({ '--color-accent': '#767676', '--color-on-accent': '#ffffff' }, scope);
    const failing = evaluateGates({ '--color-accent': '#777777', '--color-on-accent': '#ffffff' }, scope);
    expect(passing[1].ratio).toBeCloseTo(4.542, 3);
    expect(passing[1].pass).toBe(true);
    expect(failing[1].ratio).toBeCloseTo(4.478, 3);
    expect(failing[1].pass).toBe(false);
  });

  test('fails closed on an unparseable color rather than skipping the gate', () => {
    const gates = evaluateGates(
      { '--color-accent': 'not-a-color', '--color-on-accent': '#ffffff' },
      DARK_SCOPE
    );
    expect(gates.every((gate) => gate.ratio === 1 && gate.pass === false)).toBe(true);
  });
});

describe('slugifyThemeName', () => {
  test('lowercases and hyphenates whitespace', () => {
    expect(slugifyThemeName('Midnight Teal')).toBe('midnight-teal');
    expect(slugifyThemeName('  Midnight   Teal  ')).toBe('midnight-teal');
  });

  test('collapses punctuation into single dashes', () => {
    expect(slugifyThemeName('Ocean/Sky (v2)')).toBe('ocean-sky-v2');
    expect(slugifyThemeName('a___b...c')).toBe('a-b-c');
  });

  test('folds diacritics to their base letters', () => {
    expect(slugifyThemeName('Café Naïve')).toBe('cafe-naive');
  });

  test('trims leading and trailing dashes', () => {
    expect(slugifyThemeName('---Dawn---')).toBe('dawn');
    expect(slugifyThemeName('!Dawn!')).toBe('dawn');
  });

  test('keeps digits', () => {
    expect(slugifyThemeName('Theme 2026')).toBe('theme-2026');
  });

  test('yields the empty string when nothing is representable', () => {
    expect(slugifyThemeName('')).toBe('');
    expect(slugifyThemeName('   ')).toBe('');
    expect(slugifyThemeName('!!!')).toBe('');
    expect(slugifyThemeName('日本語')).toBe('');
  });

  test('is idempotent', () => {
    const once = slugifyThemeName('Ocean/Sky (v2)');
    expect(slugifyThemeName(once)).toBe(once);
  });
});

describe('checkCollision', () => {
  test('accepts a clean name', () => {
    expect(checkCollision('midnight-teal', MANIFEST)).toEqual({ collides: false, reason: null });
  });

  test('rejects an exact theme-id hit', () => {
    const result = checkCollision('dark', MANIFEST);
    expect(result.collides).toBe(true);
    expect(result.reason).toContain('theme id');
  });

  test('rejects a family-stem hit', () => {
    const result = checkCollision('high-contrast', MANIFEST);
    expect(result.collides).toBe(true);
    expect(result.reason).toContain('family');
  });

  test('rejects a name whose derived per-mode ids would collide', () => {
    // A lab theme ships as a family with one id per mode, so a slug is unusable
    // if `<slug>-dark`/`<slug>-light` already exist even when the slug itself
    // is free. Uses a manifest with the family list deliberately empty so the
    // derived-id rule is what does the rejecting.
    const result = checkCollision('apex', { ids: MANIFEST.ids, families: [] });
    expect(result.collides).toBe(true);
    expect(result.reason).toContain('apex-dark');
  });

  test('rejects the empty slug', () => {
    const result = checkCollision('', MANIFEST);
    expect(result.collides).toBe(true);
    expect(result.reason).toContain('letter or digit');
  });

  test('reads the injected manifest rather than a baked-in list', () => {
    expect(checkCollision('dark', { ids: [], families: [] }).collides).toBe(false);
    expect(checkCollision('midnight-teal', { ids: ['midnight-teal'], families: [] }).collides).toBe(
      true
    );
  });
});
