/**
 * Theme lab -- pure color math, accent-token derivation, and export spec.
 *
 * The DOM half lives in `theme-lab-ui.js`, which is what the page loads;
 * this module is import-safe and is what the unit tests exercise directly.
 *
 * This half of the module is deliberately side-effect free: no `window`, no
 * `document`, no module-level mutable state. It answers three questions the
 * lab's UI asks continuously:
 *
 *   1. What are the custom properties for a given pair of accent choices?
 *      (`deriveAccentVars`, `deriveSecondaryAccentVars`, `deriveThemeVars`)
 *   2. Do those choices clear the design system's WCAG gates?
 *      (`relativeLuminance`, `contrastRatio`, `evaluateGates`)
 *   3. Is the name the author typed usable, or does it collide with a shipped
 *      theme? (`slugifyThemeName`, `checkCollision`)
 *
 * DERIVATION RULES (read off the generated `static/css/tokens.css`, which is
 * the authority -- the default `osprey` family is the shape proposers riff on):
 *
 *   --color-accent             the chosen accent for that mode
 *   --color-accent-light       the mode's *emphasis* variant. "light" means
 *                              emphasis, not luminance: in dark mode it sits
 *                              above the base lightness, in light mode below
 *                              it. The direction is never assumed -- it comes
 *                              from that mode's own emphasis-lightness input.
 *                              NOTE this is a deliberate simplification: the
 *                              lab moves lightness only, so it cannot
 *                              reproduce a hand-authored emphasis color that
 *                              also shifts hue or saturation. The shipped
 *                              osprey dark pair does exactly that -- its base
 *                              sits near hue 179 at 51% saturation while its
 *                              emphasis sits near hue 175 at 59% -- so the lab
 *                              cannot round-trip it. A lab proposal is
 *                              internally consistent, not a re-derivation of
 *                              an existing theme; the export carries the
 *                              explicit values either way.
 *   --border-accent            emphasis variant at alpha 0.15 (dark) / 0.25 (light)
 *   --accent-tint-NN           emphasis variant at alpha NN/100, for the eight
 *                              levels 04 06 08 10 12 20 25 30
 *   --wt-accent-system-tint-04 accent BASE at alpha 0.04 (not the emphasis
 *                              variant -- the one member of the family that
 *                              tints from the base)
 *   --color-on-accent          the LAB'S OWN rule, not one read off tokens.css:
 *                              whichever of the scope's bg.primary /
 *                              text.primary / black / white has the highest
 *                              WCAG contrast against the chosen accent. Every
 *                              shipped theme hand-picks this value (only `dark`
 *                              happens to use its own bg.primary), so there is
 *                              no convention to mirror -- what the lab
 *                              guarantees is that it clears the gate whenever
 *                              any candidate can, and that the value it scores
 *                              is the value it exports.
 *
 * THE SECOND ACCENT (`--color-accent-secondary` and its family) is derived
 * separately, from its own controls, by `deriveSecondaryAccentVars`. It is a
 * distinct role rather than a shade of the accent -- every shipped family picks
 * it by hand, and the build gates its `light` slot on its own -- so the lab
 * lets it be chosen rather than inferring it. See that function for the two
 * places its rules diverge from the accent's.
 *
 * `--ansi-cursor-accent` is deliberately NOT derived here: despite the name it
 * is a near-background color, not a member of either accent family.
 *
 * The WCAG luminance and contrast math mirrors
 * `generator/validate.py`'s `relative_luminance`/`contrast_ratio` bit for bit,
 * and `GATES` mirrors every accent entry of its `WCAG_GATES` (the AA tuple
 * applied to the default family), so the lab's badges and the build-time
 * validator can never disagree.
 */


/** @typedef {{r: number, g: number, b: number}} Rgb */
/** @typedef {{h: number, s: number, l: number}} Hsl  hue in degrees, s/l in percent */
/** @typedef {'dark' | 'light'} ThemeMode */

/**
 * One mode's accent controls. `lightness` drives the base accent;
 * `emphasisLightness` drives the emphasis variant (`--color-accent-light`).
 * Both share the mode's hue and saturation.
 *
 * @typedef {{
 *   hue: number,
 *   saturation: number,
 *   lightness: number,
 *   emphasisLightness: number,
 * }} ModeAccent
 */

/** @typedef {Record<ThemeMode, ModeAccent>} LabState */

/**
 * The two colors a derivation target scope supplies about itself. Values may
 * be hex or legacy comma rgb/rgba (what `getComputedStyle` hands back).
 *
 * @typedef {{bgPrimary: string, textPrimary: string}} ScopeColors
 */

/** @typedef {{ids: string[], families: string[]}} ThemeManifest */
/** @typedef {{name: string, ratio: number, threshold: number, pass: boolean}} GateResult */
/** @typedef {{collides: boolean, reason: string | null}} CollisionResult */

/**
 * Alpha levels, as hundredths, of the `--accent-tint-NN` scale.
 *
 * Exported so the coverage spec can hold these constants against the committed
 * `tokens.css` rather than re-deriving them with the same helpers (which would
 * make the assertion tautological).
 */
export const TINT_LEVELS = [4, 6, 8, 10, 12, 20, 25, 30];

/** `--border-accent` alpha, per mode. */
export const BORDER_ALPHA = { dark: 0.15, light: 0.25 };

/** `--wt-accent-system-tint-04` alpha. */
export const SYSTEM_TINT_ALPHA = 0.04;

/**
 * Alpha levels, as hundredths, of the `--accent-secondary-tint-NN` scale.
 *
 * A different ladder from {@link TINT_LEVELS}: the second accent ships six
 * levels where the accent ships eight. Exported for the same reason -- the
 * coverage spec holds these against the committed `tokens.css` rather than
 * re-deriving them with the same helpers.
 */
export const SECONDARY_TINT_LEVELS = [4, 6, 8, 12, 15, 25];

/**
 * Lightness points between the second accent's base and its hover variant.
 *
 * THE LAB'S OWN RULE, in the same sense as `--color-on-accent`: the shipped
 * families hand-author this slot and do not agree on a derivation. Most darken
 * the base (`main` -20, `high-contrast` -15 dark / -20 light, `light` -6) while
 * `desy` simply swaps its two brand oranges (+4). Three of them also jump some
 * +50 saturation, which a lightness-only rule cannot reproduce at all -- the
 * same limitation `--color-accent-light` carries above. Darkening by a fixed
 * step follows the majority and keeps a proposal internally consistent, which
 * is what the lab promises; it is not a re-derivation of a shipped theme.
 */
export const SECONDARY_HOVER_DELTA = -14;

const HEX_PATTERN = /^#([0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/i;
const LEGACY_RGB_PATTERN =
  /^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(?:,\s*[\d.]+\s*)?\)$/i;

/**
 * Clamp `value` into `[low, high]`.
 *
 * Exported for the UI layer, which clamps slider and pointer input to the
 * same bounds this module's conversions assume.
 *
 * @param {number} value
 * @param {number} low
 * @param {number} high
 * @returns {number}
 */
export function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/**
 * Parse a hex color into 8-bit channels. Accepts 3-, 4-, 6- and 8-digit forms;
 * any alpha digits are parsed and discarded (every consumer here pairs opaque
 * colors, exactly as the validator does).
 *
 * @param {string} value
 * @returns {Rgb | null} `null` if `value` is not a hex color.
 */
export function hexToRgb(value) {
  const match = HEX_PATTERN.exec(String(value).trim());
  if (match === null) return null;
  const digits = match[1];
  const pairs =
    digits.length <= 4 ? Array.from(digits, (digit) => digit + digit) : (digits.match(/../g) ?? []);
  return {
    r: parseInt(pairs[0], 16),
    g: parseInt(pairs[1], 16),
    b: parseInt(pairs[2], 16),
  };
}

/**
 * Parse any color spelling this design system permits -- hex, or legacy comma
 * rgb/rgba -- into 8-bit channels. The legacy form is what
 * `getComputedStyle` returns, so scope colors read off a live element land
 * here rather than in {@link hexToRgb}.
 *
 * @param {string} value
 * @returns {Rgb | null} `null` if `value` is neither spelling.
 */
export function parseColor(value) {
  const text = String(value).trim();
  const hex = hexToRgb(text);
  if (hex !== null) return hex;
  const match = LEGACY_RGB_PATTERN.exec(text);
  if (match === null) return null;
  const channels = [match[1], match[2], match[3]].map((part) => Number(part));
  if (channels.some((channel) => channel > 255)) return null;
  return { r: channels[0], g: channels[1], b: channels[2] };
}

/**
 * Serialize 8-bit channels as a 6-digit lowercase hex color.
 *
 * @param {Rgb} rgb
 * @returns {string}
 */
export function rgbToHex(rgb) {
  /** @param {number} value @returns {string} */
  const channel = (value) =>
    Math.round(clamp(value, 0, 255))
      .toString(16)
      .padStart(2, '0');
  return `#${channel(rgb.r)}${channel(rgb.g)}${channel(rgb.b)}`;
}

/**
 * Convert 8-bit channels to hue/saturation/lightness.
 *
 * @param {Rgb} rgb
 * @returns {Hsl} hue in `[0, 360)`, saturation and lightness in `[0, 100]`.
 */
export function rgbToHsl(rgb) {
  const r = clamp(rgb.r, 0, 255) / 255;
  const g = clamp(rgb.g, 0, 255) / 255;
  const b = clamp(rgb.b, 0, 255) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  const l = (max + min) / 2;

  if (delta === 0) return { h: 0, s: 0, l: l * 100 };

  const s = delta / (1 - Math.abs(2 * l - 1));
  let h;
  if (max === r) {
    h = 60 * (((g - b) / delta) % 6);
  } else if (max === g) {
    h = 60 * ((b - r) / delta + 2);
  } else {
    h = 60 * ((r - g) / delta + 4);
  }
  return { h: (h + 360) % 360, s: s * 100, l: l * 100 };
}

/**
 * Convert hue/saturation/lightness to 8-bit channels.
 *
 * @param {Hsl} hsl hue in degrees (wrapped), saturation and lightness in percent.
 * @returns {Rgb}
 */
export function hslToRgb(hsl) {
  const h = (((hsl.h % 360) + 360) % 360) / 60;
  const s = clamp(hsl.s, 0, 100) / 100;
  const l = clamp(hsl.l, 0, 100) / 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h % 2) - 1));
  const m = l - c / 2;

  /** @type {[number, number, number]} */
  let triple;
  if (h < 1) triple = [c, x, 0];
  else if (h < 2) triple = [x, c, 0];
  else if (h < 3) triple = [0, c, x];
  else if (h < 4) triple = [0, x, c];
  else if (h < 5) triple = [x, 0, c];
  else triple = [c, 0, x];

  return {
    r: Math.round((triple[0] + m) * 255),
    g: Math.round((triple[1] + m) * 255),
    b: Math.round((triple[2] + m) * 255),
  };
}

/**
 * WCAG 2.x relative luminance of an opaque sRGB color.
 *
 * Mirrors `generator/validate.py::relative_luminance` exactly, including its
 * `0.03928` linear-segment cutoff, so a badge shown here and a build-time gate
 * failure can never disagree.
 *
 * @param {Rgb} rgb
 * @returns {number} luminance in `[0, 1]`.
 */
export function relativeLuminance(rgb) {
  /** @param {number} value @returns {number} */
  const channel = (value) => {
    const c = clamp(value, 0, 255) / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * channel(rgb.r) + 0.7152 * channel(rgb.g) + 0.0722 * channel(rgb.b);
}

/**
 * WCAG contrast ratio between two opaque sRGB colors. Argument order does not
 * matter -- the lighter color is always the numerator.
 *
 * @param {Rgb} a
 * @param {Rgb} b
 * @returns {number} ratio in `[1, 21]`.
 */
export function contrastRatio(a, b) {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/* hygiene-allow-color-start: color-string serializers. The two helpers below
   FORMAT a CSS color from numeric channel state -- the function-call text is
   an output template, not a hardcoded theme color. This is the single
   allowlisted span in the theme lab; every other color value in this feature
   stays numeric (h/s/l or r/g/b) until it reaches one of these two. */
/**
 * Serialize hue/saturation/lightness as a CSS `hsl` color.
 *
 * @param {Hsl} hsl
 * @returns {string}
 */
export function hslCss(hsl) {
  const h = Math.round((((hsl.h % 360) + 360) % 360) * 10) / 10;
  const s = Math.round(clamp(hsl.s, 0, 100) * 10) / 10;
  const l = Math.round(clamp(hsl.l, 0, 100) * 10) / 10;
  return `hsl(${h}, ${s}%, ${l}%)`;
}

/**
 * Serialize 8-bit channels plus an alpha as a legacy comma `rgba` color --
 * the spelling the token generator emits and the only one xterm.js accepts
 * alongside full-length hex.
 *
 * @param {Rgb} rgb
 * @param {number} alpha alpha in `[0, 1]`; emitted with two decimals.
 * @returns {string}
 */
export function rgbaCss(rgb, alpha) {
  /** @param {number} value @returns {number} */
  const channel = (value) => Math.round(clamp(value, 0, 255));
  const a = clamp(alpha, 0, 1).toFixed(2);
  return `rgba(${channel(rgb.r)}, ${channel(rgb.g)}, ${channel(rgb.b)}, ${a})`;
}
/* hygiene-allow-color-end */

/**
 * Pure black and white, always available as `--color-on-accent` candidates.
 *
 * Nothing in the design system requires `accent.on` to be one of the scope's
 * own colors, and the shipped themes prove it: of the six, only `dark` uses
 * its own `bg.primary`. `light` and both `apex` themes reuse dark's near-black
 * ink, `high-contrast-dark` ships pure black and `high-contrast-light` pure
 * white -- none of which is that scope's own background or primary text.
 * Offering only the two scope colors would make the lab reject accents that
 * ship fine: a mid grey reaches 4.22:1 against both of dark's primaries (a
 * FAIL) but 4.62:1 against black (a PASS).
 *
 * Built from numeric channels rather than written as hex literals, so this
 * file keeps its single hygiene exemption (the two serializers above).
 *
 * @type {ReadonlyArray<string>}
 */
const NEUTRAL_ON_ACCENT_CANDIDATES = [
  rgbToHex({ r: 0, g: 0, b: 0 }),
  rgbToHex({ r: 255, g: 255, b: 255 }),
];

/**
 * Pick `--color-on-accent`: whichever candidate reads best on the accent fill.
 *
 * This is the LAB'S OWN selection rule, not a rule read off `tokens.css` --
 * every shipped theme hand-picks this value. What the lab guarantees is only
 * that its choice clears the gate whenever any of its candidates can, and that
 * the value it scores is the value it exports.
 *
 * Scope colors are returned verbatim rather than re-serialized, so a value read
 * off a live computed style survives untouched.
 *
 * @param {Rgb} accent
 * @param {ScopeColors} scopeColors
 * @returns {string}
 */
function pickOnAccent(accent, scopeColors) {
  /** @type {string[]} */
  const candidates = [scopeColors.bgPrimary, scopeColors.textPrimary].filter(
    (value) => parseColor(value) !== null
  );
  candidates.push(...NEUTRAL_ON_ACCENT_CANDIDATES);

  let best = candidates[0];
  let bestRatio = -1;
  for (const candidate of candidates) {
    const rgb = parseColor(candidate);
    if (rgb === null) continue;
    const ratio = contrastRatio(accent, rgb);
    // Strictly greater, so an earlier candidate wins a tie -- the scope's own
    // colors are listed first and are the more idiomatic choice.
    if (ratio > bestRatio) {
      bestRatio = ratio;
      best = candidate;
    }
  }
  return best;
}

/**
 * Derive every accent-family custom property for one mode.
 *
 * See the module docstring for the rules and for why `--ansi-cursor-accent` is
 * not among the returned names.
 *
 * @param {LabState} state per-mode accent controls.
 * @param {ThemeMode} mode which mode to derive.
 * @param {ScopeColors} scopeColors the target scope's own bg/text primaries,
 *   used only by the `--color-on-accent` rule.
 * @returns {Record<string, string>} CSS custom-property name to value.
 */
export function deriveAccentVars(state, mode, scopeColors) {
  const accent = state[mode];
  const base = hslToRgb({ h: accent.hue, s: accent.saturation, l: accent.lightness });
  const emphasis = hslToRgb({ h: accent.hue, s: accent.saturation, l: accent.emphasisLightness });

  /** @type {Record<string, string>} */
  const vars = {
    '--color-accent': rgbToHex(base),
    '--color-accent-light': rgbToHex(emphasis),
    '--color-on-accent': pickOnAccent(base, scopeColors),
    '--border-accent': rgbaCss(emphasis, BORDER_ALPHA[mode]),
  };
  for (const level of TINT_LEVELS) {
    vars[`--accent-tint-${String(level).padStart(2, '0')}`] = rgbaCss(emphasis, level / 100);
  }
  vars['--wt-accent-system-tint-04'] = rgbaCss(base, SYSTEM_TINT_ALPHA);
  return vars;
}

/**
 * Derive the second accent family for one mode.
 *
 * The second accent is a role of its own, not a shade of the accent: every
 * family picks it by hand, and the build holds `accent-secondary.light` to its
 * own contrast gate. So it takes its own controls and derives from them with
 * the same lightness-only rules the accent uses.
 *
 * Two ways it differs from {@link deriveAccentVars}, both read off the
 * committed `tokens.css`: its tints come from the BASE rather than the emphasis
 * variant, and it carries a third opaque slot (`hover`, see
 * {@link SECONDARY_HOVER_DELTA}) where the accent carries `on`.
 *
 * @param {LabState} secondary per-mode controls for the second accent.
 * @param {ThemeMode} mode which mode to derive.
 * @returns {Record<string, string>} CSS custom-property name to value.
 */
export function deriveSecondaryAccentVars(secondary, mode) {
  const choice = secondary[mode];
  const hs = { h: choice.hue, s: choice.saturation };
  const base = hslToRgb({ ...hs, l: choice.lightness });
  const emphasis = hslToRgb({ ...hs, l: choice.emphasisLightness });
  const hover = hslToRgb({ ...hs, l: clamp(choice.lightness + SECONDARY_HOVER_DELTA, 0, 100) });

  /** @type {Record<string, string>} */
  const vars = {
    '--color-accent-secondary': rgbToHex(base),
    '--color-accent-secondary-light': rgbToHex(emphasis),
    '--color-accent-secondary-hover': rgbToHex(hover),
  };
  for (const level of SECONDARY_TINT_LEVELS) {
    vars[`--accent-secondary-tint-${String(level).padStart(2, '0')}`] = rgbaCss(base, level / 100);
  }
  return vars;
}

/**
 * Derive every custom property a proposal sets: both accent families at once.
 *
 * This is what the UI writes and what {@link evaluateGates} must be handed. The
 * third gate scores a second-accent token, so scoring an accent-only map would
 * fail closed on a value that was never derived rather than on a real shortfall.
 *
 * @param {LabState} state accent controls.
 * @param {LabState} secondary second-accent controls.
 * @param {ThemeMode} mode which mode to derive.
 * @param {ScopeColors} scopeColors passed through to the accent derivation.
 * @returns {Record<string, string>} CSS custom-property name to value.
 */
export function deriveThemeVars(state, secondary, mode, scopeColors) {
  return {
    ...deriveAccentVars(state, mode, scopeColors),
    ...deriveSecondaryAccentVars(secondary, mode),
  };
}

/**
 * The accent contrast gates, mirroring every accent entry of
 * `generator/validate.py::WCAG_GATES` (the AA tuple the default family is held
 * to). `accent.on` is gated against `accent.base` rather than the background,
 * because it is text on an accent fill; `accent-secondary.light` is the second
 * accent's text-safe slot and is gated against the background like body text.
 *
 * A test holds this list against `WCAG_GATES` itself, so a gate added to the
 * build shows up here or CI fails -- a colleague must never be told their theme
 * passes a check the build does not run, or fails one it does not enforce.
 *
 * @type {ReadonlyArray<{name: string, threshold: number}>}
 */
const GATES = [
  { name: 'accent.base vs bg.primary', threshold: 3.0 },
  { name: 'accent.on vs accent.base', threshold: 4.5 },
  { name: 'accent-secondary.light vs bg.primary', threshold: 4.5 },
];

/**
 * Score a derived accent family against the WCAG gates.
 *
 * Fails closed: if a color cannot be parsed, the gate reports the worst
 * possible ratio (1) and does not pass, rather than being skipped.
 *
 * @param {Record<string, string>} derived output of {@link deriveThemeVars} --
 *   both families, since the third gate scores a second-accent token.
 * @param {ScopeColors} scopeColors the same scope colors used to derive it.
 * @returns {GateResult[]} one entry per gate, in {@link GATES} order.
 */
export function evaluateGates(derived, scopeColors) {
  const accent = parseColor(derived['--color-accent']);
  const onAccent = parseColor(derived['--color-on-accent']);
  const secondaryLight = parseColor(derived['--color-accent-secondary-light']);
  const background = parseColor(scopeColors.bgPrimary);
  /** @type {Array<[Rgb | null, Rgb | null]>} */
  const pairs = [
    [accent, background],
    [onAccent, accent],
    [secondaryLight, background],
  ];
  return GATES.map((gate, index) => {
    const [foreground, against] = pairs[index];
    const ratio =
      foreground === null || against === null ? 1 : contrastRatio(foreground, against);
    return { name: gate.name, ratio, threshold: gate.threshold, pass: ratio >= gate.threshold };
  });
}

/**
 * Reduce a human-typed theme name to a CSS/id-safe slug.
 *
 * Diacritics are folded to their base letters; every other non-alphanumeric
 * run becomes a single dash; leading and trailing dashes are trimmed. A name
 * with no representable characters at all slugifies to the empty string, which
 * {@link checkCollision} then rejects.
 *
 * @param {string} text
 * @returns {string} a slug matching `/^[a-z0-9]+(-[a-z0-9]+)*$/`, or `''`.
 */
export function slugifyThemeName(text) {
  return String(text)
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/**
 * Check a slug against the shipped theme registry.
 *
 * A lab theme ships as a *family* with one theme per mode, so three things can
 * collide: the slug against an existing theme id, the slug against an existing
 * family name, and either derived id (`<slug>-dark` / `<slug>-light`) against
 * an existing theme id.
 *
 * The manifest is injected rather than imported so this stays pure and so
 * there is exactly one copy of the id list in the product -- callers pass
 * `THEMES`/`DEFAULTS` from the generated `tokens.js`.
 *
 * @param {string} slug output of {@link slugifyThemeName}.
 * @param {ThemeManifest} manifest known theme ids and family names.
 * @returns {CollisionResult} `reason` is `null` when there is no collision.
 */
export function checkCollision(slug, manifest) {
  if (slug === '') {
    return { collides: true, reason: 'name must contain at least one letter or digit' };
  }
  if (manifest.ids.includes(slug)) {
    return { collides: true, reason: `"${slug}" is already a theme id` };
  }
  if (manifest.families.includes(slug)) {
    return { collides: true, reason: `"${slug}" is already a theme family` };
  }
  for (const mode of ['dark', 'light']) {
    const derivedId = `${slug}-${mode}`;
    if (manifest.ids.includes(derivedId)) {
      return { collides: true, reason: `"${slug}" would produce the existing theme "${derivedId}"` };
    }
  }
  return { collides: false, reason: null };
}

/**
 * Order the derived custom properties are presented in -- the two opaque
 * colors first, then the alpha family from most to least opaque, so the table
 * reads as "the colors, then the washes".
 *
 * @type {ReadonlyArray<string>}
 */
const TOKEN_DISPLAY_ORDER = [
  '--color-accent',
  '--color-accent-light',
  '--color-on-accent',
  '--border-accent',
  '--accent-tint-30',
  '--accent-tint-25',
  '--accent-tint-20',
  '--accent-tint-12',
  '--accent-tint-10',
  '--accent-tint-08',
  '--accent-tint-06',
  '--accent-tint-04',
  '--wt-accent-system-tint-04',
  '--color-accent-secondary',
  '--color-accent-secondary-light',
  '--color-accent-secondary-hover',
  '--accent-secondary-tint-25',
  '--accent-secondary-tint-15',
  '--accent-secondary-tint-12',
  '--accent-secondary-tint-08',
  '--accent-secondary-tint-06',
  '--accent-secondary-tint-04',
];

/**
 * The design-system token dot-path each derived custom property is authored
 * as, for the export's "what to change" tables.
 *
 * `--wt-accent-system-tint-04` is deliberately absent: it lives in the
 * `web_terminal` interface document, not in a theme document, and shipped
 * families inherit it rather than authoring it (see {@link buildExportMarkdown}).
 *
 * @type {Readonly<Record<string, string>>}
 */
const TOKEN_PATHS = {
  '--color-accent': 'accent.base',
  '--color-accent-light': 'accent.light',
  '--color-on-accent': 'accent.on',
  '--border-accent': 'border.accent',
  '--accent-tint-04': 'tint.accent.04',
  '--accent-tint-06': 'tint.accent.06',
  '--accent-tint-08': 'tint.accent.08',
  '--accent-tint-10': 'tint.accent.10',
  '--accent-tint-12': 'tint.accent.12',
  '--accent-tint-20': 'tint.accent.20',
  '--accent-tint-25': 'tint.accent.25',
  '--accent-tint-30': 'tint.accent.30',
  '--color-accent-secondary': 'accent-secondary.base',
  '--color-accent-secondary-light': 'accent-secondary.light',
  '--color-accent-secondary-hover': 'accent-secondary.hover',
  '--accent-secondary-tint-04': 'tint.accent-secondary.04',
  '--accent-secondary-tint-06': 'tint.accent-secondary.06',
  '--accent-secondary-tint-08': 'tint.accent-secondary.08',
  '--accent-secondary-tint-12': 'tint.accent-secondary.12',
  '--accent-secondary-tint-15': 'tint.accent-secondary.15',
  '--accent-secondary-tint-25': 'tint.accent-secondary.25',
};

/**
 * One mode's contribution to an export: what was derived and how it scored.
 *
 * @typedef {{
 *   derived: Record<string, string>,
 *   gates: GateResult[],
 * }} ModeExport
 */

/**
 * Everything {@link buildExportMarkdown} needs, gathered by the UI layer.
 *
 * @typedef {{
 *   label: string,
 *   slug: string,
 *   collision: CollisionResult,
 *   dark: ModeExport,
 *   light: ModeExport,
 * }} ExportInput
 */

/** @param {number} ratio @returns {string} */
function formatRatio(ratio) {
  return `${ratio.toFixed(2)}:1`;
}

/**
 * Render the per-mode token table rows of the export.
 *
 * @param {ModeExport} modeExport
 * @returns {string[]} markdown lines.
 */
function exportTokenTable(modeExport) {
  const lines = ['| Token | Value |', '| --- | --- |'];
  for (const name of TOKEN_DISPLAY_ORDER) {
    const path = TOKEN_PATHS[name];
    if (path === undefined) continue;
    lines.push(`| \`${path}\` | \`${modeExport.derived[name]}\` |`);
  }
  return lines;
}

/**
 * Build the paste-into-an-issue markdown spec for a proposed theme.
 *
 * The output is deliberately self-contained: someone reading it in a GitHub
 * issue must be able to implement the theme without opening the lab. That
 * means it carries both the *values* and the complete *change set* -- every
 * file that has to move, in the shape this repository actually uses.
 *
 * Two details of that change set are easy to get wrong, so they are stated
 * explicitly in the output rather than left implied:
 *
 *   * `accent.base` / `accent.light` / `accent.on` are authored as *references*
 *     into `core.json`'s ramps (`{color.teal.300}`), not as literal hex, so a
 *     new accent needs ramp steps adding there first.
 *   * every shipped non-default family (`apex`, `high-contrast`) *inherits* the
 *     interface groups rather than authoring them, so the change set asks for
 *     `$extensions.inherits` entries -- and notes that
 *     `wt-accent-system-tint-04` therefore keeps the default value unless the
 *     author opts out of inheriting.
 *
 * @param {ExportInput} input
 * @returns {string} markdown.
 */
export function buildExportMarkdown(input) {
  const { label, slug, collision, dark, light } = input;
  const failed = [
    ...dark.gates.filter((gate) => !gate.pass).map((gate) => ['dark', gate]),
    ...light.gates.filter((gate) => !gate.pass).map((gate) => ['light', gate]),
  ];

  /** @type {string[]} */
  const lines = [];
  lines.push(`# Theme proposal: ${label || slug}`);
  lines.push('');
  lines.push(`- **Family name:** \`${slug}\``);
  lines.push(`- **Theme ids:** \`${slug}-dark\`, \`${slug}-light\``);
  lines.push(`- **Display label:** ${label || slug}`);
  lines.push('');

  if (failed.length > 0 || collision.collides) {
    lines.push('## ⚠ Warnings');
    lines.push('');
    if (collision.collides && collision.reason !== null) {
      lines.push(`- **Name collision:** ${collision.reason}. Pick a different name.`);
    }
    for (const entry of failed) {
      const mode = /** @type {string} */ (entry[0]);
      const gate = /** @type {GateResult} */ (entry[1]);
      lines.push(
        `- **Contrast (${mode}):** ${gate.name} is ${formatRatio(gate.ratio)}, ` +
          `below the required ${gate.threshold.toFixed(1)}:1.`
      );
    }
    lines.push('');
    lines.push('These do not block a proposal, but the design system enforces the');
    lines.push('contrast gates at build time -- a theme that fails them cannot ship');
    lines.push('as-is.');
    lines.push('');
  }

  for (const [mode, modeExport] of /** @type {Array<[string, ModeExport]>} */ ([
    ['Dark', dark],
    ['Light', light],
  ])) {
    lines.push(`## ${mode} mode`);
    lines.push('');
    lines.push(...exportTokenTable(modeExport));
    lines.push('');
    for (const gate of modeExport.gates) {
      const verdict = gate.pass ? 'PASS' : 'FAIL';
      lines.push(
        `- ${verdict} — ${gate.name}: ${formatRatio(gate.ratio)} ` +
          `(needs ${gate.threshold.toFixed(1)}:1)`
      );
    }
    lines.push('');
  }

  lines.push('## Implementation change set');
  lines.push('');
  lines.push('All paths are relative to `src/osprey/interfaces/design_system/`.');
  lines.push('');
  lines.push(`1. **\`tokens/core.json\`** — add ramp steps for the new hexes. The theme`);
  lines.push('   documents reference core ramps (`{color.teal.300}`) rather than');
  lines.push('   inlining hex, so add the accent base and emphasis colors above as new');
  lines.push('   steps in an existing ramp or a new one, then reference them in step 2.');
  lines.push('');
  lines.push(
    `2. **\`tokens/themes/${slug}-dark.json\`** and **\`tokens/themes/${slug}-light.json\`** —`
  );
  lines.push('   copy the same-mode sibling (`themes/dark.json` / `themes/light.json`), then');
  lines.push('   mind the two different authoring styles it uses:');
  lines.push('');
  lines.push('   - `accent.base`, `accent.light` and `accent.on` are **references** into');
  lines.push('     `core.json` (`{color.teal.300}`), not literals. Point them at the ramp');
  lines.push('     steps added in step 1 — do not paste the hex from the table here, or the');
  lines.push('     step-1 additions are left dead and the one-hop convention is broken.');
  lines.push('   - `border.accent` and the eight `tint.accent.*` entries are **literal**');
  lines.push('     alpha-composite colors. Paste those from the table exactly as tabled.');
  lines.push('');
  lines.push('   Then set');
  lines.push(
    `   \`$extensions\` to \`{"mode": "dark"|"light", "id": "${slug}-dark"|"${slug}-light",`
  );
  lines.push(`   "label": "${label || slug}", "family": "${slug}"}\` (no \`"default": true\` —`);
  lines.push('   that belongs to the shipped default theme).');
  lines.push('');
  lines.push('3. **`tokens/interfaces/*.json`** — add the new ids to each');
  lines.push('   `$extensions.inherits` map so the five interface documents');
  lines.push('   (`ariel.json`, `artifacts.json`, `channel_finder.json`,');
  lines.push('   `lattice_dashboard.json`, `web_terminal.json`) reuse the base groups:');
  lines.push('');
  lines.push('   ```json');
  lines.push(`   "${slug}-dark": "dark",`);
  lines.push(`   "${slug}-light": "light"`);
  lines.push('   ```');
  lines.push('');
  lines.push('   This is what `apex` already does in all five, and `high-contrast` in');
  lines.push('   four of them (`ariel.json` authors its own high-contrast groups). Note the');
  lines.push('   consequence: `wt-accent-system-tint-04` is authored in');
  lines.push('   `web_terminal.json`, so inheriting keeps the default value rather than');
  lines.push('   the accent-matched one below. Author a full `web_terminal.json` group');
  lines.push('   for the new ids only if that tint must match the accent:');
  lines.push('');
  lines.push(`   - dark: \`${dark.derived['--wt-accent-system-tint-04']}\``);
  lines.push(`   - light: \`${light.derived['--wt-accent-system-tint-04']}\``);
  lines.push('');
  lines.push('4. **Regenerate the artifacts** — run the generator and commit what it');
  lines.push('   writes (`static/css/tokens.css`, `static/js/tokens.js`,');
  lines.push('   `static/js/theme-boot.js`):');
  lines.push('');
  lines.push('   ```console');
  lines.push('   $ python -m osprey.interfaces.design_system.generator.build');
  lines.push('   ```');
  lines.push('');
  lines.push('5. **Check the gates** — the generator validates WCAG contrast per family.');
  lines.push('   An unrecognized family is held to the AA gates, which are the ones');
  lines.push('   scored above.');
  lines.push('');
  lines.push('_Generated by `osprey theme-lab`._');

  return lines.join('\n');
}
