/**
 * Theme lab -- DOM layer.
 *
 * The page's entry point. Everything here is impure by design: it reads the
 * lab's markup, owns the mutable accent state, and writes custom properties
 * onto the two scoped preview containers. All the math, the WCAG scoring, the
 * slug rules and the export markdown live in `theme-lab.js`, which stays
 * import-safe so the unit tests can exercise it without a document.
 *
 * The load-bearing property of this file: after boot, a color change never
 * creates or replaces a node. The mini-shell is cloned once per scope, the
 * badge and token rows are built once, and every subsequent update is a
 * `setProperty` write or a `textContent` assignment. That is what makes the
 * two previews re-skin live instead of flickering through a rebuild.
 */

import {
  buildExportMarkdown,
  checkCollision,
  clamp,
  deriveAccentVars,
  evaluateGates,
  hslCss,
  hslToRgb,
  parseColor,
  rgbToHex,
  rgbToHsl,
  rgbaCss,
  slugifyThemeName,
} from './theme-lab.js';

/** @typedef {import('./theme-lab.js').Rgb} Rgb */
/** @typedef {import('./theme-lab.js').Hsl} Hsl */
/** @typedef {import('./theme-lab.js').ThemeMode} ThemeMode */
/** @typedef {import('./theme-lab.js').LabState} LabState */
/** @typedef {import('./theme-lab.js').ScopeColors} ScopeColors */
/** @typedef {import('./theme-lab.js').GateResult} GateResult */
/** @typedef {import('./theme-lab.js').ThemeManifest} ThemeManifest */
/** @typedef {import('./theme-lab.js').ModeExport} ModeExport */
/** @typedef {import('./theme-lab.js').CollisionResult} CollisionResult */

/* The shipped theme registry, read from the generated module rather than
   transcribed: a hand-written id list here would drift the day a theme is
   added. Declared with the layer that consumes it -- ES module imports hoist,
   so the position is presentational, and the pure half above stays free of
   dependencies. */
import { THEMES, DEFAULTS, DEFAULT_FAMILY } from './tokens.js';

/** The two scopes the page renders, in DOM order. */
const MODES = /** @type {ReadonlyArray<ThemeMode>} */ (['dark', 'light']);

/**
 * Every shipped theme id and family name, for {@link checkCollision}.
 *
 * @type {ThemeManifest}
 */
const MANIFEST = {
  ids: THEMES.map((theme) => theme.id),
  families: [...new Set([DEFAULT_FAMILY, ...Object.keys(DEFAULTS), ...THEMES.map((t) => t.family)])],
};

/**
 * How far the emphasis variant (`--color-accent-light`) sits from its mode's
 * base lightness. Positive in dark mode, negative in light: "light" is the
 * token's name for *emphasis*, and emphasis reads as brighter on a dark
 * surface and deeper on a pale one. The lab has no emphasis control of its
 * own, so this pair is the whole rule.
 *
 * @type {Record<ThemeMode, number>}
 */
const EMPHASIS_DELTA = { dark: 16, light: -8 };

/**
 * How far light mode's base accent starts below dark mode's when a single
 * color is chosen for both (a preset, or a typed hex). A pale background
 * needs a deeper accent to clear the same contrast gate. It is only a
 * starting point: each mode's slider moves independently afterwards.
 */
const LIGHT_MODE_DELTA = -10;

/** Lightness the hue/saturation disc is painted at -- a fixed reference. */
const WHEEL_LIGHTNESS = 50;

/** Radius, in canvas pixels, of the disc's current-selection marker. */
const MARKER_RADIUS = 6;

/** Margin above a gate's threshold within which a pass is still flagged. */
const WARN_MARGIN = 0.5;

/** @typedef {{label: string, hsl: [number, number, number]}} Preset */

/**
 * Accent starting points, as numeric `[hue, saturation, lightness]` triples --
 * never color strings, so this file needs no second hygiene allowance. The
 * lightness is dark mode's; light mode starts {@link LIGHT_MODE_DELTA} below
 * it. All five clear both gates in both modes as shipped.
 *
 * @type {ReadonlyArray<Preset>}
 */
const PRESETS = [
  { label: 'Teal', hsl: [178, 60, 41] },
  { label: 'Amber', hsl: [45, 70, 50] },
  { label: 'Indigo', hsl: [232, 60, 55] },
  { label: 'Rose', hsl: [345, 60, 46] },
  { label: 'Slate', hsl: [210, 20, 55] },
];

/**
 * The lab's single mutable store. Every control writes here through
 * {@link setHueSaturation}/{@link setAccent}/{@link setModeLightness}, and
 * every readout is recomputed from it -- there is no second source of truth.
 *
 * @type {LabState}
 */
const state = {
  dark: { hue: 0, saturation: 0, lightness: 0, emphasisLightness: 0 },
  light: { hue: 0, saturation: 0, lightness: 0, emphasisLightness: 0 },
};

/**
 * Set one mode's base lightness, re-deriving its emphasis variant.
 *
 * @param {ThemeMode} mode
 * @param {number} lightness
 */
function setModeLightness(mode, lightness) {
  const base = clamp(lightness, 0, 100);
  state[mode] = {
    ...state[mode],
    lightness: base,
    emphasisLightness: clamp(base + EMPHASIS_DELTA[mode], 0, 100),
  };
}

/**
 * Set the hue and saturation both modes share, leaving each mode's lightness
 * alone -- dragging the wheel must not undo a slider the author has tuned.
 *
 * @param {number} hue
 * @param {number} saturation
 */
function setHueSaturation(hue, saturation) {
  for (const mode of MODES) {
    state[mode] = { ...state[mode], hue: (hue + 360) % 360, saturation: clamp(saturation, 0, 100) };
  }
}

/**
 * Set the whole accent from one color choice: shared hue/saturation, dark
 * mode at `lightness`, light mode {@link LIGHT_MODE_DELTA} below it.
 *
 * @param {number} hue
 * @param {number} saturation
 * @param {number} lightness dark mode's base lightness.
 */
function setAccent(hue, saturation, lightness) {
  setHueSaturation(hue, saturation);
  setModeLightness('dark', lightness);
  setModeLightness('light', lightness + LIGHT_MODE_DELTA);
}

/** The current accent as a hex color -- dark mode's, the mode this page runs in. */
function accentHex() {
  return rgbToHex(hslToRgb({ h: state.dark.hue, s: state.dark.saturation, l: state.dark.lightness }));
}

/* --- DOM helpers --------------------------------------------------------- */

/**
 * Look up a required element by id, narrowed to a concrete element type.
 *
 * @template {Element} T
 * @param {string} id
 * @param {new () => T} ctor
 * @returns {T}
 */
function requireElement(id, ctor) {
  const node = document.getElementById(id);
  if (!(node instanceof ctor)) {
    throw new Error(`theme lab: element "${id}" is missing or not a ${ctor.name}`);
  }
  return node;
}

/**
 * Create an element with a class and, optionally, text content.
 *
 * @param {string} tag
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className !== '') node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Read a scope's own background and primary text, which `tokens.css` gives it
 * through its `data-theme` attribute. Read once at init and cached: they are
 * inputs to the derivation, never outputs of it.
 *
 * @param {HTMLElement} element
 * @returns {ScopeColors}
 */
function readScopeColors(element) {
  const styles = getComputedStyle(element);
  return {
    bgPrimary: styles.getPropertyValue('--bg-primary').trim(),
    textPrimary: styles.getPropertyValue('--text-primary').trim(),
  };
}

/* --- Hue / saturation disc ----------------------------------------------- */

/**
 * Paint the hue/saturation disc into a reusable buffer: angle is hue, radius
 * is saturation, lightness fixed at {@link WHEEL_LIGHTNESS}. Pixels outside
 * the disc stay transparent, so the canvas's own background shows through.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} size canvas width and height, in pixels.
 * @returns {ImageData}
 */
function paintWheel(ctx, size) {
  const image = ctx.createImageData(size, size);
  const centre = size / 2;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const dx = x - centre + 0.5;
      const dy = y - centre + 0.5;
      const distance = Math.sqrt(dx * dx + dy * dy) / centre;
      if (distance > 1) continue;
      const rgb = hslToRgb({
        h: (Math.atan2(dy, dx) * 180) / Math.PI,
        s: distance * 100,
        l: WHEEL_LIGHTNESS,
      });
      const offset = (y * size + x) * 4;
      image.data[offset] = rgb.r;
      image.data[offset + 1] = rgb.g;
      image.data[offset + 2] = rgb.b;
      image.data[offset + 3] = 255;
    }
  }
  return image;
}

/**
 * Blit the disc and ring the current selection. The marker is drawn as a pale
 * circle inside a dark one so it stays visible over every hue.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {ImageData} image
 * @param {number} hue
 * @param {number} saturation
 */
function drawWheel(ctx, image, hue, saturation) {
  ctx.putImageData(image, 0, 0);
  const centre = image.width / 2;
  const angle = (hue * Math.PI) / 180;
  const distance = (clamp(saturation, 0, 100) / 100) * centre;
  const x = centre + Math.cos(angle) * distance;
  const y = centre + Math.sin(angle) * distance;
  ctx.lineWidth = 2;
  ctx.strokeStyle = rgbaCss({ r: 0, g: 0, b: 0 }, 0.6);
  ctx.beginPath();
  ctx.arc(x, y, MARKER_RADIUS + 1, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = hslCss({ h: 0, s: 0, l: 100 });
  ctx.beginPath();
  ctx.arc(x, y, MARKER_RADIUS, 0, Math.PI * 2);
  ctx.stroke();
}

/**
 * Convert a pointer position over the disc into a hue/saturation pair,
 * clamping to the rim so a drag that leaves the canvas still tracks.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {PointerEvent} event
 * @returns {{hue: number, saturation: number}}
 */
function pointerToHueSaturation(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const radius = rect.width / 2;
  const dx = event.clientX - rect.left - radius;
  const dy = event.clientY - rect.top - rect.height / 2;
  return {
    hue: (Math.atan2(dy, dx) * 180) / Math.PI,
    saturation: (Math.sqrt(dx * dx + dy * dy) / radius) * 100,
  };
}

/* --- Readouts ------------------------------------------------------------ */

/**
 * Build one scope's gate rows once. Only the badges change afterwards, so
 * only they are returned.
 *
 * @param {HTMLElement} container
 * @param {GateResult[]} results
 * @returns {HTMLElement[]} badge nodes, in gate order.
 */
function buildGates(container, results) {
  return results.map((gate) => {
    const row = make('div', 'lab-gate');
    const badge = make('span', 'lab-gate-badge');
    row.append(make('span', 'lab-gate-label', gate.name), badge);
    container.append(row);
    return badge;
  });
}

/**
 * Restate each gate as `PASS`/`FAIL` plus its ratio. A pass within
 * {@link WARN_MARGIN} of its threshold is styled as a warning: it clears the
 * build-time gate, but only just.
 *
 * @param {HTMLElement[]} badges
 * @param {GateResult[]} results
 */
function updateGates(badges, results) {
  results.forEach((gate, index) => {
    const badge = badges[index];
    const tight = gate.pass && gate.ratio < gate.threshold + WARN_MARGIN;
    badge.textContent = `${gate.pass ? 'PASS' : 'FAIL'} ${gate.ratio.toFixed(2)}`;
    badge.title = `needs ${gate.threshold.toFixed(1)}:1`;
    badge.classList.toggle('is-pass', gate.pass && !tight);
    badge.classList.toggle('is-warn', tight);
    badge.classList.toggle('is-fail', !gate.pass);
  });
}

/** @typedef {{swatch: HTMLElement, value: HTMLElement}} TokenRow */

/**
 * Build one scope's derived-token table once, one row per property name.
 *
 * @param {HTMLElement} container
 * @param {Record<string, string>} vars
 * @returns {Map<string, TokenRow>} the cells each name writes into.
 */
function buildTokens(container, vars) {
  const table = make('table', 'lab-token-table');
  const body = make('tbody', '');
  const header = make('tr', '');
  for (const title of ['', 'Token', 'Value']) header.append(make('th', '', title));
  body.append(header);

  /** @type {Map<string, TokenRow>} */
  const rows = new Map();
  for (const name of Object.keys(vars)) {
    const row = make('tr', '');
    const swatchCell = make('td', '');
    const swatch = make('span', 'lab-token-swatch');
    const value = make('td', '');
    swatchCell.append(swatch);
    row.append(swatchCell, make('td', 'lab-token-name', name), value);
    body.append(row);
    rows.set(name, { swatch, value });
  }
  table.append(body);
  container.append(table);
  return rows;
}

/**
 * Restate every derived value. Cells are written in place -- the table is
 * built once and never rebuilt, so a color change costs text and style
 * writes only.
 *
 * @param {Map<string, TokenRow>} rows
 * @param {Record<string, string>} vars
 */
function updateTokens(rows, vars) {
  for (const [name, cells] of rows) {
    const value = vars[name] ?? '';
    cells.swatch.style.background = value;
    cells.value.textContent = value;
  }
}

/* --- Page wiring --------------------------------------------------------- */

/**
 * One preview scope: its themed container, the colors that container supplies,
 * its lightness control, and the readout nodes it writes into.
 *
 * @typedef {{
 *   mode: ThemeMode,
 *   panel: HTMLElement,
 *   colors: ScopeColors,
 *   slider: HTMLInputElement,
 *   sliderValue: HTMLElement,
 *   badges: HTMLElement[],
 *   tokens: Map<string, TokenRow>,
 * }} Scope
 */

/**
 * Wire the lab page. Everything the DOM needs is created here, once; every
 * later interaction only writes custom properties and text.
 *
 * The drawing context is a parameter rather than something acquired here: it
 * is captured by the nested `render`, and a locally narrowed `const` does not
 * carry its non-nullness into a hoisted closure.
 *
 * @param {HTMLCanvasElement} wheel the hue/saturation canvas.
 * @param {CanvasRenderingContext2D} context that canvas's 2d context.
 */
function initLab(wheel, context) {
  const template = requireElement('mini-shell', HTMLTemplateElement);
  const hexInput = requireElement('hex-input', HTMLInputElement);
  const hexSwatch = requireElement('hex-swatch', HTMLElement);
  const presetRow = requireElement('preset-row', HTMLElement);
  const nameInput = requireElement('theme-name', HTMLInputElement);
  const slugWarning = requireElement('slug-warning', HTMLElement);
  const exportText = requireElement('export-text', HTMLElement);
  const copyButton = requireElement('copy-export', HTMLButtonElement);

  const first = PRESETS[0];
  setAccent(first.hsl[0], first.hsl[1], first.hsl[2]);

  /** @type {Scope[]} */
  const scopes = MODES.map((mode) => {
    const mount = requireElement(`preview-${mode}`, HTMLElement);
    const panel = mount.closest('[data-theme]');
    if (!(panel instanceof HTMLElement)) {
      throw new Error(`theme lab: preview "${mode}" is not inside a themed scope`);
    }
    mount.append(template.content.cloneNode(true));
    const colors = readScopeColors(mount);
    const vars = deriveAccentVars(state, mode, colors);
    return {
      mode,
      panel,
      colors,
      slider: requireElement(`lightness-${mode}`, HTMLInputElement),
      sliderValue: requireElement(`lightness-${mode}-value`, HTMLElement),
      badges: buildGates(requireElement(`gates-${mode}`, HTMLElement), evaluateGates(vars, colors)),
      tokens: buildTokens(requireElement(`tokens-${mode}`, HTMLElement), vars),
    };
  });

  /* Presets are static, so each button -- and its swatch -- is built once.
     The click handler is delegated off the row and finds its payload through
     this map, keyed by the node the `data-preset` hook matched. */
  /** @type {Map<Element, Preset>} */
  const presetByNode = new Map();
  PRESETS.forEach((preset, index) => {
    const button = make('button', 'lab-preset');
    button.setAttribute('type', 'button');
    button.dataset.preset = String(index);
    const swatch = make('span', 'lab-preset-swatch');
    swatch.style.background = hslCss({ h: preset.hsl[0], s: preset.hsl[1], l: preset.hsl[2] });
    button.append(swatch, document.createTextNode(preset.label));
    presetRow.append(button);
    presetByNode.set(button, preset);
  });

  /**
   * Recompute every derived value from {@link state} and write it out.
   *
   * No node is created or replaced here: each scope gets custom-property
   * writes, and the readouts get text. That is what makes the two previews
   * re-skin live rather than flicker through a rebuild.
   */
  function render() {
    /** @type {Partial<Record<ThemeMode, ModeExport>>} */
    const perMode = {};

    for (const scope of scopes) {
      const vars = deriveAccentVars(state, scope.mode, scope.colors);
      for (const [name, value] of Object.entries(vars)) {
        scope.panel.style.setProperty(name, value);
        // Dark is also the document's own theme, so the lab's chrome (buttons,
        // focus rings, preset pills) follows the accent being designed.
        if (scope.mode === 'dark') document.documentElement.style.setProperty(name, value);
      }
      const gates = evaluateGates(vars, scope.colors);
      perMode[scope.mode] = { derived: vars, gates };
      updateGates(scope.badges, gates);
      updateTokens(scope.tokens, vars);
      const lightness = Math.round(state[scope.mode].lightness);
      scope.slider.value = String(lightness);
      scope.sliderValue.textContent = `${lightness}%`;
    }

    const hex = accentHex();
    hexSwatch.style.background = hex;
    if (document.activeElement !== hexInput) {
      hexInput.value = hex;
      hexInput.classList.remove('is-invalid');
    }
    for (const [node, preset] of presetByNode) {
      const matches =
        Math.round(state.dark.hue) === preset.hsl[0] &&
        Math.round(state.dark.saturation) === preset.hsl[1] &&
        Math.round(state.dark.lightness) === preset.hsl[2];
      node.classList.toggle('is-active', matches);
    }
    drawWheel(context, wheelImage, state.dark.hue, state.dark.saturation);

    // The name field feeds two readouts -- the inline warning and the export
    // heading -- so it is resolved here rather than in its own listener, and
    // both stay consistent with whatever was last typed.
    const typed = nameInput.value.trim();
    const slug = slugifyThemeName(typed);
    const collision = checkCollision(slug, MANIFEST);
    if (typed === '') {
      slugWarning.textContent = '';
      slugWarning.className = 'lab-warning';
    } else {
      // textContent, never markup: the reason quotes what the author typed.
      slugWarning.textContent = collision.collides ? collision.reason : `${slug} — available`;
      slugWarning.className = `lab-warning ${collision.collides ? 'is-error' : 'is-ok'}`;
    }

    const dark = perMode.dark;
    const light = perMode.light;
    if (dark !== undefined && light !== undefined) {
      exportText.textContent = buildExportMarkdown({
        label: typed,
        // An unnamed proposal still exports something implementable; the
        // placeholder is obvious enough that nobody ships it by accident.
        slug: slug === '' ? 'my-theme' : slug,
        collision,
        dark,
        light,
      });
    }
  }

  const wheelImage = paintWheel(context, wheel.width);

  // The disc is a control, so it takes focus and answers the arrow keys; the
  // markup declares only its label, the interaction model belongs here.
  wheel.tabIndex = 0;

  /** @param {PointerEvent} event */
  const trackPointer = (event) => {
    const picked = pointerToHueSaturation(wheel, event);
    setHueSaturation(picked.hue, picked.saturation);
    render();
  };
  wheel.addEventListener('pointerdown', (event) => {
    wheel.setPointerCapture(event.pointerId);
    trackPointer(event);
  });
  wheel.addEventListener('pointermove', (event) => {
    if (wheel.hasPointerCapture(event.pointerId)) trackPointer(event);
  });
  wheel.addEventListener('pointerup', (event) => wheel.releasePointerCapture(event.pointerId));
  wheel.addEventListener('keydown', (event) => {
    const step = event.shiftKey ? 10 : 2;
    /** @type {Record<string, [number, number] | undefined>} */
    const moves = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    const move = moves[event.key];
    if (move === undefined) return;
    event.preventDefault();
    setHueSaturation(state.dark.hue + move[0], state.dark.saturation + move[1]);
    render();
  });

  for (const scope of scopes) {
    scope.slider.addEventListener('input', () => {
      setModeLightness(scope.mode, Number(scope.slider.value));
      render();
    });
  }

  hexInput.addEventListener('input', () => {
    const rgb = parseColor(hexInput.value);
    // Half-typed input is flagged, not acted on; the field reverts on commit.
    hexInput.classList.toggle('is-invalid', rgb === null);
    if (rgb === null) return;
    const hsl = rgbToHsl(rgb);
    setAccent(hsl.h, hsl.s, hsl.l);
    render();
  });
  const revertHex = () => {
    hexInput.classList.remove('is-invalid');
    hexInput.value = accentHex();
  };
  hexInput.addEventListener('change', revertHex);
  hexInput.addEventListener('blur', revertHex);

  presetRow.addEventListener('click', (event) => {
    const node = event.target instanceof Element ? event.target.closest('[data-preset]') : null;
    const preset = node === null ? undefined : presetByNode.get(node);
    if (preset === undefined) return;
    setAccent(preset.hsl[0], preset.hsl[1], preset.hsl[2]);
    render();
  });

  nameInput.addEventListener('input', () => render());

  copyButton.addEventListener('click', () => {
    // The <pre> is populated and selectable at all times, so a clipboard
    // denial (insecure origin, refused permission) costs the author a manual
    // select rather than the export itself.
    const clipboard = navigator.clipboard;
    if (clipboard === undefined) {
      copyButton.textContent = 'Select and copy';
      return;
    }
    clipboard.writeText(exportText.textContent ?? '').then(
      () => {
        copyButton.textContent = 'Copied';
        window.setTimeout(() => {
          copyButton.textContent = 'Copy';
        }, 1200);
      },
      () => {
        copyButton.textContent = 'Select and copy';
      }
    );
  });

  render();
}

/* The page's only side effect. This module is loaded from <head> with `type=
   "module"`, so the document is parsed by the time it runs; the guard keeps
   the module importable (a canvas-less environment simply gets no lab). */
const wheelCanvas = document.getElementById('hue-wheel');
if (wheelCanvas instanceof HTMLCanvasElement) {
  const wheelContext = wheelCanvas.getContext('2d');
  if (wheelContext !== null) initLab(wheelCanvas, wheelContext);
}
