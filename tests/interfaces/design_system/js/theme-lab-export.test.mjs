/**
 * Unit tests for `buildExportMarkdown` -- the paste-into-an-issue spec the
 * theme lab hands an author when they are happy with a proposal.
 *
 * The export is the lab's only durable output, so what this spec protects is
 * that it stays *self-contained*: someone reading it in a GitHub issue must be
 * able to implement the theme without opening the lab again. Concretely that
 * means three things, and they are what the tests below assert:
 *
 *   - the identity of the proposal (family slug, both derived theme ids, and
 *     the label to author) survives into the markdown, including when the
 *     author never typed a label;
 *   - both modes carry their derived values *paired with the token dot-path
 *     they are authored as*, plus the gate verdicts that judged them -- the
 *     table rows are checked path-and-value together, so a table that lost its
 *     values could not pass on the prose that also mentions the paths;
 *   - the change set names every real file that has to move, and the command
 *     that regenerates the artifacts.
 *
 * The warning block is checked in all three states -- failing gate, name
 * collision, and clean -- because the clean case is what stops the other two
 * from passing vacuously.
 *
 * Inputs are derived by calling the real `deriveAccentVars`/`evaluateGates`
 * rather than by hand-writing fixtures, so this spec keeps telling the truth if
 * the derivation rules change.
 *
 *   npx vitest run tests/interfaces/design_system/js/theme-lab-export.test.mjs
 */

import { test, expect, describe } from 'vitest';

import {
  buildExportMarkdown,
  deriveThemeVars,
  evaluateGates,
} from '/design-system/js/theme-lab.js';

/** @typedef {import('/design-system/js/theme-lab.js').LabState} LabState */
/** @typedef {import('/design-system/js/theme-lab.js').ThemeMode} ThemeMode */
/** @typedef {import('/design-system/js/theme-lab.js').ScopeColors} ScopeColors */
/** @typedef {import('/design-system/js/theme-lab.js').ModeExport} ModeExport */
/** @typedef {import('/design-system/js/theme-lab.js').CollisionResult} CollisionResult */

/** The osprey family's two scopes, verbatim from the generated tokens.css. */
const DARK_SCOPE = { bgPrimary: '#0a0f1a', textPrimary: '#f1f5f9' };
const LIGHT_SCOPE = { bgPrimary: '#f7f9fc', textPrimary: '#0c1322' };

/** A teal accent that clears both gates in both modes. */
/** @type {LabState} */
const PASSING_STATE = {
  dark: { hue: 190, saturation: 70, lightness: 60, emphasisLightness: 72 },
  light: { hue: 190, saturation: 65, lightness: 34, emphasisLightness: 26 },
};

/** A second accent that clears its own gate in both modes. */
/** @type {LabState} */
const PASSING_SECONDARY = {
  dark: { hue: 31, saturation: 53, lightness: 64, emphasisLightness: 80 },
  light: { hue: 31, saturation: 53, lightness: 44, emphasisLightness: 36 },
};

/** Same accent in light mode, but a near-background navy in dark mode. */
/** @type {LabState} */
const DARK_ON_DARK_STATE = {
  dark: { hue: 220, saturation: 40, lightness: 12, emphasisLightness: 20 },
  light: { hue: 190, saturation: 65, lightness: 34, emphasisLightness: 26 },
};

/** No collision -- what `checkCollision` returns for an unused name. */
/** @type {CollisionResult} */
const NO_COLLISION = { collides: false, reason: null };

/** The token dot-paths every mode table must carry, in the module's own order. */
const TOKEN_PATHS = [
  'accent.base',
  'accent.light',
  'accent.on',
  'border.accent',
  'tint.accent.04',
  'tint.accent.06',
  'tint.accent.08',
  'tint.accent.10',
  'tint.accent.12',
  'tint.accent.20',
  'tint.accent.25',
  'tint.accent.30',
  'accent-secondary.base',
  'accent-secondary.light',
  'accent-secondary.hover',
  'tint.accent-secondary.04',
  'tint.accent-secondary.06',
  'tint.accent-secondary.08',
  'tint.accent-secondary.12',
  'tint.accent-secondary.15',
  'tint.accent-secondary.25',
];

/**
 * Derive one mode the way the UI layer does, and score it.
 *
 * @param {LabState} state
 * @param {LabState} secondary
 * @param {ThemeMode} mode
 * @param {ScopeColors} scope
 * @returns {ModeExport}
 */
function modeExport(state, secondary, mode, scope) {
  const derived = deriveThemeVars(state, secondary, mode, scope);
  return { derived, gates: evaluateGates(derived, scope) };
}

/**
 * Assemble the full export input for a state, exactly as the UI layer would.
 *
 * @param {{state?: LabState, secondary?: LabState, label?: string, slug?: string,
 *   collision?: CollisionResult}} [overrides]
 * @returns {import('/design-system/js/theme-lab.js').ExportInput}
 */
function exportInput(overrides = {}) {
  const state = overrides.state ?? PASSING_STATE;
  const secondary = overrides.secondary ?? PASSING_SECONDARY;
  return {
    label: overrides.label ?? 'Harbor Teal',
    slug: overrides.slug ?? 'harbor-teal',
    collision: overrides.collision ?? NO_COLLISION,
    dark: modeExport(state, secondary, 'dark', DARK_SCOPE),
    light: modeExport(state, secondary, 'light', LIGHT_SCOPE),
  };
}

/**
 * The body of the `##` section whose heading mentions `heading`, up to the next
 * `##` heading. Lets a mode's rows and verdicts be asserted without the other
 * mode's identical-looking rows satisfying them.
 *
 * @param {string} markdown
 * @param {string} heading
 * @returns {string}
 */
function sectionOf(markdown, heading) {
  const lines = markdown.split('\n');
  const start = lines.findIndex((line) => line.startsWith('##') && line.includes(heading));
  expect(start, `export has no "${heading}" section`).toBeGreaterThanOrEqual(0);
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((line) => line.startsWith('## '));
  return (end === -1 ? rest : rest.slice(0, end)).join('\n');
}

/**
 * The one line of `section` that carries the token dot-path `path`.
 *
 * @param {string} section
 * @param {string} path
 * @returns {string}
 */
function rowFor(section, path) {
  const rows = section.split('\n').filter((line) => line.includes(`\`${path}\``));
  expect(rows, `expected exactly one row for ${path}`).toHaveLength(1);
  return rows[0];
}

describe('proposal identity', () => {
  test('names the family slug and both derived theme ids', () => {
    const markdown = buildExportMarkdown(exportInput());

    expect(markdown).toContain('harbor-teal');
    expect(markdown).toContain('harbor-teal-dark');
    expect(markdown).toContain('harbor-teal-light');
  });

  test('carries the human label into the title and the label to author', () => {
    const markdown = buildExportMarkdown(exportInput({ label: 'Harbor Teal' }));

    expect(markdown.split('\n')[0]).toContain('Harbor Teal');
    expect(markdown).toContain('"label": "Harbor Teal"');
  });

  test('falls back to the slug when no label was typed', () => {
    const markdown = buildExportMarkdown(exportInput({ label: '' }));

    expect(markdown.split('\n')[0]).toContain('harbor-teal');
    expect(markdown).toContain('"label": "harbor-teal"');
    expect(markdown).not.toContain('undefined');
  });
});

describe('per-mode values', () => {
  test('each mode tables every token dot-path against its own derived value', () => {
    const input = exportInput();
    const markdown = buildExportMarkdown(input);

    for (const [heading, modeInput] of /** @type {Array<[string, ModeExport]>} */ ([
      ['Dark mode', input.dark],
      ['Light mode', input.light],
    ])) {
      const section = sectionOf(markdown, heading);
      for (const path of TOKEN_PATHS) {
        expect(rowFor(section, path)).not.toBe('');
      }
      expect(rowFor(section, 'accent.base')).toContain(modeInput.derived['--color-accent']);
      expect(rowFor(section, 'accent.light')).toContain(modeInput.derived['--color-accent-light']);
      expect(rowFor(section, 'accent.on')).toContain(modeInput.derived['--color-on-accent']);
      expect(rowFor(section, 'border.accent')).toContain(modeInput.derived['--border-accent']);
      expect(rowFor(section, 'tint.accent.04')).toContain(modeInput.derived['--accent-tint-04']);
      expect(rowFor(section, 'tint.accent.30')).toContain(modeInput.derived['--accent-tint-30']);
    }
  });

  test('the two modes contribute distinct accent hexes, both present', () => {
    const input = exportInput();
    const markdown = buildExportMarkdown(input);

    const darkAccent = input.dark.derived['--color-accent'];
    const lightAccent = input.light.derived['--color-accent'];
    expect(darkAccent).not.toBe(lightAccent);
    expect(markdown).toContain(darkAccent);
    expect(markdown).toContain(lightAccent);
  });
});

describe('gate verdicts', () => {
  test('every gate of every mode is reported with a verdict and a ratio', () => {
    const input = exportInput();
    const markdown = buildExportMarkdown(input);

    for (const [heading, modeInput] of /** @type {Array<[string, ModeExport]>} */ ([
      ['Dark mode', input.dark],
      ['Light mode', input.light],
    ])) {
      const section = sectionOf(markdown, heading);
      expect(modeInput.gates.length).toBeGreaterThan(0);
      for (const gate of modeInput.gates) {
        const verdicts = section
          .split('\n')
          .filter((line) => line.includes(gate.name) && line.includes(gate.ratio.toFixed(2)));
        expect(verdicts, `no verdict line for ${heading} / ${gate.name}`).toHaveLength(1);
        expect(verdicts[0]).toContain(gate.pass ? 'PASS' : 'FAIL');
      }
    }
  });

  test('a failing gate is reported as FAIL in its own mode section', () => {
    const input = exportInput({ state: DARK_ON_DARK_STATE });
    const markdown = buildExportMarkdown(input);

    const failing = input.dark.gates.filter((gate) => !gate.pass);
    expect(failing.length).toBeGreaterThan(0);
    expect(sectionOf(markdown, 'Dark mode')).toContain('FAIL');
    expect(sectionOf(markdown, 'Light mode')).not.toContain('FAIL');
  });
});

describe('implementation change set', () => {
  test('names every file that has to move and the generator command', () => {
    const markdown = buildExportMarkdown(exportInput({ slug: 'harbor-teal' }));

    for (const file of [
      'core.json',
      'harbor-teal-dark.json',
      'harbor-teal-light.json',
      'ariel.json',
      'artifacts.json',
      'channel_finder.json',
      'lattice_dashboard.json',
      'web_terminal.json',
    ]) {
      expect(markdown, `change set never mentions ${file}`).toContain(file);
    }
    expect(markdown).toContain('python -m osprey.interfaces.design_system.generator.build');
  });
});

describe('warnings', () => {
  test('a failing gate raises a warning that names the gate, mode and ratio', () => {
    const input = exportInput({ state: DARK_ON_DARK_STATE });
    const markdown = buildExportMarkdown(input);

    const failing = input.dark.gates.filter((gate) => !gate.pass);
    expect(failing.length).toBeGreaterThan(0);

    const warnings = sectionOf(markdown, 'Warnings');
    for (const gate of failing) {
      expect(warnings).toContain(gate.name);
      expect(warnings).toContain(gate.ratio.toFixed(2));
    }
    expect(warnings.toLowerCase()).toContain('dark');
  });

  test('a name collision raises a warning carrying the collision reason', () => {
    const markdown = buildExportMarkdown(
      exportInput({
        slug: 'apex',
        collision: { collides: true, reason: '"apex" is already a theme family' },
      })
    );

    expect(sectionOf(markdown, 'Warnings')).toContain('"apex" is already a theme family');
  });

  test('no warning section at all when the gates pass and the name is free', () => {
    const input = exportInput();
    const markdown = buildExportMarkdown(input);

    expect([...input.dark.gates, ...input.light.gates].every((gate) => gate.pass)).toBe(true);
    expect(markdown).not.toMatch(/^#+.*Warning/mu);
    expect(markdown).not.toContain('FAIL');
  });
});
