// @ts-check
/**
 * The set-up's pure helpers, around the form of draft-model.js: the config
 * the stage draws while the form is edited (no checks: draft-model.js checks
 * on save), where each reason a save was refused belongs, a preset applied
 * over the form, and the knob filter.
 *
 * @module setup-model
 */

import { DEFAULT_OBSERVABLE_DIRECTION } from './approval-model.js';

/**
 * @typedef {import('./draft-model.js').FormState} FormState
 * @typedef {import('./draft-model.js').Preset} Preset
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 */

/**
 * The config the form stands for now, for the stage to draw. Nothing is
 * checked: a blank bound stays null, a constraint not yet complete is left
 * out. What a save sends is `buildConfig` of draft-model.js, which checks.
 *
 * @param {FormState} form
 * @param {{ environment: string, target: string }} where
 * @returns {OptimizationConfig}
 */
export function previewConfig(form, { environment, target }) {
  const selected = form.vars.filter((v) => v.include);
  /** @type {Record<string, { type: string, value: number }>} */
  const constraints = {};
  for (const c of form.constraints) {
    if (c.observable && c.value != null && Number.isFinite(c.value)) {
      constraints[c.observable] = { type: c.type, value: c.value };
    }
  }
  return {
    ...form.base,
    environment_name: environment || null,
    control_target: target || null,
    preset: form.preset || null,
    objective_name: form.objective || null,
    objective_direction: form.direction || null,
    algorithm: form.algorithm,
    n_iterations: form.nIterations ?? undefined,
    n_initial_samples: form.nInitial,
    on_complete: form.onComplete,
    variables: selected.map((v) => v.name),
    variable_overrides: Object.fromEntries(
      selected.map((v) => [v.name, { min: v.min, max: v.max, step_size: v.step }])
    ),
    constraints,
    model_file: form.modelFile || null,
  };
}

/** Is `ch` a character an identifier goes on with? */
const WORDISH = /[A-Za-z0-9_]/;

/**
 * Whether `text` names `name` as a whole identifier, not as a part of a longer one.
 *
 * @param {string} text
 * @param {string} name
 */
function names(text, name) {
  for (let at = text.indexOf(name); at !== -1; at = text.indexOf(name, at + 1)) {
    const before = at > 0 ? text[at - 1] : '';
    const after = text[at + name.length] ?? '';
    if (!WORDISH.test(before) && !WORDISH.test(after)) return true;
  }
  return false;
}

/** The words that name each field in a reason, checked in this order. */
const FIELD_WORDS = Object.freeze(
  /** @type {[RegExp, string][]} */ ([
    [/preset/, 'preset'],
    [/constraint/, 'constraints'],
    [/control.target/, 'target'],
    [/environment/, 'environment'],
    [/direction/, 'direction'],
    [/objective/, 'objective'],
    [/initial.sample/, 'initial'],
    [/iteration/, 'iterations'],
    [/algorithm.param/, 'params'],
    [/model.file/, 'model'],
    [/algorithm/, 'method'],
    [/on.complete/, 'end'],
    [/variable/, 'knobs'],
  ])
);

/**
 * The fields a reason for a refused save is about: `knob:<name>` for every
 * knob it names (a reason of the service can name several), else the one of
 * `preset`, `constraints`, `target`, `environment`, `direction`, `objective`,
 * `initial`, `iterations`, `params`, `model`, `method`, `end`, `knobs` it
 * names first; none when it names none of them.
 *
 * @param {string} text  a reason (draft-model.js or the service)
 * @param {string[]} knobNames  every knob of the environment
 * @returns {string[]}
 */
export function issueFields(text, knobNames) {
  /** @type {string[]} */
  const knobs = [];
  // Longest first, and a knob found is blanked out, so a knob whose name is
  // the start of a longer one's is not found inside it.
  let rest = text;
  for (const name of [...knobNames].sort((a, b) => b.length - a.length)) {
    if (!names(rest, name)) continue;
    knobs.push(`knob:${name}`);
    rest = rest.split(name).join(' ');
  }
  if (knobs.length) return knobs;
  const t = text.toLowerCase();
  const field = FIELD_WORDS.find(([pattern]) => pattern.test(t))?.[1];
  return field ? [field] : [];
}

/**
 * The reasons of a refused save, by the field each is about. A reason that
 * names no field is in none (the set-up lists every reason at its top).
 *
 * @param {string[]} issues
 * @param {string[]} knobNames
 * @returns {Record<string, string[]>}
 */
export function groupIssues(issues, knobNames) {
  /** @type {Record<string, string[]>} */
  const byField = {};
  for (const text of issues) {
    for (const field of issueFields(text, knobNames)) (byField[field] ||= []).push(text);
  }
  return byField;
}

/** The section of the set-up each field is in. */
const FOLDER_OF = Object.freeze(
  /** @type {Record<string, string>} */ ({
    environment: 'goal',
    target: 'goal',
    objective: 'goal',
    direction: 'goal',
    constraints: 'goal',
    preset: 'knobs',
    knobs: 'knobs',
    method: 'search',
    initial: 'search',
    iterations: 'search',
    params: 'search',
    model: 'search',
    end: 'end',
  })
);

/**
 * The sections of the set-up (`goal`, `knobs`, `search`, `end`) that hold a
 * field a refused save names.
 *
 * @param {Record<string, string[]>} byField  {@link groupIssues}
 * @returns {Set<string>}
 */
export function issueFolders(byField) {
  return new Set(Object.keys(byField).map((f) => (f.startsWith('knob:') ? 'knobs' : FOLDER_OF[f])));
}

/**
 * The reasons a refused save names a knob for, by knob name.
 *
 * @param {Record<string, string[]>} byField  {@link groupIssues}
 * @returns {Record<string, string[]>}
 */
export function knobIssues(byField) {
  return Object.fromEntries(
    Object.entries(byField)
      .filter(([f]) => f.startsWith('knob:'))
      .map(([f, list]) => [f.slice('knob:'.length), list])
  );
}

/**
 * The stored config a set-up seeds its form from, and that config's
 * revision. A set-up that edits a draft seeds only from a config read for
 * that draft: the one it was opened with, else the approval's read when it
 * is of that draft; never from the environment's defaults. A new run seeds
 * from the config it runs again, if any.
 *
 * @param {import('./store.js').State} state
 * @returns {{ config: OptimizationConfig|null, revision: string|null }}
 */
export function seedSource({ setup, approval }) {
  if (!setup) return { config: null, revision: null };
  if (setup.from) return { config: setup.from, revision: setup.jobId ? setup.revision : null };
  if (setup.jobId && approval.jobId === setup.jobId && approval.config) {
    return { config: approval.config, revision: approval.revision };
  }
  return { config: null, revision: null };
}

/**
 * Whether the draft a set-up edits changed on the service since the form was
 * seeded from (or confirmed against) revision `accepted`: the agent or
 * another panel saved it meanwhile. Not while the set-up's own save is under
 * way (its answer is the new revision).
 *
 * @param {import('./store.js').State} state
 * @param {string|null} accepted
 */
export function draftMoved({ setup, approval }, accepted) {
  if (!setup?.jobId || setup.saving || !accepted) return false;
  return approval.jobId === setup.jobId && !!approval.revision && approval.revision !== accepted;
}

/**
 * An observable's own direction (`ObservableMetadata.direction`, its default
 * when it sets none); null when the environment does not name it.
 *
 * @param {EnvironmentDetails|null|undefined} env
 * @param {string|null|undefined} name
 * @returns {string|null}
 */
export function ownDirection(env, name) {
  const observable = name ? env?.observables?.[name] : null;
  return observable ? observable.direction || DEFAULT_OBSERVABLE_DIRECTION : null;
}

/**
 * The form's direction after its objective moves from `from` to `to`. A
 * direction that was only the old objective's own is cleared, so the new
 * objective is sought in its own direction; one the operator set against the
 * old objective's own stays.
 *
 * @param {EnvironmentDetails|null|undefined} env
 * @param {string} from
 * @param {string} direction  the form's direction ('' for the objective's own)
 * @param {string} to
 * @returns {string}
 */
export function directionAfter(env, from, direction, to) {
  if (!direction || from === to) return direction;
  return direction === ownDirection(env, from) ? '' : direction;
}

/**
 * The form after a preset is picked: the preset's knobs and windows (`seeded`
 * is `formFromEnv` with the preset and the service's initial windows), and
 * the method, iterations and objective the preset names. What the preset
 * leaves unset, and everything else the form holds, stays as it was.
 *
 * @param {FormState} form
 * @param {FormState} seeded
 * @param {Preset|null} preset
 * @param {EnvironmentDetails|null} env
 * @returns {FormState}
 */
export function withPreset(form, seeded, preset, env) {
  const objective = preset?.objective_name ? seeded.objective : form.objective;
  return {
    ...form,
    preset: seeded.preset,
    vars: seeded.vars,
    objective,
    direction: directionAfter(env, form.objective, form.direction, objective),
    algorithm: preset?.algorithm ? seeded.algorithm : form.algorithm,
    nIterations: preset?.n_iterations != null ? seeded.nIterations : form.nIterations,
  };
}

/**
 * The knobs a stored config tunes that its environment no longer has: the
 * form cannot hold them, so a save leaves them out.
 *
 * @param {OptimizationConfig|null|undefined} config
 * @param {EnvironmentDetails|null|undefined} env
 * @returns {string[]}
 */
export function missingKnobs(config, env) {
  if (!config || !env) return [];
  return (config.variables || []).filter((name) => !(name in (env.variables || {})));
}

/**
 * A picker's options, with one more for `value` when none holds it, so the
 * picker shows the value a save sends, never another one.
 *
 * @template {[string, string, boolean?]} T
 * @param {T[]} options  value, label (, disabled)
 * @param {string} value
 * @returns {[string, string, boolean?][]}
 */
export function withUnlisted(options, value) {
  if (!value || options.some(([v]) => v === value)) return options;
  return [...options, [value, `${value} (not offered)`]];
}

/**
 * Whether a knob matches the picker's filter: its identifier or its
 * description holds every word of the query.
 *
 * @param {string} name
 * @param {{ description?: string|null }|null|undefined} meta
 * @param {string} query
 */
export function knobMatches(name, meta, query) {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = `${name} ${meta?.description || ''}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}
