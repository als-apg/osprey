// @ts-check
/**
 * The draft form's state, as pure data: seeding it from an environment (and a
 * preset's resolved bounds) or from a stored draft, turning it back into the
 * `OptimizationConfig` the relay sends.
 *
 * The form keeps the loaded draft's other fields (`base`) and overlays its
 * own, so fields it does not edit (surrogate model, environment config,
 * execution mode) survive an update untouched.
 *
 * @module draft-model
 */

/**
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {{ display_name?: string, description?: string, variables?: string[]|null,
 *   variable_overrides?: Record<string, { step_size?: number|null }>,
 *   algorithm?: string|null, n_iterations?: number|null, objective_name?: string|null }} Preset
 * @typedef {{ min: number, max: number, was_clamped?: boolean }} ResolvedBounds
 * @typedef {{ name: string, include: boolean, min: number|null, max: number|null, step: number|null }} FormVar
 * @typedef {{ observable: string, type: string, value: number|null }} FormConstraint
 * @typedef {{
 *   base: OptimizationConfig,
 *   preset: string,
 *   objective: string,
 *   direction: string,
 *   algorithm: string,
 *   nIterations: number|null,
 *   nInitial: number|null,
 *   onComplete: string,
 *   constraints: FormConstraint[],
 *   algorithmParams: string,
 *   modelFile: string,
 *   vars: FormVar[],
 * }} FormState
 */

export const ON_COMPLETE_OPTIONS = Object.freeze(['apply_best', 'restore', 'hold']);
export const DEFAULT_ITERATIONS = 20;

/**
 * A fresh form for an environment, optionally seeded by a preset and the
 * service's resolved initial bounds (`GET .../initial-bounds`).
 *
 * @param {EnvironmentDetails} env
 * @param {{ name?: string, preset?: Preset|null, bounds?: Record<string, ResolvedBounds>|null }} [seed]
 * @returns {FormState}
 */
export function formFromEnv(env, { name = '', preset = null, bounds = null } = {}) {
  const chosen = preset?.variables ? new Set(preset.variables) : null;
  const vars = Object.entries(env.variables || {}).map(([varName, meta]) => {
    const b = bounds?.[varName];
    return {
      name: varName,
      include: chosen ? chosen.has(varName) : true,
      min: b ? b.min : meta.min_value,
      max: b ? b.max : meta.max_value,
      step: preset?.variable_overrides?.[varName]?.step_size ?? meta.step_size ?? null,
    };
  });
  return {
    base: {},
    preset: name,
    objective: preset?.objective_name || env.default_objective || '',
    direction: '',
    algorithm: preset?.algorithm || 'random',
    nIterations: preset?.n_iterations ?? DEFAULT_ITERATIONS,
    nInitial: null,
    onComplete: 'apply_best',
    constraints: [],
    algorithmParams: '',
    modelFile: '',
    vars,
  };
}

/**
 * The form for an existing (stored) draft, so the operator edits what the
 * agent or another panel wrote.
 *
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails} env
 * @returns {FormState}
 */
export function formFromConfig(config, env) {
  const form = formFromEnv(env);
  const tuned = config.variables ? new Set(config.variables) : null;
  const overrides = config.variable_overrides || {};
  for (const v of form.vars) {
    v.include = tuned ? tuned.has(v.name) : true;
    const o = overrides[v.name];
    if (o) {
      v.min = o.min ?? v.min;
      v.max = o.max ?? v.max;
      v.step = o.step_size ?? v.step;
    }
  }
  return {
    ...form,
    base: config,
    preset: config.preset || '',
    objective: config.objective_name || '',
    direction: config.objective_direction || '',
    algorithm: config.algorithm || 'random',
    nIterations: config.n_iterations ?? DEFAULT_ITERATIONS,
    nInitial: config.n_initial_samples ?? null,
    onComplete: config.on_complete || 'apply_best',
    constraints: Object.entries(config.constraints || {}).map(([observable, c]) => ({
      observable,
      type: c.type,
      value: c.value,
    })),
    algorithmParams: config.algorithm_params ? JSON.stringify(config.algorithm_params) : '',
    modelFile: config.model_file || '',
  };
}

/**
 * The `OptimizationConfig` body the form sends, or the reasons it cannot.
 *
 * @param {FormState} form
 * @param {{ environment: string, target: string }} where
 * @returns {{ config: OptimizationConfig|null, errors: string[] }}
 */
export function buildConfig(form, { environment, target }) {
  /** @type {string[]} */
  const errors = [];
  if (!environment) errors.push('Pick an environment.');
  if (!target) errors.push('Pick a control target.');
  if (!form.objective) errors.push('Pick an objective.');
  const selected = form.vars.filter((v) => v.include);
  if (!selected.length) errors.push('Select at least one variable.');
  /** @type {Record<string, { min: number, max: number, step_size: number }>} */
  const overrides = {};
  for (const v of selected) {
    if (v.min == null || v.max == null || v.step == null) {
      errors.push(`${v.name}: min, max and step are required.`);
    } else if (v.min > v.max) {
      errors.push(`${v.name}: min exceeds max.`);
    } else if (!(v.step > 0)) {
      errors.push(`${v.name}: step must be positive.`);
    } else {
      overrides[v.name] = { min: v.min, max: v.max, step_size: v.step };
    }
  }
  if (form.nIterations == null || form.nIterations < 0)
    errors.push('Iterations must be 0 or more.');
  if (form.nInitial != null && form.nInitial < 1) errors.push('Initial samples must be 1 or more.');
  /** @type {Record<string, { type: string, value: number }>} */
  const constraints = {};
  for (const c of form.constraints) {
    if (!c.observable || !c.type || c.value == null || !Number.isFinite(c.value)) {
      errors.push('Each constraint needs an observable, a relation and a threshold.');
    } else if (c.observable in constraints) {
      errors.push(`Only one constraint per observable (${c.observable}).`);
    } else {
      constraints[c.observable] = { type: c.type, value: c.value };
    }
  }
  /** @type {Record<string, unknown>|null} */
  let algorithmParams = null;
  if (form.algorithmParams.trim()) {
    try {
      const parsed = JSON.parse(form.algorithmParams);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('object');
      algorithmParams = parsed;
    } catch {
      errors.push('Algorithm params must be a JSON object.');
    }
  }
  if (errors.length) return { config: null, errors };
  return {
    config: {
      ...form.base,
      environment_name: environment,
      control_target: target,
      preset: form.preset || null,
      objective_name: form.objective,
      objective_direction: form.direction || null,
      algorithm: form.algorithm,
      n_iterations: /** @type {number} */ (form.nIterations),
      n_initial_samples: form.nInitial,
      on_complete: form.onComplete,
      variables: selected.map((v) => v.name),
      variable_overrides: overrides,
      constraints,
      algorithm_params: algorithmParams,
      model_file: form.modelFile || null,
    },
    errors: [],
  };
}

/**
 * Parse a number input: blank → null.
 *
 * @param {string} text
 * @returns {number|null}
 */
export function parseNumber(text) {
  if (text.trim() === '') return null;
  const n = Number(text);
  return Number.isFinite(n) ? n : null;
}
