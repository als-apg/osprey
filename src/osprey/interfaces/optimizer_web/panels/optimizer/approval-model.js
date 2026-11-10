// @ts-check
/**
 * What the operator approves by clicking Start, as pure data.
 *
 * Everything here is computed from the STORED draft (`GET jobs/{id}/config`)
 * and the revision header of that same response — never from the form's local
 * state — because that stored draft is exactly what a start runs. The field review
 * of PR 2399 found these fields are bounded by no other layer, so the panel
 * is the one place a person sees them before the machine moves.
 *
 * The start gate is a small reducer: the revision Start sends is the one that
 * rendered the approval surface; a different revision, or a different draft
 * replacing one already on screen, re-renders the surface and holds Start
 * until the operator has looked again.
 *
 * @module approval-model
 */

/**
 * Algorithms whose xopt generator is a `BayesianGenerator` (the service's
 * `_resolve_n_initial` branches on that class; BAxUS subclasses it).
 */
export const BAYESIAN_ALGORITHMS = Object.freeze([
  'upper_confidence_bound',
  'expected_improvement',
  'baxus',
]);

/** Every `AlgorithmType` the service accepts, in its declaration order. */
export const ALGORITHMS = Object.freeze([
  'random',
  'latin_hypercube',
  'upper_confidence_bound',
  'expected_improvement',
  'baxus',
  'neldermead',
  'extremum_seeking',
  'rcds',
  'nsga2',
  'cnsga',
]);

/** `ObservableMetadata.direction`'s default when an observable sets none. */
export const DEFAULT_OBSERVABLE_DIRECTION = 'MAXIMIZE';

/**
 * @typedef {{ min?: number|null, max?: number|null, step_size?: number|null }} VariableOverride
 * @typedef {{ type: string, value: number }} ConstraintSpec
 * @typedef {{
 *   environment_name?: string|null,
 *   control_target?: string|null,
 *   execution_mode?: string|null,
 *   preset?: string|null,
 *   on_complete?: string,
 *   objective_name?: string|null,
 *   objective_direction?: string|null,
 *   algorithm?: string,
 *   n_iterations?: number,
 *   n_initial_samples?: number|null,
 *   variables?: string[]|null,
 *   variable_overrides?: Record<string, VariableOverride>|null,
 *   surrogate_model?: string,
 *   model_file?: string|null,
 *   algorithm_params?: Record<string, unknown>|null,
 *   constraints?: Record<string, ConstraintSpec>,
 *   environment_config?: Record<string, unknown>|null,
 * }} OptimizationConfig
 * @typedef {{
 *   min_value: number, max_value: number, units?: string, description?: string,
 *   step_size?: number, max_step?: number|null,
 * }} VariableMetadata
 * @typedef {{ kind?: string, display_name?: string, units?: string, direction?: string }} ObservableMetadata
 * @typedef {{
 *   name: string,
 *   variables: Record<string, VariableMetadata>,
 *   observables: Record<string, ObservableMetadata>,
 *   default_objective?: string,
 *   available_objectives?: string[],
 * }} EnvironmentDetails
 * @typedef {{ label: string, path: string, variables?: string[]|null }} ModelFileInfo
 */

/**
 * The initial-sample count a start will use, mirroring the service's
 * `_resolve_n_initial`: an explicit value (at least 1) wins, except that a
 * Bayesian run on a pre-trained model defaults to 1; a null count on a
 * Bayesian run is max(2, n_vars + 1); every other algorithm defaults to 1.
 *
 * @param {OptimizationConfig} config
 * @returns {{ count: number|null, derived: boolean, rule: string }}
 */
export function initialSamples(config) {
  const explicit = config.n_initial_samples;
  const bayesian = BAYESIAN_ALGORITHMS.includes(config.algorithm || '');
  if (bayesian && config.model_file) {
    if (explicit != null) return { count: explicit, derived: false, rule: 'set in the draft' };
    return { count: 1, derived: true, rule: 'pre-trained model: 1' };
  }
  if (explicit != null) {
    return { count: Math.max(1, explicit), derived: false, rule: 'set in the draft' };
  }
  if (bayesian) {
    const nVars = config.variables?.length;
    if (nVars == null) return { count: null, derived: true, rule: 'max(2, variables + 1)' };
    return { count: Math.max(2, nVars + 1), derived: true, rule: `max(2, ${nVars} + 1)` };
  }
  return { count: 1, derived: true, rule: 'the current machine state' };
}

/**
 * The objective's direction, and whether the draft overrides the observable's
 * own default (the case the operator must notice: it inverts "better").
 *
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @returns {{ effective: string|null, observableDefault: string|null, mismatch: boolean, overridden: boolean }}
 */
export function objectiveDirection(config, env) {
  const name = config.objective_name;
  const observable = name && env ? env.observables?.[name] : undefined;
  const observableDefault = observable
    ? observable.direction || DEFAULT_OBSERVABLE_DIRECTION
    : null;
  const explicit = config.objective_direction || null;
  return {
    effective: explicit || observableDefault,
    observableDefault,
    overridden: explicit !== null,
    mismatch: explicit !== null && observableDefault !== null && explicit !== observableDefault,
  };
}

/**
 * The channels a pre-trained surrogate holds fixed: its trained variables that
 * the draft does not tune (the service fixes them at their current values).
 *
 * @param {OptimizationConfig} config
 * @param {ModelFileInfo[]|null} models
 * @returns {{ known: boolean, channels: string[] }}
 */
export function fixedFeatureChannels(config, models) {
  if (!config.model_file) return { known: true, channels: [] };
  const model = (models || []).find((m) => m.path === config.model_file);
  if (!model || !Array.isArray(model.variables)) return { known: false, channels: [] };
  const tuned = new Set(config.variables || []);
  return { known: true, channels: model.variables.filter((v) => !tuned.has(v)) };
}

/**
 * The search window's width as a percentage of the variable's safety range.
 *
 * @param {number|null|undefined} min
 * @param {number|null|undefined} max
 * @param {number|null|undefined} safetyMin
 * @param {number|null|undefined} safetyMax
 * @returns {number|null}
 */
export function rangePercent(min, max, safetyMin, safetyMax) {
  if (min == null || max == null || safetyMin == null || safetyMax == null) return null;
  const span = safetyMax - safetyMin;
  if (!(span > 0) || !Number.isFinite(span)) return null;
  return ((max - min) / span) * 100;
}

/**
 * One row per tuned variable of the stored draft, with its resolved bounds and
 * step, the environment's per-write limit and the window's share of the
 * safety range.
 *
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 */
export function variableRows(config, env) {
  const overrides = config.variable_overrides || {};
  return (config.variables || []).map((name) => {
    const o = overrides[name] || {};
    const meta = env?.variables?.[name];
    return {
      name,
      min: o.min ?? null,
      max: o.max ?? null,
      step: o.step_size ?? null,
      maxStep: meta?.max_step ?? null,
      units: meta?.units || '',
      rangePct: rangePercent(o.min, o.max, meta?.min_value, meta?.max_value),
      unresolved: o.min == null || o.max == null || o.step_size == null,
    };
  });
}

/**
 * What the approval card's environment read depends on: the stored draft's
 * environment, target and tuned variables. Empty when there is nothing to read.
 *
 * @param {OptimizationConfig|null} config
 * @returns {string}
 */
export function approvalEnvKey(config) {
  if (!config?.environment_name) return '';
  const { environment_name: name, control_target: target, variables } = config;
  return [name, target || '', ...(variables || [])].join('|');
}

/* ------------------------------------------------------------------ */
/* Start gate                                                          */
/* ------------------------------------------------------------------ */

/**
 * @typedef {{
 *   jobId: string|null,
 *   revision: string|null,
 *   config: OptimizationConfig|null,
 *   needsReview: boolean,
 *   notice: string|null,
 * }} ApprovalState
 *
 * @typedef {{ type: 'draft', jobId: string, revision: string|null, config: OptimizationConfig }
 *   | { type: 'reviewed' }
 *   | { type: 'conflict', detail?: string }
 *   | { type: 'clear' }} ApprovalEvent
 */

/** @returns {ApprovalState} */
export function approvalInit() {
  return { jobId: null, revision: null, config: null, needsReview: false, notice: null };
}

export const CHANGED_NOTICE =
  'The draft changed since it was shown. Review it again before starting.';
export const SWITCHED_NOTICE =
  'A different draft replaced the one shown. Review it before starting.';
export const CONFLICT_NOTICE =
  'The draft changed before Start reached the service, so nothing started. ' +
  'It is shown again below; review it before starting.';

/**
 * @param {ApprovalState} state
 * @param {ApprovalEvent} event
 * @returns {ApprovalState}
 */
export function reduceApproval(state, event) {
  switch (event.type) {
    case 'draft':
      if (event.jobId !== state.jobId) {
        // A different draft. Its first rendering is the look only when no
        // other draft was on screen: an operator part-way through reading one
        // must not start another they never saw.
        const replacing = state.config !== null;
        return {
          jobId: event.jobId,
          revision: event.revision,
          config: event.config,
          needsReview: replacing,
          notice: replacing ? SWITCHED_NOTICE : null,
        };
      }
      if (event.revision === state.revision) return state;
      return {
        ...state,
        revision: event.revision,
        config: event.config,
        needsReview: true,
        notice: state.notice === CONFLICT_NOTICE ? CONFLICT_NOTICE : CHANGED_NOTICE,
      };
    case 'reviewed':
      return { ...state, needsReview: false, notice: null };
    case 'conflict':
      return { ...state, needsReview: true, notice: CONFLICT_NOTICE };
    case 'clear':
      return approvalInit();
    default:
      return state;
  }
}

/**
 * Whether Start may be clicked now, and why not.
 *
 * `envLoaded` says the draft's own environment has been read, so its
 * direction and per-write limits on the surface are verified, not blank.
 *
 * @param {ApprovalState} state
 * @param {{ job_id: string, status: string }|null} job
 * @param {{ starting?: boolean, envLoaded?: boolean }} [flags]
 * @returns {{ enabled: boolean, reason: string|null }}
 */
export function startGate(state, job, { starting = false, envLoaded = false } = {}) {
  if (!job) return { enabled: false, reason: 'No draft to start.' };
  if (job.status !== 'created') return { enabled: false, reason: `The job is ${job.status}.` };
  if (state.jobId !== job.job_id || !state.config) {
    return { enabled: false, reason: 'The draft is still loading.' };
  }
  if (!state.revision) {
    return { enabled: false, reason: 'The service sent no draft revision; Start cannot name one.' };
  }
  if (state.needsReview) return { enabled: false, reason: 'Review the changed draft first.' };
  if (!envLoaded) {
    return { enabled: false, reason: "Direction unverified: the draft's environment is loading." };
  }
  if (starting) return { enabled: false, reason: 'Starting…' };
  return { enabled: true, reason: null };
}

/**
 * Whether a 409 from start means "the draft changed" (as opposed to another
 * conflict: a job already running, a non-CREATED job). Matches the service's
 * `draft_changed_409` sentence.
 *
 * @param {number} status
 * @param {string|null} detail
 * @returns {boolean}
 */
export function isDraftChangedConflict(status, detail) {
  return status === 409 && /draft changed/i.test(detail || '');
}

/* ------------------------------------------------------------------ */
/* Run controls                                                        */
/* ------------------------------------------------------------------ */

/**
 * Which run controls a job status allows: pause only from running, resume only
 * from paused (the service's CAS rule), cancel from any non-terminal status
 * other than one already cancelling.
 *
 * @param {string|null|undefined} status
 * @returns {{ pause: boolean, resume: boolean, cancel: boolean }}
 */
export function runControls(status) {
  const live = ['created', 'idle', 'submitted', 'running', 'paused'];
  return {
    pause: status === 'running',
    resume: status === 'paused',
    cancel: !!status && live.includes(status),
  };
}

/**
 * The write-authorization line for a started run, from the job's
 * `write_authorization` record. `awaiting` is the case the operator must act
 * on: the chip in the web terminal.
 *
 * @param {Record<string, any>|null|undefined} auth
 * @returns {{ tone: 'warn'|'ok'|'info'|'err', text: string }|null}
 */
export function authorizationNote(auth) {
  if (!auth || typeof auth !== 'object') return null;
  switch (auth.state) {
    case 'requesting':
      return { tone: 'info', text: 'Requesting write authorization…' };
    case 'awaiting':
      return {
        tone: 'warn',
        text:
          'Waiting for write approval: approve the request in the web terminal' +
          (auth.id ? ` (request ${auth.id}).` : '.'),
      };
    case 'granted':
      return { tone: 'ok', text: 'Writes authorized.' };
    case 'ended':
      return {
        tone: 'err',
        text: `Write authorization ended${auth.reason ? `: ${auth.reason}` : ''}.`,
      };
    case 'released':
      return { tone: 'info', text: 'Write authorization released.' };
    default:
      return null;
  }
}
