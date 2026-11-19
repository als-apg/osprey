// @ts-check
/**
 * The OPTIMIZER panel's one state object, its pure transitions, and the screen
 * the panel shows for it.
 *
 * `reduce(state, event)` is the only way state changes: the poll's reads, the
 * operator's choices and the answers to the writes all arrive as events. It
 * never mutates `state` and never reads a clock: an event that needs the time
 * carries it. `deriveScreen(state)` names the screen (one of {@link SCREENS})
 * and gathers what that screen shows, so a view renders from the screen and
 * the state and holds no rule of its own.
 *
 * The rules about what Start may start live in approval-model.js and
 * approval-env.js; this module routes events to them and never restates them.
 * The approval state follows the current job: a new revision of the draft on
 * screen, or a different draft in its place, must be acknowledged again
 * before Start is enabled (reduceApproval), whoever changed it.
 *
 * "Now" is the active run while a job is active (submitted, running, paused,
 * cancelling), also when the service's current job points at a newer draft:
 * a run is always in reach with its controls. With no active job, "now"
 * follows the current job: a draft the agent made, or a run that started
 * elsewhere, is adopted by a panel that shows nothing else. A panel showing
 * a set-up or a past run keeps it, and the runs list names the new job;
 * picking a job there shows that job's own screen (an active job: its run
 * page and controls) while the set-up keeps its edits. A set-up that edits a
 * draft which has since started is closed: there is nothing left to edit.
 *
 * @module store
 */

import {
  approvalInit,
  authorizationNote,
  emptyApprovalEnv,
  reduceApproval,
  runControls,
} from './approval-model.js';
import { approvalGateFor } from './approval-env.js';
import { jobFigures, noArrivals, recordArrivals, resultFigures } from './run-model.js';
import { isActive } from './poller.js';
import { endingOf } from './words.js';

/** Every screen, in the order of a run's life. */
export const SCREENS = Object.freeze(
  /** @type {const} */ ([
    'empty',
    'setup',
    'setup-issues',
    'review',
    'review-changed',
    'review-no-owner',
    'waiting-approval',
    'running',
    'paused',
    'completed',
    'failed',
    'cancelled',
    'history',
    'result',
  ])
);

/** Screens of a run under way: no primary action, Pause and Cancel in the run header. */
export const RUN_SCREENS = Object.freeze(['waiting-approval', 'running', 'paused']);

/** The write-authorization states in which no write has been authorized yet. */
const PENDING_AUTHORIZATION = Object.freeze(['requesting', 'awaiting']);

/**
 * @typedef {typeof SCREENS[number]} ScreenId
 * @typedef {import('./approval-model.js').ApprovalState} ApprovalState
 * @typedef {import('./approval-model.js').ApprovalEnv} ApprovalEnv
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./approval-model.js').StartGate} StartGate
 * @typedef {import('./run-model.js').Arrivals} Arrivals
 * @typedef {import('./run-model.js').RunFigures} RunFigures
 *
 * What the operator looks at: the present job (or its set-up), the table of
 * past runs, one stored run, or a job of `GET jobs` other than the current.
 * @typedef {{ kind: 'present' }
 *   | { kind: 'history' }
 *   | { kind: 'run', runId: string }
 *   | { kind: 'job', jobId: string }} Selection
 *
 * A list the service sends: null until the first answer.
 * @typedef {{ list: any[]|null, error: string|null }} Listing
 * @typedef {{ details: any, job: any, loading: boolean, error: string|null }} PastRun
 * @typedef {{ details: EnvironmentDetails|null, presets: Record<string, any>|null,
 *   loading: boolean, error: string|null }} EnvEntry
 * @typedef {{ values: Record<string, number|null>, error: string|null, at: number }} Reading
 * The set-up: `jobId` names the draft it edits (null: a new run); `from` is
 * the stored config a new run starts from ("run again"), null for none. `id`
 * tells one set-up from the next, so a late save answer for an earlier one
 * never lands on it.
 * `revision` is the stored draft's revision the set-up of a draft was opened
 * with (null for a new run), so a change by someone else can be noticed.
 * @typedef {{ id: number, jobId: string|null, environment: string, target: string,
 *   issues: string[], saving: boolean, from: OptimizationConfig|null,
 *   revision: string|null }} Setup
 * The message of the last save; `moved` names the knobs whose search window
 * the service moved on the way in (draft-model.js `movedKnobs`).
 * @typedef {{ jobId: string, tone: string, text: string, moved: string[] }} SaveNote
 * @typedef {{ connected: boolean|null, message: string|null,
 *   environment: string, target: string }} Connection
 *
 * @typedef {{
 *   service: { ok: boolean|null, error: string|null },
 *   connection: Connection|null,
 *   job: any,
 *   activeJob: any,
 *   activeConfig: OptimizationConfig|null,
 *   arrivals: Arrivals,
 *   approval: ApprovalState,
 *   approvalEnv: ApprovalEnv,
 *   owner: string|null,
 *   ownerKnown: boolean|null,
 *   starting: boolean,
 *   startError: string|null,
 *   control: { pending: string|null, error: string|null, cancelArmed: boolean },
 *   environments: Listing,
 *   models: Listing,
 *   envs: Record<string, EnvEntry>,
 *   readings: Record<string, Reading>,
 *   openJobs: Listing,
 *   results: Listing,
 *   past: Record<string, PastRun>,
 *   selection: Selection,
 *   setup: Setup|null,
 *   saveNote: SaveNote|null,
 *   filter: string,
 * }} State
 *
 * @typedef {{ type: 'poll/failed', error: string|null }
 *   | { type: 'poll/job', job: any, at: number }
 *   | { type: 'poll/config', jobId: string, revision: string|null, config: OptimizationConfig }
 *   | { type: 'service', ok: boolean, error: string|null }
 *   | { type: 'connection', connection: Connection|null }
 *   | { type: 'session', owner: string|null, ownerKnown: boolean|null }
 *   | { type: 'approval/reviewed' }
 *   | { type: 'approvalEnv/reset', key: string }
 *   | { type: 'approvalEnv/loaded', env: ApprovalEnv }
 *   | { type: 'approvalEnv/readbacks', key: string, readbacks: Record<string, number|null> }
 *   | { type: 'environments' | 'models' | 'jobs' | 'results', list: any[]|null, error: string|null }
 *   | { type: 'env/loading', name: string }
 *   | { type: 'env/loaded', name: string, details: EnvironmentDetails|null,
 *       presets: Record<string, any>|null, error: string|null }
 *   | { type: 'readings', key: string, values: Record<string, number|null>,
 *       error: string|null, at: number }
 *   | { type: 'past/loading', runId: string }
 *   | { type: 'past/loaded', runId: string, details: any, job: any, error: string|null }
 *   | { type: 'select', selection: Selection }
 *   | { type: 'filter', text: string }
 *   | { type: 'setup/open', jobId: string|null, environment: string, target: string,
 *       from?: OptimizationConfig|null, id?: number, revision?: string|null }
 *   | { type: 'poll/active', job: any, config: OptimizationConfig|null, at: number }
 *   | { type: 'setup/where', environment: string, target: string }
 *   | { type: 'setup/saving', setupId?: number }
 *   | { type: 'setup/refused', issues: string[], setupId?: number }
 *   | { type: 'setup/saved', note: SaveNote, setupId?: number }
 *   | { type: 'setup/close' }
 *   | { type: 'start/sent' }
 *   | { type: 'start/answered', job?: any, conflict?: boolean, error?: string|null }
 *   | { type: 'control/arm', armed: boolean }
 *   | { type: 'control/sent', action: string }
 *   | { type: 'control/answered', job?: any, error?: string|null }} StoreEvent
 */

/** @returns {Listing} */
const unread = () => ({ list: null, error: null });

/** @returns {State} */
export function initialState() {
  return {
    service: { ok: null, error: null },
    connection: null,
    job: null,
    activeJob: null,
    activeConfig: null,
    arrivals: noArrivals(),
    approval: approvalInit(),
    approvalEnv: emptyApprovalEnv(''),
    owner: null,
    ownerKnown: null,
    starting: false,
    startError: null,
    control: { pending: null, error: null, cancelArmed: false },
    environments: unread(),
    models: unread(),
    envs: {},
    readings: {},
    openJobs: unread(),
    results: unread(),
    past: {},
    selection: { kind: 'present' },
    setup: null,
    saveNote: null,
    filter: '',
  };
}

/**
 * The key of a reading in `state.readings`: what was read (`variables` or
 * `observables`) of which environment on which target.
 *
 * @param {'variables'|'observables'} kind
 * @param {string} environment
 * @param {string} target
 * @returns {string}
 */
export function readingKey(kind, environment, target) {
  return `${kind}:${environment}@${target}`;
}

/**
 * The job whose worker holds the machine (submitted, running, paused or
 * cancelling): the current job when it is active, else the active job the
 * service lists beside a newer draft; null when no job is active.
 *
 * @param {State} state
 * @returns {any}
 */
export function runJob(state) {
  if (isActive(state.job?.status)) return state.job;
  return isActive(state.activeJob?.status) ? state.activeJob : null;
}

/**
 * The job the run controls act on: the active run, else the current job.
 *
 * @param {State} state
 * @returns {string|null}
 */
function controlledId(state) {
  return (runJob(state) || state.job)?.job_id ?? null;
}

/**
 * The state after the poll read the current job (`null`: there is none).
 *
 * @param {State} state
 * @param {any} job
 * @param {number} at
 * @returns {State}
 */
function withJob(state, job, at) {
  const id = job?.job_id ?? null;
  const next = { ...state, service: { ok: true, error: null }, job };
  // The run that was the current job stays "now" when a newer draft takes
  // the pointer, until the poll reads the run beside it (sync.followActive).
  const was = isActive(state.job?.status) ? state.job : null;
  if (was && was.job_id !== id && !isActive(job?.status)) {
    next.activeJob = was;
    next.activeConfig =
      state.approval.jobId === was.job_id ? state.approval.config : state.activeConfig;
  }
  // The evaluations are timed for the run: here, unless another job is the run.
  const other = isActive(next.activeJob?.status) && next.activeJob.job_id !== id;
  if (!other) {
    const count = job?.data?.length ?? 0;
    next.arrivals = recordArrivals(state.arrivals, id, count, at, job?.status === 'running');
  }
  if (!job) next.approval = reduceApproval(state.approval, { type: 'clear' });
  if (id !== (state.job?.job_id ?? null)) next.startError = null;
  if (controlledId(next) !== controlledId(state)) {
    next.control = { pending: null, error: null, cancelArmed: false };
  }
  // A set-up of a draft that has started is closed: there is nothing left to
  // edit. One whose draft is no longer the current job stays open.
  const setup = state.setup;
  if (setup?.jobId && setup.jobId === id && job.status !== 'created') next.setup = null;
  return next;
}

/**
 * The state after the poll read an active job that is not the current job
 * (the current job is a newer draft), with its stored config; `null`: there
 * is none. A job that has ended since is kept with its last status.
 *
 * @param {State} state
 * @param {any} job
 * @param {OptimizationConfig|null} config
 * @param {number} at
 * @returns {State}
 */
function withActive(state, job, config, at) {
  const next = { ...state, activeJob: job, activeConfig: job ? config : null };
  if (isActive(job?.status)) {
    const count = job.data?.length ?? 0;
    next.arrivals = recordArrivals(state.arrivals, job.job_id, count, at, job.status === 'running');
  }
  if (controlledId(next) !== controlledId(state)) {
    next.control = { pending: null, error: null, cancelArmed: false };
  }
  return next;
}

/**
 * Whether a save event is for the set-up now open (an event without an id is).
 *
 * @param {State} state
 * @param {number|undefined} setupId
 */
function ownSetup(state, setupId) {
  return !!state.setup && (setupId === undefined || setupId === state.setup.id);
}

/**
 * The job after a control answered: the answer is a plain status read, so it
 * updates the status fields of the job on screen and keeps its run record.
 *
 * @param {any} job  the job on screen
 * @param {any} answer
 * @returns {any}
 */
function answered(job, answer) {
  if (!answer || !job || answer.job_id !== job.job_id) return job;
  const { status, error_message, write_authorization } = answer;
  return { ...job, status, error_message, write_authorization };
}

/**
 * @param {State} state
 * @param {StoreEvent} event
 * @returns {State}
 */
export function reduce(state, event) {
  switch (event.type) {
    case 'poll/failed':
      return { ...state, service: { ok: false, error: event.error } };
    case 'poll/job':
      return withJob(state, event.job, event.at);
    case 'poll/active':
      return withActive(state, event.job, event.config, event.at);
    case 'poll/config':
      return {
        ...state,
        approval: reduceApproval(state.approval, {
          type: 'draft',
          jobId: event.jobId,
          revision: event.revision,
          config: event.config,
        }),
      };
    case 'service':
      return { ...state, service: { ok: event.ok, error: event.error } };
    case 'connection':
      return { ...state, connection: event.connection };
    case 'session':
      return { ...state, owner: event.owner, ownerKnown: event.ownerKnown };
    case 'approval/reviewed':
      return { ...state, approval: reduceApproval(state.approval, { type: 'reviewed' }) };
    case 'approvalEnv/reset':
      return { ...state, approvalEnv: emptyApprovalEnv(event.key) };
    case 'approvalEnv/loaded':
      // A read for an environment the draft no longer names is dropped.
      if (event.env.key !== state.approvalEnv.key) return state;
      return { ...state, approvalEnv: event.env };
    case 'approvalEnv/readbacks':
      if (event.key !== state.approvalEnv.key || !state.approvalEnv.details) return state;
      return { ...state, approvalEnv: { ...state.approvalEnv, readbacks: event.readbacks } };
    case 'environments':
      return { ...state, environments: { list: event.list, error: event.error } };
    case 'models':
      return { ...state, models: { list: event.list, error: event.error } };
    case 'results':
      return { ...state, results: { list: event.list, error: event.error } };
    case 'jobs':
      return { ...state, openJobs: { list: event.list, error: event.error } };
    case 'env/loading': {
      const old = state.envs[event.name];
      const entry = {
        details: old?.details ?? null,
        presets: old?.presets ?? null,
        loading: true,
        error: null,
      };
      return { ...state, envs: { ...state.envs, [event.name]: entry } };
    }
    case 'env/loaded': {
      const { details, presets, error } = event;
      const entry = { details, presets, error, loading: false };
      return { ...state, envs: { ...state.envs, [event.name]: entry } };
    }
    case 'readings': {
      // One read may name some channels only: the others keep their last value.
      const values = { ...state.readings[event.key]?.values, ...event.values };
      const reading = { values, error: event.error, at: event.at };
      return { ...state, readings: { ...state.readings, [event.key]: reading } };
    }
    case 'past/loading': {
      const old = state.past[event.runId];
      const entry = {
        details: old?.details ?? null,
        job: old?.job ?? null,
        loading: true,
        error: null,
      };
      return { ...state, past: { ...state.past, [event.runId]: entry } };
    }
    case 'past/loaded': {
      const { details, job, error } = event;
      const entry = { details, job, error, loading: false };
      return { ...state, past: { ...state.past, [event.runId]: entry } };
    }
    case 'select':
      return {
        ...state,
        selection: event.selection,
        control: { ...state.control, cancelArmed: false },
      };
    case 'filter':
      return { ...state, filter: event.text };
    case 'setup/open': {
      const { jobId, environment, target, from = null, id = 0, revision = null } = event;
      const setup = { id, jobId, environment, target, issues: [], saving: false, from, revision };
      return { ...state, setup, selection: { kind: 'present' } };
    }
    case 'setup/where':
      if (!state.setup) return state;
      return {
        ...state,
        setup: { ...state.setup, environment: event.environment, target: event.target },
      };
    case 'setup/saving':
      if (!state.setup || !ownSetup(state, event.setupId)) return state;
      return { ...state, setup: { ...state.setup, saving: true } };
    case 'setup/refused':
      if (!state.setup || !ownSetup(state, event.setupId)) return state;
      return { ...state, setup: { ...state.setup, saving: false, issues: event.issues } };
    case 'setup/saved':
      // A late answer for an earlier set-up: the draft exists, nothing else moves.
      if (!ownSetup(state, event.setupId)) return { ...state, saveNote: event.note };
      return { ...state, setup: null, saveNote: event.note, selection: { kind: 'present' } };
    case 'setup/close':
      return { ...state, setup: null, selection: { kind: 'present' } };
    case 'start/sent':
      return { ...state, starting: true, startError: null };
    case 'start/answered': {
      const next = { ...state, starting: false, startError: event.error ?? null };
      if (event.job) next.job = event.job;
      if (event.conflict) next.approval = reduceApproval(state.approval, { type: 'conflict' });
      return next;
    }
    case 'control/arm':
      return { ...state, control: { ...state.control, cancelArmed: event.armed } };
    case 'control/sent':
      return { ...state, control: { pending: event.action, error: null, cancelArmed: false } };
    case 'control/answered':
      return {
        ...state,
        job: answered(state.job, event.job),
        activeJob: answered(state.activeJob, event.job),
        control: { pending: null, error: event.error ?? null, cancelArmed: false },
      };
    default:
      return state;
  }
}

/**
 * The screen a job's status calls for. A draft (`created`) is the review of
 * the current job's approval state.
 *
 * @param {State} state
 * @param {any} job
 * @returns {ScreenId}
 */
function statusScreen(state, job) {
  const { approval } = state;
  const status = job.status;
  if (status === 'created') {
    if (approval.needsReview) return 'review-changed';
    const readonly = approval.config?.execution_mode === 'readonly';
    if (state.ownerKnown === false && !readonly) return 'review-no-owner';
    return 'review';
  }
  if (status === 'running' || status === 'submitted') {
    const auth = job.write_authorization;
    if (PENDING_AUTHORIZATION.includes(auth?.state)) return 'waiting-approval';
    // Started, but no write approval asked yet and nothing evaluated: the
    // worker is still starting, and nothing has been written.
    const readonly = configOf(state, job)?.execution_mode === 'readonly';
    if (!auth && !job.data?.length && !readonly) return 'waiting-approval';
    return 'running';
  }
  if (status === 'paused') return 'paused';
  // An ended job shows its outcome (words.js `ENDINGS`).
  return endingOf(status)?.kind ?? 'running';
}

/**
 * The stored config of a job, when the panel has read it.
 *
 * @param {State} state
 * @param {any} job
 * @returns {OptimizationConfig|null}
 */
function configOf(state, job) {
  if (job.job_id === state.activeJob?.job_id) return state.activeConfig;
  return state.approval.jobId === job.job_id ? state.approval.config : null;
}

/**
 * The screen of the service's current job alone ('empty' when there is none).
 *
 * @param {State} state
 * @returns {ScreenId}
 */
export function currentScreen(state) {
  return state.job ? statusScreen(state, state.job) : 'empty';
}

/**
 * The screen of "now", as if nothing else were selected: the active run when
 * there is one, also when the current job is a newer draft; else the current
 * job.
 *
 * @param {State} state
 * @returns {ScreenId}
 */
export function presentScreen(state) {
  const run = runJob(state);
  return run ? statusScreen(state, run) : currentScreen(state);
}

/**
 * @typedef {{
 *   id: ScreenId,
 *   loading: boolean,
 *   job: any,
 *   config: OptimizationConfig|null,
 *   subject: { environment: string|null, target: string|null },
 *   gate: StartGate|null,
 *   shown: { jobId: string|null, revision: string|null }|null,
 *   controls: { pause: boolean, resume: boolean, cancel: boolean },
 *   cancelling: boolean,
 *   authorization: { tone: string, text: string }|null,
 *   figures: RunFigures|null,
 *   past: { runId: string|null, info: any, entry: PastRun|null, job: any }|null,
 * }} Screen
 */

const NO_CONTROLS = Object.freeze({ pause: false, resume: false, cancel: false });

/**
 * @param {OptimizationConfig|null|undefined} config
 * @param {string|null} [environment]
 */
function subjectOf(config, environment = null) {
  return {
    environment: config?.environment_name || environment || null,
    target: config?.control_target || null,
  };
}

/**
 * The screen for a stored run or another job of `GET jobs`.
 *
 * @param {State} state
 * @param {Selection} selection
 * @returns {Screen}
 */
function pastScreen(state, selection) {
  const runId = selection.kind === 'run' ? selection.runId : null;
  const info = runId ? (state.results.list || []).find((r) => r.run_id === runId) || null : null;
  const entry = runId ? state.past[runId] || null : null;
  const job =
    selection.kind === 'job'
      ? [...(state.openJobs.list || []), state.activeJob].find(
          (j) => j && j.job_id === selection.jobId
        ) || null
      : entry?.job || null;
  const config = entry?.details?.config || null;
  return {
    ...blankScreen(state, 'result'),
    job,
    config,
    subject: subjectOf(config, info?.environment),
    figures: entry?.details ? resultFigures(entry.details) : null,
    past: { runId, info, entry, job },
  };
}

/**
 * @param {State} state
 * @param {ScreenId} id
 * @returns {Screen}
 */
function blankScreen(state, id) {
  return {
    id,
    loading: state.service.ok === null,
    job: null,
    config: null,
    subject: { environment: null, target: null },
    gate: null,
    shown: null,
    controls: NO_CONTROLS,
    cancelling: false,
    authorization: null,
    figures: null,
    past: null,
  };
}

/**
 * The screen of one job: the review of the current draft, the run page of an
 * active job, the outcome of an ended current job. The run controls are bound
 * to this job's own status.
 *
 * @param {State} state
 * @param {any} job
 * @returns {Screen}
 */
function jobScreen(state, job) {
  const { approval } = state;
  const id = statusScreen(state, job);
  const config = configOf(state, job);
  const screen = {
    ...blankScreen(state, id),
    job,
    config,
    subject: subjectOf(config, job.environment_name),
  };
  if (id.startsWith('review')) {
    return {
      ...screen,
      gate: approvalGateFor(state),
      shown: { jobId: approval.jobId, revision: approval.revision },
    };
  }
  const busy = state.control.pending !== null || state.starting;
  return {
    ...screen,
    controls: busy ? NO_CONTROLS : runControls(job.status),
    cancelling: job.status === 'cancelling',
    authorization: authorizationNote(job.write_authorization),
    figures: jobFigures(job, config, state.arrivals),
  };
}

/**
 * The screen to show for `state`, with what it shows.
 *
 * While a job is active, "now" is its run, also when the service's current
 * job is a newer draft; that draft shows when it is picked in the runs list.
 * A job picked in the runs list shows its own screen: an active job its run
 * page with its controls, the current job its review or outcome, any other
 * job its status and settings (`result`).
 *
 * @param {State} state
 * @returns {Screen}
 */
export function deriveScreen(state) {
  const { selection, setup, job, approval } = state;
  const run = runJob(state);
  if (selection.kind === 'history') return blankScreen(state, 'history');
  if (selection.kind === 'run') return pastScreen(state, selection);
  if (selection.kind === 'job') {
    if (run && selection.jobId === run.job_id) return jobScreen(state, run);
    if (job && selection.jobId === job.job_id) return jobScreen(state, job);
    return pastScreen(state, selection);
  }
  if (setup) {
    return {
      ...blankScreen(state, setup.issues.length ? 'setup-issues' : 'setup'),
      job: setup.jobId && job?.job_id === setup.jobId ? job : null,
      config: setup.jobId && approval.jobId === setup.jobId ? approval.config : null,
      subject: { environment: setup.environment || null, target: setup.target || null },
    };
  }
  if (run) return jobScreen(state, run);
  return job ? jobScreen(state, job) : blankScreen(state, 'empty');
}

/**
 * @typedef {{
 *   readonly state: State,
 *   dispatch: (event: StoreEvent) => void,
 *   subscribe: (listener: (state: State) => void) => () => void,
 * }} Store
 */

/**
 * The store: the state, a dispatch that runs {@link reduce}, and listeners
 * called after each event that changed the state.
 *
 * @param {State} [initial]
 * @returns {Store}
 */
export function createStore(initial = initialState()) {
  let state = initial;
  /** @type {Set<(state: State) => void>} */
  const listeners = new Set();
  return {
    get state() {
      return state;
    },
    dispatch(event) {
      const next = reduce(state, event);
      if (next === state) return;
      state = next;
      for (const listener of [...listeners]) listener(state);
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
