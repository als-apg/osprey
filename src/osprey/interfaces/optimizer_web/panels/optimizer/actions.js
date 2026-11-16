// @ts-check
/**
 * What the operator can do, and the reads a view asks for. A view never calls
 * the sidecar: it calls one of these, and the answer arrives in the store.
 *
 * Every write goes to `relay/...`, where the sidecar adds the owner and, for
 * Start and Resume, the launch token. A refusal is stored with the service's
 * own sentence; no write is retried.
 *
 * - Start starts the draft a view showed, and only that one: it takes the job
 *   and revision the view rendered (`screen.shown`), sends nothing when the
 *   store has moved on since, and asks the Start gate of approval-env.js. The
 *   revision is the one that rendered the view, never a fresh read.
 * - Cancel has two steps: `requestCancel` arms it (the view shows the
 *   confirmation: the machine keeps the last point written, nothing is
 *   restored), and only `confirmCancel` on an armed cancel sends it.
 * - Saving a draft writes nothing to the machine. The answer is re-read from
 *   the service, which may clamp or snap the bounds; draft-model.js names
 *   every bound it moved.
 *
 * @module actions
 */

import { readbackValues, saveDraft as save, seg } from './api-client.js';
import { TERMINAL_STATUSES, isDraftChangedConflict, runControls } from './approval-model.js';
import { approvalGateFor } from './approval-env.js';
import { movedKnobs, savedMessage } from './draft-model.js';
import { readingKey, runJob } from './store.js';
import { readCatalog } from './sync.js';

/** Every field of the service's `OptimizationConfig` (it refuses any other). */
const CONFIG_FIELDS = Object.freeze([
  'environment_name',
  'environment_config',
  'preset',
  'control_target',
  'execution_mode',
  'on_complete',
  'objective_name',
  'objective_direction',
  'algorithm',
  'n_iterations',
  'n_initial_samples',
  'variables',
  'variable_overrides',
  'surrogate_model',
  'model_file',
  'algorithm_params',
  'constraints',
]);

/**
 * A stored config as a body the service accepts: its `OptimizationConfig`
 * fields only. A stored run's config also carries `snapshots`, which a new
 * draft must not send.
 *
 * @param {Record<string, any>} config
 * @returns {import('./approval-model.js').OptimizationConfig}
 */
function configBody(config) {
  return Object.fromEntries(CONFIG_FIELDS.filter((k) => k in config).map((k) => [k, config[k]]));
}

/**
 * @typedef {import('./store.js').Selection} Selection
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 */

/**
 * @param {{
 *   api: import('./api-client.js').Api,
 *   store: import('./store.js').Store,
 *   refresh: () => Promise<void>,
 *   now?: () => number,
 * }} options  `refresh` runs the poll at once (after a write)
 */
export function createActions({ api, store, refresh, now = Date.now }) {
  const { dispatch } = store;
  /** The last set-up id handed out: each set-up gets its own. */
  let setups = 0;
  /** The catalogue read in flight, so a second request waits for it. */
  /** @type {Promise<void>|null} */
  let catalog = null;

  /** Read the environments and the model files again; one read at a time. */
  function loadCatalog() {
    catalog ||= readCatalog(api, store).finally(() => (catalog = null));
    return catalog;
  }

  /**
   * Open a set-up. When the last catalogue read failed, it is read again: the
   * set-up needs both lists.
   *
   * @param {Extract<import('./store.js').StoreEvent, { type: 'setup/open' }>} event
   */
  function openSetup(event) {
    dispatch(event);
    const { environments, models } = store.state;
    if (environments.error || models.error) void loadCatalog();
  }

  /**
   * Send a run control the job's status allows, one at a time. It acts on the
   * active run (else the current job); a view passes the id of the job it
   * shows, and a control for another job sends nothing.
   *
   * @param {'pause'|'resume'|'cancel'} action
   * @param {string} [jobId]  the job the view showed
   * @returns {Promise<boolean>}
   */
  async function control(action, jobId) {
    const { control: held } = store.state;
    const job = runJob(store.state) || store.state.job;
    if (!job || held.pending || !runControls(job.status)[action]) return false;
    if (jobId !== undefined && jobId !== job.job_id) return false;
    dispatch({ type: 'control/sent', action });
    const res = await api.post(`relay/jobs/${seg(job.job_id)}/${action}`);
    dispatch({
      type: 'control/answered',
      job: res.ok ? res.data : null,
      error: res.ok ? null : res.error,
    });
    await refresh();
    return res.ok;
  }

  /**
   * Load the stored run `runId` and, while the service still holds it, the
   * job that made it (its status).
   *
   * @param {string} runId
   */
  async function loadResult(runId) {
    const held = store.state.past[runId];
    // Read again while the run's job has not ended (or could not be read).
    const ended = TERMINAL_STATUSES.includes(held?.job?.status);
    if (held && (held.loading || (held.details && ended))) return;
    dispatch({ type: 'past/loading', runId });
    const info = (store.state.results.list || []).find((r) => r.run_id === runId);
    const [details, job] = await Promise.all([
      api.result(runId),
      info?.job_id ? api.get(`jobs/${seg(info.job_id)}`) : null,
    ]);
    dispatch({
      type: 'past/loaded',
      runId,
      details: details.ok ? details.data : null,
      job: job?.ok ? job.data : null,
      error: details.ok ? null : details.error,
    });
  }

  return {
    /**
     * Look at the present job, the past runs, a stored run or another job.
     * A stored run is loaded when it is first selected.
     *
     * @param {Selection} selection
     */
    select(selection) {
      dispatch({ type: 'select', selection });
      if (selection.kind === 'run') void loadResult(selection.runId);
    },
    /** @param {string} text  the runs list's filter */
    setFilter(text) {
      dispatch({ type: 'filter', text });
    },
    /** Open an empty set-up. It names no target: the operator picks one. */
    newRun() {
      openSetup({ type: 'setup/open', jobId: null, environment: '', target: '', id: ++setups });
    },
    /**
     * Open a new set-up from a stored config ("run again"): its environment
     * and target, with the config in `state.setup.from` to seed the form.
     * Nothing is written. A null config does nothing.
     *
     * @param {OptimizationConfig|null} config
     */
    runAgain(config) {
      if (!config) return;
      openSetup({
        type: 'setup/open',
        jobId: null,
        environment: config.environment_name || '',
        target: config.control_target || '',
        from: configBody(config),
        id: ++setups,
      });
    },
    /** Open the set-up of the current draft. */
    editDraft() {
      const { job, approval } = store.state;
      if (job?.status !== 'created' || approval.jobId !== job.job_id) return;
      openSetup({
        type: 'setup/open',
        jobId: job.job_id,
        environment: approval.config?.environment_name || '',
        target: approval.config?.control_target || '',
        from: approval.config,
        revision: approval.revision,
        id: ++setups,
      });
    },
    closeSetup() {
      dispatch({ type: 'setup/close' });
    },
    /**
     * The environment and target the set-up names, so the toolbar names them.
     *
     * @param {string} environment
     * @param {string} target
     */
    setSetupWhere(environment, target) {
      dispatch({ type: 'setup/where', environment, target });
    },
    /**
     * Hold the set-up with the reasons it cannot be saved (checked in the panel).
     *
     * @param {string[]} issues
     */
    refuseSetup(issues) {
      dispatch({ type: 'setup/refused', issues, setupId: store.state.setup?.id });
    },
    /**
     * Save `config` as a new draft, or over the draft the set-up edits.
     *
     * @param {OptimizationConfig} config
     * @param {'create'|'update'} kind
     * @returns {Promise<boolean>}
     */
    async saveDraft(config, kind) {
      const setup = store.state.setup;
      // One save at a time: a second call while one is in flight sends nothing.
      if (!setup || setup.saving) return false;
      const setupId = setup.id;
      // Captured before the write: a poll during it may move the current job.
      const jobId = kind === 'update' ? setup.jobId : null;
      if (kind === 'update' && !jobId) {
        const issues = ['There is no stored draft to update.'];
        dispatch({ type: 'setup/refused', issues, setupId });
        return false;
      }
      dispatch({ type: 'setup/saving', setupId });
      const { error, issues, id, stored, revision } = await save(api, jobId, config);
      if (error || !id) {
        // Each reason on its own, so each shows at the field it names.
        const reasons = issues?.length ? issues : [error || 'the draft was refused'];
        dispatch({ type: 'setup/refused', issues: reasons, setupId });
        return false;
      }
      // The revision of the read-back is the draft's from now on: a Start
      // before the next poll names it, not the one before the save.
      if (stored && revision) {
        dispatch({ type: 'poll/config', jobId: id, revision, config: stored });
      }
      const note = savedMessage(kind, config, stored);
      const moved = stored ? movedKnobs(config, stored) : [];
      dispatch({ type: 'setup/saved', note: { jobId: id, ...note, moved }, setupId });
      await refresh();
      return true;
    },
    /** The operator has looked at the changed draft again. */
    markReviewed() {
      dispatch({ type: 'approval/reviewed' });
    },
    /**
     * Start the draft the view showed, and only that one.
     *
     * @param {{ jobId: string|null, revision: string|null }} shown
     * @returns {Promise<boolean>}  whether the service started it
     */
    async start(shown) {
      const { approval } = store.state;
      if (!shown.jobId || shown.jobId !== approval.jobId) return false;
      if (!shown.revision || shown.revision !== approval.revision) return false;
      if (!approvalGateFor(store.state).enabled) return false;
      dispatch({ type: 'start/sent' });
      const res = await api.post(`relay/jobs/${seg(shown.jobId)}/start`, undefined, {
        config_revision: shown.revision,
      });
      const conflict = !res.ok && isDraftChangedConflict(res.status, res.error);
      dispatch({
        type: 'start/answered',
        job: res.ok ? res.data : null,
        conflict,
        error: res.ok || conflict ? null : res.error,
      });
      await refresh();
      return res.ok;
    },
    /** @param {string} [jobId]  the job the view showed */
    pause: (jobId) => control('pause', jobId),
    /** @param {string} [jobId]  the job the view showed */
    resume: (jobId) => control('resume', jobId),
    /** Arm Cancel: the view shows the confirmation. */
    requestCancel() {
      dispatch({ type: 'control/arm', armed: true });
    },
    dismissCancel() {
      dispatch({ type: 'control/arm', armed: false });
    },
    /**
     * Send the armed cancel; an unarmed one sends nothing.
     *
     * @param {string} [jobId]  the job the view showed
     */
    async confirmCancel(jobId) {
      if (!store.state.control.cancelArmed) return false;
      return control('cancel', jobId);
    },
    /**
     * Load an environment's details and presets into `state.envs`. An entry
     * that was read is kept; one that failed is read again when asked again.
     *
     * @param {string} name
     */
    async loadEnvironment(name) {
      const held = store.state.envs[name];
      if (!name || (held && (held.loading || held.details))) return;
      dispatch({ type: 'env/loading', name });
      const [details, presets] = await Promise.all([
        api.get(`environments/${seg(name)}`),
        api.get(`environments/${seg(name)}/presets`),
      ]);
      dispatch({
        type: 'env/loaded',
        name,
        details: details.ok ? details.data : null,
        presets: presets.ok ? presets.data || {} : {},
        error: details.ok ? null : details.error || 'environment unavailable',
      });
    },
    /**
     * Read what `names` read now on `target` into `state.readings`.
     *
     * @param {'variables'|'observables'} kind
     * @param {string} environment
     * @param {string} target
     * @param {string[]} names
     */
    async read(kind, environment, target, names) {
      if (!environment || !target || !names.length) return;
      const res =
        kind === 'observables'
          ? await api.observables(environment, names, target)
          : await api.get(`environments/${seg(environment)}/variables`, { names, target });
      dispatch({
        type: 'readings',
        key: readingKey(kind, environment, target),
        values: res.ok ? readbackValues(res.data) : {},
        error: res.ok ? null : res.error,
        at: now(),
      });
    },
    /**
     * The service's initial search windows for a set-up (one answer, not stored).
     *
     * @param {string} environment
     * @param {string} target
     * @param {string} preset
     * @returns {Promise<{ bounds: Record<string, any>|null, error: string|null }>}
     */
    async initialBounds(environment, target, preset) {
      const res = await api.get(`environments/${seg(environment)}/initial-bounds`, {
        target,
        preset: preset || null,
      });
      return { bounds: res.ok ? res.data?.bounds || null : null, error: res.ok ? null : res.error };
    },
    loadResult,
    loadCatalog,
    /** Read the service now (the poll's next tick, at once). */
    refresh,
  };
}

/** @typedef {ReturnType<typeof createActions>} Actions */
