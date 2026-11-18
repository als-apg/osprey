// @ts-check
/**
 * The panel's reads, turned into store events: the poll tick, and the slower
 * reads beside it (the runs list, the machine connection, the catalogue).
 * Nothing here writes.
 *
 * The tick follows `jobs/current`, so a draft the agent creates over MCP (the
 * service points the current job at every new draft) reaches the panel by
 * itself. While the current job is a CREATED draft the tick re-reads its
 * stored config, and the `X-Config-Revision` of THAT response is the revision
 * Start sends (approval-model.js holds the rule). A started job is read with
 * `detail=full`, which carries its evaluations. While a draft waits, the
 * approval readbacks are re-read in the rhythm approval-env.js sets.
 *
 * `GET jobs` and `GET results` are read every {@link LIST_MS} and whenever the
 * current job's status changes (a run that ends adds a stored run).
 *
 * @module sync
 */

import { seg } from './api-client.js';
import { approvalEnvKey } from './approval-model.js';
import { fetchApprovalEnv, readApprovalReadbacks } from './approval-env.js';
import { isActive } from './poller.js';
import { deriveScreen, runJob } from './store.js';

/** How often the runs list is re-read while nothing changes. */
const LIST_MS = 15000;
/** How often the machine connection of the screen's environment is re-read. */
const CONNECTION_MS = 10000;

/**
 * Read the catalogue: the environments and the model files. A list that
 * fails keeps the last one read (null before the first) and names the error.
 * The panel reads it at start-up; actions.js reads it again on request.
 *
 * @param {import('./api-client.js').Api} api
 * @param {import('./store.js').Store} store
 */
export async function readCatalog(api, store) {
  const [environments, models] = await Promise.all([api.get('environments'), api.get('models')]);
  const held = store.state;
  store.dispatch({
    type: 'environments',
    list: environments.ok ? environments.data || [] : held.environments.list,
    error: environments.ok ? null : environments.error,
  });
  store.dispatch({
    type: 'models',
    list: models.ok ? models.data || [] : held.models.list,
    error: models.ok ? null : models.error,
  });
}

/**
 * @param {{
 *   api: import('./api-client.js').Api,
 *   store: import('./store.js').Store,
 *   now?: () => number,
 * }} options
 */
export function createSync({ api, store, now = Date.now }) {
  const { dispatch } = store;
  // The readback rhythm approval-env.js keeps: it writes the two times and
  // reads the rest live from the store, so a reply for an environment the
  // card no longer shows is dropped.
  const readback = {
    approvalReadAt: 0,
    approvalReading: false,
    get job() {
      return store.state.job;
    },
    get approval() {
      return store.state.approval;
    },
    get approvalEnv() {
      return store.state.approvalEnv;
    },
    get starting() {
      return store.state.starting;
    },
  };
  let listsAt = -Infinity;
  let connectionAt = -Infinity;
  /** @type {string|null|undefined} */
  let lastStatus;
  /** The stored config of the active job beside a newer draft, read once per job. */
  /** @type {{ id: string, config: any }|null} */
  let activeConfig = null;

  /**
   * While the current job is not active, look for a job that is: the service
   * points the current job at every new draft, also while a run goes on. That
   * run is read with `detail=full` on every tick, so its page and its
   * controls stay in reach. A run that has left the list is read once more,
   * so its end shows.
   */
  async function followActive() {
    const current = store.state.job;
    if (isActive(current?.status)) {
      if (store.state.activeJob)
        dispatch({ type: 'poll/active', job: null, config: null, at: now() });
      return;
    }
    const listed = await api.jobs();
    if (listed.ok) dispatch({ type: 'jobs', list: listed.data || [], error: null });
    const held = store.state.activeJob;
    const found = (store.state.openJobs.list || []).find(
      (j) => isActive(j.status) && j.job_id !== current?.job_id
    );
    const id = found?.job_id || (isActive(held?.status) ? held.job_id : null);
    if (!id) return;
    const full = await api.get(`jobs/${seg(id)}`, { detail: 'full' });
    if (full.status === 404) {
      dispatch({ type: 'poll/active', job: null, config: null, at: now() });
      return;
    }
    if (!full.ok) return;
    // A failed config read is not kept: the next tick reads it again.
    if (activeConfig?.id !== id) {
      const cfg = await api.get(`jobs/${seg(id)}/config`);
      activeConfig = cfg.ok ? { id, config: cfg.data } : null;
    }
    const kept = store.state.activeJob?.job_id === id ? store.state.activeConfig : null;
    const config = activeConfig?.config ?? kept;
    dispatch({ type: 'poll/active', job: full.data, config, at: now() });
  }

  /**
   * Read the stored draft's own environment (details, models, readbacks on
   * its own target) once per draft environment. A failed read is retried on
   * the next tick; until one lands the direction reads unverified and Start
   * is held (approvalGateFor).
   */
  async function loadApprovalEnv() {
    const config = store.state.approval.config;
    const key = approvalEnvKey(config);
    const held = store.state.approvalEnv;
    if (key === held.key && !held.error) return;
    dispatch({ type: 'approvalEnv/reset', key });
    if (!config || !key) return;
    readback.approvalReadAt = now();
    dispatch({ type: 'approvalEnv/loaded', env: await fetchApprovalEnv(api, config, key) });
  }

  async function refreshApprovalReadbacks() {
    const readbacks = await readApprovalReadbacks(api, readback, now);
    if (!readbacks) return; // not due, failed, or the draft moved on
    dispatch({ type: 'approvalEnv/readbacks', key: store.state.approvalEnv.key, readbacks });
  }

  /** `GET jobs` and `GET results`; a failed read keeps the last list and names the error. */
  async function refreshLists() {
    listsAt = now();
    const [jobs, results] = await Promise.all([api.jobs(), api.results()]);
    const { openJobs, results: stored } = store.state;
    dispatch({
      type: 'jobs',
      list: jobs.ok ? jobs.data || [] : openJobs.list,
      error: jobs.ok ? null : jobs.error,
    });
    dispatch({
      type: 'results',
      list: results.ok ? results.data || [] : stored.list,
      error: results.ok ? null : results.error,
    });
  }

  /** Whether the environment the screen names is connected on its target. */
  async function refreshConnection() {
    connectionAt = now();
    const { environment, target } = deriveScreen(store.state).subject;
    if (!environment || !target) return dispatch({ type: 'connection', connection: null });
    const res = await api.get(`environments/${seg(environment)}/status`, { target });
    const data = res.ok ? res.data || {} : {};
    dispatch({
      type: 'connection',
      connection: {
        environment,
        target,
        connected: typeof data.connected === 'boolean' ? data.connected : null,
        message: res.ok ? data.message || null : res.error,
      },
    });
  }

  /** The environments and the model files, read at start-up ({@link readCatalog}). */
  const loadCatalog = () => readCatalog(api, store);

  /** @returns {Promise<import('./poller.js').TickResult>} */
  async function tick() {
    const cur = await api.get('jobs/current');
    if (!cur.ok) {
      dispatch({ type: 'poll/failed', error: cur.error });
      return { failed: true };
    }
    const id = cur.data?.job_id || null;
    // A pointer at a draft that expired (404) reads as no current job.
    const basic = id ? await api.get(`jobs/${seg(id)}`) : null;
    if (!id || !basic || basic.status === 404) {
      dispatch({ type: 'poll/job', job: null, at: now() });
    } else if (!basic.ok) {
      dispatch({ type: 'poll/failed', error: basic.error });
      return { failed: true };
    } else {
      let job = basic.data;
      if (job.status !== 'created') {
        const full = await api.get(`jobs/${seg(id)}`, { detail: 'full' });
        if (full.ok) job = full.data;
      }
      dispatch({ type: 'poll/job', job, at: now() });
      if (job.status === 'created' || store.state.approval.jobId !== id) {
        const cfg = await api.get(`jobs/${seg(id)}/config`);
        if (cfg.ok) {
          dispatch({ type: 'poll/config', jobId: id, revision: cfg.revision, config: cfg.data });
          void loadApprovalEnv();
        }
      }
    }
    await followActive();
    if (!store.state.approval.config) void loadApprovalEnv();
    void refreshApprovalReadbacks();
    // The cadence follows the run when there is one, else the current job.
    const status = runJob(store.state)?.status ?? store.state.job?.status ?? null;
    if (status !== lastStatus || now() - listsAt >= LIST_MS) void refreshLists();
    if (now() - connectionAt >= CONNECTION_MS) void refreshConnection();
    lastStatus = status;
    return { status };
  }

  return { tick, refreshLists, refreshConnection, loadCatalog };
}

/** @typedef {ReturnType<typeof createSync>} Sync */
