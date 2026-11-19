/**
 * Tests for the panel's reads (sync.js), driven against a stubbed sidecar and
 * a real store: what one poll tick reads and in which order, what it does not
 * read again, the status it gives the poll for its rhythm, an active job
 * beside a newer draft, the review's own read of the draft's environment and
 * its readbacks while a draft waits, the runs list, the machine connection
 * and the catalogue. A failed read keeps what the store had and names the
 * error.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { flush } from './_views.mjs';
import { withQuery } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/api-client.js';
import {
  ACTIVE_MS,
  DRAFT_MS,
  FAILED_MS,
  IDLE_MS,
  pollDelay,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/poller.js';
import {
  createStore,
  deriveScreen,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';
import {
  createSync,
  readCatalog,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/sync.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/api-client.js').ApiResult} ApiResult */

/**
 * @param {any} data
 * @param {string|null} [revision]
 * @returns {ApiResult}
 */
const answer = (data, revision = null) => ({
  ok: true,
  status: 200,
  data,
  revision,
  error: null,
  issues: [],
});

/**
 * @param {number} status
 * @param {string} error  the reason the service gives
 * @returns {ApiResult}
 */
const refusal = (status, error) => ({
  ok: false,
  status,
  data: { detail: error },
  revision: null,
  error,
  issues: [error],
});

/** A promise and the function that resolves it. */
function deferred() {
  /** @type {() => void} */
  let open = () => {};
  const promise = new Promise((resolve) => (open = () => resolve(undefined)));
  return { promise: /** @type {Promise<void>} */ (promise), open };
}

/** The sidecar stub: a mutable world, every read it saw (`path?query`), and the api over it. */
function service() {
  const world = {
    /** The job `jobs/current` points at. */
    /** @type {string|null} */
    current: 'j1',
    /** @type {Record<string, { status: string, revision: string, config: any, data?: any[] }>} */
    jobs: { j1: { status: 'created', revision: 'rev-1', config: storedConfig() } },
    /** What `environments/{name}/variables` reads off the machine. */
    /** @type {Record<string, number>} */
    readings: { phase_a: 1 },
    /** What `environments/{name}/status` answers besides the name. */
    /** @type {{ connected?: unknown, message?: string|null }} */
    status: { connected: true },
    /** @type {any[]} */
    openJobs: [],
    /** @type {any[]} */
    results: [],
    /** @type {any[]} */
    environments: [{ name: 'injector' }],
    /** @type {any[]|null} */
    models: [],
    /** A read (`path?query`) answered with this refusal instead. */
    /** @type {Record<string, ApiResult>} */
    refuse: {},
    /** Paths whose answer waits until the test resolves the promise. */
    /** @type {Record<string, Promise<void>|undefined>} */
    held: {},
  };

  /**
   * @param {string} path
   * @param {Record<string, any>|undefined} query
   * @returns {ApiResult}
   */
  function route(path, query) {
    if (path === 'jobs/current') return answer({ job_id: world.current });
    if (path === 'jobs') return answer(world.openJobs);
    if (path === 'results') return answer(world.results);
    if (path === 'environments') return answer(world.environments);
    if (path === 'models') return answer(world.models);
    const job = path.match(/^jobs\/([^/]+)(\/config)?$/);
    if (job) {
      const stored = world.jobs[job[1]];
      if (!stored) return refusal(404, `no job ${job[1]}`);
      if (job[2]) return answer(stored.config, stored.revision);
      // Only `detail=full` carries the evaluations.
      const full = query?.detail === 'full' ? { data: stored.data ?? [] } : {};
      return answer({ job_id: job[1], status: stored.status, ...full });
    }
    const env = path.match(/^environments\/([^/]+)(?:\/(variables|status))?$/);
    if (env?.[2] === 'variables') {
      const readings = Object.entries(world.readings).map(([name, value]) => [
        name,
        { value, connected: true },
      ]);
      return answer(Object.fromEntries(readings));
    }
    if (env?.[2] === 'status') return answer({ name: env[1], ...world.status });
    if (env) return answer({ ...envDetails(), name: env[1] });
    return refusal(404, `no route ${path}`);
  }

  /** @type {string[]} */
  const reads = [];
  /**
   * @param {string} path
   * @param {Record<string, any>} [query]
   * @returns {Promise<ApiResult>}
   */
  async function get(path, query) {
    const read = withQuery(path, query);
    reads.push(read);
    if (world.held[path]) await world.held[path];
    return world.refuse[read] ?? route(path, query);
  }
  const api = /** @type {any} */ ({ get, jobs: () => get('jobs'), results: () => get('results') });
  return { world, reads, api };
}

/** The service still runs r9; the current job is the newer draft j1. */
function beside() {
  const svc = service();
  svc.world.jobs.r9 = {
    status: 'running',
    revision: 'rev-r9',
    config: storedConfig({ control_target: 'standin' }),
    data: [{ charge: 1, phase_a: 0, sol_b: 3 }],
  };
  svc.world.openJobs = [
    { job_id: 'j1', status: 'created' },
    { job_id: 'r9', status: 'running' },
  ];
  return svc;
}

/**
 * createSync over `svc` and a real store, on a clock the test sets.
 *
 * @param {ReturnType<typeof service>} [svc]
 */
function syncOver(svc = service()) {
  let t = 0;
  const store = createStore();
  const sync = createSync({ api: svc.api, store, now: () => t });
  return {
    ...svc,
    store,
    sync,
    state: () => store.state,
    /**
     * Tick at `ms`, and let the reads the tick started answer.
     *
     * @param {number} ms
     */
    async tickAt(ms) {
      t = ms;
      const seen = await sync.tick();
      await flush();
      return seen;
    },
  };
}

describe('one tick', () => {
  test("a waiting draft: the current job, its config, the draft's environment, the open jobs, the runs list and the connection", async () => {
    const { sync, reads, state } = syncOver();
    expect(await sync.tick()).toEqual({ status: 'created' });
    await flush();
    expect(reads).toEqual([
      'jobs/current',
      'jobs/j1',
      'jobs/j1/config',
      'environments/injector',
      'models',
      'environments/injector/variables?names=phase_a&names=sol_b&target=va',
      // The open jobs: once to look for a run beside the draft, once with the runs list.
      'jobs',
      'jobs',
      'results',
      'environments/injector/status?target=va',
    ]);
    expect(state().job).toEqual({ job_id: 'j1', status: 'created' });
    // Start sends the revision of the config response that drew the review.
    expect(state().approval).toMatchObject({ jobId: 'j1', revision: 'rev-1', needsReview: false });
    expect(state().approvalEnv).toMatchObject({
      key: 'injector|va|phase_a|sol_b',
      details: { name: 'injector' },
      readbacks: { phase_a: 1 },
      error: null,
    });
    expect(state().openJobs).toEqual({ list: [], error: null });
    expect(state().results).toEqual({ list: [], error: null });
    expect(state().connection).toEqual({
      environment: 'injector',
      target: 'va',
      connected: true,
      message: null,
    });
  });

  test('a started job is read with detail=full and its config once; the next tick reads only the job', async () => {
    const svc = service();
    Object.assign(svc.world.jobs.j1, {
      status: 'running',
      data: [{ charge: 1, phase_a: 0, sol_b: 3 }],
    });
    const { reads, state, tickAt } = syncOver(svc);
    expect(await tickAt(0)).toEqual({ status: 'running' });
    expect(reads.slice(0, 4)).toEqual([
      'jobs/current',
      'jobs/j1',
      'jobs/j1?detail=full',
      'jobs/j1/config',
    ]);
    expect(state().job.data).toHaveLength(1);
    expect(state().approval).toMatchObject({ jobId: 'j1', revision: 'rev-1' });
    reads.length = 0;
    await tickAt(1000);
    // No config again, no look for another run, no list or connection due.
    expect(reads).toEqual(['jobs/current', 'jobs/j1', 'jobs/j1?detail=full']);
  });

  test('a started job whose full read fails is shown from its plain read', async () => {
    const svc = service();
    svc.world.jobs.j1.status = 'running';
    svc.world.refuse['jobs/j1?detail=full'] = refusal(503, 'busy');
    const { state, tickAt } = syncOver(svc);
    expect(await tickAt(0)).toEqual({ status: 'running' });
    expect(state().job).toEqual({ job_id: 'j1', status: 'running' });
    expect(state().service).toEqual({ ok: true, error: null });
  });

  test("a waiting draft's config is read on every tick, and a new revision reaches the review state", async () => {
    const { world, reads, state, tickAt } = syncOver();
    await tickAt(0);
    world.jobs.j1.revision = 'rev-2';
    world.jobs.j1.config = storedConfig({ n_iterations: 77 });
    await tickAt(1000);
    expect(reads.filter((r) => r === 'jobs/j1/config')).toHaveLength(2);
    expect(state().approval).toMatchObject({ jobId: 'j1', revision: 'rev-2', needsReview: true });
    expect(state().approval.config?.n_iterations).toBe(77);
  });

  test("no current job, or a pointer at a draft that has expired, is none, and the draft's environment goes with it", async () => {
    const { world, reads, state, tickAt } = syncOver();
    await tickAt(0);
    world.current = 'gone';
    reads.length = 0;
    expect(await tickAt(1000)).toEqual({ status: null });
    expect(reads.slice(0, 2)).toEqual(['jobs/current', 'jobs/gone']);
    expect(state().job).toBeNull();
    expect(state().service).toEqual({ ok: true, error: null });
    expect(state().approvalEnv.key).toBe('');
    world.current = null;
    reads.length = 0;
    expect(await tickAt(2000)).toEqual({ status: null });
    expect(reads.filter((r) => r.startsWith('jobs/'))).toEqual(['jobs/current']);
  });

  test('a failed read of the current job names the error, keeps the job, and reads nothing more', async () => {
    const { world, reads, state, tickAt } = syncOver();
    await tickAt(0);
    world.refuse['jobs/current'] = refusal(503, 'optimizer service unreachable');
    reads.length = 0;
    const seen = await tickAt(1000);
    expect(seen).toEqual({ failed: true });
    expect(pollDelay(seen)).toBe(FAILED_MS);
    expect(reads).toEqual(['jobs/current']);
    expect(state().service).toEqual({ ok: false, error: 'optimizer service unreachable' });
    expect(state().job?.job_id).toBe('j1');

    delete world.refuse['jobs/current'];
    world.refuse['jobs/j1'] = refusal(500, 'internal error');
    reads.length = 0;
    expect(await tickAt(2000)).toEqual({ failed: true });
    expect(reads).toEqual(['jobs/current', 'jobs/j1']);
    expect(state().service).toEqual({ ok: false, error: 'internal error' });
    expect(state().job?.job_id).toBe('j1');
  });
});

describe('the rhythm of the poll', () => {
  test('the tick gives the status of the run beside a newer draft, else of the current job', async () => {
    expect(pollDelay(await syncOver().sync.tick())).toBe(DRAFT_MS);
    const run = await syncOver(beside()).sync.tick();
    expect(run).toEqual({ status: 'running' });
    expect(pollDelay(run)).toBe(ACTIVE_MS);
    const idle = service();
    idle.world.current = null;
    expect(pollDelay(await syncOver(idle).sync.tick())).toBe(IDLE_MS);
  });
});

describe('an active job beside a newer draft', () => {
  test('is read with detail=full on every tick and its config once; the connection follows it', async () => {
    const { reads, state, tickAt } = syncOver(beside());
    await tickAt(0);
    await tickAt(1000);
    expect(reads.filter((r) => r.startsWith('jobs/r9'))).toEqual([
      'jobs/r9?detail=full',
      'jobs/r9/config',
      'jobs/r9?detail=full',
    ]);
    expect(state().activeJob).toMatchObject({ job_id: 'r9', status: 'running' });
    expect(state().activeJob.data).toHaveLength(1);
    expect(state().activeConfig?.control_target).toBe('standin');
    expect(deriveScreen(state())).toMatchObject({ id: 'running', job: { job_id: 'r9' } });
    expect(reads).toContain('environments/injector/status?target=standin');
  });

  test('a config read that failed is read again on the next tick', async () => {
    const { world, state, tickAt } = syncOver(beside());
    world.refuse['jobs/r9/config'] = refusal(502, 'bad gateway');
    await tickAt(0);
    expect(state().activeJob).toMatchObject({ job_id: 'r9' });
    expect(state().activeConfig).toBeNull();
    delete world.refuse['jobs/r9/config'];
    await tickAt(1000);
    expect(state().activeConfig?.control_target).toBe('standin');
  });

  test('a run that has left the open jobs is read once more, so its end shows', async () => {
    const { world, reads, state, tickAt } = syncOver(beside());
    await tickAt(0);
    world.jobs.r9.status = 'completed';
    world.openJobs = [{ job_id: 'j1', status: 'created' }];
    expect(await tickAt(1000)).toEqual({ status: 'created' });
    expect(state().activeJob).toMatchObject({ job_id: 'r9', status: 'completed' });
    await tickAt(2000);
    expect(reads.filter((r) => r === 'jobs/r9?detail=full')).toHaveLength(2);
    expect(state().activeJob?.status).toBe('completed');
  });

  test('a run the service no longer holds is let go', async () => {
    const { world, state, tickAt } = syncOver(beside());
    await tickAt(0);
    delete world.jobs.r9;
    world.openJobs = [{ job_id: 'j1', status: 'created' }];
    await tickAt(1000);
    expect(state().activeJob).toBeNull();
    expect(state().activeConfig).toBeNull();
  });

  test('a run that is the current job again is no longer kept beside it', async () => {
    const { world, state, tickAt } = syncOver(beside());
    await tickAt(0);
    world.current = 'r9';
    expect(await tickAt(1000)).toEqual({ status: 'running' });
    expect(state().job?.job_id).toBe('r9');
    expect(state().activeJob).toBeNull();
  });
});

describe("the review's read of the draft's own environment", () => {
  test('is read once per environment, target and knob set; a failed read is read again on the next tick', async () => {
    const { world, reads, state, tickAt } = syncOver();
    world.refuse['environments/injector'] = refusal(503, 'timed out');
    await tickAt(0);
    expect(state().approvalEnv).toMatchObject({
      key: 'injector|va|phase_a|sol_b',
      details: null,
      error: 'timed out',
    });
    delete world.refuse['environments/injector'];
    await tickAt(1000);
    expect(state().approvalEnv).toMatchObject({ details: { name: 'injector' }, error: null });
    await tickAt(2000);
    expect(reads.filter((r) => r === 'environments/injector')).toHaveLength(2);
  });
});

describe("the review's readbacks while a draft waits", () => {
  const VARIABLES = 'environments/injector/variables';
  const readsOf = (/** @type {string[]} */ reads) =>
    reads.filter((r) => r.startsWith(VARIABLES)).length;

  test('are re-read on the poll, at most every 5 s', async () => {
    const { world, reads, state, tickAt } = syncOver();
    await tickAt(0);
    expect(state().approvalEnv.readbacks.phase_a).toBe(1);
    world.readings = { phase_a: 4.7, sol_b: 1.7 };
    const before = readsOf(reads);
    await tickAt(1000);
    expect(readsOf(reads)).toBe(before);
    expect(state().approvalEnv.readbacks.phase_a).toBe(1);
    await tickAt(6000);
    expect(state().approvalEnv.readbacks).toEqual({ phase_a: 4.7, sol_b: 1.7 });
  });

  test('a read still in flight holds the next one back', async () => {
    const { world, reads, state, tickAt } = syncOver();
    await tickAt(0);
    expect(state().approvalEnv.readbacks.phase_a).toBe(1);
    const held = deferred();
    world.held[VARIABLES] = held.promise;
    const before = readsOf(reads);
    await tickAt(6000);
    expect(readsOf(reads)).toBe(before + 1);
    // A slow machine read: later ticks skip rather than pile reads up.
    await tickAt(12000);
    await tickAt(18000);
    expect(readsOf(reads)).toBe(before + 1);
    world.readings = { phase_a: 5 };
    held.open();
    await flush();
    expect(state().approvalEnv.readbacks.phase_a).toBe(5);
  });

  test('nothing is re-read while a Start is pending', async () => {
    const { store, reads, state, tickAt } = syncOver();
    await tickAt(0);
    store.dispatch({ type: 'start/sent' });
    expect(state().starting).toBe(true);
    const before = readsOf(reads);
    await tickAt(6000);
    expect(readsOf(reads)).toBe(before);
    // Once the Start has answered, the read that was due is made.
    store.dispatch({ type: 'start/answered', error: 'refused' });
    expect(state().starting).toBe(false);
    await tickAt(6000);
    expect(readsOf(reads)).toBe(before + 1);
  });

  test("a late reply for the review's previous environment is dropped", async () => {
    const { world, reads, state, tickAt } = syncOver();
    await tickAt(0);
    expect(state().approvalEnv.readbacks.phase_a).toBe(1);
    const held = deferred();
    world.held[VARIABLES] = held.promise;
    const before = readsOf(reads);
    await tickAt(6000);
    expect(readsOf(reads)).toBe(before + 1);
    delete world.held[VARIABLES]; // only the refresh already sent is held
    // The draft moves to another variable set: a new environment key.
    world.jobs.j1.revision = 'rev-2';
    world.jobs.j1.config = storedConfig({ variables: ['phase_a'] });
    world.readings = { phase_a: 2 };
    await tickAt(6000);
    expect(state().approvalEnv.readbacks.phase_a).toBe(2);
    const key = state().approvalEnv.key;
    expect(key).toBe('injector|va|phase_a');
    // The old read answers last, with a reading the review must not show.
    world.readings = { phase_a: 99 };
    held.open();
    await flush();
    expect(state().approvalEnv.key).toBe(key);
    expect(state().approvalEnv.readbacks.phase_a).toBe(2);
  });

  test('a refresh leaves the review state, the revision and Start as they were', async () => {
    const { world, store, state, tickAt } = syncOver();
    await tickAt(0);
    world.jobs.j1.revision = 'rev-2';
    await tickAt(0);
    const gate = () => deriveScreen(state()).gate;
    expect(state().approval).toMatchObject({ revision: 'rev-2', needsReview: true });
    expect(gate()?.enabled).toBe(false);

    // Still unreviewed after a refresh: new readings are no reason to skip a review.
    world.readings = { phase_a: 2 };
    await tickAt(6000);
    expect(state().approvalEnv.readbacks.phase_a).toBe(2);
    expect(state().approval).toMatchObject({ revision: 'rev-2', needsReview: true });
    expect(gate()?.enabled).toBe(false);

    // Reviewed, then refreshed: it stays reviewed, and Start sends the same job and revision.
    store.dispatch({ type: 'approval/reviewed' });
    await tickAt(7000); // the poll that adopts the reviewed state; no read is due
    const shown = deriveScreen(state()).shown;
    expect(gate()?.enabled).toBe(true);
    world.readings = { phase_a: 3 };
    await tickAt(12000);
    expect(state().approvalEnv.readbacks.phase_a).toBe(3);
    expect(state().approval).toMatchObject({ revision: 'rev-2', needsReview: false });
    expect(deriveScreen(state()).shown).toEqual(shown);
    expect(shown).toEqual({ jobId: 'j1', revision: 'rev-2' });
    expect(gate()?.enabled).toBe(true);
  });
});

describe('the runs list', () => {
  test('the open jobs and the stored runs are read again when the status changes, and every 15 s', async () => {
    const { world, reads, tickAt } = syncOver();
    const listReads = () => reads.filter((r) => r === 'results').length;
    await tickAt(0);
    expect(listReads()).toBe(1);
    await tickAt(14999);
    expect(listReads()).toBe(1);
    await tickAt(15000);
    expect(listReads()).toBe(2);
    // A run that starts or ends changes what the lists hold.
    world.jobs.j1.status = 'running';
    await tickAt(16000);
    expect(listReads()).toBe(3);
  });

  test('a failed read keeps the last list and names the error; the next answer clears it', async () => {
    const { world, sync, state } = syncOver();
    world.refuse.results = refusal(503, 'results unavailable');
    await sync.refreshLists();
    expect(state().results).toEqual({ list: null, error: 'results unavailable' });
    expect(state().openJobs).toEqual({ list: [], error: null });

    delete world.refuse.results;
    const run = { run_id: 'injector/2026-11-16_21-15-04', job_id: 'old' };
    const open = { job_id: 'j1', status: 'created' };
    world.results = [run];
    world.openJobs = [open];
    await sync.refreshLists();
    expect(state().results).toEqual({ list: [run], error: null });

    world.refuse.results = refusal(503, 'results unavailable');
    world.refuse.jobs = refusal(502, 'bad gateway');
    world.results = [];
    world.openJobs = [];
    await sync.refreshLists();
    expect(state().results).toEqual({ list: [run], error: 'results unavailable' });
    expect(state().openJobs).toEqual({ list: [open], error: 'bad gateway' });
  });
});

describe('the machine connection', () => {
  test("is read for the screen's environment on its target, at most every 10 s", async () => {
    const { world, reads, state, tickAt } = syncOver();
    const statusReads = () => reads.filter((r) => r.includes('/status')).length;
    await tickAt(0);
    expect(state().connection).toEqual({
      environment: 'injector',
      target: 'va',
      connected: true,
      message: null,
    });
    world.status = { connected: false, message: 'IOC not answering' };
    await tickAt(9999);
    expect(statusReads()).toBe(1);
    expect(state().connection?.connected).toBe(true);
    await tickAt(10000);
    expect(statusReads()).toBe(2);
    expect(state().connection).toEqual({
      environment: 'injector',
      target: 'va',
      connected: false,
      message: 'IOC not answering',
    });
  });

  test('a screen that names no environment has none; an unclear or failed read leaves connected unknown', async () => {
    const { world, store, sync, reads, state, tickAt } = syncOver();
    await tickAt(0);
    expect(state().connection?.connected).toBe(true);
    store.dispatch({ type: 'setup/open', jobId: null, environment: '', target: '' });
    reads.length = 0;
    await sync.refreshConnection();
    expect(state().connection).toBeNull();
    expect(reads).toEqual([]);

    store.dispatch({ type: 'setup/close' });
    world.status = { message: 'warming up' };
    await sync.refreshConnection();
    expect(state().connection).toEqual({
      environment: 'injector',
      target: 'va',
      connected: null,
      message: 'warming up',
    });

    world.refuse['environments/injector/status?target=va'] = refusal(503, 'gateway down');
    await sync.refreshConnection();
    expect(state().connection).toEqual({
      environment: 'injector',
      target: 'va',
      connected: null,
      message: 'gateway down',
    });
  });
});

describe('the catalogue', () => {
  test('reads the environments and the model files, and gives each list its own event', async () => {
    const { world, reads, api } = service();
    world.environments = [{ name: 'injector' }, { name: 'linac' }];
    world.models = null; // an empty answer is an empty list
    const store = createStore();
    const dispatch = vi.spyOn(store, 'dispatch');
    await readCatalog(api, store);
    expect(reads).toEqual(['environments', 'models']);
    expect(dispatch.mock.calls.map(([event]) => event)).toEqual([
      { type: 'environments', list: [{ name: 'injector' }, { name: 'linac' }], error: null },
      { type: 'models', list: [], error: null },
    ]);
  });

  test('a list that fails keeps the last one read and names the error; the other list is still taken', async () => {
    const { world, sync, state } = syncOver();
    world.models = [{ name: 'gp.pt' }];
    await sync.loadCatalog();
    expect(state().models).toEqual({ list: [{ name: 'gp.pt' }], error: null });
    world.environments = [{ name: 'injector' }, { name: 'linac' }];
    world.refuse.models = refusal(503, 'model store down');
    await sync.loadCatalog();
    expect(state().models).toEqual({ list: [{ name: 'gp.pt' }], error: 'model store down' });
    expect(state().environments).toEqual({
      list: [{ name: 'injector' }, { name: 'linac' }],
      error: null,
    });
  });
});
