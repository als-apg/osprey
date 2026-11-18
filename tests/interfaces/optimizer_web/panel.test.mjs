/**
 * Integration tests for the OPTIMIZER panel (panel.js), its wiring: the real
 * index.html markup, the real views and a stubbed sidecar. The panel follows
 * `jobs/current` to an agent's draft, Start sends the revision of the config
 * response that drew the review (never one fetched at click time), a changed
 * draft must be acknowledged again, a session with no account keeps Start
 * enabled, a refusal is shown and not sent again, Pause, Resume and the
 * two-step Cancel reach the relay for the job shown, a readback refresh
 * redraws the review and keeps its Start, a run started elsewhere is adopted,
 * an active run stays "now" beside a newer draft, and every write goes
 * through `relay/`. What the reads fetch, and when, is tested in sync.test.mjs.
 */

import { readFileSync } from 'node:fs';
import { cwd } from 'node:process';

import { describe, test, expect, beforeEach, vi } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { createApi } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/api-client.js';
import { createPanel } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/panel.js';
import { STOP_WORDS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

const INDEX = `${cwd()}/src/osprey/interfaces/optimizer_web/panels/optimizer/index.html`;

/**
 * @typedef {{ status: string, revision: string, config: any, start_notice?: string|null,
 *   write_authorization?: any, data?: any[] }} StubJob
 */

/** The sidecar stub: a mutable world, and every request it saw. */
function sidecar() {
  const world = {
    current: /** @type {string|null} */ ('j1'),
    /** @type {Record<string, StubJob>} */
    jobs: { j1: { status: 'created', revision: 'rev-1', config: storedConfig() } },
    /** @type {{ status: number, body: any }|null} */
    startReply: null,
    /** @type {{ status: number, body: any }|null} */
    controlReply: null,
    /** Paths whose answer waits until the test resolves the promise. */
    /** @type {Record<string, Promise<void>|undefined>} */
    held: {},
    /** What `relay/session` answers; unset, it is not routed (the read fails). */
    /** @type {{ owner: string|null }|undefined} */
    session: undefined,
    /** The service's clamp on a created draft's bounds: its ceiling for every max. */
    clampMax: Infinity,
    /** The last draft body the panel created. */
    /** @type {any} */
    created: null,
    /** What `environments/injector/variables` reads off the machine. */
    /** @type {Record<string, { value: number, connected: boolean }>} */
    readings: { phase_a: { value: 1, connected: true } },
    /** @type {any[]} */
    openJobs: [],
    /** @type {any[]} */
    results: [],
    /** @type {Record<string, any>} */
    stored: {},
  };
  /** @type {{ method: string, url: string }[]} */
  const calls = [];
  /** @type {Record<string, () => any>} */
  const routes = {
    'GET /environments': () => [
      { name: 'injector', display_name: 'Injector', valid: true },
      { name: 'linac', display_name: 'Linac', valid: true },
    ],
    'GET /environments/injector': () => envDetails(),
    'GET /environments/linac': () => ({ ...envDetails(), name: 'linac' }),
    'GET /environments/injector/presets': () => ({}),
    'GET /environments/linac/presets': () => ({}),
    'GET /environments/injector/status': () => ({ name: 'injector', connected: true }),
    'GET /environments/linac/status': () => ({ name: 'linac', connected: true }),
    'GET /environments/injector/variables': () => world.readings,
    'GET /environments/linac/variables': () => world.readings,
    'GET /environments/injector/observables': () => ({ charge: { value: 2, connected: true } }),
    'GET /environments/linac/observables': () => ({ charge: { value: 2, connected: true } }),
    'GET /environments/injector/initial-bounds': () => ({ bounds: {}, current_values: {} }),
    'GET /models': () => [],
    'GET /jobs/current': () => ({ job_id: world.current }),
    'GET /jobs': () => world.openJobs,
    'GET /results': () => world.results,
  };
  /** @param {unknown} body @param {number} [status] @param {Record<string, string>} [headers] */
  const reply = (body, status = 200, headers = {}) =>
    new Response(JSON.stringify(body), { status, headers });
  /** @type {typeof fetch} */
  const fetchImpl = async (input, init) => {
    const url = String(input);
    const method = init?.method || 'GET';
    calls.push({ method, url });
    const path = url.split('?')[0];
    if (world.held[path]) await world.held[path];
    const startMatch = method === 'POST' && path.match(/^\/relay\/jobs\/([^/]+)\/start$/);
    if (startMatch) {
      const answer = world.startReply || {
        status: 200,
        body: { job_id: startMatch[1], status: 'submitted' },
      };
      return reply(answer.body, answer.status);
    }
    const controlMatch =
      method === 'POST' && path.match(/^\/relay\/jobs\/([^/]+)\/(pause|resume|cancel)$/);
    if (controlMatch) {
      const next = { pause: 'paused', resume: 'running', cancel: 'cancelling' }[controlMatch[2]];
      const answer = world.controlReply || {
        status: 200,
        body: { job_id: controlMatch[1], status: next },
      };
      return reply(answer.body, answer.status);
    }
    if (method === 'GET' && path === '/relay/session' && world.session) return reply(world.session);
    if (method === 'POST' && path === '/relay/optimizations') {
      const sent = JSON.parse(String(init?.body));
      world.created = sent;
      const overrides = Object.fromEntries(
        Object.entries(sent.variable_overrides).map(([name, o]) => [
          name,
          { .../** @type {any} */ (o), max: Math.min(/** @type {any} */ (o).max, world.clampMax) },
        ])
      );
      world.jobs.j2 = {
        status: 'created',
        revision: 'rev-j2',
        config: { ...sent, variable_overrides: overrides },
      };
      return reply({ job_id: 'j2', status: 'created' });
    }
    const putMatch = method === 'PUT' && path.match(/^\/relay\/jobs\/([^/]+)\/config$/);
    if (putMatch && world.jobs[putMatch[1]]) {
      world.jobs[putMatch[1]].config = JSON.parse(String(init?.body));
      return reply({ job_id: putMatch[1], status: 'created' });
    }
    const resultMatch = method === 'GET' && path.match(/^\/results\/([^/]+)\/([^/]+)$/);
    if (resultMatch) {
      const id = `${decodeURIComponent(resultMatch[1])}/${decodeURIComponent(resultMatch[2])}`;
      const details = world.stored[id];
      return details ? reply(details) : reply({ detail: 'not found' }, 404);
    }
    const jobMatch = method === 'GET' && path.match(/^\/jobs\/([^/]+)(\/config)?$/);
    if (jobMatch && jobMatch[1] !== 'current') {
      const job = world.jobs[jobMatch[1]];
      if (!job) return reply({ detail: 'no job' }, 404);
      if (jobMatch[2]) return reply(job.config, 200, { 'X-Config-Revision': job.revision });
      return reply({
        job_id: jobMatch[1],
        status: job.status,
        job_type: 'optimization',
        start_notice: job.start_notice ?? null,
        write_authorization: job.write_authorization ?? null,
        // A started job has read one evaluation, unless the test says otherwise.
        data: job.data ?? (job.status === 'created' ? null : [{ charge: 1, phase_a: 0, sol_b: 3 }]),
      });
    }
    const route = routes[`${method} ${path}`];
    if (!route) return reply({ detail: 'no route' }, 404);
    return reply(route());
  };
  return { world, calls, fetchImpl };
}

/** A promise and the function that resolves it. */
function gate() {
  /** @type {() => void} */
  let open = () => {};
  const promise = new Promise((resolve) => (open = () => resolve(undefined)));
  return { promise, open };
}

/** Let the requests already sent answer. */
async function settle() {
  for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

/** @param {{ method: string, url: string }[]} calls @param {string} jobId */
const startPosts = (calls, jobId) =>
  calls.filter((c) => c.method === 'POST' && c.url.startsWith(`/relay/jobs/${jobId}/start`));

/** @param {{ method: string, url: string }[]} calls */
const posts = (calls) => calls.filter((c) => c.method === 'POST').map((c) => c.url);

const $ = (/** @type {string} */ sel) =>
  /** @type {HTMLElement|null} */ (document.querySelector(sel));
const button = (/** @type {string} */ sel) =>
  /** @type {HTMLButtonElement|null} */ (document.querySelector(sel));
const screenId = () => $('#optimizer-root')?.dataset.screen;
/** A notice of the review's inspector (`changed`, `refused`, `saved`, `start-warning`, …). */
const notice = (/** @type {string} */ kind) => $(`#inspector [data-notice="${kind}"]`);
/** A stored value of the review's inspector. */
const field = (/** @type {string} */ key) => $(`#inspector [data-field="${key}"]`);

beforeEach(() => {
  const html = readFileSync(INDEX, 'utf8');
  // The body markup without its module script (the test boots the panel itself).
  document.body.innerHTML = html
    .slice(html.indexOf('<body>') + 6, html.indexOf('</body>'))
    .replace(/<script[^>]*><\/script>/g, '');
});

/**
 * @param {{ now?: () => number, stub?: ReturnType<typeof sidecar>, waitStart?: boolean }} [options]
 */
async function bootPanel({ now, stub = sidecar(), waitStart = true } = {}) {
  const panel = createPanel({
    api: createApi({ prefix: '', fetchImpl: stub.fetchImpl }),
    setTimer: vi.fn(() => 1),
    clearTimer: vi.fn(),
    ...(now ? { now } : {}),
  });
  await panel.boot();
  const start = () => button('#start-btn');
  if (waitStart) await vi.waitFor(() => expect(start()?.disabled).toBe(false));
  return { ...stub, panel, start, state: () => panel.store.state };
}

test("follows jobs/current to the agent's draft and names its target and environment", async () => {
  await bootPanel();
  expect(field('method')?.textContent).toContain('upper_confidence_bound');
  expect($('#target-slot')?.textContent).toBe('Simulator');
  expect($('#env-text')?.textContent).toBe('Injector');
  expect(screenId()).toBe('review');
});

test('Start sends the rendered revision and fetches nothing at click time', async () => {
  const { start, calls, world } = await bootPanel();
  world.jobs.j1.revision = 'rev-SERVER-NOW';
  const before = calls.length;
  start()?.click();
  await vi.waitFor(() => expect(calls.some((c) => c.method === 'POST')).toBe(true));
  expect(calls.slice(before)[0]).toEqual({
    method: 'POST',
    url: '/relay/jobs/j1/start?config_revision=rev-1',
  });
});

test("the toolbar's Start starts the same revision, and is held when the review's is", async () => {
  const { calls, world, panel } = await bootPanel();
  const toolbar = () => button('#toolbar-start-btn');
  expect(toolbar()?.disabled).toBe(false);
  world.jobs.j1.revision = 'rev-2';
  await panel.poller.now();
  await vi.waitFor(() => expect(toolbar()?.disabled).toBe(true));
  expect(button('#start-btn')?.disabled).toBe(true);
  button('#reviewed-check')?.click();
  await vi.waitFor(() => expect(toolbar()?.disabled).toBe(false));
  toolbar()?.click();
  await vi.waitFor(() => expect(startPosts(calls, 'j1')).toHaveLength(1));
  expect(startPosts(calls, 'j1')[0].url).toBe('/relay/jobs/j1/start?config_revision=rev-2');
});

test('a new revision seen by the poll holds Start until the operator has looked again', async () => {
  const { start, world, panel } = await bootPanel();
  world.jobs.j1.revision = 'rev-2';
  world.jobs.j1.config = storedConfig({ n_iterations: 77 });
  await panel.poller.now();
  await vi.waitFor(() => expect(start()?.disabled).toBe(true));
  expect(field('iterations')?.textContent).toContain('77');
  expect(notice('changed')?.textContent).toContain('changed since it was shown');
  expect($('#status-label')?.textContent).toBe('Draft changed');
  button('#reviewed-check')?.click();
  await vi.waitFor(() => expect(start()?.disabled).toBe(false));
});

test('a 409 draft-changed tells the operator, holds Start and sends nothing again', async () => {
  const { start, world, calls } = await bootPanel();
  world.startReply = {
    status: 409,
    body: {
      detail:
        "Job j1's draft changed since it was shown; read it again (GET /jobs/j1/config) before starting it.",
    },
  };
  world.jobs.j1.revision = 'rev-3';
  start()?.click();
  await vi.waitFor(() => expect(notice('changed')?.textContent).toContain('nothing started'));
  expect(start()?.disabled).toBe(true);
  await settle();
  expect(startPosts(calls, 'j1')).toHaveLength(1);
});

test('no current job: the empty screen, no Start and no run', async () => {
  const stub = sidecar();
  stub.world.current = null;
  const { state } = await bootPanel({ stub, waitStart: false });
  await vi.waitFor(() => expect(state().service.ok).toBe(true));
  expect(screenId()).toBe('empty');
  expect($('#start-btn')).toBeNull();
  expect($('.opt-rp-head')).toBeNull();
  expect(button('#new-run-primary')?.disabled).toBe(false);
});

test('a panel opened during a run shows the run: no Start anywhere, the controls live', async () => {
  const stub = sidecar();
  stub.world.jobs.j1.status = 'running';
  await bootPanel({ stub, waitStart: false });
  await vi.waitFor(() => expect($('.opt-rp-head')).not.toBeNull());
  expect(screenId()).toBe('running');
  expect($('#target-slot')?.textContent).toBe('Simulator');
  expect($('#start-btn')).toBeNull();
  expect($('#primary-slot')?.children).toHaveLength(0);
  expect($('#status-label')?.textContent).toContain('Running');
  expect(button('[data-control="pause"]')?.disabled).toBe(false);
  expect(button('[data-control="cancel"]')?.disabled).toBe(false);
});

test('a Start clicked on a review the poll has replaced starts nothing', async () => {
  const { start, calls, world, panel, state } = await bootPanel();
  const staleStart = /** @type {HTMLButtonElement} */ (start());
  // j1 goes away, then j2 arrives on an empty surface: no review is due, so
  // only the button's own record of what it showed can refuse the click.
  world.current = null;
  await panel.poller.now();
  world.jobs.j2 = { status: 'created', revision: 'rev-j2', config: storedConfig() };
  world.current = 'j2';
  await panel.poller.now();
  await vi.waitFor(() => expect(start()?.disabled).toBe(false));
  expect(state().approval).toMatchObject({ jobId: 'j2', needsReview: false });

  staleStart.click();
  await settle();
  expect(startPosts(calls, 'j2')).toEqual([]);
  expect(startPosts(calls, 'j1')).toEqual([]);

  start()?.click();
  await vi.waitFor(() => expect(startPosts(calls, 'j2')).toHaveLength(1));
  expect(startPosts(calls, 'j2')[0].url).toBe('/relay/jobs/j2/start?config_revision=rev-j2');
});

test('a draft replacing the one on screen is shown at once and must be reviewed', async () => {
  const { start, calls, world, panel } = await bootPanel();
  const linac = gate();
  world.held['/environments/linac'] = linac.promise;
  world.jobs.j2 = {
    status: 'created',
    revision: 'rev-j2',
    config: storedConfig({ environment_name: 'linac', n_iterations: 55 }),
  };
  world.current = 'j2';
  const ticking = panel.poller.now();
  await vi.waitFor(() => expect(field('iterations')?.textContent).toContain('55'));
  expect(notice('changed')?.textContent).toContain('A different draft');
  expect(start()?.disabled).toBe(true);
  start()?.click();
  linac.open();
  await ticking;
  await vi.waitFor(() => expect(field('direction')?.textContent).not.toContain('Not verified'));
  expect(start()?.disabled).toBe(true);
  button('#reviewed-check')?.click();
  await vi.waitFor(() => expect(start()?.disabled).toBe(false));
  expect(startPosts(calls, 'j2')).toEqual([]);
});

test("the review reads the draft's own environment, not the set-up's picker", async () => {
  const { start, world, panel } = await bootPanel();
  const perWrite = () => $('#inspector [data-knob="phase_a"]')?.textContent;
  expect(perWrite()).toContain('at most 20 per write');
  // The operator opens the set-up and points it at another environment, then
  // leaves it: the review keeps the per-write limits read for the draft's.
  const linac = gate();
  world.held['/environments/linac'] = linac.promise;
  button('#edit-draft-btn')?.click();
  await vi.waitFor(() => expect($('select[aria-label="Environment"]')).not.toBeNull());
  const env = /** @type {HTMLSelectElement} */ ($('select[aria-label="Environment"]'));
  env.value = 'linac';
  env.dispatchEvent(new Event('change'));
  expect($('#env-text')?.textContent).toBe('Linac');
  button('#opt-su-discard')?.click();
  await vi.waitFor(() => expect(screenId()).toBe('review'));
  expect(perWrite()).toContain('at most 20 per write');
  expect(field('direction')?.textContent).not.toContain('Not verified');
  expect(start()?.disabled).toBe(false);
  linac.open();

  // A draft whose environment has not loaded yet holds Start, unverified.
  const injector = gate();
  world.held['/environments/injector'] = injector.promise;
  world.jobs.j3 = {
    status: 'created',
    revision: 'rev-j3',
    config: storedConfig({ objective_direction: 'MINIMIZE' }),
  };
  world.current = null;
  await panel.poller.now();
  world.current = 'j3';
  const ticking = panel.poller.now();
  await vi.waitFor(() => expect(field('direction')?.textContent).toContain('Not verified'));
  expect(start()?.disabled).toBe(true);
  expect($('#inspector [data-why]')?.textContent).toBe(
    "Direction unverified: the draft's environment is loading."
  );
  injector.open();
  await ticking;
  await vi.waitFor(() =>
    expect(field('direction')?.textContent).toContain("Differs from the observable's own direction")
  );
  expect(start()?.disabled).toBe(false);
});

test("a start the service did not answer in time says so, not 'unreachable'", async () => {
  const { start, world, calls } = await bootPanel();
  const detail =
    'the optimizer service did not answer in time; it may still be working — check the job before retrying';
  world.startReply = { status: 504, body: { detail } };
  start()?.click();
  await vi.waitFor(() => expect(notice('refused')?.textContent).toContain(detail));
  expect(notice('refused')?.textContent).not.toContain('unreachable');
  await settle();
  expect(startPosts(calls, 'j1')).toHaveLength(1);
});

test('a session naming no account warns before Start, and Start stays enabled', async () => {
  const stub = sidecar();
  stub.world.session = { owner: null };
  const { state } = await bootPanel({ stub, waitStart: false });
  await vi.waitFor(() =>
    expect(notice('start-warning')?.textContent).toContain('names no account')
  );
  expect(state().ownerKnown).toBe(false);
  await vi.waitFor(() => expect(button('#start-btn')?.disabled).toBe(false));
  expect(button('#toolbar-start-btn')?.disabled).toBe(false);
  expect($('#status-label')?.textContent).toBe('No account');
});

test('a start notice that comes or goes on the same revision is shown and goes again', async () => {
  const { start, world, panel, state } = await bootPanel();
  const warnings = () =>
    Array.from(document.querySelectorAll('#inspector [data-notice="start-warning"]')).map(
      (n) => n.textContent
    );
  expect(warnings()).toEqual([]);
  const text = 'Nelder-Mead may not move from the current setpoints.';
  world.jobs.j1.start_notice = text;
  await panel.poller.now();
  await vi.waitFor(() => expect(warnings()).toEqual([text]));
  expect(state().approval.revision).toBe('rev-1');
  expect(start()?.disabled).toBe(false);
  world.jobs.j1.start_notice = null;
  await panel.poller.now();
  await vi.waitFor(() => expect(warnings()).toEqual([]));
});

test('a named session says whose the run is; a failed session read says nothing', async () => {
  const named = sidecar();
  named.world.session = { owner: 'op1' };
  await bootPanel({ stub: named, waitStart: false });
  await vi.waitFor(() => expect($('#inspector')?.textContent).toContain('as op1'));
  const { state } = await bootPanel();
  expect(state().ownerKnown).toBeNull();
  expect($('#inspector')?.textContent).not.toContain('no account');
});

test('a new draft is saved through the relay, and the review names the windows the service moved', async () => {
  const { world, calls, state } = await bootPanel();
  world.clampMax = 10;
  world.current = 'j2';
  button('#new-run')?.click();
  await vi.waitFor(() => expect($('#opt-su-target-va')).not.toBeNull());
  button('#opt-su-target-va')?.click();
  const env = /** @type {HTMLSelectElement} */ ($('select[aria-label="Environment"]'));
  env.value = 'injector';
  env.dispatchEvent(new Event('change'));
  await vi.waitFor(() => expect(button('#save-draft-btn')?.disabled).toBe(false));
  button('#save-draft-btn')?.click();
  await vi.waitFor(() => expect(state().saveNote?.jobId).toBe('j2'));
  expect(posts(calls)).toEqual(['/relay/optimizations']);
  // The knobs the service moved: those sent with a max above its ceiling.
  const sent = /** @type {Record<string, { max: number }>} */ (world.created.variable_overrides);
  const moved = Object.keys(sent).filter((name) => sent[name].max > 10);
  expect(moved.length).toBeGreaterThan(0);
  expect(state().saveNote?.moved).toEqual(moved);
  await vi.waitFor(() => expect(screenId()).toBe('review-changed'));
  expect(notice('saved')?.textContent).toContain('→ 10');
  expect(notice('changed')?.textContent).toContain('A different draft');
});

test('a set-up the panel refuses stays open with the reason', async () => {
  const { panel } = await bootPanel();
  button('#new-run')?.click();
  panel.actions.refuseSetup(['Pick an objective.']);
  await vi.waitFor(() => expect(screenId()).toBe('setup-issues'));
  expect($('#inspector [role="alert"]')?.textContent).toContain('Pick an objective.');
  expect($('#status-label')?.textContent).toBe('1 problem');
});

describe('an update a poll overtakes', () => {
  /**
   * Boot, open the set-up of j1, hold the PUT, let `during` change the world
   * and poll, then release.
   *
   * @param {(world: any, panel: any) => Promise<void>} during
   */
  async function updateWhilePolled(during) {
    const booted = await bootPanel();
    const held = gate();
    booted.world.held['/relay/jobs/j1/config'] = held.promise;
    button('#edit-draft-btn')?.click();
    await vi.waitFor(() => expect(button('#opt-su-save')?.disabled).toBe(false));
    button('#opt-su-save')?.click();
    await vi.waitFor(() =>
      expect(
        booted.calls.some((c) => c.method === 'PUT' && c.url === '/relay/jobs/j1/config')
      ).toBe(true)
    );
    await during(booted.world, booted.panel);
    held.open();
    return booted;
  }

  test('the job cleared mid-write: the save lands, nothing throws', async () => {
    const { calls, state } = await updateWhilePolled(async (world, panel) => {
      world.current = null;
      await panel.poller.now();
      expect(panel.store.state.job).toBeNull();
    });
    await vi.waitFor(() => expect(state().saveNote?.jobId).toBe('j1'));
    const put = calls.findIndex((c) => c.method === 'PUT');
    // The read-back names the job written, not whatever the poll left.
    expect(calls.slice(put).some((c) => c.method === 'GET' && c.url === '/jobs/j1/config')).toBe(
      true
    );
    expect(state().setup).toBeNull();
  });

  test('the job switched mid-write: the read-back is of the draft written; the new one must be reviewed', async () => {
    const { calls, state } = await updateWhilePolled(async (world, p) => {
      world.jobs.j3 = { status: 'created', revision: 'rev-j3', config: storedConfig() };
      world.current = 'j3';
      await p.poller.now();
      // The set-up of j1 keeps the operator's edits while another draft is current.
      expect(p.store.state.setup?.jobId).toBe('j1');
    });
    await vi.waitFor(() => expect(state().saveNote?.jobId).toBe('j1'));
    expect(calls.some((c) => c.method === 'GET' && c.url === '/jobs/j1/config')).toBe(true);
    expect(state().job.job_id).toBe('j3');
    await vi.waitFor(() => expect(screenId()).toBe('review-changed'));
  });
});

describe('a run', () => {
  /** @param {Partial<StubJob>} [job] */
  async function bootRun(job = {}) {
    const stub = sidecar();
    Object.assign(stub.world.jobs.j1, { status: 'running', ...job });
    const booted = await bootPanel({ stub, waitStart: false });
    await vi.waitFor(() => expect($('.opt-rp-head')).not.toBeNull());
    return booted;
  }

  const confirmButton = () => button('.opt-confirm [data-confirm="cancel"]');
  const confirmShown = () => !!$('.opt-confirm') && !$('.opt-confirm')?.hidden;

  test('Cancel asks first; keep sends nothing; only the confirmation sends it', async () => {
    const { calls } = await bootRun();
    button('[data-control="cancel"]')?.click();
    await vi.waitFor(() => expect(confirmShown()).toBe(true));
    expect($('.opt-confirm')?.textContent).toContain(STOP_WORDS.now);
    expect(posts(calls)).toEqual([]);
    button('.opt-confirm [data-confirm="keep"]')?.click();
    await vi.waitFor(() => expect(confirmShown()).toBe(false));
    await settle();
    expect(posts(calls)).toEqual([]);
    button('[data-control="cancel"]')?.click();
    await vi.waitFor(() => expect(confirmShown()).toBe(true));
    confirmButton()?.click();
    await vi.waitFor(() => expect(posts(calls)).toEqual(['/relay/jobs/j1/cancel']));
  });

  test('Pause, then Resume, reach the relay for this job', async () => {
    const { calls, world, panel } = await bootRun();
    button('[data-control="pause"]')?.click();
    await vi.waitFor(() => expect(posts(calls)).toEqual(['/relay/jobs/j1/pause']));
    world.jobs.j1.status = 'paused';
    await panel.poller.now();
    await vi.waitFor(() => expect(screenId()).toBe('paused'));
    button('[data-control="pause"]')?.click();
    await vi.waitFor(() =>
      expect(posts(calls)).toEqual(['/relay/jobs/j1/pause', '/relay/jobs/j1/resume'])
    );
  });

  test('a refused pause shows the reason and sends nothing again', async () => {
    const { calls, world } = await bootRun();
    world.controlReply = { status: 409, body: { detail: 'Cannot pause job j1: it is paused' } };
    button('[data-control="pause"]')?.click();
    await vi.waitFor(() =>
      expect($('.opt-rp-refusal')?.textContent).toContain('Cannot pause job j1: it is paused')
    );
    await settle();
    expect(posts(calls)).toEqual(['/relay/jobs/j1/pause']);
  });

  test('a run waiting for its write approval says who approves, and is not running', async () => {
    await bootRun({ write_authorization: { state: 'awaiting', owner: 'op1' } });
    await vi.waitFor(() => expect(screenId()).toBe('waiting-approval'));
    expect($('#status-label')?.textContent).toBe('Waiting for approval');
    expect($('.opt-rp-sentence')?.textContent).toContain('Waiting for op1 to approve the writes');
    expect($('.opt-rp-sentence')?.textContent).toContain('Nothing has been written');
    expect($('.opt-wait-card')).not.toBeNull();
  });

  test('a cancelling run keeps its controls in place, both disabled', async () => {
    const { calls } = await bootRun({ status: 'cancelling' });
    expect($('#status-label')?.textContent).toBe('Cancelling');
    expect(button('[data-control="cancel"]')?.disabled).toBe(true);
    expect(button('[data-control="pause"]')?.disabled).toBe(true);
    button('[data-control="cancel"]')?.click();
    button('[data-control="pause"]')?.click();
    await settle();
    expect(posts(calls)).toEqual([]);
  });

  test('a run that starts elsewhere while a new set-up is open: the set-up stays, one click shows the run', async () => {
    const { world, panel } = await bootPanel();
    button('#new-run')?.click();
    world.jobs.j1.status = 'running';
    await panel.poller.now();
    expect(screenId()).toBe('setup');
    const runRow = () => $('[data-row="job:j1"]');
    await vi.waitFor(() => expect(runRow()?.textContent).toContain('running'));
    runRow()?.click();
    await vi.waitFor(() => expect($('.opt-rp-head')).not.toBeNull());
    expect(button('[data-control="cancel"]')?.disabled).toBe(false);
    expect(panel.store.state.setup).not.toBeNull();
    $('[data-row="setup"]')?.click();
    expect(screenId()).toBe('setup');
  });
});

describe('the runs list', () => {
  test('lists the stored runs; a click reads that run and shows it', async () => {
    const stub = sidecar();
    stub.world.current = null;
    stub.world.results = [
      {
        run_id: 'injector/2026-11-16_21-15-04',
        environment: 'injector',
        timestamp: '2026-11-16_21-15-04',
        objective_name: 'charge',
        has_model: false,
        job_id: 'old-job',
      },
    ];
    stub.world.stored['injector/2026-11-16_21-15-04'] = {
      timestamp: 'injector/2026-11-16_21-15-04',
      config: storedConfig({ objective_direction: 'MAXIMIZE' }),
      best_point: {},
      initial_point: { charge: 1, phase_a: 0, sol_b: 3 },
      data_records: [
        { charge: 1.5, phase_a: 1, sol_b: 3 },
        { charge: 2, phase_a: 2, sol_b: 3 },
      ],
    };
    const { calls } = await bootPanel({ stub, waitStart: false });
    const row = () => $('[data-row="run:injector/2026-11-16_21-15-04"]');
    await vi.waitFor(() => expect(row()).not.toBeNull());
    expect(row()?.textContent).toContain('Charge');
    row()?.click();
    await vi.waitFor(() => expect($('#stage .opt-facts')).not.toBeNull());
    expect(calls.some((c) => c.url === '/results/injector/2026-11-16_21-15-04')).toBe(true);
    expect(calls.some((c) => c.url === '/jobs/old-job')).toBe(true);
    const facts = Object.fromEntries(
      Array.from(document.querySelectorAll('#stage [data-fact]')).map((n) => [
        /** @type {HTMLElement} */ (n).dataset.fact,
        n.querySelector('.osprey-fact-value')?.textContent,
      ])
    );
    expect(facts).toMatchObject({ evaluations: '2 of 33', before: '1', best: '2' });
    expect(facts.status).toBeUndefined();
    expect(screenId()).toBe('result');
    expect($('#target-slot')?.textContent).toBe('Simulator');
  });
});

describe('a run active beside a newer draft', () => {
  async function bootBeside() {
    const stub = sidecar();
    // The current job is the draft j1; the service still runs r9.
    stub.world.jobs.r9 = {
      status: 'running',
      revision: 'rev-r9',
      config: storedConfig({ control_target: 'standin' }),
    };
    stub.world.openJobs = [
      { job_id: 'j1', status: 'created' },
      { job_id: 'r9', status: 'running' },
    ];
    const booted = await bootPanel({ stub, waitStart: false });
    await vi.waitFor(() => expect($('.opt-rp-head')).not.toBeNull());
    return booted;
  }

  test('now is the run, with its controls bound to its own id', async () => {
    const { calls } = await bootBeside();
    expect(screenId()).toBe('running');
    expect($('#target-slot')?.textContent).toBe('Rehearsal');
    button('[data-control="cancel"]')?.click();
    await vi.waitFor(() => expect(button('.opt-confirm [data-confirm="cancel"]')).not.toBeNull());
    button('.opt-confirm [data-confirm="cancel"]')?.click();
    await vi.waitFor(() => expect(posts(calls)).toEqual(['/relay/jobs/r9/cancel']));
  });

  test('the newer draft opens as its review when picked; a refused Start shows the reason', async () => {
    const { world, calls } = await bootBeside();
    const draftRow = () => $('[data-row="job:j1"]');
    await vi.waitFor(() => expect(draftRow()).not.toBeNull());
    draftRow()?.click();
    await vi.waitFor(() => expect(button('#start-btn')?.disabled).toBe(false));
    const detail = 'Another job (r9) is already running';
    world.startReply = { status: 409, body: { detail } };
    button('#start-btn')?.click();
    await vi.waitFor(() => expect(notice('refused')?.textContent).toContain(detail));
    await settle();
    expect(startPosts(calls, 'j1')).toHaveLength(1);
    $('[data-row="job:r9"]')?.click();
    await vi.waitFor(() => expect($('.opt-rp-head')).not.toBeNull());
    expect(button('[data-control="pause"]')?.disabled).toBe(false);
  });
});

test('run again opens a new set-up from a stored config and writes nothing', async () => {
  const { panel, calls } = await bootPanel();
  const writes = () => calls.filter((c) => c.method !== 'GET').length;
  const before = writes();
  panel.actions.runAgain(storedConfig({ control_target: 'standin' }));
  expect(screenId()).toBe('setup');
  expect(panel.store.state.setup).toMatchObject({ jobId: null, target: 'standin' });
  expect(panel.store.state.setup?.from?.control_target).toBe('standin');
  panel.actions.runAgain(null);
  expect(writes()).toBe(before);
});

test('every write goes through relay/', async () => {
  const { calls, start, world, panel } = await bootPanel();
  start()?.click();
  await vi.waitFor(() => expect(startPosts(calls, 'j1')).toHaveLength(1));
  world.jobs.j1.status = 'running';
  await panel.poller.now();
  await vi.waitFor(() => expect(button('[data-control="pause"]')?.disabled).toBe(false));
  button('[data-control="pause"]')?.click();
  await settle();
  const writes = calls.filter((c) => c.method !== 'GET');
  expect(writes.length).toBeGreaterThan(1);
  expect(writes.every((c) => c.url.startsWith('/relay/'))).toBe(true);
});

test("a readback refresh redraws the review's values and keeps its Start button", async () => {
  let t = 0;
  const { world, panel, start } = await bootPanel({ now: () => t });
  /** The value "now" the review shows for a knob, as a number string. */
  const readback = (/** @type {string} */ name) =>
    $(`#stage .opt-knob-row[data-knob="${name}"] [data-now]`)
      ?.textContent?.replace(/^now /, '')
      .split(' ')[0];
  await vi.waitFor(() => expect(readback('phase_a')).toBe('1'));
  const startButton = start();
  world.readings = {
    phase_a: { value: 4.7, connected: true },
    sol_b: { value: 1.7, connected: true },
  };
  t = 6000;
  await panel.poller.now();
  await vi.waitFor(() => expect(readback('phase_a')).toBe('4.7'));
  expect(readback('sol_b')).toBe('1.7');
  expect(start()).toBe(startButton);
  expect(startButton?.disabled).toBe(false);
});
