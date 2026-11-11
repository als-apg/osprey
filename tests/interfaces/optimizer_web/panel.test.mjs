/**
 * Integration tests for the OPTIMIZER panel shell (panel.js) against its real
 * index.html markup and a stubbed sidecar: the panel follows `jobs/current`
 * to an agent's draft, and Start sends the revision of the config response
 * that rendered the approval surface — never one fetched at click time.
 */

import { readFileSync } from 'node:fs';
import { cwd } from 'node:process';

import { describe, test, expect, beforeEach, vi } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { createApi } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/api-client.js';
import { createPanel } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/panel.js';

const INDEX = `${cwd()}/src/osprey/interfaces/optimizer_web/panels/optimizer/index.html`;

/** The sidecar stub: a mutable world, and every request it saw. */
function sidecar() {
  const world = {
    current: /** @type {string|null} */ ('j1'),
    /** @type {Record<string, { status: string, revision: string, config: any }>} */
    jobs: { j1: { status: 'created', revision: 'rev-1', config: storedConfig() } },
    /** @type {{ status: number, body: any }|null} */
    startReply: null,
    /** Paths whose answer waits until the test resolves the promise. */
    /** @type {Record<string, Promise<void>|undefined>} */
    held: {},
    /** What `relay/session` answers; unset, it is not routed (the read fails). */
    /** @type {{ owner: string|null }|undefined} */
    session: undefined,
    /** The service's clamp on a created draft's bounds: its ceiling for every max. */
    clampMax: Infinity,
    /** What `environments/injector/variables` reads off the machine. */
    /** @type {Record<string, { value: number, connected: boolean }>} */
    readings: { phase_a: { value: 1, connected: true } },
  };
  /** @type {{ method: string, url: string }[]} */
  const calls = [];
  /** @type {Record<string, () => any>} */
  const routes = {
    'GET /service/health': () => ({ status: 'ok' }),
    'GET /environments': () => [
      { name: 'injector', display_name: 'Injector', valid: true },
      { name: 'linac', display_name: 'Linac', valid: true },
    ],
    'GET /environments/injector': () => envDetails(),
    'GET /environments/linac': () => ({ ...envDetails(), name: 'linac' }),
    'GET /environments/injector/presets': () => ({}),
    'GET /environments/linac/presets': () => ({}),
    'GET /environments/injector/status': () => ({ name: 'injector', connected: true }),
    'GET /environments/injector/variables': () => world.readings,
    'GET /environments/injector/initial-bounds': () => ({ bounds: {}, current_values: {} }),
    'GET /models': () => [],
    'GET /jobs/current': () => ({ job_id: world.current }),
  };
  /** @type {typeof fetch} */
  const fetchImpl = async (input, init) => {
    const url = String(input);
    const method = init?.method || 'GET';
    calls.push({ method, url });
    const path = url.split('?')[0];
    if (world.held[path]) await world.held[path];
    const startMatch = method === 'POST' && path.match(/^\/relay\/jobs\/([^/]+)\/start$/);
    if (startMatch) {
      const reply = world.startReply || {
        status: 200,
        body: { job_id: startMatch[1], status: 'submitted' },
      };
      return new Response(JSON.stringify(reply.body), { status: reply.status });
    }
    if (method === 'GET' && path === '/relay/session' && world.session) {
      return new Response(JSON.stringify(world.session), { status: 200 });
    }
    if (method === 'POST' && path === '/relay/optimizations') {
      const sent = JSON.parse(String(init?.body));
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
      return new Response(JSON.stringify({ job_id: 'j2', status: 'created' }), { status: 200 });
    }
    const putMatch = method === 'PUT' && path.match(/^\/relay\/jobs\/([^/]+)\/config$/);
    if (putMatch && world.jobs[putMatch[1]]) {
      world.jobs[putMatch[1]].config = JSON.parse(String(init?.body));
      return new Response(JSON.stringify({ job_id: putMatch[1], status: 'created' }), {
        status: 200,
      });
    }
    const jobMatch = method === 'GET' && path.match(/^\/jobs\/([^/]+)(\/config)?$/);
    if (jobMatch && jobMatch[1] !== 'current') {
      const job = world.jobs[jobMatch[1]];
      if (!job) return new Response(JSON.stringify({ detail: 'no job' }), { status: 404 });
      if (jobMatch[2]) {
        return new Response(JSON.stringify(job.config), {
          status: 200,
          headers: { 'X-Config-Revision': job.revision },
        });
      }
      const body = { job_id: jobMatch[1], status: job.status, job_type: 'optimization' };
      return new Response(JSON.stringify(body), { status: 200 });
    }
    const route = routes[`${method} ${path}`];
    if (!route) return new Response(JSON.stringify({ detail: 'no route' }), { status: 404 });
    return new Response(JSON.stringify(route()), { status: 200 });
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

/** @param {string} jobId */
const startPosts = (/** @type {{ method: string, url: string }[]} */ calls, jobId) =>
  calls.filter((c) => c.method === 'POST' && c.url.startsWith(`/relay/jobs/${jobId}/start`));

beforeEach(() => {
  const html = readFileSync(INDEX, 'utf8');
  // The body markup without its module script (the test boots the panel itself).
  document.body.innerHTML = html
    .slice(html.indexOf('<body>') + 6, html.indexOf('</body>'))
    .replace(/<script[^>]*><\/script>/g, '');
});

/** @param {{ now?: () => number }} [options] */
async function bootPanel({ now } = {}) {
  const stub = sidecar();
  const timers = { setTimer: vi.fn(() => 1), clearTimer: vi.fn() };
  const panel = createPanel({
    api: createApi({ prefix: '', fetchImpl: stub.fetchImpl }),
    ...timers,
    ...(now ? { now } : {}),
  });
  const led = await panel.boot();
  clearInterval(led);
  const start = () => /** @type {HTMLButtonElement|null} */ (document.querySelector('#start-btn'));
  await vi.waitFor(() => expect(start()?.disabled).toBe(false));
  return { ...stub, panel, start };
}

test("follows jobs/current to the agent's draft and adopts its target and environment", async () => {
  const { panel } = await bootPanel();
  expect(/** @type {HTMLSelectElement} */ (document.querySelector('#target-select')).value).toBe(
    'va'
  );
  expect(/** @type {HTMLSelectElement} */ (document.querySelector('#env-select')).value).toBe(
    'injector'
  );
  expect(panel.state.approval.revision).toBe('rev-1');
  expect(document.querySelector('#approval-card')?.textContent).toContain('upper_confidence_bound');
});

test('Start sends the rendered revision and fetches nothing at click time', async () => {
  const { start, calls, world } = await bootPanel();
  world.jobs.j1.revision = 'rev-SERVER-NOW';
  const before = calls.length;
  start()?.click();
  await vi.waitFor(() => expect(calls.some((c) => c.method === 'POST')).toBe(true));
  const sinceClick = calls.slice(before);
  expect(sinceClick[0]).toEqual({
    method: 'POST',
    url: '/relay/jobs/j1/start?config_revision=rev-1',
  });
});

test('a new revision seen by the poll holds Start until the operator has looked again', async () => {
  const { start, world, panel } = await bootPanel();
  world.jobs.j1.revision = 'rev-2';
  world.jobs.j1.config = storedConfig({ n_iterations: 77 });
  await panel.poller.now();
  await vi.waitFor(() => expect(start()?.disabled).toBe(true));
  expect(document.querySelector('#approval-card')?.textContent).toContain('77');
  /** @type {HTMLButtonElement} */ (document.querySelector('#reviewed-btn')).click();
  expect(start()?.disabled).toBe(false);
});

test('a 409 draft-changed tells the operator and re-renders', async () => {
  const { start, world } = await bootPanel();
  world.startReply = {
    status: 409,
    body: {
      detail:
        "Job j1's draft changed since it was shown; read it again (GET /jobs/j1/config) before starting it.",
    },
  };
  world.jobs.j1.revision = 'rev-3';
  start()?.click();
  await vi.waitFor(() =>
    expect(document.querySelector('#approval-card .banner-warn')?.textContent).toContain(
      'nothing started'
    )
  );
  expect(start()?.disabled).toBe(true);
});

test('no current job: an empty approval state and no run card', async () => {
  const stub = sidecar();
  stub.world.current = null;
  const panel = createPanel({
    api: createApi({ prefix: '', fetchImpl: stub.fetchImpl }),
    setTimer: () => 1,
    clearTimer: () => {},
  });
  clearInterval(await panel.boot());
  await vi.waitFor(() => expect(stub.calls.some((c) => c.url === '/jobs/current')).toBe(true));
  expect(document.querySelector('#start-btn')).toBeNull();
  expect(/** @type {HTMLElement} */ (document.querySelector('#run-card')).hidden).toBe(true);
});

test('a Start clicked on a card the poll has replaced starts nothing', async () => {
  const { start, calls, world, panel } = await bootPanel();
  const staleStart = /** @type {HTMLButtonElement} */ (start());
  // j1 goes away, then j2 arrives on an empty surface: no review is due, so
  // only the button's own record of what it showed can refuse the click.
  world.current = null;
  await panel.poller.now();
  world.jobs.j2 = { status: 'created', revision: 'rev-j2', config: storedConfig() };
  world.current = 'j2';
  await panel.poller.now();
  await vi.waitFor(() => expect(start()?.disabled).toBe(false));
  expect(panel.state.approval).toMatchObject({ jobId: 'j2', needsReview: false });

  staleStart.click();
  await Promise.resolve();
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
  // The new card is on screen while the form is still following the draft.
  await vi.waitFor(() =>
    expect(document.querySelector('#approval-card')?.textContent).toContain('55')
  );
  expect(document.querySelector('#approval-card .banner-warn')?.textContent).toContain(
    'A different draft'
  );
  expect(start()?.disabled).toBe(true);
  start()?.click();
  linac.open();
  await ticking;
  await vi.waitFor(() =>
    expect(document.querySelector('#approval-card')?.textContent).not.toContain(
      'direction unverified'
    )
  );
  expect(start()?.disabled).toBe(true);
  /** @type {HTMLButtonElement} */ (document.querySelector('#reviewed-btn')).click();
  expect(start()?.disabled).toBe(false);
  expect(startPosts(calls, 'j2')).toEqual([]);
});

test("the approval card reads the draft's own environment, not the form's picker", async () => {
  const { start, world, panel } = await bootPanel();
  const linac = gate();
  world.held['/environments/linac'] = linac.promise;
  // The operator points the form at another environment: the card keeps its
  // direction check and per-write limits, read for the draft's environment.
  const envSelect = /** @type {HTMLSelectElement} */ (document.querySelector('#env-select'));
  envSelect.value = 'linac';
  envSelect.dispatchEvent(new Event('change'));
  const card = () => document.querySelector('#approval-card');
  expect(card()?.querySelector('.appr-vars tbody td:nth-child(5)')?.textContent).toBe('20');
  expect(card()?.textContent).not.toContain('direction unverified');
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
  await vi.waitFor(() => expect(card()?.textContent).toContain('direction unverified'));
  expect(start()?.disabled).toBe(true);
  expect(card()?.textContent).toContain(
    "Direction unverified: the draft's environment is loading."
  );
  injector.open();
  await ticking;
  await vi.waitFor(() =>
    expect(card()?.textContent).toContain("differs from the observable's MAXIMIZE")
  );
  expect(start()?.disabled).toBe(false);
});

test("a start the service did not answer in time says so, not 'unreachable'", async () => {
  const { start, world } = await bootPanel();
  const detail =
    'the optimizer service did not answer in time; it may still be working ' +
    '— check the job before retrying';
  world.startReply = { status: 504, body: { detail } };
  start()?.click();
  await vi.waitFor(() =>
    expect(document.querySelector('#approval-card .banner-err')?.textContent).toBe(detail)
  );
});

test('a session naming no account warns before Start, and Start stays enabled', async () => {
  const stub = sidecar();
  stub.world.session = { owner: null };
  const panel = createPanel({
    api: createApi({ prefix: '', fetchImpl: stub.fetchImpl }),
    setTimer: vi.fn(() => 1),
    clearTimer: vi.fn(),
  });
  clearInterval(await panel.boot());
  const card = () => document.querySelector('#approval-card');
  await vi.waitFor(() => expect(card()?.textContent).toContain('names no account'));
  expect(panel.state.ownerKnown).toBe(false);
  await vi.waitFor(() =>
    expect(/** @type {HTMLButtonElement} */ (document.querySelector('#start-btn')).disabled).toBe(
      false
    )
  );
});

test('a named session says whose the run is; a failed session read says nothing', async () => {
  const named = sidecar();
  named.world.session = { owner: 'op1' };
  const panel = createPanel({
    api: createApi({ prefix: '', fetchImpl: named.fetchImpl }),
    setTimer: vi.fn(() => 1),
    clearTimer: vi.fn(),
  });
  clearInterval(await panel.boot());
  await vi.waitFor(() =>
    expect(document.querySelector('#approval-card .start-owner')?.textContent).toBe(
      'Starts as op1.'
    )
  );

  const { panel: unknown } = await bootPanel();
  expect(unknown.state.ownerKnown).toBeNull();
  expect(document.querySelector('#approval-card')?.textContent).not.toContain('names no account');
});

test('a created draft re-seeds the form from the stored draft and names a clamped bound', async () => {
  const { world } = await bootPanel();
  world.clampMax = 10;
  world.current = 'j2';
  /** @type {HTMLButtonElement} */ (document.querySelector('#create-draft-btn')).click();
  const form = () => document.querySelector('#form-card');
  await vi.waitFor(() =>
    expect(form()?.textContent).toContain(
      "Draft created — phase_a max 18 → 10 (adjusted by the service to the environment's range and step)."
    )
  );
  const max = /** @type {HTMLInputElement} */ (form()?.querySelector('[aria-label="phase_a max"]'));
  expect(max.value).toBe('10');
});

describe('an update a poll overtakes', () => {
  /**
   * Boot, hold the PUT, let `during` change the world and poll, then release.
   *
   * @param {(world: any, panel: any) => Promise<void>} during
   */
  async function updateWhilePolled(during) {
    const booted = await bootPanel();
    const held = gate();
    booted.world.held['/relay/jobs/j1/config'] = held.promise;
    const update = () =>
      /** @type {HTMLButtonElement} */ (document.querySelector('#update-draft-btn'));
    await vi.waitFor(() => expect(update().disabled).toBe(false));
    update().click();
    await vi.waitFor(() =>
      expect(
        booted.calls.some((c) => c.method === 'PUT' && c.url === '/relay/jobs/j1/config')
      ).toBe(true)
    );
    await during(booted.world, booted.panel);
    held.open();
    return booted;
  }

  test('the job cleared mid-write: the saved message shows, nothing throws', async () => {
    const { calls } = await updateWhilePolled(async (world, panel) => {
      world.current = null;
      await panel.poller.now();
      expect(panel.state.job).toBeNull();
    });
    const put = calls.findIndex((c) => c.method === 'PUT');
    await vi.waitFor(() =>
      expect(document.querySelector('#form-card')?.textContent).toContain('Draft updated.')
    );
    // The read-back still names the job written, not whatever the poll left.
    expect(calls.slice(put).some((c) => c.method === 'GET' && c.url === '/jobs/j1/config')).toBe(
      true
    );
  });

  test("the job switched mid-write: the form is not re-seeded from either draft's read-back", async () => {
    const { calls, panel } = await updateWhilePolled(async (world, p) => {
      world.jobs.j3 = {
        status: 'created',
        revision: 'rev-j3',
        config: storedConfig({
          variable_overrides: {
            phase_a: { min: -7, max: 7, step_size: 0.5 },
            sol_b: { min: 2, max: 4, step_size: 0.01 },
          },
        }),
      };
      world.current = 'j3';
      await p.poller.now();
      await vi.waitFor(() => expect(p.state.formJobId).toBe('j3'));
    });
    await vi.waitFor(() =>
      expect(document.querySelector('#form-card')?.textContent).toContain('Draft updated.')
    );
    // The read-back is of the draft that was written, and the form stays on j3.
    expect(calls.some((c) => c.method === 'GET' && c.url === '/jobs/j1/config')).toBe(true);
    const max = /** @type {HTMLInputElement} */ (
      document.querySelector('#form-card [aria-label="phase_a max"]')
    );
    expect(max.value).toBe('7');
    expect(panel.state.job.job_id).toBe('j3');
  });
});

describe("the approval card's readbacks while a draft waits", () => {
  const readback = (/** @type {string} */ name) =>
    document.querySelector(`#approval-card td[data-readback="${name}"]`)?.textContent;

  test('are re-read on the poll, at most every 5 s', async () => {
    let t = 0;
    const { world, panel, calls } = await bootPanel({ now: () => t });
    await vi.waitFor(() => expect(readback('phase_a')).toBe('1'));
    const reads = () =>
      calls.filter((c) => c.url.startsWith('/environments/injector/variables')).length;
    world.readings = {
      phase_a: { value: 4.7, connected: true },
      sol_b: { value: 1.7, connected: true },
    };
    const before = reads();
    t = 1000;
    await panel.poller.now();
    expect(reads()).toBe(before);
    expect(readback('phase_a')).toBe('1');
    t = 6000;
    await panel.poller.now();
    await vi.waitFor(() => expect(readback('phase_a')).toBe('4.7'));
    expect(readback('sol_b')).toBe('1.7');
  });

  const VARIABLES = '/environments/injector/variables';
  const readsOf = (/** @type {{ url: string }[]} */ calls) =>
    calls.filter((c) => c.url.startsWith(VARIABLES)).length;

  test('a read still in flight holds the next one back', async () => {
    let t = 0;
    const { world, panel, calls } = await bootPanel({ now: () => t });
    await vi.waitFor(() => expect(readback('phase_a')).toBe('1'));
    const held = gate();
    world.held[VARIABLES] = held.promise;
    const before = readsOf(calls);
    t = 6000;
    await panel.poller.now();
    expect(readsOf(calls)).toBe(before + 1);
    // A slow machine read: later ticks skip rather than pile reads up.
    t = 12000;
    await panel.poller.now();
    t = 18000;
    await panel.poller.now();
    expect(readsOf(calls)).toBe(before + 1);
    world.readings = { phase_a: { value: 5, connected: true } };
    held.open();
    await vi.waitFor(() => expect(readback('phase_a')).toBe('5'));
  });

  test('nothing is re-read while a Start is pending', async () => {
    let t = 0;
    const { world, panel, calls, start } = await bootPanel({ now: () => t });
    const held = gate();
    world.held['/relay/jobs/j1/start'] = held.promise;
    start()?.click();
    await vi.waitFor(() => expect(panel.state.starting).toBe(true));
    const before = readsOf(calls);
    t = 6000;
    await panel.poller.now();
    expect(readsOf(calls)).toBe(before);
    held.open();
    await vi.waitFor(() => expect(panel.state.starting).toBe(false));
  });

  test("a late reply for the card's previous environment is dropped", async () => {
    let t = 0;
    const { world, panel, calls } = await bootPanel({ now: () => t });
    await vi.waitFor(() => expect(readback('phase_a')).toBe('1'));
    const held = gate();
    world.held[VARIABLES] = held.promise;
    const before = readsOf(calls);
    t = 6000;
    await panel.poller.now();
    expect(readsOf(calls)).toBe(before + 1);
    delete world.held[VARIABLES]; // only the refresh already sent is held
    // The draft moves to another variable set: a new environment key.
    world.jobs.j1.revision = 'rev-2';
    world.jobs.j1.config = storedConfig({ variables: ['phase_a'] });
    world.readings = { phase_a: { value: 2, connected: true } };
    await panel.poller.now();
    await vi.waitFor(() => expect(readback('phase_a')).toBe('2'));
    const key = panel.state.approvalEnv.key;
    expect(key).toBe('injector|va|phase_a');
    // The old read answers last, with a reading the card must not show.
    world.readings = { phase_a: { value: 99, connected: true } };
    held.open();
    await vi.waitFor(() => expect(panel.state.approvalReading).toBe(false));
    expect(readback('phase_a')).toBe('2');
    expect(panel.state.approvalEnv.key).toBe(key);
    expect(panel.state.approvalEnv.readbacks.phase_a).not.toBe(99);
  });

  test('a refresh leaves the review state, the revision and Start as they were', async () => {
    let t = 0;
    const { world, panel, start } = await bootPanel({ now: () => t });
    world.jobs.j1.revision = 'rev-2';
    await panel.poller.now();
    await vi.waitFor(() => expect(start()?.disabled).toBe(true));
    expect(panel.state.approval).toMatchObject({ revision: 'rev-2', needsReview: true });

    // Still unreviewed after a refresh: new readings are no reason to skip a review.
    world.readings = { phase_a: { value: 2, connected: true } };
    t = 6000;
    await panel.poller.now();
    await vi.waitFor(() => expect(readback('phase_a')).toBe('2'));
    expect(panel.state.approval).toMatchObject({ revision: 'rev-2', needsReview: true });
    expect(start()?.disabled).toBe(true);

    // Reviewed, then refreshed: it stays reviewed, and Start is the same button.
    /** @type {HTMLButtonElement} */ (document.querySelector('#reviewed-btn')).click();
    t = 7000;
    await panel.poller.now(); // the poll that adopts the reviewed card; no read is due
    const button = start();
    expect(button?.disabled).toBe(false);
    world.readings = { phase_a: { value: 3, connected: true } };
    t = 12000;
    await panel.poller.now();
    await vi.waitFor(() => expect(readback('phase_a')).toBe('3'));
    expect(panel.state.approval).toMatchObject({ revision: 'rev-2', needsReview: false });
    expect(start()).toBe(button);
    expect(button?.disabled).toBe(false);
  });
});
