/**
 * Integration tests for the OPTIMIZER panel shell (panel.js) against its real
 * index.html markup and a stubbed sidecar: the panel follows `jobs/current`
 * to an agent's draft, and Start sends the revision of the config response
 * that rendered the approval surface — never one fetched at click time.
 */

import { readFileSync } from 'node:fs';
import { cwd } from 'node:process';

import { test, expect, beforeEach, vi } from 'vitest';

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
    'GET /environments/injector/variables': () => ({ phase_a: { value: 1, connected: true } }),
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

async function bootPanel() {
  const stub = sidecar();
  const timers = { setTimer: vi.fn(() => 1), clearTimer: vi.fn() };
  const panel = createPanel({
    api: createApi({ prefix: '', fetchImpl: stub.fetchImpl }),
    ...timers,
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
