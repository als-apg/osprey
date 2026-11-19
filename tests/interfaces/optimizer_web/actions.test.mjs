/**
 * Behaviour tests for the panel's actions (actions.js): what a call sends to
 * the sidecar and what the store learns from the answer.
 */

import { test, expect, vi } from 'vitest';

import { storedConfig } from './_fixtures.mjs';
import { createActions } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/actions.js';
import { createApi } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/api-client.js';
import { createStore } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

/**
 * Actions on a fresh store over a sidecar whose stored draft j1 has
 * `revision` and whose write answers wait for `hold`.
 *
 * @param {{ revision?: string, hold?: Promise<void> }} [options]
 */
function setup({ revision = 'rev-after-save', hold = Promise.resolve() } = {}) {
  /** @type {{ method: string, url: string, body: any }[]} */
  const calls = [];
  /** @type {typeof fetch} */
  const fetchImpl = async (input, init) => {
    const method = init?.method || 'GET';
    const url = String(input);
    calls.push({ method, url, body: init?.body ? JSON.parse(String(init.body)) : null });
    if (method !== 'GET') {
      await hold;
      return new Response(JSON.stringify({ job_id: 'j1', status: 'created' }), { status: 200 });
    }
    return new Response(JSON.stringify(storedConfig({ n_iterations: 50 })), {
      status: 200,
      headers: { 'X-Config-Revision': revision },
    });
  };
  const store = createStore();
  const actions = createActions({
    api: createApi({ prefix: '', fetchImpl }),
    store,
    refresh: vi.fn(async () => {}),
  });
  const writes = () => calls.filter((c) => c.method !== 'GET');
  return { store, actions, calls, writes };
}

test('run again from a stored run sends only fields the service accepts', async () => {
  const { store, actions, writes } = setup();
  // A stored run's config carries the initial point the service added to it.
  const stored = /** @type {any} */ ({ ...storedConfig(), snapshots: { initial: { phase_a: 0 } } });
  actions.runAgain(stored);
  const from = /** @type {Record<string, unknown>} */ (store.state.setup?.from);
  expect('snapshots' in from).toBe(false);
  await actions.saveDraft(/** @type {any} */ ({ ...from }), 'create');
  expect(Object.keys(writes()[0].body)).not.toContain('snapshots');
});

test('a second save while one is in flight sends nothing', async () => {
  /** @type {() => void} */
  let release = () => {};
  const hold = new Promise((resolve) => (release = () => resolve(undefined)));
  const { actions, writes } = setup({ hold });
  actions.newRun();
  const first = actions.saveDraft(storedConfig(), 'create');
  expect(await actions.saveDraft(storedConfig(), 'create')).toBe(false);
  release();
  expect(await first).toBe(true);
  expect(writes()).toHaveLength(1);
});

test('after an update the store holds the revision the service stored, before the next poll', async () => {
  const { store, actions } = setup({ revision: 'rev-after-save' });
  store.dispatch({ type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 0 });
  store.dispatch({
    type: 'poll/config',
    jobId: 'j1',
    revision: 'rev-before',
    config: storedConfig(),
  });
  actions.editDraft();
  expect(await actions.saveDraft(storedConfig({ n_iterations: 50 }), 'update')).toBe(true);
  expect(store.state.approval).toMatchObject({ jobId: 'j1', revision: 'rev-after-save' });
  // The operator's own change must be looked at again before Start.
  expect(store.state.approval.needsReview).toBe(true);
});

/**
 * Actions over a sidecar that answers each GET path from `answers` (a
 * status and a body; each path answers the next of its list, then the last).
 *
 * @param {Record<string, [number, unknown][]>} answers
 */
function routed(answers) {
  /** @type {string[]} */
  const reads = [];
  /** @type {typeof fetch} */
  const fetchImpl = async (input) => {
    const path = String(input).replace(/^\//, '').split('?')[0];
    reads.push(path);
    const list = answers[path] || [[404, { detail: 'not found' }]];
    const [status, body] = list.length > 1 ? /** @type {any} */ (list.shift()) : list[0];
    return new Response(JSON.stringify(body), { status });
  };
  const store = createStore();
  const actions = createActions({
    api: createApi({ prefix: '', fetchImpl }),
    store,
    refresh: vi.fn(async () => {}),
  });
  return { store, actions, reads };
}

test('an environment that could not be read is read again when asked; one that was read is kept', async () => {
  const { store, actions, reads } = routed({
    'environments/injector': [
      [503, { detail: 'timed out' }],
      [200, { name: 'injector', variables: {} }],
    ],
    'environments/injector/presets': [[200, {}]],
  });
  await actions.loadEnvironment('injector');
  expect(store.state.envs.injector).toMatchObject({ details: null, error: 'timed out' });
  await actions.loadEnvironment('injector');
  expect(store.state.envs.injector).toMatchObject({ details: { name: 'injector' }, error: null });
  const count = reads.length;
  await actions.loadEnvironment('injector');
  expect(reads).toHaveLength(count);
});

test('a set-up that opens after a failed catalogue read reads it again', async () => {
  const { store, actions, reads } = routed({
    environments: [
      [503, { detail: 'down' }],
      [200, [{ name: 'injector' }]],
    ],
    models: [[200, []]],
  });
  await actions.loadCatalog();
  expect(store.state.environments).toEqual({ list: null, error: 'down' });
  actions.newRun();
  await vi.waitFor(() => expect(store.state.environments.list).toEqual([{ name: 'injector' }]));
  expect(store.state.environments.error).toBeNull();
  expect(reads.filter((p) => p === 'environments')).toHaveLength(2);
  // A set-up after a read that answered reads nothing more.
  actions.closeSetup();
  actions.newRun();
  expect(reads.filter((p) => p === 'environments')).toHaveLength(2);
});

test('a save refused with a 422 list holds one issue per item', async () => {
  const detail = [
    { loc: ['body', 'n_iterations'], msg: 'Input should be greater than 0' },
    { loc: ['body', 'objective_name'], msg: 'Field required' },
  ];
  /** @type {typeof fetch} */
  const fetchImpl = async () => new Response(JSON.stringify({ detail }), { status: 422 });
  const store = createStore();
  const actions = createActions({
    api: createApi({ prefix: '', fetchImpl }),
    store,
    refresh: vi.fn(async () => {}),
  });
  actions.newRun();
  expect(await actions.saveDraft(storedConfig(), 'create')).toBe(false);
  expect(store.state.setup?.issues).toEqual([
    'n_iterations: Input should be greater than 0',
    'objective_name: Field required',
  ]);
});
