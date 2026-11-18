/**
 * Tests for the outcome view (outcome-view.js): what the band of the
 * completed, failed and cancelled screens and of a past run says for a given
 * state, which actions it offers and which action a click calls, the switch
 * between the chart area's views, the reads it asks for and what it redraws.
 * "Run again" opens a set-up and starts nothing; there is no Restore and no
 * run control; the service's reason for a stop is shown. The inspector's
 * parts are tested in outcome-inspector.test.mjs, the Knobs view in
 * outcome-knobs.test.mjs.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import { position } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/knob-bar.js';
import { mountOutcomeView } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-view.js';
import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';
import { STOP_WORDS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent} StoreEvent
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/view-host.js').ViewContext} ViewContext
 */

const INITIAL = { phase_a: 0, sol_b: 3, charge: 0.8 };
const RUN_ID = 'injector/2026-11-16_14-12-00';

function context() {
  const stage = document.createElement('section');
  const inspector = document.createElement('aside');
  const setPrimary = vi.fn();
  const actions = {
    runAgain: vi.fn(),
    select: vi.fn(),
    loadEnvironment: vi.fn(async () => {}),
    loadResult: vi.fn(async () => {}),
    start: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    requestCancel: vi.fn(),
    confirmCancel: vi.fn(),
  };
  /** @type {ViewContext} */
  const ctx = {
    stage,
    inspector,
    actions: /** @type {any} */ (actions),
    shell: {
      setPrimary,
      fillInspector: (parts) => inspector.replaceChildren(...inspectorFrame(parts, () => {})),
      setDrawer: vi.fn(),
      setDrawerTop: vi.fn(),
    },
  };
  return { ctx, stage, inspector, setPrimary, actions };
}

/** @param {StoreEvent[]} events */
const at = (events) => {
  const state = events.reduce(reduce, initialState());
  return { state, screen: deriveScreen(state) };
};

/**
 * The events of an ended present job with its stored config and environment.
 *
 * @param {string} status
 * @param {{ job?: Record<string, unknown>, config?: Record<string, unknown>|null }} [options]
 * @returns {StoreEvent[]}
 */
function endedRun(status, { job = {}, config = {} } = {}) {
  const j = runningJob({ status, snapshots: { initial: INITIAL }, ...job });
  /** @type {StoreEvent[]} */
  const events = [
    { type: 'poll/job', job: j, at: 0 },
    { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
  ];
  if (config) {
    events.push({
      type: 'poll/config',
      jobId: j.job_id,
      revision: 'rev-1',
      config: storedConfig({ objective_direction: 'MAXIMIZE', ...config }),
    });
  }
  return events;
}

/** @param {ReturnType<typeof context>} c @param {StoreEvent[]} events */
function show(c, events) {
  const view = mountOutcomeView(c.ctx);
  const { state, screen } = at(events);
  view.update(state, screen);
  return { view, state, screen };
}

/** @param {HTMLElement} root */
const banner = (root) => /** @type {HTMLElement} */ (root.querySelector('[data-part="outcome"]'));

/** @param {HTMLElement} root */
const buttonTexts = (root) => Array.from(root.querySelectorAll('button')).map((b) => b.textContent);

/**
 * Each knob of "what changed": its name, the value before and after.
 *
 * @param {HTMLElement} root
 */
const changeRows = (root) =>
  Array.from(root.querySelectorAll('[data-section="changes"] [data-knob]')).map((r) => {
    const d = /** @type {HTMLElement} */ (r).dataset;
    return [d.knob, Number(d.from), Number(d.to)];
  });

describe('a completed run', () => {
  test('says the best point was applied when its end action applies it', () => {
    const c = context();
    const { screen } = show(c, endedRun('completed'));
    expect(screen.id).toBe('completed');
    expect(banner(c.stage).textContent).toMatch(/best point \(evaluation 2\) was applied/);
  });

  test('with the end action "hold" it does not say the best point was applied', () => {
    const c = context();
    show(c, endedRun('completed', { config: { on_complete: 'hold' } }));
    expect(banner(c.stage).textContent).not.toMatch(/applied/);
  });

  test('a readonly run says nothing was written', () => {
    const c = context();
    show(c, endedRun('completed', { config: { execution_mode: 'readonly' } }));
    expect(banner(c.stage).textContent).toMatch(/Nothing was written/);
  });

  test('a run with no value before it shows no start and no change in its facts', () => {
    const c = context();
    show(c, endedRun('completed', { job: { snapshots: {} } }));
    expect(c.stage.querySelector('[data-fact="before"]')).toBeNull();
    expect(c.stage.querySelector('[data-fact="change"]')).toBeNull();
  });

  test('the knobs view is given the windows the job used (variable_bounds)', () => {
    const c = context();
    // The service clamps and snaps the stored windows; variable_bounds are the ones it used.
    const variable_bounds = { phase_a: [-9, 9], sol_b: [2, 4] };
    show(c, endedRun('completed', { job: { variable_bounds } }));
    /** @type {HTMLElement} */ (c.stage.querySelector('[data-view="knobs"]')).click();
    const win = /** @type {HTMLElement} */ (
      c.stage.querySelector('.opt-out-knob[data-knob="phase_a"] [data-mark="window"]')
    );
    const safe = envDetails().variables.phase_a;
    expect(win.style.left).toBe(`${position(-9, safe.min_value, safe.max_value)?.toFixed(2)}%`);
  });
});

describe('a cancelled run', () => {
  test('says the machine keeps the last point written and nothing is restored, and never that the best was applied', () => {
    const c = context();
    show(c, endedRun('cancelled'));
    const text = banner(c.stage).textContent || '';
    expect(text).toContain(STOP_WORDS.after);
    expect(text).toMatch(/was not applied/);
    expect(text).not.toMatch(/was applied/);
  });

  test('a run cancelled before its writes were approved says nothing was written', () => {
    const c = context();
    const auth = { state: 'ended', reason: 'released', detail: 'The request is withdrawn.' };
    show(c, endedRun('cancelled', { job: { data: [], write_authorization: auth } }));
    expect(banner(c.stage).textContent).toMatch(/Nothing was written/);
  });
});

describe('a failed run', () => {
  test('shows the service reason in the stage, as an alert', () => {
    const c = context();
    const job = {
      error_message: 'A write was outside the approved range',
      write_authorization: { state: 'ended', reason: 'auth_expired', detail: null },
    };
    show(c, endedRun('error', { job }));
    const b = banner(c.stage);
    expect(b.getAttribute('role')).toBe('alert');
    expect(b.textContent).toContain('A write was outside the approved range');
    expect(b.textContent).toContain('auth_expired');
  });
});

describe('the actions of an outcome', () => {
  test('Run again opens a set-up from the stored config; nothing starts, nothing restores', () => {
    const c = context();
    const { state } = show(c, endedRun('cancelled'));
    const spec = c.setPrimary.mock.calls.at(-1)?.[0];
    expect(spec).toMatchObject({ label: 'Run again', disabled: false });
    spec.onClick();
    /** @type {HTMLElement} */ (c.inspector.querySelector('#run-again-btn')).click();
    expect(c.actions.runAgain).toHaveBeenCalledTimes(2);
    expect(c.actions.runAgain).toHaveBeenCalledWith(state.approval.config);
    for (const name of ['start', 'pause', 'resume', 'requestCancel', 'confirmCancel']) {
      expect(c.actions[/** @type {'start'} */ (name)]).not.toHaveBeenCalled();
    }
    const all = [...buttonTexts(c.stage), ...buttonTexts(c.inspector)].join(' | ');
    expect(all).not.toMatch(/Restore|Pause|Resume|Cancel/);
  });

  test('without a stored config, Run again is held and says why', () => {
    const c = context();
    show(c, endedRun('completed', { config: null }));
    expect(c.setPrimary.mock.calls.at(-1)?.[0]).toMatchObject({
      label: 'Run again',
      disabled: true,
    });
    // No end action is claimed without the config.
    expect(banner(c.stage).textContent).not.toMatch(/applied/);
  });

  test('the view switch shows the knobs and the evaluations of the run', () => {
    const c = context();
    show(c, endedRun('completed'));
    const switchTo = (/** @type {string} */ id) =>
      /** @type {HTMLElement} */ (c.stage.querySelector(`[data-view="${id}"]`)).click();
    switchTo('knobs');
    const knobs = Array.from(c.stage.querySelectorAll('.opt-out-knob[data-knob]')).map(
      (n) => /** @type {HTMLElement} */ (n).dataset.knob
    );
    expect(knobs).toEqual(['phase_a', 'sol_b']);
    switchTo('evaluations');
    expect(c.stage.querySelectorAll('tbody tr')).toHaveLength(3);
    switchTo('objective');
    expect(c.stage.querySelector('svg')).not.toBeNull();
  });

  test('a poll with nothing new redraws nothing', () => {
    const c = context();
    const { view, state, screen } = show(c, endedRun('completed'));
    const drawn = banner(c.stage);
    const foot = c.inspector.querySelector('#run-again-btn');
    view.update({ ...state }, { ...screen });
    expect(banner(c.stage)).toBe(drawn);
    expect(c.inspector.querySelector('#run-again-btn')).toBe(foot);
  });
});

describe('a job picked in the runs list', () => {
  test('a run that ended beside a newer draft shows its own outcome and runs again from its config', () => {
    const c = context();
    const run = runningJob({ status: 'completed', snapshots: { initial: INITIAL } });
    const config = storedConfig({ objective_direction: 'MAXIMIZE', on_complete: 'hold' });
    const draft = { job_id: 'draft-2', status: 'created', environment_name: 'injector' };
    const { screen } = show(c, [
      { type: 'poll/job', job: draft, at: 0 },
      { type: 'poll/active', job: run, config, at: 0 },
      { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
      { type: 'select', selection: { kind: 'job', jobId: run.job_id } },
    ]);
    expect(screen.id).toBe('result');
    const text = banner(c.stage).textContent || '';
    expect(text).toMatch(/last point tried/);
    expect(text).not.toMatch(/was applied/);
    c.setPrimary.mock.calls.at(-1)?.[0].onClick();
    expect(c.actions.runAgain).toHaveBeenCalledWith(config);
  });

  test('an open job that is not the job of now says so; with no config, Run again is held', () => {
    const c = context();
    const other = { job_id: 'other-1', status: 'created', environment_name: 'injector' };
    show(c, [
      { type: 'poll/job', job: null, at: 0 },
      { type: 'jobs', list: [other], error: null },
      { type: 'select', selection: { kind: 'job', jobId: 'other-1' } },
    ]);
    expect(banner(c.stage).textContent).toMatch(/This job is created/);
    expect(c.setPrimary.mock.calls.at(-1)?.[0]).toMatchObject({ disabled: true });
  });
});

describe('a stored run', () => {
  const info = {
    run_id: RUN_ID,
    environment: 'injector',
    timestamp: '2026-11-16_14-12-00',
    objective_name: 'charge',
    has_model: false,
    job_id: 'job-old',
  };
  /** @param {Record<string, unknown>} details @param {any} [job] @returns {StoreEvent[]} */
  const storedRun = (details, job = null) => [
    { type: 'poll/job', job: null, at: 0 },
    { type: 'results', list: [info], error: null },
    { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
    { type: 'select', selection: { kind: 'run', runId: RUN_ID } },
    { type: 'past/loaded', runId: RUN_ID, details, job, error: null },
  ];
  // As the service sends a stored run: the config carries `snapshots` and a
  // filled direction; best_point is the full best row of a completed run.
  const details = (/** @type {Record<string, unknown>} */ overrides = {}) => ({
    timestamp: RUN_ID,
    config: storedConfig({ objective_direction: 'MAXIMIZE', snapshots: { initial: INITIAL } }),
    best_point: { ...runningJob().data[1] },
    initial_point: INITIAL,
    data_records: runningJob().data,
    ...overrides,
  });

  test('a completed stored run shows the best feasible point the service applied', () => {
    // MAXIMIZE charge with loss < 1: evaluation 2 has the highest charge but
    // breaks the constraint; the service applied evaluation 3 and stored it.
    const records = [
      { phase_a: 0, sol_b: 3, charge: 1, loss: 0.5 },
      { phase_a: 5, sol_b: 2, charge: 9, loss: 3 },
      { phase_a: -9, sol_b: 4, charge: 4, loss: 0.2 },
    ];
    const config = storedConfig({
      objective_direction: 'MAXIMIZE',
      constraints: { loss: { type: 'LESS_THAN', value: 1 } },
      snapshots: { initial: INITIAL },
    });
    const c = context();
    show(
      c,
      storedRun(details({ config, data_records: records, best_point: { ...records[2] } }), {
        job_id: 'job-old',
        status: 'completed',
      })
    );
    expect(changeRows(c.inspector)).toEqual([
      ['phase_a', 0, -9],
      ['sol_b', 3, 4],
    ]);
  });

  test('a run still under way is never shown as a past run: no figures, and a way to its run page', () => {
    const c = context();
    const running = { job_id: 'job-old', status: 'running', write_authorization: null };
    show(c, storedRun(details({ data_records: runningJob().data.slice(0, 1) }), running));
    expect(c.stage.querySelector('[data-fact]')).toBeNull();
    expect(c.stage.querySelector('svg')).toBeNull();
    expect(c.inspector.querySelector('[data-part="outcome"]')).toBeNull();
    /** @type {HTMLElement} */ (c.stage.querySelector('#show-run-btn')).click();
    expect(c.actions.select).toHaveBeenCalledWith({ kind: 'job', jobId: 'job-old' });
  });

  test('without its status, it says how it ended is not known and claims nothing about the machine', () => {
    const c = context();
    const { screen } = show(c, storedRun(details()));
    expect(screen.id).toBe('result');
    const text = banner(c.stage).textContent || '';
    expect(text).toMatch(/not known/);
    expect(text).not.toMatch(/was applied|machine holds|written back/);
    expect(c.stage.querySelector('svg')).not.toBeNull();
  });

  test('with its status still held, it says what the end action left', () => {
    const c = context();
    show(c, storedRun(details(), { job_id: 'job-old', status: 'completed' }));
    expect(banner(c.stage).textContent).toMatch(/was applied/);
  });

  test('with no evaluation stored, one sentence says so and the facts it has remain', () => {
    const c = context();
    show(c, storedRun(details({ data_records: [], best_point: {} })));
    expect(c.stage.textContent).toContain('No evaluation was stored for this run.');
    expect(c.stage.querySelector('svg')).toBeNull();
    expect(c.stage.querySelector('[data-fact="before"]')).not.toBeNull();
    expect(c.stage.querySelector('[data-fact="best"]')).toBeNull();
  });

  test('offers the way back to now, and Run again from the stored config', () => {
    const c = context();
    show(c, storedRun(details()));
    /** @type {HTMLElement} */ (c.stage.querySelector('#back-to-now-btn')).click();
    expect(c.actions.select).toHaveBeenCalledWith({ kind: 'present' });
    c.setPrimary.mock.calls.at(-1)?.[0].onClick();
    expect(c.actions.runAgain).toHaveBeenCalledWith(details().config);
  });

  test('a stored run not read yet is asked for once, however often the view updates', () => {
    const c = context();
    const view = mountOutcomeView(c.ctx);
    const { state, screen } = at(storedRun(details()).slice(0, 4));
    view.update(state, screen);
    view.update({ ...state }, { ...screen });
    expect(c.actions.loadResult).toHaveBeenCalledTimes(1);
    expect(c.stage.textContent).toContain('Reading the stored run');
  });

  test('a stored run whose job is still under way is read again when the runs list is, not on every poll', () => {
    const c = context();
    const view = mountOutcomeView(c.ctx);
    const running = { job_id: 'job-old', status: 'running', write_authorization: null };
    const events = storedRun(details(), running);
    const first = at(events);
    view.update(first.state, first.screen);
    view.update({ ...first.state }, { ...first.screen });
    expect(c.actions.loadResult).toHaveBeenCalledTimes(1);
    const relisted = at([...events, { type: 'results', list: [info], error: null }]);
    view.update(relisted.state, relisted.screen);
    expect(c.actions.loadResult).toHaveBeenCalledTimes(2);
  });
});
