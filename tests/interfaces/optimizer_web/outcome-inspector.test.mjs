/**
 * Unit tests for the inspector of an ended or past run (outcome-inspector.js):
 * the service's reasons for the end, the objective before the run and at its
 * best, what the run left on the machine, what changed there per knob (or
 * the last evaluation recorded, for a stopped run), the run's facts and
 * settings, and the "Run again" foot, held without a stored config.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import {
  NO_CONFIG_WHY,
  outcomeInspector,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-inspector.js';
import { outcomeOf } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-model.js';
import {
  jobFigures,
  resultFigures,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-model.js';
import { runTime } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/runs-list.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-inspector.js').Source} Source
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-model.js').Outcome} Outcome
 */

const INITIAL = { phase_a: 0, sol_b: 3, charge: 0.8 };
const RUN_ID = 'injector/2026-11-16_14-12-00';

/**
 * The source and the outcome of an ended present job with its stored config,
 * built the way the outcome view builds them.
 *
 * @param {string} status
 * @param {{ job?: Record<string, unknown>, config?: Record<string, unknown>|null }} [options]
 * @returns {{ source: Source, outcome: Outcome }}
 */
function presentRun(status, { job = {}, config = {} } = {}) {
  /** @type {any} */
  const j = runningJob({ status, snapshots: { initial: INITIAL }, ...job });
  /** @type {any} */
  const c = config ? storedConfig({ objective_direction: 'MAXIMIZE', ...config }) : null;
  const figures = jobFigures(j, c);
  /** @type {Source} */
  const source = {
    key: `job:${j.job_id}`,
    stored: false,
    runId: null,
    info: null,
    entry: null,
    job: j,
    status: j.status,
    config: c,
    figures,
    records: j.data || [],
    variables: c?.variables || j.variable_names || [],
    environment: 'injector',
    started: null,
    applied: null,
    bounds: j.variable_bounds || null,
  };
  const outcome = outcomeOf({ status: j.status, job: j, config: c, figures });
  return { source, outcome };
}

/**
 * The source and the outcome of a stored run whose job the service no longer
 * holds, built the way the outcome view builds them.
 *
 * @param {{ timestamp?: string|null }} [options]  the runs list's timestamp of the run
 * @returns {{ source: Source, outcome: Outcome }}
 */
function storedRun({ timestamp = '2026-11-16_14-12-00' } = {}) {
  const info = { run_id: RUN_ID, environment: 'injector', timestamp, job_id: 'job-old' };
  const details = {
    timestamp: RUN_ID,
    config: storedConfig({ objective_direction: 'MAXIMIZE', snapshots: { initial: INITIAL } }),
    best_point: { ...runningJob().data[1] },
    initial_point: INITIAL,
    data_records: runningJob().data,
  };
  /** @type {any} */
  const config = details.config;
  const figures = resultFigures(details);
  /** @type {Source} */
  const source = {
    key: `run:${RUN_ID}`,
    stored: true,
    runId: RUN_ID,
    info,
    entry: /** @type {any} */ ({ details, loading: false, error: null, job: null }),
    job: null,
    status: null,
    config,
    figures,
    records: details.data_records,
    variables: config.variables,
    environment: 'injector',
    started: runTime(info.timestamp),
    applied: details.best_point,
    bounds: null,
  };
  const outcome = outcomeOf({
    status: null,
    job: null,
    config,
    figures,
    applied: source.applied,
  });
  return { source, outcome };
}

/**
 * The inspector's parts for `source` and `outcome`, its body in one element.
 *
 * @param {{ source: Source, outcome: Outcome }} run
 */
function inspect({ source, outcome }) {
  const onRunAgain = vi.fn();
  const parts = outcomeInspector({
    source,
    outcome,
    objective: { title: 'Charge', units: '' },
    env: /** @type {any} */ (envDetails()),
    models: null,
    environmentLabel: null,
    onRunAgain,
  });
  const body = document.createElement('div');
  const nodes = /** @type {(Node|null|false)[]} */ (parts.body);
  body.append(.../** @type {Node[]} */ (nodes.filter(Boolean)));
  const foot = /** @type {HTMLButtonElement} */ (parts.foot);
  return { parts, body, foot, onRunAgain };
}

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

describe('what changed on the machine', () => {
  test('a completed run whose end action applies the best point shows each change', () => {
    const { body } = inspect(presentRun('completed'));
    const changes = body.querySelector('[data-section="changes"]');
    expect(changes?.textContent).toContain('What changed on the machine');
    expect(changeRows(body)).toEqual([
      ['phase_a', 0, 9],
      ['sol_b', 3, 2],
    ]);
  });

  test('with the end action "hold" it does not say the best point was applied', () => {
    const { body } = inspect(presentRun('completed', { config: { on_complete: 'hold' } }));
    expect(body.textContent).not.toMatch(/best point.*was applied/);
    // The machine holds the last point tried (evaluation 3).
    expect(changeRows(body)).toEqual([
      ['phase_a', 0, -9],
      ['sol_b', 3, 4],
    ]);
  });

  test('a readonly run lists no change', () => {
    const { body } = inspect(presentRun('completed', { config: { execution_mode: 'readonly' } }));
    expect(body.querySelector('[data-section="changes"]')).toBeNull();
  });

  test('a run with no value before it shows no change from before', () => {
    const { body } = inspect(presentRun('completed', { job: { snapshots: {} } }));
    expect(body.querySelector('[data-section="changes"]')).toBeNull();
  });

  test('a cancelled run shows its last evaluation recorded as such, not as the machine state', () => {
    const { body } = inspect(presentRun('cancelled'));
    const changes = body.querySelector('[data-section="changes"]');
    expect(changes?.textContent).toContain('At the last evaluation recorded');
    expect(changes?.textContent).not.toContain('What changed on the machine');
    expect(changeRows(body)).toEqual([
      ['phase_a', 0, -9],
      ['sol_b', 3, 4],
    ]);
  });

  test('a stopped run whose writes are not known to stand lists no knob value', () => {
    // A failed run that would have written the starting setpoints back may have failed in that write.
    const failed = inspect(presentRun('error', { config: { on_complete: 'restore' } }));
    expect(failed.body.querySelector('[data-section="changes"]')).toBeNull();
    // Cancelled before its writes were approved: nothing was written.
    const auth = { state: 'ended', reason: 'released', detail: 'The request is withdrawn.' };
    const released = inspect(presentRun('cancelled', { job: { write_authorization: auth } }));
    expect(released.body.querySelector('[data-section="changes"]')).toBeNull();
  });

  test('a run whose end is not known lists no change', () => {
    const { body } = inspect(storedRun());
    expect(body.querySelector('[data-section="changes"]')).toBeNull();
  });
});

describe('the end and the machine', () => {
  test('a failed run shows the service reasons, one by one', () => {
    const job = {
      error_message: 'A write was outside the approved range',
      write_authorization: { state: 'ended', reason: 'auth_expired', detail: null },
    };
    const { body } = inspect(presentRun('error', { job }));
    const reasons = Array.from(body.querySelectorAll('[data-part="reason"]')).map(
      (n) => n.textContent
    );
    expect(reasons[0]).toBe('A write was outside the approved range');
    expect(reasons[1]).toMatch(/auth_expired/);
  });

  test('a completed run shows no reason', () => {
    const { body } = inspect(presentRun('completed', { job: { error_message: 'stale' } }));
    expect(body.querySelector('[data-part="reason"]')).toBeNull();
  });

  test('says what the run left on the machine', () => {
    const { body } = inspect(presentRun('completed'));
    const machine = /** @type {HTMLElement} */ (body.querySelector('[data-part="machine"]'));
    expect(machine.textContent).toContain(presentRun('completed').outcome.machine.text);
  });

  test('an open job that is not the job of now has no machine banner', () => {
    const other = { job_id: 'other-1', status: 'created', environment_name: 'injector' };
    /** @type {Source} */
    const source = {
      key: 'job:other-1',
      stored: false,
      runId: null,
      info: null,
      entry: null,
      job: other,
      status: 'created',
      config: null,
      figures: null,
      records: [],
      variables: [],
      environment: 'injector',
      started: null,
      applied: null,
      bounds: null,
    };
    const outcome = outcomeOf({ status: 'created', job: other, config: null, figures: null });
    const { body, parts } = inspect({ source, outcome });
    expect(body.querySelector('[data-part="machine"]')).toBeNull();
    expect(parts.title).toBe('Run');
  });

  test('a run whose end is not known has no machine banner', () => {
    const { body } = inspect(storedRun());
    expect(body.querySelector('[data-part="machine"]')).toBeNull();
  });

  test('shows the objective before the run and at its best, and nothing without figures', () => {
    const run = presentRun('completed');
    const card = /** @type {HTMLElement} */ (
      inspect(run).body.querySelector('[data-part="outcome"]')
    );
    // charge 0.8 before the run, 2.5 at its best (evaluation 2 of 3).
    expect(card.textContent).toContain('0.8 → 2.5');
    expect(card.textContent).toContain('evaluation 2 of 3');
    const none = { ...run, source: { ...run.source, figures: null, records: [] } };
    expect(inspect(none).body.querySelector('[data-part="outcome"]')).toBeNull();
  });
});

describe('the run and its settings', () => {
  test('a stored run is described by its start as written, else by its run id', () => {
    expect(inspect(storedRun()).parts.desc).toBe('2026-11-16 14:12');
    expect(inspect(storedRun({ timestamp: null })).parts.desc).toBe(RUN_ID);
  });

  test("the run's facts name the stored run and its status only for a stored run", () => {
    const stored = inspect(storedRun()).body;
    expect(stored.querySelector('[data-field="status"]')?.textContent).toContain('not known');
    expect(stored.querySelector('[data-field="run-id"]')?.textContent).toContain(RUN_ID);
    expect(stored.querySelector('[data-field="started"]')).not.toBeNull();
    const present = inspect(presentRun('completed')).body;
    expect(present.querySelector('[data-field="status"]')).toBeNull();
    expect(present.querySelector('[data-field="run-id"]')).toBeNull();
    expect(present.querySelector('[data-field="job"]')?.textContent).toContain('job-1234-abcd');
  });

  test('the settings of the run are shown only with its stored config', () => {
    expect(
      inspect(presentRun('completed')).body.querySelector('[data-part="settings"]')
    ).not.toBeNull();
    const bare = inspect(presentRun('completed', { config: null })).body;
    expect(bare.querySelector('[data-part="settings"]')).toBeNull();
    expect(bare.querySelector('[data-field="method"]')).toBeNull();
  });
});

describe('Run again', () => {
  test('the foot calls onRunAgain', () => {
    const { foot, onRunAgain } = inspect(presentRun('completed'));
    expect(foot.disabled).toBe(false);
    foot.click();
    expect(onRunAgain).toHaveBeenCalledTimes(1);
  });

  test('without a stored config, Run again is held and says why', () => {
    const { foot, onRunAgain } = inspect(presentRun('completed', { config: null }));
    expect(foot.disabled).toBe(true);
    expect(foot.title).toMatch(/not read/);
    expect(foot.title).toBe(NO_CONFIG_WHY);
    foot.click();
    expect(onRunAgain).not.toHaveBeenCalled();
  });
});
