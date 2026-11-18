/**
 * Unit tests for the knobs of an ended run (outcome-knobs.js): the range and
 * the search window each knob is drawn in, which marks and values appear for
 * the run and its outcome, the traces of the knobs at each evaluation, the
 * key to the drawings, and the rows of what changed per knob.
 */

import { describe, test, expect } from 'vitest';

import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import { position } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/knob-bar.js';
import {
  changeRows,
  knobsKey,
  knobsView,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-knobs.js';
import {
  knobChanges,
  outcomeOf,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-model.js';
import { jobFigures } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-model.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-model.js').MachineState} MachineState
 */

const INITIAL = { phase_a: 0, sol_b: 3, charge: 0.8 };

/**
 * What the outcome view gives the Knobs view for an ended present job with
 * its stored config: the job's records and windows, its figures and what
 * the outcome proves is on the machine.
 *
 * @param {string} status
 * @param {{ job?: Record<string, unknown>, config?: Record<string, unknown>, env?: any }} [options]
 */
function ended(status, { job = {}, config = {}, env = envDetails() } = {}) {
  const j = runningJob({ status, snapshots: { initial: INITIAL }, ...job });
  /** @type {any} */
  const c = storedConfig({ objective_direction: 'MAXIMIZE', ...config });
  const figures = jobFigures(j, c);
  const { machine } = outcomeOf({ status, job: j, config: c, figures });
  return {
    variables: /** @type {string[]} */ (c.variables),
    config: c,
    bounds: /** @type {Record<string, number[]>|null} */ (j.variable_bounds || null),
    env,
    figures,
    machine,
    records: /** @type {Record<string, unknown>[]} */ (j.data),
  };
}

/** @param {HTMLElement} root  every number drawn in the knob traces */
const traceCoordinates = (root) =>
  Array.from(root.querySelectorAll('.opt-out-spark *')).flatMap((n) =>
    ['x1', 'x2', 'y1', 'y2', 'd'].map((a) => n.getAttribute(a)).filter((v) => v !== null)
  );

/**
 * @param {HTMLElement} root
 * @param {string} knob
 * @param {string} mark
 */
const markOf = (root, knob, mark) =>
  /** @type {HTMLElement|null} */ (
    root.querySelector(`.opt-out-knob[data-knob="${knob}"] [data-mark="${mark}"]`)
  );

/**
 * The values written beside a knob's bar.
 *
 * @param {HTMLElement} root
 * @param {string} knob
 */
const valuesOf = (root, knob) =>
  Array.from(
    root.querySelectorAll(`.opt-out-knob[data-knob="${knob}"] .opt-out-knob-vals span`)
  ).map((n) => n.textContent);

/** @param {Partial<MachineState>} over @returns {MachineState} */
const machineState = (over = {}) => ({
  tone: 'ok',
  text: '',
  written: true,
  values: null,
  unapplied: null,
  ...over,
});

/** @param {HTMLElement} legend */
const keysOf = (legend) =>
  Array.from(legend.querySelectorAll('[data-key]')).map(
    (n) => /** @type {HTMLElement} */ (n).dataset.key
  );

describe('knobsView', () => {
  test('the knobs of the current job are drawn in the windows the run used', () => {
    // The service clamps and snaps the stored windows; variable_bounds are the ones it used.
    const variable_bounds = { phase_a: [-9, 9], sol_b: [2, 4] };
    const view = knobsView(ended('completed', { job: { variable_bounds } }));
    const win = /** @type {HTMLElement} */ (markOf(view, 'phase_a', 'window'));
    const safe = envDetails().variables.phase_a;
    expect(win.style.left).toBe(`${position(-9, safe.min_value, safe.max_value)?.toFixed(2)}%`);
  });

  test("without the run's windows, the stored config's window is drawn", () => {
    // The stored config's phase_a window is -18 … 18.
    const view = knobsView(ended('completed', { job: { variable_bounds: null } }));
    const win = /** @type {HTMLElement} */ (markOf(view, 'phase_a', 'window'));
    const safe = envDetails().variables.phase_a;
    expect(win.style.left).toBe(`${position(-18, safe.min_value, safe.max_value)?.toFixed(2)}%`);
  });

  test('without the safe ranges, each knob is drawn across its search window and no window is marked', () => {
    const view = knobsView(
      ended('completed', {
        job: { variable_bounds: { phase_a: [-9, 9], sol_b: [2, 4] } },
        env: null,
      })
    );
    expect(view.querySelector('[data-mark="window"]')).toBeNull();
    // phase_a started at 0, the middle of -9 … 9.
    expect(/** @type {HTMLElement} */ (markOf(view, 'phase_a', 'start')).style.left).toBe('50.00%');
  });

  test('a knob with neither a safe range nor a window gets no bar', () => {
    const view = knobsView(
      ended('completed', {
        job: { snapshots: { initial: { ...INITIAL, extra: 1 } } },
        config: { variables: ['phase_a', 'sol_b', 'extra'] },
        env: null,
      })
    );
    expect(view.querySelector('.opt-out-knob[data-knob="extra"] [role="img"]')).toBeNull();
    expect(view.querySelector('.opt-out-knob[data-knob="phase_a"] [role="img"]')).not.toBeNull();
  });

  test('a knob with no window and no value read draws no broken coordinates', () => {
    const view = knobsView(
      ended('completed', {
        job: { snapshots: { initial: { ...INITIAL, extra: 1 } } },
        config: { variables: ['phase_a', 'sol_b', 'extra'] },
      })
    );
    expect(view.querySelectorAll('.opt-out-trace')).toHaveLength(3);
    expect(traceCoordinates(view).filter((v) => v.includes('NaN'))).toEqual([]);
  });

  test('a cancelled run marks no point as the one on the machine', () => {
    const view = knobsView(ended('cancelled'));
    expect(view.querySelectorAll('.opt-out-knob[data-knob]')).toHaveLength(2);
    expect(view.querySelector('[data-mark="current"]')).toBeNull();
  });

  test('without a value proven on the machine, each knob gives its last evaluation recorded', () => {
    const view = knobsView(ended('cancelled'));
    // The last record has phase_a -9.
    expect(valuesOf(view, 'phase_a')).toEqual([
      'start 0\u00a0deg',
      'best 9\u00a0deg',
      'last -9\u00a0deg',
    ]);
  });

  test('the best point applied is marked on each knob as the value on the machine', () => {
    const view = knobsView(ended('completed'));
    expect(view.querySelectorAll('[data-mark="current"]')).toHaveLength(2);
    // phase_a: the best (evaluation 2) is 9, on a -180 … 180 range.
    const current = /** @type {HTMLElement} */ (markOf(view, 'phase_a', 'current'));
    expect(current.style.left).toBe(`${position(9, -180, 180)?.toFixed(2)}%`);
    expect(valuesOf(view, 'phase_a')).toEqual([
      'start 0\u00a0deg',
      'best 9\u00a0deg',
      'now 9\u00a0deg',
    ]);
  });

  test('a value the run does not hold is left out of the values and the marks', () => {
    const view = knobsView(ended('cancelled', { job: { snapshots: {} } }));
    expect(markOf(view, 'phase_a', 'start')).toBeNull();
    expect(valuesOf(view, 'phase_a')).toEqual(['best 9\u00a0deg', 'last -9\u00a0deg']);
  });

  test('a run that names no knob draws no knob and no trace', () => {
    const view = knobsView({ ...ended('completed'), variables: [] });
    expect(view.querySelector('[data-knob]')).toBeNull();
    expect(view.querySelector('.opt-out-trace')).toBeNull();
  });

  test('the knobs at each evaluation need two evaluations recorded', () => {
    const one = ended('completed', { job: { data: [runningJob().data[0]] } });
    expect(knobsView(one).querySelector('.opt-out-trace')).toBeNull();
    const two = ended('completed', { job: { data: runningJob().data.slice(0, 2) } });
    expect(knobsView(two).querySelectorAll('.opt-out-trace')).toHaveLength(2);
  });

  test('each trace marks the best evaluation and draws one point per evaluation', () => {
    const view = knobsView(ended('completed'));
    const trace = /** @type {HTMLElement} */ (view.querySelector('.opt-out-trace'));
    const best = /** @type {Element} */ (trace.querySelector('.opt-out-spark-best'));
    // The best is evaluation 2 of a planned 10: the middle of its tenth of the width.
    expect(Number(best.getAttribute('x1'))).toBeCloseTo(15);
    const line = /** @type {Element} */ (trace.querySelector('.opt-out-spark-line'));
    expect((line.getAttribute('d') || '').split('L')).toHaveLength(3);
  });
});

describe('knobsKey', () => {
  test('names the safe range and the window only when the knobs are drawn on their safe ranges', () => {
    expect(keysOf(knobsKey(machineState(), true))).toEqual(['track', 'window', 'start', 'best']);
    expect(keysOf(knobsKey(machineState(), false))).toEqual(['start', 'best']);
  });

  test('names the value on the machine only when the outcome proves one', () => {
    const proven = machineState({ values: { phase_a: 9 } });
    expect(keysOf(knobsKey(proven, true))).toEqual(['track', 'window', 'start', 'best', 'current']);
    expect(keysOf(knobsKey(proven, false))).toEqual(['start', 'best', 'current']);
  });
});

describe('changeRows', () => {
  const changes = knobChanges(
    ['phase_a', 'sol_b'],
    { phase_a: 0, sol_b: 3 },
    { phase_a: 9, sol_b: 2 }
  );

  test('gives one row per change, with its value before and after', () => {
    const rows = Array.from(
      changeRows(changes, /** @type {any} */ (envDetails())).querySelectorAll('[data-knob]')
    ).map((r) => {
      const d = /** @type {HTMLElement} */ (r).dataset;
      return [d.knob, Number(d.from), Number(d.to)];
    });
    expect(rows).toEqual([
      ['phase_a', 0, 9],
      ['sol_b', 3, 2],
    ]);
    expect(changeRows([], null).querySelector('[data-knob]')).toBeNull();
  });

  test('names a knob by its description, with its channel beside it, and by its channel without one', () => {
    const root = changeRows(changes, /** @type {any} */ (envDetails()));
    const phase = /** @type {HTMLElement} */ (root.querySelector('[data-knob="phase_a"]'));
    expect(phase.textContent).toContain('Buncher phase');
    expect(phase.textContent).toContain('phase_a');
    const sol = /** @type {HTMLElement} */ (root.querySelector('[data-knob="sol_b"]'));
    expect(sol.textContent).toContain('sol_b');
    // Without the environment read, the channel is the name.
    const bare = /** @type {HTMLElement} */ (
      changeRows(changes, null).querySelector('[data-knob="phase_a"]')
    );
    expect(bare.textContent).not.toContain('Buncher phase');
    expect(bare.textContent).toContain('phase_a');
  });

  test('gives each change its signed difference, in the units of the knob', () => {
    const phase = /** @type {HTMLElement} */ (
      changeRows(changes, /** @type {any} */ (envDetails())).querySelector('[data-knob="phase_a"]')
    );
    expect(phase.textContent).toContain('deg');
    expect(phase.textContent).toContain('+9');
  });
});
