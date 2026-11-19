/**
 * Unit tests for outcome-model.js: what the panel says a run left on the
 * machine follows from the fields the service sent, and from nothing else.
 */

import { describe, test, expect } from 'vitest';

import { runningJob, storedConfig } from './_fixtures.mjs';
import {
  endReasons,
  knobChanges,
  leftOnMachine,
  outcomeOf,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/outcome-model.js';
import { STOP_WORDS } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';
import { jobFigures } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-model.js';

const INITIAL = { phase_a: 0, sol_b: 3, charge: 0.8 };
const VARIABLES = ['phase_a', 'sol_b'];

/** @param {string} status @param {Record<string, unknown>} [overrides] */
function ended(status, overrides = {}) {
  return runningJob({ status, snapshots: { initial: INITIAL }, ...overrides });
}

/**
 * @param {string} status
 * @param {Record<string, unknown>} [config]
 * @param {Record<string, unknown>} [job]
 */
function machine(status, config = {}, job = {}) {
  const j = ended(status, job);
  // The service fills objective_direction into every stored config.
  const c = storedConfig({ objective_direction: 'MAXIMIZE', ...config });
  return outcomeOf({ status, job: j, config: c, figures: jobFigures(j, c) }).machine;
}

describe('what a completed run left on the machine', () => {
  test('apply_best: the best point, with its values', () => {
    const m = machine('completed', { on_complete: 'apply_best' });
    expect(m.text).toMatch(/best point \(evaluation 2\) was applied/);
    expect(m.values).toEqual({ phase_a: 9, sol_b: 2 });
  });

  test('apply_best of a stored run: the knob values are the best point the service stored', () => {
    // The service applies its best FEASIBLE row and stores it as best_point;
    // here the unconstrained best record (evaluation 2) is not that row.
    const records = runningJob().data;
    const applied = { ...records[2] };
    const config = storedConfig({ objective_direction: 'MAXIMIZE', on_complete: 'apply_best' });
    const figures = jobFigures(ended('completed'), config);
    const m = outcomeOf({ status: 'completed', job: null, config, figures, applied }).machine;
    expect(m.values).toEqual({ phase_a: -9, sol_b: 4 });
    expect(m.written).toBe(true);
  });

  test('restore: the starting setpoints, never "applied"', () => {
    const m = machine('completed', { on_complete: 'restore' });
    expect(m.text).not.toMatch(/applied/);
    expect(m.values).toEqual({ phase_a: 0, sol_b: 3 });
  });

  test('hold: the last point tried, never "applied"', () => {
    const m = machine('completed', { on_complete: 'hold' });
    expect(m.text).not.toMatch(/applied/);
    expect(m.values).toEqual({ phase_a: -9, sol_b: 4 });
  });

  test('without a config read, no end action is claimed and no values are given', () => {
    const j = ended('completed');
    const m = outcomeOf({
      status: 'completed',
      job: j,
      config: null,
      figures: jobFigures(j, null),
    }).machine;
    expect(m.text).not.toMatch(/applied|written back|holds/);
    expect(m.values).toBeNull();
  });
});

describe('what a stopped run left on the machine', () => {
  test('cancelled: the last point written, nothing restored, the best not applied', () => {
    const m = machine('cancelled');
    expect(m.text).toContain(STOP_WORDS.after);
    expect(m.text).toMatch(/best point \(evaluation 2\) was not applied/);
    // A stop can land inside an evaluation after its write: no recorded point
    // is given as the machine's state.
    expect(m.values).toBeNull();
  });

  test('a failed run whose end action writes the start back: what the machine holds is not known', () => {
    const m = machine('error', { on_complete: 'restore' });
    expect(m.written).toBeNull();
    expect(m.values).toBeNull();
    expect(machine('error', { on_complete: 'apply_best' }).written).toBe(true);
  });

  test('a cancelled run leaves the best point unapplied only when its end action would apply it', () => {
    expect(machine('cancelled', { on_complete: 'apply_best' }).unapplied).toBe(2);
    expect(machine('cancelled', { on_complete: 'hold' }).unapplied).toBeNull();
    expect(machine('cancelled', { on_complete: 'restore' }).unapplied).toBeNull();
  });

  test('a stop before any evaluation names no point', () => {
    const m = machine('cancelled', {}, { data: [] });
    expect(m.text).toMatch(/No evaluation was recorded/);
    expect(m.values).toBeNull();
  });
});

describe('runs that wrote nothing', () => {
  test('a readonly run', () => {
    const m = machine('completed', { execution_mode: 'readonly' });
    expect(m.text).toMatch(/Nothing was written/);
    expect(m.written).toBe(false);
    expect(m.values).toBeNull();
  });

  test('a run whose write request ended before anyone approved it', () => {
    const auth = { state: 'ended', reason: 'released', detail: 'The request is withdrawn.' };
    const m = machine('cancelled', {}, { data: [], write_authorization: auth });
    expect(m.text).toMatch(/Nothing was written/);
  });
});

describe('a stored run without its status', () => {
  test('says that how it ended and what it left are not known', () => {
    const out = outcomeOf({ status: null, job: null, config: storedConfig(), figures: null });
    expect(out.kind).toBe('unknown');
    expect(out.machine.text).toMatch(/not known/);
    expect(out.machine.values).toBeNull();
    expect(
      leftOnMachine({ kind: 'unknown', config: null, figures: null, variables: [] }).written
    ).toBeNull();
  });
});

describe('the reasons for the end', () => {
  test('the job error and an authorization that ended with a reason', () => {
    const job = ended('error', {
      error_message: 'A write was outside the approved range',
      write_authorization: { state: 'ended', reason: 'auth_expired', detail: 'The lease lapsed.' },
    });
    const reasons = endReasons(job);
    expect(reasons[0]).toBe('A write was outside the approved range');
    expect(reasons[1]).toMatch(/auth_expired/);
    expect(reasons[1]).toMatch(/The lease lapsed/);
  });

  test('the chart marks where a stopped run ended', () => {
    const j = ended('cancelled');
    const c = storedConfig();
    const out = outcomeOf({ status: 'cancelled', job: j, config: c, figures: jobFigures(j, c) });
    expect(out.end).toEqual({ index: 2, label: 'cancelled', tone: 'warning' });
  });
});

test('knob changes need a value before and after', () => {
  expect(knobChanges(VARIABLES, { phase_a: 0 }, { phase_a: 9, sol_b: 2 })).toEqual([
    { name: 'phase_a', from: 0, to: 9, delta: 9 },
  ]);
  expect(knobChanges(VARIABLES, null, { phase_a: 9 })).toEqual([]);
});
