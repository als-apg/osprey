/**
 * Unit tests for a run's figures (run-model.js): only fields the service sends
 * become a figure; a missing one is null.
 */

import { describe, test, expect } from 'vitest';

import { runningJob, storedConfig } from './_fixtures.mjs';
import {
  MIN_TIMED_EVALUATIONS,
  figures,
  jobFigures,
  minutesLeft,
  noArrivals,
  recordArrivals,
  resultFigures,
  runningBest,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/run-model.js';

describe('runningBest', () => {
  test('follows the direction, skips nulls, refuses an unknown direction', () => {
    expect(runningBest([null, 2, 1, 3], 'MAXIMIZE')).toEqual([null, 2, 2, 3]);
    expect(runningBest([2, 1, null, 3], 'MINIMIZE')).toEqual([2, 1, 1, 1]);
    expect(runningBest([1, 2], null)).toBeNull();
  });
});

describe('figures', () => {
  const base = { objective: 'q', direction: 'MAXIMIZE', total: 5, initial: { q: 2, a: 0 } };

  test('best so far, the evaluation that found it, and the change from before the run', () => {
    const f = figures({
      ...base,
      data: [
        { q: 1, a: 1 },
        { q: 3, a: 2 },
        { q: 3, a: 9 },
        { q: 2, a: 4 },
      ],
    });
    expect(f).toMatchObject({
      done: 4,
      total: 5,
      best: 3,
      bestIndex: 1,
      before: 2,
      change: 1,
      changePct: 50,
    });
    expect(f.bestRecord).toEqual({ q: 3, a: 2 });
    expect(f.lastRecord).toEqual({ q: 2, a: 4 });
  });

  test('no value before the run: no before, no change', () => {
    const f = figures({ ...base, initial: null, data: [{ q: 1 }] });
    expect(f).toMatchObject({ before: null, change: null, changePct: null, best: 1 });
  });

  test('a value of zero before the run gives a change but no percentage', () => {
    const f = figures({ ...base, initial: { q: 0 }, data: [{ q: 1 }] });
    expect(f).toMatchObject({ before: 0, change: 1, changePct: null });
  });

  test('an unknown direction gives no best at all', () => {
    const f = figures({ ...base, direction: null, data: [{ q: 1 }, { q: 2 }] });
    expect(f).toMatchObject({ bests: null, best: null, bestIndex: null, change: null });
    expect(f.values).toEqual([1, 2]);
  });

  test('unreadable objectives are null, and a run with none read has no best', () => {
    const f = figures({ ...base, data: [{ q: null }, { q: 'nan' }] });
    expect(f).toMatchObject({ done: 2, values: [null, null], best: null, bestIndex: null });
  });

  test('no data: nothing done, no record', () => {
    expect(figures({ ...base, data: null })).toMatchObject({
      done: 0,
      lastRecord: null,
      bestRecord: null,
    });
  });
});

describe('the time left', () => {
  test('is timed only from evaluations the panel saw arrive', () => {
    let a = recordArrivals(noArrivals(), 'j', 4, 0);
    expect(a).toEqual({ jobId: 'j', seen: 4, first: null, last: null });
    expect(recordArrivals(a, 'j', 4, 10)).toBe(a);
    a = recordArrivals(a, 'j', 5, 60000);
    a = recordArrivals(a, 'j', 7, 180000);
    expect(a).toMatchObject({ first: { count: 5, at: 60000 }, last: { count: 7, at: 180000 } });
    expect(minutesLeft(a, 7, 10)).toBeNull();
    a = recordArrivals(a, 'j', 8, 240000);
    expect(MIN_TIMED_EVALUATIONS).toBe(3);
    expect(minutesLeft(a, 8, 10)).toBe(2);
  });

  test('starts again for another job or a count that went down', () => {
    const a = recordArrivals(recordArrivals(noArrivals(), 'j', 1, 0), 'j', 2, 5);
    expect(recordArrivals(a, 'k', 2, 9)).toEqual({ jobId: 'k', seen: 2, first: null, last: null });
    expect(recordArrivals(a, 'j', 0, 9)).toEqual({ jobId: 'j', seen: 0, first: null, last: null });
  });

  test('none with no total, or with nothing left', () => {
    const a = { jobId: 'j', seen: 0, first: { count: 1, at: 0 }, last: { count: 5, at: 4000 } };
    expect(minutesLeft(a, 5, null)).toBeNull();
    expect(minutesLeft(a, 5, 5)).toBeNull();
    expect(minutesLeft(a, 5, 6)).toBe(1);
  });
});

describe('jobFigures and resultFigures', () => {
  test("a job's own total wins; before it reports one, the stored config plans it", () => {
    expect(jobFigures(runningJob(), storedConfig()).total).toBe(10);
    const planned = jobFigures(runningJob({ total_evaluations: null }), storedConfig());
    expect(planned.total).toBe(33); // 30 iterations + max(2, 2 + 1) initial samples
    expect(jobFigures(runningJob({ total_evaluations: null }), null).total).toBeNull();
  });

  test('the objective and direction fall back to the stored config', () => {
    const job = { job_id: 'j', status: 'running', data: [{ charge: 2, phase_a: 0, sol_b: 3 }] };
    const f = jobFigures(job, storedConfig({ objective_direction: 'MAXIMIZE' }));
    expect(f).toMatchObject({ objective: 'charge', direction: 'MAXIMIZE', best: 2 });
  });

  test("the job's initial snapshot is the value before the run", () => {
    expect(jobFigures(runningJob({ snapshots: { initial: { charge: 0.5 } } }), null).before).toBe(
      0.5
    );
    expect(jobFigures(runningJob({ snapshots: {} }), null).before).toBeNull();
  });

  test('a stored run: its records, its initial point and its planned total', () => {
    const f = resultFigures({
      config: storedConfig({ objective_direction: 'MINIMIZE' }),
      initial_point: { charge: 4 },
      data_records: [
        { charge: 3, phase_a: 0, sol_b: 3 },
        { charge: 1, phase_a: 1, sol_b: 3 },
      ],
    });
    expect(f).toMatchObject({
      done: 2,
      total: 33,
      best: 1,
      before: 4,
      change: -3,
      minutesLeft: null,
    });
  });
});

describe('the best is the one the service applies', () => {
  // MAXIMIZE q under loss < 1, as `best_feasible_row` judges it.
  const config = storedConfig({
    objective_name: 'q',
    objective_direction: 'MAXIMIZE',
    variables: ['a'],
    constraints: { loss: { type: 'LESS_THAN', value: 1 } },
  });
  const data = [
    { a: 1, q: 1, loss: 0.5 },
    { a: 5, q: 9, loss: 3 },
    { a: 2, q: 4, loss: 0.2 },
    { a: 3, q: 8, loss: null },
    { a: null, q: 7, loss: 0.1 },
  ];

  test('an infeasible evaluation, a missing constraint reading or a non-finite knob is never the best', () => {
    const f = jobFigures({ job_id: 'j', status: 'running', data }, config);
    expect(f).toMatchObject({ best: 4, bestIndex: 2, bestRecord: { a: 2 } });
    expect(f.feasible).toEqual([true, false, true, false, true]);
    expect(f.bests).toEqual([1, 1, 4, 4, 4]);
    expect(f.values).toEqual([1, 9, 4, 8, 7]);
  });

  test('a stored run is judged under its own constraints; no feasible evaluation, no best', () => {
    const none = resultFigures({ config, data_records: [data[1], data[3]], initial_point: null });
    expect(none).toMatchObject({ best: null, bestIndex: null, bestRecord: null });
  });

  test('without constraints, feasibility is not judged', () => {
    expect(
      figures({ data, objective: 'q', direction: 'MAXIMIZE', total: 5, initial: null }).feasible
    ).toBeNull();
  });
});

test('time spent paused is not timed', () => {
  let a = recordArrivals(noArrivals(), 'j', 0, 0);
  for (const [n, at] of [
    [1, 60000],
    [2, 120000],
    [3, 180000],
    [4, 240000],
  ])
    a = recordArrivals(a, 'j', n, at);
  expect(minutesLeft(a, 4, 10)).toBe(6);
  // A pause of an hour, then the run goes on.
  a = recordArrivals(a, 'j', 4, 3840000, false);
  a = recordArrivals(a, 'j', 5, 3900000);
  expect(minutesLeft(a, 5, 10)).toBeNull();
  for (const [n, at] of [
    [6, 3960000],
    [7, 4020000],
    [8, 4080000],
  ])
    a = recordArrivals(a, 'j', n, at);
  expect(minutesLeft(a, 8, 10)).toBe(2);
});
