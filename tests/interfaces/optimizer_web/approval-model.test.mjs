/**
 * Unit tests for what the operator approves by clicking Start
 * (approval-model.js): the derived initial-sample count, the direction flag,
 * fixed-feature channels, range %, the revision/Start gate, run controls and
 * the write-authorization line. Pure — no DOM, no network.
 */

import { test, expect, describe } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import {
  CHANGED_NOTICE,
  CONFLICT_NOTICE,
  OWNER_UNKNOWN_NOTICE,
  REVIEWED_NOTICE,
  SWITCHED_NOTICE,
  approvalInit,
  authorizationNote,
  fixedFeatureChannels,
  initialSamples,
  plannedEvaluations,
  isDraftChangedConflict,
  objectiveDirection,
  rangePercent,
  reduceApproval,
  runControls,
  startGate,
  variableRows,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';

test('plannedEvaluations: iterations plus initial samples, as the runner totals them', () => {
  expect(plannedEvaluations(storedConfig())).toBe(33);
  expect(plannedEvaluations(storedConfig({ n_iterations: undefined }))).toBe(23);
  expect(plannedEvaluations(storedConfig({ algorithm: 'random', n_iterations: 5 }))).toBe(6);
  expect(plannedEvaluations(storedConfig({ variables: undefined }))).toBeNull();
  expect(plannedEvaluations(null)).toBeNull();
});

describe('initialSamples (mirrors the service _resolve_n_initial)', () => {
  test('Bayesian with null count: max(2, n_vars + 1)', () => {
    expect(initialSamples(storedConfig())).toMatchObject({ count: 3, derived: true });
    expect(
      initialSamples(storedConfig({ algorithm: 'expected_improvement', variables: ['a'] }))
    ).toMatchObject({ count: 2, derived: true });
    expect(
      initialSamples(storedConfig({ algorithm: 'baxus', variables: ['a', 'b', 'c', 'd'] }))
    ).toMatchObject({ count: 5, derived: true });
  });
  test('Bayesian on a pre-trained model defaults to 1', () => {
    expect(initialSamples(storedConfig({ model_file: 'm.joblib' }))).toMatchObject({
      count: 1,
      derived: true,
    });
  });
  test('non-Bayesian algorithms default to 1 (the current machine state)', () => {
    for (const algorithm of ['random', 'neldermead', 'rcds', 'extremum_seeking', 'nsga2']) {
      expect(initialSamples(storedConfig({ algorithm }))).toMatchObject({
        count: 1,
        derived: true,
      });
    }
  });
  test('an explicit count wins and is not marked derived', () => {
    expect(initialSamples(storedConfig({ n_initial_samples: 7 }))).toMatchObject({
      count: 7,
      derived: false,
    });
  });
});

describe('objectiveDirection', () => {
  test('null direction follows the observable default, unflagged', () => {
    expect(objectiveDirection(storedConfig(), envDetails())).toEqual({
      effective: 'MAXIMIZE',
      observableDefault: 'MAXIMIZE',
      overridden: false,
      mismatch: false,
    });
  });
  test('a direction that inverts the observable default is flagged', () => {
    const d = objectiveDirection(storedConfig({ objective_direction: 'MINIMIZE' }), envDetails());
    expect(d).toMatchObject({
      effective: 'MINIMIZE',
      mismatch: true,
      observableDefault: 'MAXIMIZE',
    });
  });
  test('an observable with no direction defaults to MAXIMIZE', () => {
    const d = objectiveDirection(
      storedConfig({ objective_name: 'plain', objective_direction: 'MINIMIZE' }),
      envDetails()
    );
    expect(d.mismatch).toBe(true);
  });
  test('the same direction stated explicitly is not a mismatch', () => {
    const d = objectiveDirection(
      storedConfig({ objective_name: 'loss', objective_direction: 'MINIMIZE' }),
      envDetails()
    );
    expect(d).toMatchObject({ overridden: true, mismatch: false });
  });
});

describe('fixedFeatureChannels', () => {
  const models = [{ label: 'm', path: 'runs/m.joblib', variables: ['phase_a', 'sol_b', 'kick_c'] }];
  test('model variables the draft does not tune are held fixed', () => {
    expect(fixedFeatureChannels(storedConfig({ model_file: 'runs/m.joblib' }), models)).toEqual({
      known: true,
      channels: ['kick_c'],
    });
  });
  test('an unlisted model is reported unknown, not empty', () => {
    expect(fixedFeatureChannels(storedConfig({ model_file: 'other' }), models).known).toBe(false);
  });
  test('no model file: nothing fixed', () => {
    expect(fixedFeatureChannels(storedConfig(), models)).toEqual({ known: true, channels: [] });
  });
});

describe('rangePercent and variableRows', () => {
  test('window width over the safety span', () => {
    expect(rangePercent(-18, 18, -180, 180)).toBeCloseTo(10);
    expect(rangePercent(0, 1, 5, 5)).toBeNull();
    expect(rangePercent(null, 1, 0, 1)).toBeNull();
  });
  test('an inverted window (max below min) has no range %', () => {
    expect(rangePercent(18, -18, -180, 180)).toBeNull();
    expect(rangePercent(5, 5, 0, 10)).toBe(0);
  });
  test('rows carry the resolved bounds, step, max step and range %', () => {
    const rows = variableRows(storedConfig(), envDetails());
    expect(rows[0]).toMatchObject({
      name: 'phase_a',
      min: -18,
      max: 18,
      step: 0.5,
      maxStep: 20,
      unresolved: false,
    });
    expect(rows[0].rangePct).toBeCloseTo(10);
    expect(rows[1]).toMatchObject({ name: 'sol_b', maxStep: null });
    expect(rows[1].rangePct).toBeCloseTo(20);
  });
  test('an override missing a field is flagged unresolved', () => {
    const rows = variableRows(
      storedConfig({ variables: ['phase_a'], variable_overrides: { phase_a: { min: 1 } } }),
      null
    );
    expect(rows[0].unresolved).toBe(true);
  });
});

describe('the revision / Start gate', () => {
  const created = { job_id: 'j1', status: 'created' };
  const ready = { envLoaded: true };
  const draft = (/** @type {string} */ revision, jobId = 'j1') =>
    /** @type {const} */ ({ type: 'draft', jobId, revision, config: storedConfig() });

  test('the first rendering of a draft enables Start with that revision', () => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    expect(s.revision).toBe('r1');
    expect(startGate(s, created, ready)).toEqual({ enabled: true, reason: null, warnings: [] });
  });

  test('the same revision again changes nothing', () => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    expect(reduceApproval(s, draft('r1'))).toBe(s);
  });

  test('a new revision re-renders and holds Start until reviewed', () => {
    let s = reduceApproval(approvalInit(), draft('r1'));
    s = reduceApproval(s, draft('r2'));
    expect(s).toMatchObject({ revision: 'r2', needsReview: true, notice: CHANGED_NOTICE });
    expect(startGate(s, created, ready).enabled).toBe(false);
    s = reduceApproval(s, { type: 'reviewed' });
    expect(startGate(s, created, ready).enabled).toBe(true);
    expect(s.revision).toBe('r2');
    // The notice keeps its place, so nothing below it (Start) moves.
    expect(s.notice).toBe(REVIEWED_NOTICE);
  });

  test('a 409 holds Start; the refetched revision keeps the conflict notice', () => {
    let s = reduceApproval(approvalInit(), draft('r1'));
    s = reduceApproval(s, { type: 'conflict' });
    expect(startGate(s, created, ready).enabled).toBe(false);
    s = reduceApproval(s, draft('r2'));
    expect(s).toMatchObject({ needsReview: true, notice: CONFLICT_NOTICE, revision: 'r2' });
  });

  test('a different draft replacing one on screen holds Start until reviewed', () => {
    let s = reduceApproval(approvalInit(), draft('r1'));
    s = reduceApproval(s, { type: 'conflict' });
    s = reduceApproval(s, draft('x', 'j2'));
    expect(s).toMatchObject({ jobId: 'j2', needsReview: true, notice: SWITCHED_NOTICE });
    expect(startGate(s, { job_id: 'j2', status: 'created' }, ready).enabled).toBe(false);
  });

  test('a draft arriving on an empty surface is its own first look', () => {
    let s = reduceApproval(approvalInit(), draft('r1'));
    s = reduceApproval(s, { type: 'clear' });
    s = reduceApproval(s, draft('x', 'j2'));
    expect(s).toMatchObject({ jobId: 'j2', needsReview: false, notice: null });
  });

  test("Start is held until the draft's environment has loaded", () => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    expect(startGate(s, created)).toEqual({
      enabled: false,
      reason: "Direction unverified: the draft's environment is loading.",
      warnings: [],
    });
  });

  test('Start stays disabled for a non-created job, a missing revision, or in flight', () => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    expect(startGate(s, { job_id: 'j1', status: 'running' }).enabled).toBe(false);
    expect(startGate(s, { job_id: 'other', status: 'created' }).enabled).toBe(false);
    expect(startGate(s, null).enabled).toBe(false);
    expect(startGate(s, created, { ...ready, starting: true }).enabled).toBe(false);
    const noRev = reduceApproval(approvalInit(), { ...draft('r1'), revision: null });
    expect(startGate(noRev, created, ready).enabled).toBe(false);
  });

  test.each(['completed', 'error', 'cancelled'])(
    'an ended (%s) job asks for a new draft',
    (status) => {
      const s = reduceApproval(approvalInit(), draft('r1'));
      expect(startGate(s, { job_id: 'j1', status }, ready).reason).toBe(
        `This job has ended (${status}). Create a new draft.`
      );
    }
  );

  test('a job under way says so', () => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    expect(startGate(s, { job_id: 'j1', status: 'running' }, ready).reason).toBe(
      'This job is already under way (running); Start is only for a new draft.'
    );
  });

  test('a session naming no account warns before Start but does not hold it', () => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    expect(startGate(s, created, { ...ready, ownerKnown: false })).toEqual({
      enabled: true,
      reason: null,
      warnings: [OWNER_UNKNOWN_NOTICE],
    });
  });

  test.each([
    ['a named owner', true],
    ['an unanswered session read', null],
  ])('%s neither warns nor holds', (_label, ownerKnown) => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    expect(startGate(s, created, { ...ready, ownerKnown })).toEqual({
      enabled: true,
      reason: null,
      warnings: [],
    });
  });

  test('no owner warning for a readonly draft or a job that is not a draft', () => {
    const readonly = reduceApproval(approvalInit(), {
      ...draft('r1'),
      config: storedConfig({ execution_mode: 'readonly' }),
    });
    expect(startGate(readonly, created, { ...ready, ownerKnown: false }).warnings).toEqual([]);
    const s = reduceApproval(approvalInit(), draft('r1'));
    const ended = { job_id: 'j1', status: 'error' };
    expect(startGate(s, ended, { ...ready, ownerKnown: false }).warnings).toEqual([]);
  });

  test("a draft that may not move warns beside Start, next to the owner's, and never holds it", () => {
    const s = reduceApproval(approvalInit(), draft('r1'));
    const stuck = {
      ...created,
      start_notice: 'Nelder-Mead may not move from the current setpoints.',
    };
    expect(startGate(s, stuck, ready)).toEqual({
      enabled: true,
      reason: null,
      warnings: ['Nelder-Mead may not move from the current setpoints.'],
    });
    expect(startGate(s, stuck, { ...ready, ownerKnown: false }).warnings).toEqual([
      OWNER_UNKNOWN_NOTICE,
      'Nelder-Mead may not move from the current setpoints.',
    ]);
    expect(startGate(s, { ...stuck, start_notice: null }, ready).warnings).toEqual([]);
    expect(startGate(s, { ...stuck, status: 'running' }, ready).warnings).toEqual([]);
  });

  test('only the service draft-changed 409 counts as a draft change', () => {
    expect(
      isDraftChangedConflict(409, "Job j's draft changed since it was shown; read it again")
    ).toBe(true);
    expect(isDraftChangedConflict(409, 'Another optimization is already running')).toBe(false);
    expect(isDraftChangedConflict(422, 'draft changed')).toBe(false);
  });
});

describe('runControls', () => {
  test('pause from running, resume from paused, cancel while non-terminal', () => {
    expect(runControls('running')).toEqual({ pause: true, resume: false, cancel: true });
    expect(runControls('paused')).toEqual({ pause: false, resume: true, cancel: true });
    expect(runControls('created')).toEqual({ pause: false, resume: false, cancel: true });
    for (const status of ['cancelling', 'completed', 'error', 'cancelled', null]) {
      expect(runControls(status)).toEqual({ pause: false, resume: false, cancel: false });
    }
  });
});

describe('authorizationNote', () => {
  test('awaiting asks for the web-terminal approval, naming the request', () => {
    expect(authorizationNote({ state: 'awaiting', id: 'req-9' })).toEqual({
      tone: 'warn',
      text: 'Waiting for write approval: approve the request in the web terminal (request req-9).',
    });
  });
  test('ended carries the reason; none or unknown says nothing', () => {
    expect(authorizationNote({ state: 'ended', reason: 'denied' })?.text).toContain('denied');
    expect(authorizationNote(null)).toBeNull();
    expect(authorizationNote({ state: 'mystery' })).toBeNull();
  });
});
