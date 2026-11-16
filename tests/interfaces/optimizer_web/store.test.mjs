/**
 * Unit tests for the OPTIMIZER panel's store (store.js): every transition, and
 * the screen derived from the job, the draft, the approval state and the
 * operator's selection.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import { approvalEnvKey } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';
import {
  SCREENS,
  createStore,
  currentScreen,
  deriveScreen,
  initialState,
  presentScreen,
  readingKey,
  reduce,
  runJob,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').State} State
 * @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent} StoreEvent
 */

/**
 * Run `events` from the initial state.
 *
 * @param {StoreEvent[]} events
 * @param {State} [from]
 * @returns {State}
 */
function run(events, from = initialState()) {
  return events.reduce(reduce, from);
}

/** @param {Record<string, unknown>} [overrides] */
const draftJob = (overrides = {}) => ({ job_id: 'j1', status: 'created', ...overrides });

/**
 * A created draft j1 on screen, its environment read, the session naming op1.
 *
 * @param {{ config?: any, revision?: string }} [options]
 * @returns {StoreEvent[]}
 */
function draftOnScreen({ config = storedConfig(), revision = 'rev-1' } = {}) {
  const key = approvalEnvKey(config);
  return [
    { type: 'session', owner: 'op1', ownerKnown: true },
    { type: 'poll/job', job: draftJob(), at: 0 },
    { type: 'poll/config', jobId: 'j1', revision, config },
    { type: 'approvalEnv/reset', key },
    {
      type: 'approvalEnv/loaded',
      env: { key, details: envDetails(), models: [], readbacks: {}, error: null },
    },
  ];
}

/** @param {State} state */
const screenOf = (state) => deriveScreen(state).id;

describe('the start of the panel', () => {
  test('before the first read the screen is empty and says it is still reading', () => {
    const screen = deriveScreen(initialState());
    expect(screen).toMatchObject({ id: 'empty', loading: true, job: null, gate: null });
  });

  test('a first read with no current job is empty and no longer loading', () => {
    const state = run([{ type: 'poll/job', job: null, at: 0 }]);
    expect(deriveScreen(state)).toMatchObject({ id: 'empty', loading: false });
    expect(state.service).toEqual({ ok: true, error: null });
  });

  test('a failed read names the error and keeps the job it had', () => {
    const state = run([
      { type: 'poll/job', job: draftJob(), at: 0 },
      { type: 'poll/failed', error: 'optimizer service unreachable' },
    ]);
    expect(state.service).toEqual({ ok: false, error: 'optimizer service unreachable' });
    expect(state.job.job_id).toBe('j1');
  });

  test('the screens are the fourteen of a run, in order', () => {
    expect(SCREENS).toHaveLength(14);
    expect(SCREENS[0]).toBe('empty');
    expect(SCREENS.at(-1)).toBe('result');
  });
});

describe('a draft and its review', () => {
  test('the first draft on an empty panel is ready, with Start enabled for its revision', () => {
    const screen = deriveScreen(run(draftOnScreen()));
    expect(screen.id).toBe('review');
    expect(screen.shown).toEqual({ jobId: 'j1', revision: 'rev-1' });
    expect(screen.gate).toMatchObject({ enabled: true, reason: null });
    expect(screen.subject).toEqual({ environment: 'injector', target: 'va' });
  });

  test('a new revision of the draft on screen must be acknowledged before Start', () => {
    const changed = run([
      ...draftOnScreen(),
      {
        type: 'poll/config',
        jobId: 'j1',
        revision: 'rev-2',
        config: storedConfig({ n_iterations: 77 }),
      },
    ]);
    const screen = deriveScreen(changed);
    expect(screen.id).toBe('review-changed');
    expect(screen.gate?.enabled).toBe(false);
    expect(screen.shown).toEqual({ jobId: 'j1', revision: 'rev-2' });

    const reviewed = reduce(changed, { type: 'approval/reviewed' });
    expect(deriveScreen(reviewed)).toMatchObject({ id: 'review', gate: { enabled: true } });
  });

  test('the same revision read again changes nothing', () => {
    const state = run(draftOnScreen());
    const again = reduce(state, {
      type: 'poll/config',
      jobId: 'j1',
      revision: 'rev-1',
      config: storedConfig(),
    });
    expect(again.approval).toBe(state.approval);
  });

  test('a different draft in place of the one on screen must be acknowledged too', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'poll/job', job: draftJob({ job_id: 'j2' }), at: 1 },
      { type: 'poll/config', jobId: 'j2', revision: 'rev-j2', config: storedConfig() },
    ]);
    expect(screenOf(state)).toBe('review-changed');
    expect(state.approval.notice).toContain('different draft');
  });

  test('while the new job is read but its draft is not, Start is held', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'poll/job', job: draftJob({ job_id: 'j2' }), at: 1 },
    ]);
    const screen = deriveScreen(state);
    expect(screen.gate).toMatchObject({ enabled: false, reason: 'The draft is still loading.' });
    expect(screen.config).toBeNull();
  });

  test('a session that names no account warns, and Start stays enabled', () => {
    const state = run([...draftOnScreen(), { type: 'session', owner: null, ownerKnown: false }]);
    const screen = deriveScreen(state);
    expect(screen.id).toBe('review-no-owner');
    expect(screen.gate?.enabled).toBe(true);
    expect(screen.gate?.warnings.join(' ')).toContain('names no account');
  });

  test('a readonly draft is not warned about a missing account', () => {
    const config = storedConfig({ execution_mode: 'readonly' });
    const state = run([
      ...draftOnScreen({ config }),
      { type: 'session', owner: null, ownerKnown: false },
    ]);
    expect(screenOf(state)).toBe('review');
  });

  test('a changed draft comes before a missing account', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'session', owner: null, ownerKnown: false },
      { type: 'poll/config', jobId: 'j1', revision: 'rev-2', config: storedConfig() },
    ]);
    expect(screenOf(state)).toBe('review-changed');
  });

  test("until the draft's environment is read, the direction is unverified and Start held", () => {
    const state = run(draftOnScreen().slice(0, 3));
    expect(deriveScreen(state).gate?.reason).toContain('Direction unverified');
  });

  test('a read of an environment the draft no longer names is dropped', () => {
    const state = run(draftOnScreen());
    const stale = reduce(state, {
      type: 'approvalEnv/loaded',
      env: { key: 'linac|va|x', details: null, models: null, readbacks: {}, error: 'gone' },
    });
    expect(stale).toBe(state);
    const late = reduce(state, {
      type: 'approvalEnv/readbacks',
      key: 'linac|va|x',
      readbacks: { x: 1 },
    });
    expect(late).toBe(state);
    const fresh = reduce(state, {
      type: 'approvalEnv/readbacks',
      key: state.approvalEnv.key,
      readbacks: { phase_a: 4 },
    });
    expect(fresh.approvalEnv.readbacks).toEqual({ phase_a: 4 });
    expect(fresh.approval).toBe(state.approval);
  });

  test('a readback before the environment is read is dropped', () => {
    const state = run(draftOnScreen().slice(0, 4));
    expect(
      reduce(state, {
        type: 'approvalEnv/readbacks',
        key: state.approvalEnv.key,
        readbacks: { a: 1 },
      })
    ).toBe(state);
  });
});

describe('Start', () => {
  test('a pending start holds Start; a refusal is kept with its reason', () => {
    const sent = run([...draftOnScreen(), { type: 'start/sent' }]);
    expect(sent.starting).toBe(true);
    expect(deriveScreen(sent).gate).toMatchObject({ enabled: false, reason: 'Starting…' });
    const refused = reduce(sent, { type: 'start/answered', error: 'owner_unknown' });
    expect(refused).toMatchObject({ starting: false, startError: 'owner_unknown' });
  });

  test('a draft-changed answer asks for a new look', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'start/sent' },
      { type: 'start/answered', conflict: true },
    ]);
    expect(state.approval.needsReview).toBe(true);
    expect(state.approval.notice).toContain('nothing started');
    expect(screenOf(state)).toBe('review-changed');
  });

  test('an accepted start takes the job the service answered with', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'start/sent' },
      {
        type: 'start/answered',
        job: { job_id: 'j1', status: 'running', write_authorization: null },
      },
    ]);
    expect(state.job.status).toBe('running');
    // Started: no approval asked yet and nothing evaluated, so nothing is written.
    expect(screenOf(state)).toBe('waiting-approval');
  });

  test('a start error goes when another job comes', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'start/answered', error: 'refused' },
      { type: 'poll/job', job: draftJob({ job_id: 'j2' }), at: 2 },
    ]);
    expect(state.startError).toBeNull();
  });
});

describe('a run and its end', () => {
  /** @param {Record<string, unknown>} overrides */
  const runOf = (overrides) =>
    run([
      { type: 'poll/job', job: runningJob({ job_id: 'j1', ...overrides }), at: 0 },
      { type: 'poll/config', jobId: 'j1', revision: 'rev-1', config: storedConfig() },
    ]);

  test.each([
    [{ status: 'submitted', data: [] }, 'waiting-approval'],
    [{ status: 'running', write_authorization: { state: 'requesting' } }, 'waiting-approval'],
    [
      { status: 'running', write_authorization: { state: 'awaiting', owner: 'op1' } },
      'waiting-approval',
    ],
    [{ status: 'running', write_authorization: { state: 'granted' } }, 'running'],
    [{ status: 'running', write_authorization: null }, 'running'],
    [{ status: 'paused' }, 'paused'],
    [{ status: 'cancelling' }, 'running'],
    [{ status: 'completed' }, 'completed'],
    [{ status: 'error', error_message: 'boom' }, 'failed'],
    [{ status: 'cancelled' }, 'cancelled'],
  ])('%j is the %s screen', (overrides, id) => {
    expect(screenOf(runOf(overrides))).toBe(id);
  });

  test('waiting for approval names who approves, and Pause and Cancel are the service rules', () => {
    const screen = deriveScreen(
      runOf({
        status: 'running',
        write_authorization: { state: 'awaiting', owner: 'op1', id: 'abcdef1234' },
      })
    );
    expect(screen.authorization?.text).toContain('op1');
    expect(screen.controls).toEqual({ pause: true, resume: false, cancel: true });
  });

  test('cancelling is the running screen with no control left', () => {
    const screen = deriveScreen(runOf({ status: 'cancelling' }));
    expect(screen).toMatchObject({
      cancelling: true,
      controls: { pause: false, resume: false, cancel: false },
    });
  });

  test('paused allows Resume and Cancel', () => {
    expect(deriveScreen(runOf({ status: 'paused' })).controls).toEqual({
      pause: false,
      resume: true,
      cancel: true,
    });
  });

  test('an ended run has no control', () => {
    expect(deriveScreen(runOf({ status: 'completed' })).controls).toEqual({
      pause: false,
      resume: false,
      cancel: false,
    });
  });

  test("the run's figures come from its record and its config", () => {
    const screen = deriveScreen(
      runOf({ status: 'running', snapshots: { initial: { charge: 1.5 } } })
    );
    expect(screen.figures).toMatchObject({ done: 3, total: 10, best: 2.5, before: 1.5, change: 1 });
    expect(screen.subject).toEqual({ environment: 'injector', target: 'va' });
  });

  test('a control in flight disables every control; its answer keeps the run record', () => {
    const sent = reduce(runOf({ status: 'running' }), { type: 'control/sent', action: 'pause' });
    expect(deriveScreen(sent).controls).toEqual({ pause: false, resume: false, cancel: false });
    const answered = reduce(sent, {
      type: 'control/answered',
      job: {
        job_id: 'j1',
        status: 'paused',
        data: null,
        error_message: null,
        write_authorization: null,
      },
    });
    expect(answered.job.status).toBe('paused');
    expect(answered.job.data).toHaveLength(3);
    expect(answered.control).toEqual({ pending: null, error: null, cancelArmed: false });
  });

  test('a refused control keeps the reason and the job as it was', () => {
    const state = run(
      [
        { type: 'control/sent', action: 'resume' },
        { type: 'control/answered', error: 'switch_off' },
      ],
      runOf({ status: 'paused' })
    );
    expect(state.control.error).toBe('switch_off');
    expect(state.job.status).toBe('paused');
  });

  test('an answer for another job leaves the job alone', () => {
    const before = runOf({ status: 'running' });
    const state = reduce(before, {
      type: 'control/answered',
      job: { job_id: 'other', status: 'paused' },
    });
    expect(state.job).toBe(before.job);
  });

  test('cancel is armed and disarmed; a selection or another job disarms it', () => {
    const armed = reduce(runOf({ status: 'running' }), { type: 'control/arm', armed: true });
    expect(armed.control.cancelArmed).toBe(true);
    expect(reduce(armed, { type: 'control/arm', armed: false }).control.cancelArmed).toBe(false);
    expect(
      reduce(armed, { type: 'select', selection: { kind: 'history' } }).control.cancelArmed
    ).toBe(false);
    expect(
      reduce(armed, { type: 'poll/job', job: runningJob({ job_id: 'j9' }), at: 1 }).control
        .cancelArmed
    ).toBe(false);
  });

  test('the arrivals of evaluations are timed per job', () => {
    const job = (/** @type {number} */ n) =>
      runningJob({ data: Array.from({ length: n }, () => ({ charge: 1 })) });
    const state = run([
      { type: 'poll/job', job: job(2), at: 0 },
      { type: 'poll/job', job: job(3), at: 60000 },
      { type: 'poll/job', job: job(6), at: 240000 },
    ]);
    expect(state.arrivals).toMatchObject({
      seen: 2,
      first: { count: 3, at: 60000 },
      last: { count: 6, at: 240000 },
    });
    expect(deriveScreen(state).figures?.minutesLeft).toBe(4);
  });
});

describe('adopting the current job', () => {
  test('a run that started elsewhere is shown by a panel that shows nothing else', () => {
    const state = run([
      { type: 'poll/job', job: null, at: 0 },
      { type: 'poll/job', job: runningJob({ job_id: 'j7' }), at: 1 },
    ]);
    expect(screenOf(state)).toBe('running');
  });

  test('a panel with a new set-up open keeps it when a run starts elsewhere', () => {
    const state = run([
      { type: 'setup/open', jobId: null, environment: 'injector', target: 'va' },
      { type: 'poll/job', job: runningJob({ job_id: 'j7' }), at: 1 },
    ]);
    expect(screenOf(state)).toBe('setup');
    expect(state.setup?.jobId).toBeNull();
  });

  test('picking the current run beside an open set-up shows the run; the set-up keeps its place', () => {
    const state = run([
      { type: 'setup/open', jobId: null, environment: 'injector', target: 'va' },
      { type: 'poll/job', job: runningJob({ job_id: 'j7' }), at: 1 },
      { type: 'select', selection: { kind: 'job', jobId: 'j7' } },
    ]);
    expect(deriveScreen(state)).toMatchObject({ id: 'running', controls: { cancel: true } });
    expect(state.setup).not.toBeNull();
    expect(screenOf(reduce(state, { type: 'select', selection: { kind: 'present' } }))).toBe(
      'setup'
    );
    const closed = reduce(state, { type: 'setup/close' });
    expect(closed.selection).toEqual({ kind: 'present' });
  });

  test('a panel showing a past run keeps it when a draft arrives', () => {
    const state = run([
      { type: 'select', selection: { kind: 'run', runId: 'injector/2026-11-16_21-15-04' } },
      { type: 'poll/job', job: draftJob(), at: 1 },
    ]);
    expect(screenOf(state)).toBe('result');
  });

  test('the set-up of a draft that has started is closed; the run shows', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'setup/open', jobId: 'j1', environment: 'injector', target: 'va' },
      { type: 'poll/job', job: runningJob({ job_id: 'j1' }), at: 2 },
    ]);
    expect(state.setup).toBeNull();
    expect(screenOf(state)).toBe('running');
  });

  test('the set-up of a draft that is no longer current keeps the edits', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'setup/open', jobId: 'j1', environment: 'injector', target: 'va' },
      { type: 'poll/job', job: draftJob({ job_id: 'j2' }), at: 2 },
    ]);
    expect(state.setup?.jobId).toBe('j1');
    // The job on screen is another one; the stored draft of j1 is still the one read.
    expect(deriveScreen(state)).toMatchObject({
      id: 'setup',
      job: null,
      config: { n_iterations: 30 },
    });
  });

  test('no current job clears the draft on screen', () => {
    const state = run([...draftOnScreen(), { type: 'poll/job', job: null, at: 3 }]);
    expect(state.approval.jobId).toBeNull();
    expect(screenOf(state)).toBe('empty');
  });
});

describe('an active run beside a newer draft', () => {
  const RUN_CONFIG = storedConfig({ control_target: 'standin', n_iterations: 40 });

  /**
   * The service's current job is the draft j1; job r9 is active beside it.
   *
   * @param {Record<string, unknown>} [overrides]
   * @returns {StoreEvent[]}
   */
  const besideDraft = (overrides = {}) => [
    ...draftOnScreen(),
    {
      type: 'poll/active',
      job: runningJob({ job_id: 'r9', ...overrides }),
      config: RUN_CONFIG,
      at: 5,
    },
  ];

  test.each([
    [{ status: 'running' }, 'running'],
    [{ status: 'running', write_authorization: { state: 'awaiting' } }, 'waiting-approval'],
    [{ status: 'submitted', data: [] }, 'waiting-approval'],
    [{ status: 'paused' }, 'paused'],
    [{ status: 'cancelling' }, 'running'],
  ])('now is the run (%j), not the newer draft', (overrides, id) => {
    const state = run(besideDraft(overrides));
    expect(presentScreen(state)).toBe(id);
    const screen = deriveScreen(state);
    expect(screen).toMatchObject({ id, job: { job_id: 'r9' }, config: RUN_CONFIG });
    expect(screen.subject).toEqual({ environment: 'injector', target: 'standin' });
    expect(runJob(state)?.job_id).toBe('r9');
  });

  test("the run's controls follow its own status", () => {
    expect(deriveScreen(run(besideDraft())).controls).toEqual({
      pause: true,
      resume: false,
      cancel: true,
    });
    expect(deriveScreen(run(besideDraft({ status: 'paused' }))).controls).toEqual({
      pause: false,
      resume: true,
      cancel: true,
    });
    expect(deriveScreen(run(besideDraft({ status: 'cancelling' }))).cancelling).toBe(true);
  });

  test('the newer draft opens as its review when picked; its Start keeps the gate rules', () => {
    const state = run([
      ...besideDraft(),
      { type: 'select', selection: { kind: 'job', jobId: 'j1' } },
    ]);
    const screen = deriveScreen(state);
    expect(screen).toMatchObject({ id: 'review', job: { job_id: 'j1' } });
    expect(screen.shown).toEqual({ jobId: 'j1', revision: 'rev-1' });
    expect(screen.gate?.enabled).toBe(true);
    expect(currentScreen(state)).toBe('review');
  });

  test('picking the active job always gives its run page, never the stored-run screen', () => {
    const listed = run([
      ...besideDraft({ status: 'paused' }),
      { type: 'jobs', list: [{ job_id: 'r9', status: 'paused' }], error: null },
      { type: 'select', selection: { kind: 'job', jobId: 'r9' } },
    ]);
    expect(deriveScreen(listed)).toMatchObject({
      id: 'paused',
      job: { job_id: 'r9' },
      controls: { resume: true, cancel: true },
    });
    const withSetup = run([
      ...besideDraft(),
      { type: 'setup/open', jobId: null, environment: '', target: '' },
      { type: 'select', selection: { kind: 'job', jobId: 'r9' } },
    ]);
    expect(deriveScreen(withSetup)).toMatchObject({ id: 'running', job: { job_id: 'r9' } });
  });

  test('the current job when it is the run: picked or not, its run page', () => {
    const state = run([
      { type: 'poll/job', job: runningJob({ job_id: 'j1' }), at: 0 },
      { type: 'select', selection: { kind: 'job', jobId: 'j1' } },
    ]);
    expect(deriveScreen(state)).toMatchObject({ id: 'running', controls: { cancel: true } });
  });

  test('an armed cancel survives a newer draft; it is disarmed when the run changes', () => {
    const armed = run([...besideDraft(), { type: 'control/arm', armed: true }]);
    const draft = reduce(armed, { type: 'poll/job', job: draftJob({ job_id: 'j2' }), at: 6 });
    expect(draft.control.cancelArmed).toBe(true);
    const other = reduce(armed, {
      type: 'poll/active',
      job: runningJob({ job_id: 'r10' }),
      config: null,
      at: 6,
    });
    expect(other.control.cancelArmed).toBe(false);
  });

  test('the evaluations are timed for the run, not for the draft', () => {
    const job = (/** @type {number} */ n) =>
      runningJob({ job_id: 'r9', data: Array.from({ length: n }, () => ({ charge: 1 })) });
    const state = run([
      ...draftOnScreen(),
      { type: 'poll/active', job: job(2), config: RUN_CONFIG, at: 0 },
      { type: 'poll/job', job: draftJob(), at: 1000 },
      { type: 'poll/active', job: job(3), config: RUN_CONFIG, at: 60000 },
    ]);
    expect(state.arrivals).toMatchObject({ jobId: 'r9', first: { count: 3, at: 60000 } });
  });

  test("a control's answer updates the run beside the draft", () => {
    const state = run([
      ...besideDraft(),
      { type: 'control/sent', action: 'pause' },
      { type: 'control/answered', job: { job_id: 'r9', status: 'paused' } },
    ]);
    expect(state.activeJob.status).toBe('paused');
    expect(state.activeJob.data).toHaveLength(3);
    expect(screenOf(state)).toBe('paused');
  });

  test('once the run has ended, now is the draft again; picking the ended run shows its status', () => {
    const ended = run([
      ...besideDraft(),
      {
        type: 'poll/active',
        job: runningJob({ job_id: 'r9', status: 'completed' }),
        config: RUN_CONFIG,
        at: 9,
      },
    ]);
    expect(runJob(ended)).toBeNull();
    expect(screenOf(ended)).toBe('review');
    const picked = reduce(ended, { type: 'select', selection: { kind: 'job', jobId: 'r9' } });
    expect(deriveScreen(picked)).toMatchObject({ id: 'result', job: { status: 'completed' } });
    const cleared = reduce(ended, { type: 'poll/active', job: null, config: null, at: 10 });
    expect(cleared).toMatchObject({ activeJob: null, activeConfig: null });
  });
});

describe('the set-up', () => {
  test('a new run opens an empty set-up that names no target', () => {
    const state = run([{ type: 'setup/open', jobId: null, environment: '', target: '' }]);
    expect(deriveScreen(state)).toMatchObject({
      id: 'setup',
      subject: { environment: null, target: null },
    });
    expect(state.setup?.from).toBeNull();
  });

  test('a set-up opened from a stored config keeps it, and names its environment and target', () => {
    const from = storedConfig({ control_target: 'standin' });
    const state = run([
      { type: 'setup/open', jobId: null, environment: 'injector', target: 'standin', from },
    ]);
    expect(state.setup).toMatchObject({ jobId: null, from });
    expect(deriveScreen(state).subject).toEqual({ environment: 'injector', target: 'standin' });
  });

  test('the toolbar follows what the set-up names', () => {
    const state = run([
      { type: 'setup/open', jobId: null, environment: '', target: '' },
      { type: 'setup/where', environment: 'linac', target: 'standin' },
    ]);
    expect(deriveScreen(state).subject).toEqual({ environment: 'linac', target: 'standin' });
  });

  test('a refused save holds the set-up with its reasons; a save closes it', () => {
    const refused = run([
      { type: 'setup/open', jobId: null, environment: 'injector', target: 'va' },
      { type: 'setup/saving' },
      { type: 'setup/refused', issues: ['Pick an objective.'] },
    ]);
    expect(refused.setup).toMatchObject({ saving: false, issues: ['Pick an objective.'] });
    expect(screenOf(refused)).toBe('setup-issues');
    const note = { jobId: 'j2', tone: 'ok', text: 'Draft created.', moved: [] };
    const saved = reduce(refused, { type: 'setup/saved', note });
    expect(saved).toMatchObject({ setup: null, saveNote: note, selection: { kind: 'present' } });
  });

  test('set-up events with no set-up open change nothing', () => {
    const state = initialState();
    for (const event of /** @type {StoreEvent[]} */ ([
      { type: 'setup/where', environment: 'x', target: 'va' },
      { type: 'setup/saving' },
      { type: 'setup/refused', issues: ['x'] },
    ])) {
      expect(reduce(state, event)).toBe(state);
    }
  });

  test('a set-up opened from a past run goes to the present', () => {
    const state = run([
      { type: 'select', selection: { kind: 'history' } },
      { type: 'setup/open', jobId: null, environment: '', target: '' },
    ]);
    expect(state.selection).toEqual({ kind: 'present' });
    expect(reduce(state, { type: 'setup/close' }).setup).toBeNull();
  });

  test('a set-up of the current draft shows that draft and its config', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'setup/open', jobId: 'j1', environment: 'injector', target: 'va' },
    ]);
    expect(deriveScreen(state)).toMatchObject({
      id: 'setup',
      job: { job_id: 'j1' },
      config: { n_iterations: 30 },
    });
  });
});

describe('the selection, the lists and the reads', () => {
  const info = {
    run_id: 'injector/2026-11-16_21-15-04',
    environment: 'injector',
    timestamp: '2026-11-16_21-15-04',
    job_id: 'j0',
  };

  test('history and a past run are their own screens; the present returns', () => {
    const history = run([{ type: 'select', selection: { kind: 'history' } }]);
    expect(screenOf(history)).toBe('history');
    expect(screenOf(reduce(history, { type: 'select', selection: { kind: 'present' } }))).toBe(
      'empty'
    );
  });

  test('a stored run shows its entry, its figures and its own subject', () => {
    const details = {
      config: storedConfig({ objective_direction: 'MAXIMIZE' }),
      initial_point: { charge: 1 },
      best_point: {},
      data_records: [
        { charge: 2, phase_a: 0, sol_b: 3 },
        { charge: 3, phase_a: 1, sol_b: 3 },
      ],
    };
    const state = run([
      { type: 'results', list: [info], error: null },
      { type: 'select', selection: { kind: 'run', runId: info.run_id } },
      { type: 'past/loading', runId: info.run_id },
      {
        type: 'past/loaded',
        runId: info.run_id,
        details,
        job: { job_id: 'j0', status: 'completed' },
        error: null,
      },
    ]);
    const screen = deriveScreen(state);
    expect(screen.id).toBe('result');
    expect(screen.past).toMatchObject({ runId: info.run_id, info, job: { status: 'completed' } });
    expect(screen.figures).toMatchObject({ done: 2, best: 3, before: 1, change: 2 });
    expect(screen.subject).toEqual({ environment: 'injector', target: 'va' });
  });

  test('a stored run still loading keeps what it had', () => {
    const state = run([
      { type: 'past/loaded', runId: 'a/b', details: { config: {} }, job: null, error: null },
      { type: 'past/loading', runId: 'a/b' },
    ]);
    expect(state.past['a/b']).toEqual({
      details: { config: {} },
      job: null,
      loading: true,
      error: null,
    });
  });

  test('another open job is shown as a past run; the current job as the present', () => {
    const other = { job_id: 'j5', status: 'created' };
    const state = run([
      { type: 'jobs', list: [other], error: null },
      ...draftOnScreen(),
      { type: 'select', selection: { kind: 'job', jobId: 'j5' } },
    ]);
    expect(deriveScreen(state)).toMatchObject({ id: 'result', past: { job: other, runId: null } });
    const current = reduce(state, { type: 'select', selection: { kind: 'job', jobId: 'j1' } });
    expect(screenOf(current)).toBe('review');
  });

  test('the lists keep the service answer and its error', () => {
    const state = run([
      { type: 'environments', list: [{ name: 'injector' }], error: null },
      { type: 'models', list: null, error: 'down' },
      { type: 'results', list: [], error: null },
      { type: 'jobs', list: [], error: null },
      {
        type: 'connection',
        connection: { environment: 'injector', target: 'va', connected: true, message: null },
      },
      { type: 'filter', text: 'charge' },
    ]);
    expect(state.environments.list).toEqual([{ name: 'injector' }]);
    expect(state.models).toEqual({ list: null, error: 'down' });
    expect(state.openJobs.list).toEqual([]);
    expect(state.connection?.connected).toBe(true);
    expect(state.filter).toBe('charge');
  });

  test('an environment read keeps what it had while it loads again', () => {
    const state = run([
      { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
      { type: 'env/loading', name: 'injector' },
    ]);
    expect(state.envs.injector).toMatchObject({
      loading: true,
      error: null,
      details: { name: 'injector' },
    });
  });

  test('readings of some channels keep the last value of the others', () => {
    const key = readingKey('variables', 'injector', 'va');
    expect(key).toBe('variables:injector@va');
    const state = run([
      { type: 'readings', key, values: { a: 1, b: 2 }, error: null, at: 1 },
      { type: 'readings', key, values: { a: 3 }, error: 'slow', at: 2 },
    ]);
    expect(state.readings[key]).toEqual({ values: { a: 3, b: 2 }, error: 'slow', at: 2 });
  });

  test('an unknown event changes nothing', () => {
    const state = initialState();
    expect(reduce(state, /** @type {any} */ ({ type: 'nothing' }))).toBe(state);
  });
});

describe('presentScreen', () => {
  test('ignores the selection and the set-up', () => {
    const state = run([
      ...draftOnScreen(),
      { type: 'select', selection: { kind: 'history' } },
      { type: 'setup/open', jobId: 'j1', environment: 'injector', target: 'va' },
    ]);
    expect(presentScreen(state)).toBe('review');
  });
});

describe('createStore', () => {
  test('a change calls every listener once; no change calls none; a listener can leave', () => {
    const store = createStore();
    const listener = vi.fn();
    const leave = store.subscribe(listener);
    store.dispatch({ type: 'filter', text: 'x' });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0]).toBe(store.state);
    store.dispatch(/** @type {any} */ ({ type: 'nothing' }));
    expect(listener).toHaveBeenCalledTimes(1);
    leave();
    store.dispatch({ type: 'filter', text: 'y' });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.state.filter).toBe('y');
  });
});

describe('a run starts, a draft arrives, a save answers late', () => {
  test('a run just started, before any approval is asked or any evaluation read, waits: nothing is written', () => {
    const starting = runningJob({
      job_id: 'j1',
      status: 'running',
      write_authorization: null,
      data: [],
    });
    const state = run([
      { type: 'poll/job', job: starting, at: 0 },
      { type: 'poll/config', jobId: 'j1', revision: 'r', config: storedConfig() },
    ]);
    expect(screenOf(state)).toBe('waiting-approval');
    const readonly = run([
      { type: 'poll/job', job: starting, at: 0 },
      {
        type: 'poll/config',
        jobId: 'j1',
        revision: 'r',
        config: storedConfig({ execution_mode: 'readonly' }),
      },
    ]);
    expect(screenOf(readonly)).toBe('running');
    const evaluated = reduce(state, {
      type: 'poll/job',
      job: runningJob({ job_id: 'j1', write_authorization: null }),
      at: 1,
    });
    expect(screenOf(evaluated)).toBe('running');
  });

  test('a draft taking the pointer from a running run never shows its review in between', () => {
    const state = run([
      {
        type: 'poll/job',
        job: runningJob({ job_id: 'r9', write_authorization: { state: 'granted' } }),
        at: 0,
      },
      {
        type: 'poll/config',
        jobId: 'r9',
        revision: 'r',
        config: storedConfig({ control_target: 'standin' }),
      },
      { type: 'control/arm', armed: true },
      { type: 'poll/job', job: draftJob({ job_id: 'j2' }), at: 1 },
    ]);
    expect(deriveScreen(state)).toMatchObject({ id: 'running', job: { job_id: 'r9' } });
    expect(deriveScreen(state).subject.target).toBe('standin');
    const config = reduce(state, {
      type: 'poll/config',
      jobId: 'j2',
      revision: 'r2',
      config: storedConfig(),
    });
    expect(screenOf(config)).toBe('running');
    expect(config.control.cancelArmed).toBe(true);
  });

  test('a late save answer never lands on another set-up nor moves the selection', () => {
    const state = run([
      { type: 'setup/open', jobId: null, environment: 'injector', target: 'va', id: 1 },
      { type: 'setup/saving', setupId: 1 },
      { type: 'setup/open', jobId: null, environment: '', target: '', id: 2 },
      { type: 'select', selection: { kind: 'history' } },
      { type: 'setup/refused', issues: ['old'], setupId: 1 },
    ]);
    expect(state.setup).toMatchObject({ id: 2, issues: [], saving: false });
    const note = { jobId: 'j9', tone: 'ok', text: 'Draft created.', moved: [] };
    const saved = reduce(state, { type: 'setup/saved', note, setupId: 1 });
    expect(saved).toMatchObject({
      setup: { id: 2 },
      selection: { kind: 'history' },
      saveNote: note,
    });
  });

  test('the run beside the draft keeps the stored run of now out of the past runs', async () => {
    const { pastRuns } =
      await import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/runs-list.js');
    const entry = (/** @type {string} */ job_id, /** @type {string} */ ts) => ({
      run_id: `injector/${ts}`,
      environment: 'injector',
      timestamp: ts,
      job_id,
    });
    const state = run([
      { type: 'poll/job', job: runningJob({ job_id: 'r9' }), at: 0 },
      {
        type: 'results',
        list: [entry('r9', '2026-11-17_10-00-00'), entry('old', '2026-11-16_10-00-00')],
        error: null,
      },
    ]);
    expect(pastRuns(state).map((r) => r.job_id)).toEqual(['old']);
    const ended = reduce(state, {
      type: 'poll/job',
      job: runningJob({ job_id: 'r9', status: 'completed' }),
      at: 1,
    });
    expect(pastRuns(ended).map((r) => r.job_id)).toEqual(['r9', 'old']);
  });
});
