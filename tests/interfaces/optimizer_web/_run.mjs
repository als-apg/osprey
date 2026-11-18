/**
 * The screens of a run for the run page's tests (run-page, run-header,
 * run-knobs, run-page-model): the state and screen after the poll read a job,
 * its stored config and its environment.
 *
 * The fixtures follow what the service sends: a stored draft always carries
 * the objective direction (the service fills it in), a started run has no
 * write authorization until it asks for one, and a pause shows at once while
 * the runner only checks for it before each guided step.
 */

import { envDetails, runningJob, storedConfig } from './_fixtures.mjs';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent} StoreEvent */

/** The record of a request that waits for a person. */
export const awaiting = () => ({
  state: 'awaiting',
  target: 'va',
  channels: { 'A:SP': [-18, 18], 'B:SP': [2, 4] },
  steps: 11,
  completion: 'apply_best',
  id: 'req-0123456789',
  owner: 'carol',
  approver: null,
});

/** The same request, approved for an hour from now. */
export const granted = () => ({
  ...awaiting(),
  state: 'granted',
  approver: 'chip',
  expires_at_ns: Date.now() * 1e6 + 3.6e12,
});

/** The knobs and the objective before the run. */
export const INITIAL = { phase_a: 1, sol_b: 3, charge: 0.5 };

/**
 * The state and screen after the poll read `job`, its stored config (with
 * `config` over it) and its environment, then `extra`. The session is carol's.
 *
 * @param {Record<string, unknown>} job
 * @param {StoreEvent[]} [extra]
 * @param {Record<string, unknown>} [config]
 */
export function runAt(job, extra = [], config = {}) {
  const full = runningJob({ snapshots: { initial: INITIAL }, ...job });
  /** @type {StoreEvent[]} */
  const events = [
    { type: 'poll/job', job: full, at: 1000 },
    {
      type: 'poll/config',
      jobId: full.job_id,
      revision: 'rev-abcdef12345',
      // The service writes the resolved direction into every stored draft.
      config: storedConfig({ objective_direction: 'MAXIMIZE', ...config }),
    },
    { type: 'env/loaded', name: 'injector', details: envDetails(), presets: null, error: null },
    { type: 'session', owner: 'carol', ownerKnown: true },
    ...extra,
  ];
  const state = events.reduce(reduce, initialState());
  return { state, screen: deriveScreen(state) };
}

/** A run that waits for carol to approve its writes. */
export const waiting = () => runAt({ data: [], write_authorization: awaiting() });
/** A run under way: 3 of 10 evaluations recorded. */
export const running = () => runAt({ write_authorization: granted() });
/** The same run, paused. */
export const paused = () => runAt({ status: 'paused', write_authorization: granted() });
/** Started a moment ago: no write authorization asked yet, nothing evaluated. */
export const starting = () => runAt({ data: [], write_authorization: null });

/**
 * A run of 3 first tries (UCB on two knobs) and 30 guided steps, paused
 * with `done` evaluations recorded.
 *
 * @param {number} done
 */
export function pausedAfter(done) {
  const data = Array.from({ length: done }, (_, i) => ({ phase_a: i, sol_b: 3, charge: i }));
  return runAt({ status: 'paused', write_authorization: granted(), total_evaluations: 33, data });
}
