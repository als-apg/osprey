// @ts-check
/**
 * The approval card's own read of the stored draft's environment: details,
 * models and readbacks on the draft's own target, keyed by `approvalEnvKey`
 * and independent of the form's pickers.
 *
 * Readbacks are what the machine reads NOW, not part of the approved config:
 * while a draft waits they are refreshed on the poll (at most every
 * {@link READBACK_REFRESH_MS}) and written into the card's Readback cells in
 * place, so a refresh never rebuilds the card, never moves Start and never
 * touches the review state or the revision Start sends.
 *
 * @module approval-env
 */

import { readbackValues, seg } from './api-client.js';
import { approvalEnvKey, startGate } from './approval-model.js';

/** The fastest the approval card re-reads the machine while a draft waits. */
export const READBACK_REFRESH_MS = 5000;

/**
 * @typedef {import('./approval-model.js').ApprovalEnv} ApprovalEnv
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 */

/**
 * Read the readbacks of the draft's variables on the draft's own target.
 *
 * @param {import('./api-client.js').Api} api
 * @param {OptimizationConfig} config
 * @returns {Promise<Record<string, number|null>|null>}  `null` when there is
 *   nothing to read or the read failed
 */
export async function fetchApprovalReadbacks(api, config) {
  const names = config.variables || [];
  if (!config.environment_name || !config.control_target || !names.length) return null;
  const res = await api.get(`environments/${seg(config.environment_name)}/variables`, {
    names,
    target: config.control_target,
  });
  return res.ok ? readbackValues(res.data) : null;
}

/**
 * Read everything the approval card shows about the draft's environment.
 *
 * @param {import('./api-client.js').Api} api
 * @param {OptimizationConfig} config
 * @param {string} key  `approvalEnvKey(config)`
 * @returns {Promise<ApprovalEnv>}
 */
export async function fetchApprovalEnv(api, config, key) {
  const name = /** @type {string} */ (config.environment_name);
  const [details, models, readbacks] = await Promise.all([
    api.get(`environments/${seg(name)}`),
    api.get('models'),
    fetchApprovalReadbacks(api, config),
  ]);
  return {
    key,
    details: details.ok ? details.data : null,
    models: models.ok ? models.data || [] : null,
    readbacks: readbacks || {},
    error: details.ok ? null : details.error || 'environment unavailable',
  };
}

/**
 * Whether the poll should re-read the approval card's readbacks now: a
 * created draft whose environment read has landed, last read at least
 * {@link READBACK_REFRESH_MS} ago, with no readback read still in flight (a
 * slow machine read skips ticks rather than piling reads up) and no Start
 * pending.
 *
 * @param {{
 *   job: any,
 *   approval: import('./approval-model.js').ApprovalState,
 *   approvalEnv: ApprovalEnv,
 *   starting?: boolean,
 *   approvalReading?: boolean,
 * }} state
 * @param {number} readAt  when the readbacks were last read (ms)
 * @param {number} now
 * @returns {boolean}
 */
export function readbacksDue(state, readAt, now) {
  const { job, approval, approvalEnv } = state;
  if (state.starting || state.approvalReading) return false;
  if (job?.status !== 'created' || approval.jobId !== job.job_id || !approval.config) return false;
  if (!approvalEnv.details || approvalEnv.key !== approvalEnvKey(approval.config)) return false;
  return now - readAt >= READBACK_REFRESH_MS;
}

/**
 * Re-read the waiting draft's readbacks when {@link readbacksDue}, one read
 * at a time. A reply for an environment the card no longer shows (the draft
 * moved on, or its environment was re-read meanwhile) is dropped.
 *
 * @param {import('./api-client.js').Api} api
 * @param {{
 *   job: any,
 *   approval: import('./approval-model.js').ApprovalState,
 *   approvalEnv: ApprovalEnv,
 *   approvalReadAt: number,
 *   approvalReading: boolean,
 *   starting: boolean,
 * }} state  the panel's state; `approvalReading` and `approvalReadAt` are written
 * @param {() => number} now
 * @returns {Promise<Record<string, number|null>|null>}  the readings; null when not due, failed or stale
 */
export async function readApprovalReadbacks(api, state, now) {
  const config = state.approval.config;
  if (!config || !readbacksDue(state, state.approvalReadAt, now())) return null;
  const env = state.approvalEnv;
  state.approvalReading = true;
  const readbacks = await fetchApprovalReadbacks(api, config).finally(() => {
    state.approvalReading = false;
    state.approvalReadAt = now();
  });
  if (!readbacks || state.approvalEnv !== env || state.approvalEnv.key !== env.key) return null;
  return readbacks;
}

/**
 * The Start gate for the panel's state, naming a failed environment read.
 *
 * @param {{
 *   approval: import('./approval-model.js').ApprovalState,
 *   approvalEnv: ApprovalEnv,
 *   job: any,
 *   starting: boolean,
 *   ownerKnown: boolean|null,
 * }} state
 * @returns {import('./approval-model.js').StartGate}
 */
export function approvalGateFor(state) {
  const loaded =
    !!state.approvalEnv.details && state.approvalEnv.key === approvalEnvKey(state.approval.config);
  const gate = startGate(state.approval, state.job, {
    starting: state.starting,
    envLoaded: loaded,
    ownerKnown: state.ownerKnown,
  });
  const failed = !loaded && state.approvalEnv.error && gate.reason?.startsWith('Direction');
  return failed
    ? { ...gate, enabled: false, reason: `Direction unverified: ${state.approvalEnv.error}` }
    : gate;
}
