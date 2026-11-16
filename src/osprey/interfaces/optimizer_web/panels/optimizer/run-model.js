// @ts-check
/**
 * A run's figures as pure data: evaluations done and planned, the objective of
 * each evaluation, the best so far, the value before the run and the change
 * from it, and the time left.
 *
 * Every figure comes from a field the service sends (`GET jobs/{id}?detail=full`
 * for the present run, `GET results/{env}/{timestamp}` for a stored one) or
 * from the times this panel saw the evaluations arrive. A figure with no
 * source is `null`; a view leaves its element out, it never estimates it.
 *
 * - The best is the service's own: among the evaluations whose knob values
 *   are all finite and that meet every constraint of the config (a strict
 *   inequality; a missing or non-finite constraint reading fails it), the
 *   best finite objective in the run's direction. With no such evaluation
 *   there is no best. This is the point `apply_best` writes.
 * - The value before the run is the objective in the run's initial point
 *   (`snapshots.initial` of the job, `initial_point` of a stored run).
 * - The time left is "about N min" from the mean time of the evaluations this
 *   panel saw arrive while the run was running. The service records no time
 *   per evaluation, so after a reload or a pause the count starts again, and
 *   fewer than {@link MIN_TIMED_EVALUATIONS} timed evaluations give no figure.
 * - The service says nothing about the step inside an evaluation (writing,
 *   settling, reading); there is no figure for it.
 *
 * @module run-model
 */

import { plannedEvaluations } from './approval-model.js';

/** The fewest timed evaluations that give a time left. */
export const MIN_TIMED_EVALUATIONS = 3;

/**
 * @typedef {Record<string, unknown>} DataRecord
 * @typedef {Record<string, { type: string, value: number }>} Constraints
 *
 * When the panel saw the evaluation count of one job grow while it ran.
 * `first` is the first growth it saw (the count and the time), `last` the
 * latest; `seen` is the count at the first read, which was not an arrival
 * the panel timed.
 * @typedef {{
 *   jobId: string|null,
 *   seen: number,
 *   first: { count: number, at: number }|null,
 *   last: { count: number, at: number }|null,
 * }} Arrivals
 *
 * `feasible` is null when the config has no constraint; else one entry per
 * evaluation: true when it meets every constraint. `bests` is the running
 * best over the evaluations that count (finite knobs, feasible); null before
 * the first one, and null as a whole when the direction is unknown.
 * @typedef {{
 *   done: number,
 *   total: number|null,
 *   objective: string|null,
 *   direction: string|null,
 *   values: (number|null)[],
 *   feasible: boolean[]|null,
 *   bests: (number|null)[]|null,
 *   best: number|null,
 *   bestIndex: number|null,
 *   bestRecord: DataRecord|null,
 *   lastRecord: DataRecord|null,
 *   initial: Record<string, number>|null,
 *   before: number|null,
 *   change: number|null,
 *   changePct: number|null,
 *   minutesLeft: number|null,
 * }} RunFigures
 */

/**
 * @param {unknown} value
 * @returns {number|null}
 */
export function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The running best of `values` in `direction`; null until the first finite
 * value. The whole result is null when the direction is unknown: an inverted
 * best is worse than none.
 *
 * @param {(number|null)[]} values
 * @param {string|null|undefined} direction
 * @returns {(number|null)[]|null}
 */
export function runningBest(values, direction) {
  if (direction !== 'MAXIMIZE' && direction !== 'MINIMIZE') return null;
  const better = direction === 'MAXIMIZE' ? Math.max : Math.min;
  /** @type {number|null} */
  let best = null;
  return values.map((v) => {
    if (v !== null) best = best === null ? v : better(best, v);
    return best;
  });
}

/**
 * Whether a record meets every constraint, as the service judges it: a
 * strict inequality per observable; a missing or non-finite reading fails.
 *
 * @param {DataRecord} record
 * @param {Constraints} constraints
 * @returns {boolean}
 */
function meetsConstraints(record, constraints) {
  return Object.entries(constraints).every(([name, c]) => {
    const v = finite(record[name]);
    if (v === null) return false;
    return c.type === 'LESS_THAN' ? v < c.value : v > c.value;
  });
}

/** @returns {Arrivals} */
export function noArrivals() {
  return { jobId: null, seen: 0, first: null, last: null };
}

/**
 * Record a read of `jobId` that showed `count` evaluations at time `at`. A
 * different job, a count that went down, or a read while the run was not
 * running (paused, waiting) starts the record again, so no idle time is timed.
 *
 * @param {Arrivals} arrivals
 * @param {string|null} jobId
 * @param {number} count
 * @param {number} at
 * @param {boolean} [running]  whether the job was running at this read
 * @returns {Arrivals}
 */
export function recordArrivals(arrivals, jobId, count, at, running = true) {
  const latest = arrivals.last?.count ?? arrivals.seen;
  if (jobId !== arrivals.jobId || count < latest || !running) {
    const fresh = jobId === arrivals.jobId && arrivals.seen === count && !arrivals.first;
    return fresh ? arrivals : { jobId, seen: count, first: null, last: null };
  }
  if (count === latest) return arrivals;
  const point = { count, at };
  return { ...arrivals, first: arrivals.first ?? point, last: point };
}

/**
 * About how many minutes the evaluations left take, from the mean time of the
 * evaluations the panel timed; null with too few of them or no total.
 *
 * @param {Arrivals} arrivals
 * @param {number} done
 * @param {number|null} total
 * @returns {number|null}
 */
export function minutesLeft(arrivals, done, total) {
  const { first, last } = arrivals;
  if (!first || !last || total == null || done >= total) return null;
  const timed = last.count - first.count;
  if (timed < MIN_TIMED_EVALUATIONS) return null;
  const mean = (last.at - first.at) / timed;
  return Math.max(1, Math.round((mean * (total - done)) / 60000));
}

/**
 * @param {{
 *   data: DataRecord[]|null|undefined,
 *   objective: string|null,
 *   direction: string|null,
 *   total: number|null,
 *   initial: Record<string, number>|null,
 *   constraints?: Constraints|null,
 *   variables?: string[]|null,
 *   arrivals?: Arrivals|null,
 * }} source
 * @returns {RunFigures}
 */
export function figures({
  data,
  objective,
  direction,
  total,
  initial,
  constraints = null,
  variables = null,
  arrivals = null,
}) {
  const records = data || [];
  const values = objective ? records.map((r) => finite(r[objective])) : [];
  const limits = constraints && Object.keys(constraints).length ? constraints : null;
  const feasible = limits ? records.map((r) => meetsConstraints(r, limits)) : null;
  // The evaluations the service picks its best from: finite knobs, feasible.
  const counted = values.map((v, i) => {
    if (feasible && !feasible[i]) return null;
    if ((variables || []).some((name) => finite(records[i][name]) === null)) return null;
    return v;
  });
  const bests = runningBest(counted, direction);
  let bestIndex = null;
  if (bests) {
    for (let i = 0; i < bests.length; i += 1) {
      if (bests[i] !== null && (i === 0 || bests[i] !== bests[i - 1])) bestIndex = i;
    }
  }
  const best = bestIndex === null ? null : values[bestIndex];
  const before = objective && initial ? finite(initial[objective]) : null;
  const change = best !== null && before !== null ? best - before : null;
  const done = records.length;
  return {
    done,
    total,
    objective,
    direction,
    values,
    feasible,
    bests,
    best,
    bestIndex,
    bestRecord: bestIndex === null ? null : records[bestIndex],
    lastRecord: done ? records[done - 1] : null,
    initial,
    before,
    change,
    changePct:
      change !== null && before ? (change / Math.abs(/** @type {number} */ (before))) * 100 : null,
    minutesLeft: arrivals ? minutesLeft(arrivals, done, total) : null,
  };
}

/**
 * The figures of the present job (a `detail=full` JobResponse) with its stored
 * config, which names the objective, the constraints and the knobs, and plans
 * the total before the runner reports its own.
 *
 * @param {any} job
 * @param {import('./approval-model.js').OptimizationConfig|null} config
 * @param {Arrivals|null} [arrivals]
 * @returns {RunFigures}
 */
export function jobFigures(job, config, arrivals = null) {
  const objective = job?.objective_name || config?.objective_name || null;
  return figures({
    data: job?.data,
    objective,
    direction: job?.objective_direction || config?.objective_direction || null,
    total: finite(job?.total_evaluations) ?? plannedEvaluations(config),
    initial: job?.snapshots?.initial || null,
    constraints: /** @type {Constraints|null} */ (config?.constraints || null),
    variables: job?.variable_names || config?.variables || null,
    arrivals,
  });
}

/**
 * The figures of a stored run (`HistoricalRunDetails`). Its config holds the
 * direction the run used and its constraints; the total is what it planned.
 * The point the run applied is `details.best_point`.
 *
 * @param {any} details
 * @returns {RunFigures}
 */
export function resultFigures(details) {
  const config = details?.config || null;
  return figures({
    data: details?.data_records,
    objective: config?.objective_name || null,
    direction: config?.objective_direction || null,
    total: plannedEvaluations(config),
    initial: details?.initial_point || null,
    constraints: config?.constraints || null,
    variables: config?.variables || null,
  });
}
