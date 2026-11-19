// @ts-check
/**
 * How a run ended and what it left on the machine, as pure data, for the
 * outcome of the present job and for a past run.
 *
 * Every sentence follows from a field the service sends; where the fields do
 * not tell, the sentence says that it is not known:
 *
 * - A completed run ends with its `on_complete` (from its stored config):
 *   `apply_best` writes the best feasible point (the stored run's
 *   `best_point` when the service stored one), `restore` writes the starting
 *   setpoints back, `hold` leaves the last point tried. A completed run with
 *   no config read names no end action.
 * - A cancelled or failed run writes nothing after the stop: the machine
 *   holds the last point written, and nothing restores by itself. The stop
 *   can land inside an evaluation after its write, so the last evaluation
 *   recorded is not claimed as the point on the machine. A failed run whose
 *   end action is `restore` may have failed in that write: what it left is
 *   not known.
 * - A readonly run, and a run whose write request ended before anyone
 *   approved it, wrote nothing.
 * - A stored run whose job the service no longer holds has no status: how it
 *   ended, and so what it left, is not known. A job that has not ended is
 *   under way, never a past run.
 *
 * @module outcome-model
 */

import { authorizationNote } from './approval-model.js';
import { isActive } from './poller.js';
import { STOP_WORDS, endedWord, endingOf } from './words.js';

/**
 * @typedef {import('./run-model.js').RunFigures} RunFigures
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {'completed'|'failed'|'cancelled'|'open'|'unknown'} OutcomeKind
 * @typedef {'ok'|'warn'|'err'|'info'} Tone
 * @typedef {import('./objective-chart.js').EndTone} EndTone
 *
 * What the machine holds after the run. `values` are the knob values the
 * fields prove are on the machine; null when they do not, which is always
 * the case for a stopped run (a stop inside a later evaluation may have
 * overtaken its last record). `unapplied` is the evaluation (from 1) of the
 * best point a cancelled `apply_best` run did not get to write.
 * @typedef {{
 *   tone: Tone,
 *   text: string,
 *   written: boolean|null,
 *   values: Record<string, number>|null,
 *   unapplied: number|null,
 * }} MachineState
 *
 * @typedef {{
 *   kind: OutcomeKind,
 *   status: string|null,
 *   word: string,
 *   tone: Tone,
 *   lead: string,
 *   machine: MachineState,
 *   reasons: string[],
 *   end: { index: number, label: string, tone: EndTone }|null,
 * }} Outcome
 */

/**
 * @param {string|null|undefined} status
 * @returns {OutcomeKind}
 */
function outcomeKind(status) {
  if (!status) return 'unknown';
  return endingOf(status)?.kind ?? 'open';
}

/**
 * The run's word for its outcome: "Completed", "Failed at 11", "Cancelled at 29".
 *
 * @param {string|null} status
 * @param {number} done
 * @returns {string}
 */
function outcomeWord(status, done) {
  const ending = endingOf(status);
  if (ending) return endedWord(ending, done);
  return status ? String(status) : 'Status unknown';
}

/**
 * Whether the run's write request ended before anyone approved it: the
 * service then records `ended` with reason `released`, and nothing was written.
 *
 * @param {any} auth  the job's `write_authorization`
 * @returns {boolean}
 */
function neverApproved(auth) {
  return auth?.state === 'ended' && auth.reason === 'released';
}

/**
 * The knob values of `record` for the config's variables; null when it has none.
 *
 * @param {Record<string, unknown>|null|undefined} record
 * @param {string[]} names
 * @returns {Record<string, number>|null}
 */
export function pointOf(record, names) {
  if (!record) return null;
  /** @type {Record<string, number>} */
  const out = {};
  for (const name of names) {
    const v = record[name];
    if (typeof v === 'number' && Number.isFinite(v)) out[name] = v;
  }
  return Object.keys(out).length ? out : null;
}

/**
 * Whether two points name the same value for every knob of `names`.
 *
 * @param {Record<string, number>|null} a
 * @param {Record<string, number>|null} b
 * @param {string[]} names
 */
function samePoint(a, b, names) {
  return !!a && !!b && names.every((n) => a[n] === b[n]);
}

/**
 * What the run left on the machine.
 *
 * `applied` is the point a completed `apply_best` run wrote when the service
 * stored it (`best_point` of a stored run); without it the best feasible
 * evaluation of the figures stands for it.
 *
 * @param {{
 *   kind: OutcomeKind,
 *   config: OptimizationConfig|null,
 *   figures: RunFigures|null,
 *   auth?: any,
 *   variables: string[],
 *   applied?: Record<string, unknown>|null,
 * }} source
 * @returns {MachineState}
 */
export function leftOnMachine({ kind, config, figures, auth = null, variables, applied = null }) {
  const done = figures?.done ?? 0;
  const onComplete = config?.on_complete;
  /** @type {(tone: Tone, text: string, written: boolean|null, values?: Record<string, number>|null, unapplied?: number|null) => MachineState} */
  const state = (tone, text, written, values = null, unapplied = null) => ({
    tone,
    text,
    written,
    values,
    unapplied,
  });
  if (kind === 'open') return state('info', 'The run has not ended.', null);
  if (kind === 'unknown') {
    return state(
      'info',
      'The service no longer holds the status of this run, so how it ended and what it left on the machine are not known.',
      null
    );
  }
  if (config?.execution_mode === 'readonly') {
    return state('info', 'Nothing was written: the run could only read the machine.', false);
  }
  if (neverApproved(auth)) {
    return state(
      'info',
      'Nothing was written: the run ended before its writes were approved.',
      false
    );
  }
  if (kind === 'completed') {
    switch (onComplete) {
      case 'apply_best': {
        const best = pointOf(figures?.bestRecord, variables);
        const stored = pointOf(applied, variables);
        const point = stored || best;
        // The evaluation is named only when the figures' best is the point written.
        const at =
          figures?.bestIndex != null && (!stored || samePoint(stored, best, variables))
            ? ` (evaluation ${figures.bestIndex + 1})`
            : '';
        return state('ok', `The best point${at} was applied at the end.`, true, point);
      }
      case 'restore':
        return state(
          'ok',
          'The starting setpoints were written back at the end.',
          true,
          pointOf(figures?.initial, variables)
        );
      case 'hold':
        return state(
          'ok',
          `The machine holds the last point tried${done ? ` (evaluation ${done})` : ''}.`,
          true,
          pointOf(figures?.lastRecord, variables)
        );
      default:
        return state(
          'info',
          'The run completed. Its settings were not read, so its end action is not known.',
          true
        );
    }
  }
  // A failed run whose end action writes the starting setpoints back may have
  // failed in that write: some, all or none of them were written.
  if (kind === 'failed' && onComplete === 'restore') {
    return state(
      'warn',
      'What the machine holds is not known: if the run failed while it wrote the starting setpoints back, some, all or none of them were written. Check the machine.',
      null
    );
  }
  // Cancelled or failed: no write follows the stop. The stop can land inside
  // an evaluation after its write, so no recorded point is claimed.
  if (!done) {
    return state(
      'warn',
      'No evaluation was recorded. Nothing was restored: any knob the run wrote keeps its value.',
      null
    );
  }
  // Only an end action that would have applied the best point leaves it unapplied.
  const best = figures?.bestIndex;
  const unapplied =
    kind === 'cancelled' && onComplete === 'apply_best' && best != null && best !== done - 1
      ? best + 1
      : null;
  const notBest = unapplied ? ` The best point (evaluation ${unapplied}) was not applied.` : '';
  return state('warn', `${STOP_WORDS.after}${notBest}`, true, null, unapplied);
}

/**
 * The reasons the service gave for the end: the job's error, and a write
 * authorization that ended with a reason.
 *
 * @param {any} job
 * @returns {string[]}
 */
export function endReasons(job) {
  /** @type {string[]} */
  const reasons = [];
  const error = typeof job?.error_message === 'string' ? job.error_message.trim() : '';
  if (error) reasons.push(error);
  const auth = job?.write_authorization;
  if (auth?.state === 'ended') {
    const note = authorizationNote(auth)?.text;
    const detail = typeof auth.detail === 'string' ? auth.detail.trim() : '';
    const text = [note, detail].filter(Boolean).join(' ');
    if (text && !reasons.includes(text)) reasons.push(text);
  }
  return reasons;
}

/**
 * How the run ended, in words: the lead of the banner, what the machine
 * holds, the reasons and the chart's end mark.
 *
 * @param {{
 *   status: string|null,
 *   job: any,
 *   config: OptimizationConfig|null,
 *   figures: RunFigures|null,
 *   applied?: Record<string, unknown>|null,
 * }} source  `applied`: the stored run's `best_point`, when there is one
 * @returns {Outcome}
 */
export function outcomeOf({ status, job, config, figures, applied = null }) {
  const kind = outcomeKind(status);
  const done = figures?.done ?? 0;
  const variables = config?.variables || job?.variable_names || [];
  const machine = leftOnMachine({
    kind,
    config,
    figures,
    auth: job?.write_authorization,
    variables,
    applied,
  });
  /** @type {Record<OutcomeKind, string>} */
  const leads = {
    completed: 'Completed.',
    failed: done ? `Stopped at evaluation ${done}.` : 'Failed.',
    cancelled: done ? `Cancelled at evaluation ${done}.` : 'Cancelled.',
    open: isActive(status) ? `This run is under way (${status}).` : `This job is ${status}.`,
    unknown: 'Status unknown.',
  };
  const ending = endingOf(status);
  /** @type {Record<string, { label: string, tone: EndTone }>} */
  const marks = {
    failed: { label: 'stopped', tone: 'error' },
    cancelled: { label: 'cancelled', tone: 'warning' },
  };
  const mark = marks[kind];
  return {
    kind,
    status,
    word: outcomeWord(status, done),
    tone: !ending || (kind === 'completed' && machine.tone !== 'ok') ? 'info' : ending.tone,
    lead: leads[kind],
    machine,
    reasons: kind === 'failed' || kind === 'cancelled' ? endReasons(job) : [],
    end: mark && done ? { index: done - 1, ...mark } : null,
  };
}

/**
 * The change of each knob from before the run to the values on the machine,
 * in the config's order; knobs with no value on either side are left out.
 *
 * @param {string[]} names
 * @param {Record<string, number>|null} before
 * @param {Record<string, number>|null} after
 * @returns {{ name: string, from: number, to: number, delta: number }[]}
 */
export function knobChanges(names, before, after) {
  if (!before || !after) return [];
  return names
    .filter((n) => typeof before[n] === 'number' && typeof after[n] === 'number')
    .map((name) => ({
      name,
      from: before[name],
      to: after[name],
      delta: after[name] - before[name],
    }));
}
