// @ts-check
/**
 * The panel's words for what the service names, in one place: the methods,
 * the end actions, what a stop leaves on the machine, when a run may write,
 * the status of a job, the direction of an objective, a change, and an
 * instant. Every view takes these words from here, so one thing reads the
 * same on every screen. No state; an instant reads the facility zone stamped
 * on the page (facility-time.js), the rest touches no DOM.
 *
 * @module words
 */

import { facilityDayKey, formatFacilityTime } from '/design-system/js/facility-time.js';
import { fmt } from './hyperscript.js';

/**
 * @typedef {'ok'|'err'|'warn'} EndedTone
 * @typedef {{ kind: 'completed'|'failed'|'cancelled', word: string, tone: EndedTone }} Ending
 * @typedef {{ tone: string, label: string }} Chip
 * @typedef {{ word: string, seek: string, better: string }} DirectionWords
 */

/** Every `AlgorithmType`, in the operator's words. */
const METHODS = Object.freeze(
  /** @type {Record<string, string>} */ ({
    random: 'Random search',
    latin_hypercube: 'Latin hypercube',
    upper_confidence_bound: 'Bayesian (UCB)',
    expected_improvement: 'Bayesian (EI)',
    baxus: 'Bayesian (BAxUS)',
    neldermead: 'Nelder–Mead',
    extremum_seeking: 'Extremum seeking',
    rcds: 'RCDS',
    nsga2: 'NSGA-II',
    cnsga: 'Constrained NSGA-II',
  })
);

/**
 * A method's name. A config that names none runs the service's default,
 * random search. An unknown method keeps its identifier.
 *
 * @param {string|null|undefined} algorithm
 * @returns {string}
 */
export function methodLabel(algorithm) {
  const id = algorithm || 'random';
  return METHODS[id] ?? id;
}

/** What each `on_complete` does with the knobs when the run completes. */
const END_ACTIONS = Object.freeze(
  /** @type {Record<string, { short: string, sentence: string }>} */ ({
    apply_best: {
      short: 'Apply best',
      sentence: 'At the end the knobs are set to the best point found.',
    },
    restore: {
      short: 'Restore start',
      sentence: 'At the end the knobs are set back to where they started.',
    },
    hold: { short: 'Hold last', sentence: 'At the end the knobs stay at the last point tried.' },
  })
);

/**
 * An end action's words. A config that names none has the service's
 * default, `apply_best`. An unknown action keeps its identifier and has no
 * sentence.
 *
 * @param {string|null|undefined} onComplete
 * @returns {{ short: string, sentence: string }}
 */
export function endWords(onComplete) {
  const id = onComplete || 'apply_best';
  return END_ACTIONS[id] ?? { short: id, sentence: '' };
}

/**
 * What a cancel or a failure leaves, whatever the end action: no write
 * follows the stop and nothing restores by itself. `ahead`: before a run;
 * `now`: while a cancel is asked or under way; `after`: once the run has
 * stopped; `short`: the value of a settings row.
 */
export const STOP_WORDS = Object.freeze({
  ahead:
    'If the run is cancelled or fails, nothing is restored: the machine keeps the last point written.',
  now: 'The machine keeps the last point written. Nothing is restored.',
  after: 'The machine keeps the last point written. Nothing was restored.',
  short: 'Keeps the last point written',
});

/**
 * When a run may write: only after a person starts it, and only under a
 * write authorization. The owner approves its request, or the owner's write
 * switch already allows it and the service grants it at once.
 */
export const BEFORE_A_WRITE =
  'Nothing is written before a person starts the run and the run holds a write ' +
  "authorization: the owner approves its request, or the owner's write switch already allows it.";

/** How a job ended, by its status. A status not listed has not ended. */
export const ENDINGS = Object.freeze(
  /** @type {Record<string, Ending>} */ ({
    completed: { kind: 'completed', word: 'Completed', tone: 'ok' },
    error: { kind: 'failed', word: 'Failed', tone: 'err' },
    cancelled: { kind: 'cancelled', word: 'Cancelled', tone: 'warn' },
  })
);

/**
 * @param {string|null|undefined} status  a job status
 * @returns {Ending|null}  null for a job that has not ended, or no status
 */
export function endingOf(status) {
  return (status && ENDINGS[status]) || null;
}

/**
 * The word of an ended run, with the evaluation it stopped at when it did
 * not complete: "Completed", "Failed at 11", "Cancelled at 29".
 *
 * @param {Ending} ending
 * @param {number} [done]  the evaluations recorded
 * @returns {string}
 */
export function endedWord(ending, done = 0) {
  return ending.kind !== 'completed' && done ? `${ending.word} at ${done}` : ending.word;
}

/** An ended tone as the tone of each primitive that shows it. */
const CHIP_TONES = Object.freeze({ ok: 'done', err: 'failed', warn: 'warn' });
/** The status dot has no warning colour: a stop shows by its shape, hollow. */
const DOT_TONES = Object.freeze({ ok: 'ok', err: 'err', warn: 'hollow' });

/**
 * The status of each screen of a job: its chip tone, the toolbar's label
 * and the runs list's short word.
 */
export const SCREEN_STATUS = Object.freeze(
  /** @type {Record<string, { tone: string, label: string, short: string }>} */ ({
    review: { tone: 'info', label: 'Ready to start', short: 'ready' },
    'review-changed': { tone: 'stale', label: 'Draft changed', short: 'changed' },
    'review-no-owner': { tone: 'warn', label: 'No account', short: 'no account' },
    'waiting-approval': { tone: 'pending', label: 'Waiting for approval', short: 'waiting' },
    running: { tone: 'running', label: 'Running', short: 'running' },
    paused: { tone: 'pending', label: 'Paused', short: 'paused' },
    ...Object.fromEntries(
      Object.values(ENDINGS).map((e) => [
        e.kind,
        { tone: CHIP_TONES[e.tone], label: e.word, short: e.word.toLowerCase() },
      ])
    ),
  })
);

/**
 * The chip of an ended run.
 *
 * @param {Ending} ending
 * @param {number} [done]
 * @returns {Chip}
 */
export function endedChip(ending, done = 0) {
  return { tone: CHIP_TONES[ending.tone], label: endedWord(ending, done) };
}

/**
 * The status dot of an ended run.
 *
 * @param {Ending} ending
 * @returns {Chip}
 */
export function endedDot(ending) {
  return { tone: DOT_TONES[ending.tone], label: ending.word };
}

/** Each objective direction: its name, the value sought, and which is better. */
export const DIRECTIONS = Object.freeze(
  /** @type {Record<string, DirectionWords>} */ ({
    MAXIMIZE: { word: 'Maximise', seek: 'highest', better: 'higher is better' },
    MINIMIZE: { word: 'Minimise', seek: 'lowest', better: 'lower is better' },
  })
);

/**
 * @param {string|null|undefined} direction  `MAXIMIZE` or `MINIMIZE`, any case
 * @returns {DirectionWords|null}  null for none or an unknown direction
 */
export function directionWords(direction) {
  return DIRECTIONS[String(direction || '').toUpperCase()] ?? null;
}

/**
 * A direction's name; an unknown direction as it is.
 *
 * @param {string|null} direction
 * @returns {string|null}
 */
export function directionWord(direction) {
  return directionWords(direction)?.word ?? direction;
}

/**
 * A signed number to four significant figures: "+0.358", "−0.166", and
 * "±0" for zero. Empty for no number.
 *
 * @param {number|null|undefined} value
 * @returns {string}
 */
export function signed(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '';
  if (value === 0) return '±0';
  return `${value > 0 ? '+' : '\u2212'}${fmt(Math.abs(value), 4)}`;
}

/**
 * A change in percent: one decimal below 10 %, whole numbers above.
 * "+60 %", "−4.2 %", "±0 %"; empty for no number.
 *
 * @param {number|null|undefined} pct
 * @returns {string}
 */
export function percent(pct) {
  if (typeof pct !== 'number' || !Number.isFinite(pct)) return '';
  const size = Math.abs(pct);
  const text = size >= 9.95 ? size.toFixed(0) : size.toFixed(1);
  if (Number(text) === 0) return '±0\u00a0%';
  return `${pct > 0 ? '+' : '\u2212'}${text}\u00a0%`;
}

/**
 * The best so far against the value before the run: a percentage of the
 * value before, else (the value before is zero) the difference. Null when
 * either value is missing.
 *
 * @param {{ change: number|null, changePct: number|null, before: number|null }} f
 * @returns {string|null}
 */
export function changeText(f) {
  if (f.change === null || f.before === null) return null;
  return f.changePct !== null ? percent(f.changePct) : signed(f.change);
}

/** @type {import('/design-system/js/facility-time.js').FacilityTimeOptions} */
const CLOCK = Object.freeze({ hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });

/** @type {import('/design-system/js/facility-time.js').FacilityTimeOptions} */
const DATED = Object.freeze({ day: 'numeric', month: 'short', ...CLOCK });

/**
 * An instant on the facility clock, in the viewer's locale: its time of day
 * when it falls on the facility's today, else its date and time of day.
 * Empty for no instant or an unreadable one.
 *
 * @param {Date|string|number|null|undefined} value  a Date, an ISO time or epoch ms
 * @param {Date|number} now  the instant "today" is read from
 * @returns {string}
 */
export function instantText(value, now) {
  const day = facilityDayKey(value);
  if (!day) return '';
  return formatFacilityTime(value, day === facilityDayKey(now) ? CLOCK : DATED);
}
