// @ts-check
/**
 * OSPREY Channel Finder — the panel's one severity vocabulary
 *
 * Every severity the panel draws is one of {@link SEVERITIES}: the alarm
 * server's words, OK, and NONE for no state. {@link severityOf} reads any
 * spelling the routes carry, {@link worstSeverity} is the one comparison of
 * severities, and {@link severityMark} the one mark: a shape and a colour per
 * severity (severity.css), with the word always printed beside it.
 */

import { icon } from '/design-system/js/icons.js';
import { esc } from './utils.js';

/**
 * The severities, worst first: the order of the connectors' severity rank.
 * @type {readonly ['UNDEFINED', 'INVALID', 'MAJOR', 'MINOR', 'OK', 'NONE']}
 */
export const SEVERITIES = Object.freeze(
  /** @type {const} */ (['UNDEFINED', 'INVALID', 'MAJOR', 'MINOR', 'OK', 'NONE'])
);

/** @typedef {typeof SEVERITIES[number]} Severity */

/**
 * The alarm server's severities in the order a band of counts shows them (the
 * mockup's order, not the rank).
 * @type {readonly ['MAJOR', 'INVALID', 'MINOR', 'UNDEFINED']}
 */
export const BAND_SEVERITIES = Object.freeze(
  /** @type {const} */ (['MAJOR', 'INVALID', 'MINOR', 'UNDEFINED'])
);

/**
 * Whether a band of counts shows `severity` at `count`: UNDEFINED, which no
 * record reports, only when something is counted under it.
 * @param {Severity} severity
 * @param {number} count
 * @returns {boolean}
 */
export function inBand(severity, count) {
  return severity !== 'UNDEFINED' || count > 0;
}

/** EPICS record severities by their wire number. */
const RECORD_SEVERITIES = /** @type {const} */ (['OK', 'MINOR', 'MAJOR', 'INVALID']);

/** The alarm server's words, each also spelled with `_ACK` once acknowledged. */
const SERVER_WORDS = new Set(['UNDEFINED', 'INVALID', 'MAJOR', 'MINOR']);

/**
 * The severity of `value`: an alarm server severity, its `_ACK` form, or an
 * EPICS record severity number 0-3 as the values route carries it. No value is
 * NONE; anything else is UNDEFINED.
 * @param {unknown} value
 * @returns {Severity}
 */
export function severityOf(value) {
  if (value === null || value === undefined) return 'NONE';
  if (typeof value === 'number') {
    return Number.isInteger(value) && value >= 0 && value < RECORD_SEVERITIES.length
      ? RECORD_SEVERITIES[value]
      : 'UNDEFINED';
  }
  if (typeof value !== 'string') return 'UNDEFINED';
  if (value === 'OK') return 'OK';
  const word = value.endsWith('_ACK') ? value.slice(0, -'_ACK'.length) : value;
  return SERVER_WORDS.has(word) ? /** @type {Severity} */ (word) : 'UNDEFINED';
}

/**
 * The worst of `severities`; NONE for none.
 * @param {Iterable<Severity>} severities
 * @returns {Severity}
 */
export function worstSeverity(severities) {
  let worst = SEVERITIES.length - 1;
  for (const severity of severities) {
    const rank = SEVERITIES.indexOf(severity);
    if (rank >= 0 && rank < worst) worst = rank;
  }
  return SEVERITIES[worst];
}

/**
 * The mark of a severity: its shape and its word, or `text` in place of the
 * word. A severity outside the vocabulary draws as UNDEFINED.
 * @param {Severity} severity
 * @param {string} [text]
 * @returns {string}
 */
export function severityMark(severity, text) {
  const known = SEVERITIES.includes(severity) ? severity : 'UNDEFINED';
  const shape = known === 'OK' ? icon('check').outerHTML : '';
  return (
    `<span class="cf-sev cf-sev--${known.toLowerCase()}">` +
    `<i aria-hidden="true">${shape}</i>${esc(text ?? known)}</span>`
  );
}
