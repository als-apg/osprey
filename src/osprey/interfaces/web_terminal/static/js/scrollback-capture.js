// @ts-check
/* OSPREY Web Terminal — Terminal Scrollback Capture
 *
 * Turns the live xterm buffer into the plain-text tail that rides along with a
 * feedback report. The server keeps no PTY backlog — pty_manager only hands out
 * a live iterator — so the browser is the only place this text exists, and the
 * capture has to travel in the request body.
 *
 * Two things make the raw buffer unusable as-is:
 *
 *   - xterm stores PHYSICAL rows. A command longer than the terminal is width
 *     hard-wrapped across several rows, each flagged `isWrapped`. Reporting
 *     those as separate lines shreds every long command and stack trace, so
 *     continuation rows are rejoined into one logical line here.
 *   - every row is padded out to the column count with spaces, which would
 *     otherwise dominate the byte budget.
 *
 * Both caps keep the NEWEST content and drop the oldest, because the failure a
 * report is about is at the bottom of the buffer. When anything is dropped the
 * capture is headed with a marker line so a reader never mistakes a truncated
 * capture for the whole session.
 *
 * The `term` argument is deliberately duck-typed and nothing here imports
 * xterm: the module is exercised against a hand-built fake buffer in vitest,
 * where no real Terminal can exist.
 */

/** Header line prepended when either cap dropped something. */
const TRUNCATION_MARKER = '[scrollback truncated]';

const DEFAULT_MAX_LINES = 2000;
const DEFAULT_MAX_BYTES = 1_000_000;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/**
 * One row of an xterm buffer, as much of it as this module touches.
 *
 * @typedef {object} BufferLineLike
 * @property {boolean} [isWrapped] - true when the row continues the row above
 * @property {(trimRight?: boolean) => string} translateToString
 */

/**
 * An xterm buffer (`term.buffer.active`), as much of it as this module touches.
 *
 * @typedef {object} BufferLike
 * @property {number} length - rows held, scrollback plus viewport
 * @property {(index: number) => (BufferLineLike | undefined)} getLine
 */

/**
 * An xterm `Terminal`, as much of it as this module touches.
 *
 * @typedef {object} TerminalLike
 * @property {{active?: BufferLike}} [buffer]
 */

/**
 * UTF-8 byte length — the unit the byte cap is measured in, since the capture
 * is budgeted against a request body and not against JS string length.
 *
 * @param {string} text
 * @returns {number}
 */
function utf8Length(text) {
  return encoder.encode(text).length;
}

/**
 * The longest suffix of `text` that fits in `maxBytes` UTF-8 bytes.
 *
 * @param {string} text
 * @param {number} maxBytes
 * @returns {string}
 */
function utf8Tail(text, maxBytes) {
  if (maxBytes <= 0) return '';
  const encoded = encoder.encode(text);
  if (encoded.length <= maxBytes) return text;
  let start = encoded.length - maxBytes;
  // Walk forward off any continuation byte (10xxxxxx) so the slice starts on a
  // character boundary — cutting inside a multi-byte character would decode to
  // a replacement character.
  while (start < encoded.length && (encoded[start] & 0b1100_0000) === 0b1000_0000) {
    start += 1;
  }
  return decoder.decode(encoded.subarray(start));
}

/**
 * Rebuild the buffer's logical lines: rejoin hard-wrapped rows, drop the column
 * padding, and discard the empty rows below the cursor that xterm keeps to fill
 * the viewport.
 *
 * @param {BufferLike} buffer
 * @returns {string[]} logical lines, oldest first
 */
function readLogicalLines(buffer) {
  /** @type {string[]} */
  const lines = [];
  const rows = Number.isFinite(buffer.length) ? buffer.length : 0;
  for (let i = 0; i < rows; i += 1) {
    const row = buffer.getLine(i);
    // translateToString(true) drops the row's trailing padding. A wrapped row
    // is a continuation of a row that filled the width, so nothing meaningful
    // sits between the two halves.
    const text = row ? row.translateToString(true) : '';
    if (row && row.isWrapped === true && lines.length > 0) {
      lines[lines.length - 1] += text;
    } else {
      lines.push(text);
    }
  }
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }
  return lines;
}

/**
 * Keep logical lines from the newest end until either cap is reached.
 *
 * With `tailFill`, a single line that alone blows the byte budget is kept as
 * its newest tail rather than dropped outright — losing a 2 MB stack trace
 * entirely would defeat the capture. Without it the selection only measures,
 * which is what the first (does-this-even-truncate) pass needs.
 *
 * @param {string[]} lines - logical lines, oldest first
 * @param {number[]} sizes - `lines[i]`'s UTF-8 byte length
 * @param {number} maxLines
 * @param {number} maxBytes
 * @param {boolean} tailFill
 * @returns {{text: string, truncated: boolean}}
 */
function takeNewest(lines, sizes, maxLines, maxBytes, tailFill) {
  /** @type {string[]} */
  const kept = [];
  let used = 0;
  let truncated = false;

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    if (kept.length >= maxLines) {
      truncated = true;
      break;
    }
    // Every line but the newest also pays for the newline that joins it.
    const cost = sizes[i] + (kept.length > 0 ? 1 : 0);
    if (used + cost > maxBytes) {
      if (tailFill && kept.length === 0) {
        const tail = utf8Tail(lines[i], maxBytes);
        if (tail !== '') kept.push(tail);
      }
      truncated = true;
      break;
    }
    used += cost;
    kept.push(lines[i]);
  }

  kept.reverse();
  return { text: kept.join('\n'), truncated };
}

/**
 * Capture the terminal's scrollback as plain text, newest-first-preserving.
 *
 * Hard-wrapped rows are rejoined into logical lines and trailing padding is
 * dropped. At most `maxLines` logical lines and `maxBytes` UTF-8 bytes are
 * returned, counting the marker line; the oldest content goes first, and a
 * `[scrollback truncated]` line heads the result whenever anything was dropped.
 *
 * A missing or buffer-less terminal captures the empty string rather than
 * throwing: a report without scrollback is still worth sending.
 *
 * @param {TerminalLike | null | undefined} term - an xterm `Terminal` (duck-typed)
 * @param {{maxLines?: number, maxBytes?: number}} [options]
 * @returns {string} the captured scrollback, `''` when there is nothing to capture
 */
export function captureScrollback(term, options = {}) {
  const { maxLines = DEFAULT_MAX_LINES, maxBytes = DEFAULT_MAX_BYTES } = options;

  const buffer = term && term.buffer ? term.buffer.active : null;
  if (!buffer || typeof buffer.getLine !== 'function') return '';

  const lines = readLogicalLines(buffer);
  if (lines.length === 0) return '';
  const sizes = lines.map(utf8Length);

  const whole = takeNewest(lines, sizes, maxLines, maxBytes, false);
  if (!whole.truncated) return whole.text;

  // The marker is part of the payload, so it comes out of the same budget.
  const budget = maxBytes - utf8Length(TRUNCATION_MARKER) - 1;
  if (budget <= 0) return TRUNCATION_MARKER;

  const { text } = takeNewest(lines, sizes, maxLines, budget, true);
  return text === '' ? TRUNCATION_MARKER : `${TRUNCATION_MARKER}\n${text}`;
}
