// @ts-check
/**
 * BLUESKY panel — CSV export for the Results view.
 *
 * The Results table is a bounded PREVIEW: `GET /runs/{id}/data` serves 100 rows
 * by default while reporting the run's true `row_count` separately. This module
 * is the other half of that split — the path by which an operator gets the
 * whole run out, as a file, in one click.
 *
 * Two rules shape everything here.
 *
 * First, the CSV is an ANALYSIS artifact, not a rendering of the table. The
 * table's `formatCell` writes `—` for an absent value and rounds numbers to six
 * decimals, both correct for a human reading a screen and both wrong in a file
 * someone fits a model to. So this module formats independently: absent becomes
 * an empty field, and numbers keep every digit the wire carried.
 *
 * Second, the file is built in the browser rather than served by a new route.
 * `read_proxy.py` is a GET-only, verbatim JSON passthrough that recomputes
 * nothing; a `/csv` route would be the first response it had to format, and the
 * first place it could disagree with the bridge. Fetching the same JSON with a
 * larger `max_rows` and formatting it here keeps that boundary intact.
 *
 * Nothing in this file issues a write verb, and nothing in it decides what to
 * export — `results-view.js` owns the fetch and hands the values down.
 */

/** Fields containing any of these cannot be written bare (RFC 4180 §2.6). */
const NEEDS_QUOTING = /[",\r\n]/;

/**
 * One wire value as a CSV field.
 *
 * `null`/`undefined` become an empty field — the CSV spelling of "no value".
 * Numbers go through `String`, which is what preserves full precision;
 * `formatCell`'s `toFixed(6)` is display rounding and must not reach a file.
 * Non-finite numbers are written as their names rather than blanked, because a
 * `NaN` in a run is a reading that failed, which is a different fact from a
 * channel that was never read at all.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function csvField(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  if (!NEEDS_QUOTING.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

/**
 * A full CSV document: the header row, then one row per record, each line
 * terminated by CRLF per RFC 4180 (which is also what keeps Excel on Windows
 * from reading the whole file as a single row).
 *
 * Rows are padded to the header's width. The bridge should never serve a ragged
 * window, but a row that arrived short would otherwise shift every subsequent
 * column of that line one place left — a corruption that reads as plausible
 * data rather than as an error.
 *
 * No UTF-8 BOM is emitted: the columns are channel names, and a BOM confuses
 * naive parsers more often than it helps here.
 *
 * @param {string[]} columns
 * @param {Array<Array<unknown>>} rows
 * @returns {string}
 */
export function toCsv(columns, rows) {
  if (columns.length === 0) return '';
  const lines = [columns.map(csvField).join(',')];
  for (const row of rows) {
    const fields = [];
    for (let index = 0; index < columns.length; index += 1) fields.push(csvField(row[index]));
    lines.push(fields.join(','));
  }
  return `${lines.join('\r\n')}\r\n`;
}

/**
 * Reduce one component to something safe to put in a filename: anything outside
 * `[A-Za-z0-9._-]` collapses to a single dash, and leading/trailing dashes and
 * dots are trimmed off.
 *
 * This matters beyond tidiness. The plan name arrives from the bridge, so it is
 * not this panel's to trust — without this, a name carrying `../` would be
 * handed to the save picker as a relative path.
 *
 * @param {string} text
 * @returns {string}
 */
function slug(text) {
  return text
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^[-.]+|[-.]+$/g, '');
}

/**
 * The name the save dialog opens pre-filled with.
 *
 * The run id is truncated to its first 8 characters: run ids are UUIDs, and the
 * full 36 characters push the meaningful part of the name off the end of every
 * file listing. Eight is enough to tell one run from another by eye.
 *
 * @param {string|null|undefined} planName
 * @param {string} runId
 * @returns {string}
 */
export function exportFilename(planName, runId) {
  const plan = planName ? slug(planName) : '';
  const id = slug(runId).slice(0, 8);
  return `${plan || 'run'}-${id}.csv`;
}

/** @typedef {{saved: boolean, method: 'picker'|'download'}} SaveResult */

/**
 * Put `text` on the operator's disk under `filename`.
 *
 * Chromium browsers expose `showSaveFilePicker`, which opens the real OS save
 * dialog and lets the operator choose any folder. Firefox and Safari have no
 * equivalent — there, a file can only go to the Downloads folder — so this
 * falls back to a plain anchor download. The caller is expected to report which
 * path ran (`method`), because "it saved" and "it saved somewhere you did not
 * choose" are different things to be told.
 *
 * The two failure modes at the picker are deliberately NOT treated alike:
 *
 * - `AbortError` is the operator closing the dialog. That is a decision, and it
 *   is honored — returning `saved: false` rather than falling through, because
 *   a fallback download here would drop an unwanted file in Downloads every
 *   time someone changed their mind.
 * - Anything else (a `SecurityError` from an embedding context that denies the
 *   API, a transport failure) is an environment limit rather than a choice, so
 *   the download path still runs and the operator still gets their data.
 *
 * @param {string} filename
 * @param {string} text
 * @param {{picker?: unknown}} [options] `picker` defaults to the browser's
 *   `showSaveFilePicker`. It is injected rather than read inline so both
 *   branches are reachable under a DOM shim that has no File System Access API.
 * @returns {Promise<SaveResult>}
 */
export async function saveCsv(filename, text, options = {}) {
  const picker =
    'picker' in options ? options.picker : /** @type {any} */ (globalThis).showSaveFilePicker;

  if (typeof picker === 'function') {
    try {
      const handle = await picker({
        suggestedName: filename,
        types: [{ description: 'CSV file', accept: { 'text/csv': ['.csv'] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(text);
      await writable.close();
      return { saved: true, method: 'picker' };
    } catch (error) {
      if (/** @type {any} */ (error)?.name === 'AbortError') {
        return { saved: false, method: 'picker' };
      }
      // Fall through to the download path.
    }
  }

  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = /** @type {HTMLAnchorElement} */ (document.createElement('a'));
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  anchor.remove();
  // Revoked on the next turn rather than immediately: some browsers have not
  // finished reading the blob when `click()` returns.
  setTimeout(() => URL.revokeObjectURL(url), 0);
  return { saved: true, method: 'download' };
}
