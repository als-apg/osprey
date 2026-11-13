/**
 * A new session's title during its first turn, for the session model.
 *
 * The listing has a session's first prompt only some seconds after the turn
 * starts, and the hub names a session when a listing shows that prompt
 * (session_naming.py). Until then the session has no title. So while the
 * active session is untitled, the chase re-lists at {@link TITLE_CHASE_MS}
 * after the operator sends (or the turn records its transcript), and in the
 * meantime the prompt the chat sent stands in as the title, shortened as the
 * hub shortens a first prompt.
 */

/** The re-listings that look for a new session's first prompt, after a send. */
export const TITLE_CHASE_MS = [3000, 10_000, 30_000];

/** The longest stand-in title made from a prompt (the hub's `PROMPT_TITLE_CHARS`). */
export const PROMPT_TITLE_CHARS = 60;

/**
 * A prompt as a title, the way the hub shortens a first prompt: one line, at
 * most {@link PROMPT_TITLE_CHARS}, cut at a word in its second half.
 * @param {string} prompt
 * @returns {string}
 */
export function shortenedPrompt(prompt) {
  const text = prompt.split(/\s+/).filter(Boolean).join(' ');
  if (text.length <= PROMPT_TITLE_CHARS) return text;
  let cut = text.slice(0, PROMPT_TITLE_CHARS - 1);
  const space = cut.lastIndexOf(' ');
  if (space > PROMPT_TITLE_CHARS / 2) cut = cut.slice(0, space);
  return `${cut.trimEnd()}\u2026`;
}

/**
 * @typedef {object} ChaseRow
 * @property {string|null} session_key
 * @property {boolean} [current_for_key]
 * @property {string} [title]
 * @property {string|null} [title_source]
 * @property {boolean} [synthesized]
 */

/** @param {ChaseRow | null} row  whether the listing has no title for it yet */
const untitled = (row) => Boolean(row) && Boolean(row?.synthesized || !row?.title_source);

/**
 * @param {object} model
 * @param {() => string|null} model.pointer  the tab's session key
 * @param {() => ChaseRow|null} model.activeRow
 * @param {() => ChaseRow[]} model.held  every listed row the model holds
 * @param {() => void} model.refresh  a throttled re-listing
 * @param {() => void} model.changed  tell the model's listeners the rows changed
 */
export function createTitleChase({ pointer, activeRow, held, refresh, changed }) {
  /** @type {{key: string, title: string} | null} */
  let standIn = null;
  /** @type {{key: string, step: number} | null} */
  let chase = null;
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let timer;

  /** Put the stand-in title on the active session's untitled listed rows. */
  function paint() {
    if (!standIn) return false;
    const { key, title } = standIn;
    const rows = held().filter(
      (r) => r.session_key === key && r.current_for_key && !r.title_source
    );
    for (const row of rows) row.title = title;
    return rows.length > 0;
  }

  function stop() {
    clearTimeout(timer);
    timer = undefined;
    chase = null;
  }

  /** Re-list at the next delay while the active session is untitled. */
  function next() {
    const key = pointer();
    if (!key || !untitled(activeRow())) return;
    if (chase?.key === key && timer) return;
    if (chase?.key !== key) chase = { key, step: 0 };
    if (chase.step >= TITLE_CHASE_MS.length) return;
    const delay = TITLE_CHASE_MS[chase.step];
    chase.step += 1;
    timer = setTimeout(() => {
      timer = undefined;
      refresh();
    }, delay);
  }

  return {
    /**
     * The chat sent a prompt: on an untitled session it stands in as the title,
     * and the re-listings start.
     * @param {{text?: string, replay?: boolean}} detail
     */
    sent(detail) {
      const key = pointer();
      const text = detail?.text?.trim() ?? '';
      if (detail?.replay || !key || !text || text.startsWith('/')) return;
      if (!untitled(activeRow())) return;
      standIn = { key, title: shortenedPrompt(text) };
      if (paint()) changed();
      next();
    },
    /** The turn on the tab's session recorded its transcript. */
    moved: next,
    /** @param {string|null} key  the tab moved to another session */
    pointed(key) {
      if (standIn && standIn.key !== key) standIn = null;
      if (chase && chase.key !== key) stop();
    },
    /** After a listing: a titled active session ends the chase and the stand-in. */
    listed() {
      if (!untitled(activeRow())) {
        standIn = null;
        stop();
        return;
      }
      paint();
      if (chase && !timer) next();
    },
    /** @param {string} key @returns {string|null} the stand-in title for a synthesized row */
    standInFor: (key) => (standIn?.key === key ? standIn.title : null),
    reset() {
      stop();
      standIn = null;
    },
  };
}
