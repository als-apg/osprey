// @ts-check
/**
 * The title of a new session during its first turn (session-title-chase.js):
 * the prompt the chat sent stands in as the title until a listing names the
 * session, shortened to one line of at most 60 characters; the re-listings
 * run at 3 s, 10 s and 30 s, one step at a time, and end when the session is
 * titled, when the tab moves to another session, or on reset.
 *
 *   npx vitest run tests/interfaces/web_terminal/session-title-chase.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import {
  createTitleChase,
  shortenedPrompt,
  PROMPT_TITLE_CHARS,
  TITLE_CHASE_MS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/session-title-chase.js';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/session-title-chase.js').ChaseRow} ChaseRow */

const KEY = 'sess-a';

/**
 * A small session model around the chase: a tab key, a list of held rows, and
 * an active row that is the held current row of the key, or a synthesized one.
 * @param {string|null} [key]
 */
function makeModel(key = KEY) {
  const model = {
    /** @type {string|null} */
    key,
    /** @type {ChaseRow[]} */
    rows: [],
    refresh: vi.fn(),
    changed: vi.fn(),
  };
  const chase = createTitleChase({
    pointer: () => model.key,
    activeRow: () =>
      model.rows.find((r) => r.session_key === model.key && r.current_for_key) ??
      (model.key ? { session_key: model.key, current_for_key: true, synthesized: true } : null),
    held: () => model.rows,
    refresh: model.refresh,
    changed: model.changed,
  });
  return { model, chase };
}

/** @returns {ChaseRow} a listed current row of the key that the hub has not titled */
const untitledRow = () => ({ session_key: KEY, current_for_key: true, title: '' });

/** @returns {ChaseRow} a listed current row of the key that the hub has titled */
const titledRow = () => ({
  session_key: KEY,
  current_for_key: true,
  title: 'Fix the orbit',
  title_source: 'prompt',
});

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('shortenedPrompt', () => {
  test('a short prompt is kept and whitespace and newlines collapse to single spaces', () => {
    expect(shortenedPrompt('  Fix   the\n\norbit\tfeedback  ')).toBe('Fix the orbit feedback');
    expect(shortenedPrompt('Hello')).toBe('Hello');
  });

  test('a long prompt is cut at a word boundary within the limit and ends with an ellipsis', () => {
    const prompt = 'Please compare the horizontal corrector strengths across the last three fills';
    const title = shortenedPrompt(prompt);

    expect(title.length).toBeLessThanOrEqual(PROMPT_TITLE_CHARS);
    expect(title.endsWith('\u2026')).toBe(true);
    const kept = title.slice(0, -1);
    expect(prompt.startsWith(kept)).toBe(true);
    expect(prompt[kept.length]).toBe(' ');
    expect(kept.endsWith(' ')).toBe(false);
  });

  test('a long prompt with no space in its second half is cut hard at the limit', () => {
    const title = shortenedPrompt(`ab ${'x'.repeat(100)}`);

    expect(title).toBe(`ab ${'x'.repeat(PROMPT_TITLE_CHARS - 4)}\u2026`);
    expect(title.length).toBe(PROMPT_TITLE_CHARS);
  });
});

describe('a prompt that stands in for the title', () => {
  test('on an untitled session it becomes the stand-in and is painted on the untitled held row', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];

    chase.sent({ text: 'Fix   the orbit\nfeedback' });

    expect(chase.standInFor(KEY)).toBe('Fix the orbit feedback');
    expect(model.rows[0].title).toBe('Fix the orbit feedback');
    expect(model.changed).toHaveBeenCalledTimes(1);
  });

  test('a held row that already has a title source, or belongs to another session, is not touched', () => {
    const { model, chase } = makeModel();
    const mine = untitledRow();
    const titled = { ...titledRow(), title: 'Kept title' };
    const other = { session_key: 'sess-b', current_for_key: true, title: 'Other' };
    model.rows = [mine, titled, other];

    chase.sent({ text: 'Fix the orbit' });

    expect(mine.title).toBe('Fix the orbit');
    expect(titled.title).toBe('Kept title');
    expect(other.title).toBe('Other');
  });

  test.each([
    ['a replayed prompt', { text: 'Fix the orbit', replay: true }, KEY, untitledRow],
    ['an empty text', { text: '' }, KEY, untitledRow],
    ['a whitespace-only text', { text: ' \n\t ' }, KEY, untitledRow],
    ['a slash command', { text: '/clear' }, KEY, untitledRow],
    ['a tab with no session key', { text: 'Fix the orbit' }, null, untitledRow],
    ['a session that is already titled', { text: 'Fix the orbit' }, KEY, titledRow],
  ])('%s is ignored: no stand-in and no re-listing', (_name, detail, key, makeRow) => {
    const { model, chase } = makeModel(key);
    model.rows = [makeRow()];

    chase.sent(detail);
    vi.advanceTimersByTime(TITLE_CHASE_MS.reduce((a, b) => a + b, 0) * 2);

    expect(chase.standInFor(KEY)).toBeNull();
    expect(model.refresh).not.toHaveBeenCalled();
    expect(model.changed).not.toHaveBeenCalled();
  });
});

describe('the re-listings', () => {
  test('they run at 3 s, 10 s and 30 s, each one after a listing that still has no title, and then stop', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];
    chase.sent({ text: 'Fix the orbit' });

    vi.advanceTimersByTime(TITLE_CHASE_MS[0] - 1);
    expect(model.refresh).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(model.refresh).toHaveBeenCalledTimes(1);

    chase.listed();
    vi.advanceTimersByTime(TITLE_CHASE_MS[1] - 1);
    expect(model.refresh).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1);
    expect(model.refresh).toHaveBeenCalledTimes(2);

    chase.listed();
    vi.advanceTimersByTime(TITLE_CHASE_MS[2] - 1);
    expect(model.refresh).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(1);
    expect(model.refresh).toHaveBeenCalledTimes(3);

    chase.listed();
    vi.advanceTimersByTime(10 * 60_000);
    expect(model.refresh).toHaveBeenCalledTimes(3);
  });

  test('the chase waits for the answer of a re-listing: no second step without a listing, failure or move', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];
    chase.sent({ text: 'Fix the orbit' });

    vi.advanceTimersByTime(10 * 60_000);

    expect(model.refresh).toHaveBeenCalledTimes(1);
  });

  test('a failed re-listing goes on to the next step, so one failure does not end the chase', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];
    chase.sent({ text: 'Fix the orbit' });
    vi.advanceTimersByTime(TITLE_CHASE_MS[0]);
    expect(model.refresh).toHaveBeenCalledTimes(1);

    chase.failed();
    vi.advanceTimersByTime(TITLE_CHASE_MS[1]);

    expect(model.refresh).toHaveBeenCalledTimes(2);
  });

  test('a listing that titles the session ends the stand-in and the chase', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];
    chase.sent({ text: 'Fix the orbit' });
    vi.advanceTimersByTime(TITLE_CHASE_MS[0]);
    expect(model.refresh).toHaveBeenCalledTimes(1);

    model.rows = [titledRow()];
    chase.listed();
    vi.advanceTimersByTime(10 * 60_000);

    expect(chase.standInFor(KEY)).toBeNull();
    expect(model.refresh).toHaveBeenCalledTimes(1);
  });

  test('a listing that is still untitled paints the stand-in on the fresh row that replaced the old one', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];
    chase.sent({ text: 'Fix the orbit' });
    vi.advanceTimersByTime(TITLE_CHASE_MS[0]);

    const fresh = untitledRow();
    model.rows = [fresh];
    chase.listed();

    expect(fresh.title).toBe('Fix the orbit');
  });
});

describe('the tab moves, the model resets, or a turn records its transcript', () => {
  test('moving to another session drops the stand-in and stops the chase; the same session keeps both', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];
    chase.sent({ text: 'Fix the orbit' });

    chase.pointed(KEY);
    expect(chase.standInFor(KEY)).toBe('Fix the orbit');
    vi.advanceTimersByTime(TITLE_CHASE_MS[0]);
    expect(model.refresh).toHaveBeenCalledTimes(1);

    chase.failed();
    chase.pointed('sess-b');
    expect(chase.standInFor(KEY)).toBeNull();
    vi.advanceTimersByTime(10 * 60_000);
    expect(model.refresh).toHaveBeenCalledTimes(1);
  });

  test('a reset stops a pending step and forgets the stand-in', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];
    chase.sent({ text: 'Fix the orbit' });

    chase.reset();
    vi.advanceTimersByTime(10 * 60_000);

    expect(chase.standInFor(KEY)).toBeNull();
    expect(model.refresh).not.toHaveBeenCalled();
  });

  test('a recorded transcript on an untitled session with no chase running starts one', () => {
    const { model, chase } = makeModel();
    model.rows = [untitledRow()];

    chase.moved();
    vi.advanceTimersByTime(TITLE_CHASE_MS[0]);

    expect(model.refresh).toHaveBeenCalledTimes(1);
  });

  test('a recorded transcript on a titled session starts nothing', () => {
    const { model, chase } = makeModel();
    model.rows = [titledRow()];

    chase.moved();
    vi.advanceTimersByTime(10 * 60_000);

    expect(model.refresh).not.toHaveBeenCalled();
  });
});
