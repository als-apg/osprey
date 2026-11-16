// @ts-check
/**
 * The session list's view (sidebar-session-view.js): Last activity hides
 * older rows and, past its cutoff, the next page; Date and State group the
 * rows in their listed order; With artifacts only hides sessions that made
 * none; the session in use and a row that cannot be judged always stay. The
 * view is the operator's, stored per user, and the button shows a dot while
 * it differs from the defaults.
 *
 *   npx vitest run tests/interfaces/web_terminal/sidebar-session-view.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-session-view.js')} */
let view;

// Thursday 13 November 2026, 15:00 local time.
const NOW = new Date(2026, 10, 13, 15, 0, 0);
const HOUR = 60 * 60 * 1000;
/** @param {number} hours  @returns {string} an ISO time that many hours before NOW */
const ago = (hours) => new Date(NOW.getTime() - hours * HOUR).toISOString();

/**
 * @param {string} id  @param {Partial<any>} [extra]
 * @returns {any}
 */
const row = (id, extra = {}) => ({
  transcript_id: id,
  session_key: null,
  title: id,
  title_source: 'auto',
  state: 'idle',
  ...extra,
});

/** Newest activity first, as the listing gives them. */
const ROWS = [
  row('now', { last_ts: ago(1), state: 'live' }),
  row('yesterday', { last_ts: ago(20) }),
  row('tuesday', { last_ts: ago(50) }),
  row('last-week', { last_ts: ago(24 * 6) }),
  row('last-month', { last_ts: ago(24 * 20) }),
  row('old', { last_ts: ago(24 * 90) }),
];

/** @type {Record<string, number>} */
let counts;

/**
 * @param {any[]} rows  @param {string | null} [activeId]  @param {Date} [now]
 */
function arrange(rows, activeId = null, now = NOW) {
  return view.arrangeSessions(rows, {
    idOf: (r) => String(r.transcript_id),
    activeId,
    countOf: (id) => counts[id],
    now,
  });
}

/** @param {ReturnType<typeof arrange>} shown  @returns {string[]} labels as `# x`, rows by id */
const flat = (shown) =>
  shown.items.map((i) => ('label' in i ? `# ${i.label}` : String(i.row.transcript_id)));

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  document.documentElement.removeAttribute('data-facility-timezone');
  document.documentElement.removeAttribute('data-session-counts-off');
  counts = {};
  view = await import(`${JS}/sidebar-session-view.js`);
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('Last activity', () => {
  test('All shows every row and lets the next page load', () => {
    const shown = arrange(ROWS);
    expect(flat(shown)).toEqual(ROWS.map((r) => r.transcript_id));
    expect(shown.more).toBe(true);
  });

  test('3 days hides older rows, and the next page, all older still, is not offered', () => {
    view.setSessionView({ activity: '3d' });
    const shown = arrange(ROWS);
    expect(flat(shown)).toEqual(['now', 'yesterday', 'tuesday']);
    expect(shown.more).toBe(false);
  });

  test('a cutoff the loaded rows have not reached yet still offers the next page', () => {
    view.setSessionView({ activity: '30d' });
    expect(arrange(ROWS.slice(0, 4)).more).toBe(true);
  });

  test('Today starts at midnight, not 24 hours back', () => {
    view.setSessionView({ activity: '1d' });
    const early = row('early', { last_ts: new Date(2026, 10, 13, 0, 30).toISOString() });
    const late = row('late', { last_ts: new Date(2026, 10, 12, 23, 30).toISOString() });
    expect(flat(arrange([early, late]))).toEqual(['early']);
  });

  test('Today and Yesterday are days on the facility clock, not on the viewer’s', () => {
    document.documentElement.setAttribute('data-facility-timezone', 'Asia/Tokyo');
    // 10:00 on 13 November in Tokyo, where that day started at 15:00 UTC on the 12th.
    const now = new Date('2026-11-13T01:00:00Z');
    const rows = [
      row('after-midnight', { last_ts: '2026-11-12T16:00:00Z' }),
      row('before-midnight', { last_ts: '2026-11-12T14:00:00Z' }),
    ];
    view.setSessionView({ activity: '1d' });
    expect(flat(arrange(rows, null, now))).toEqual(['after-midnight']);
    view.setSessionView({ activity: 'all', group: 'date' });
    expect(flat(arrange(rows, null, now))).toEqual([
      '# Today',
      'after-midnight',
      '# Yesterday',
      'before-midnight',
    ]);
  });

  test('the session in use stays however old, and a row with no time yet stays', () => {
    view.setSessionView({ activity: '1d' });
    const undated = row('undated');
    expect(flat(arrange([...ROWS, undated], 'old'))).toEqual(['now', 'old', 'undated']);
  });

  test('a view that hides every loaded row says so', () => {
    view.setSessionView({ activity: '3d', withArtifacts: true });
    counts = { now: 0, yesterday: 0, tuesday: 0 };
    const shown = arrange(ROWS);
    expect(shown.items).toEqual([]);
    expect(shown.empty).toBe('No sessions with artifacts in 3 days');
  });
});

describe('With artifacts only', () => {
  test('hides a session that made none, keeps one not counted yet and the one in use', () => {
    view.setSessionView({ withArtifacts: true });
    counts = { now: 0, yesterday: 2, tuesday: 0, 'last-week': 0, 'last-month': 1 };
    expect(flat(arrange(ROWS, 'tuesday'))).toEqual(['yesterday', 'tuesday', 'last-month', 'old']);
  });
});

describe('Group by', () => {
  test('Date: Today, Yesterday, Last 7 days, Last 30 days, Older, in listed order', () => {
    view.setSessionView({ group: 'date' });
    expect(flat(arrange(ROWS))).toEqual([
      '# Today',
      'now',
      '# Yesterday',
      'yesterday',
      '# Last 7 days',
      'tuesday',
      'last-week',
      '# Last 30 days',
      'last-month',
      '# Older',
      'old',
    ]);
  });

  test('Date puts a new session, which has no time yet, under Today', () => {
    view.setSessionView({ group: 'date' });
    const fresh = row('fresh', { transcript_id: null, synthesized: true });
    expect(flat(arrange([{ ...fresh, transcript_id: 'key:k' }, ...ROWS.slice(1, 2)]))).toEqual([
      '# Today',
      'key:k',
      '# Yesterday',
      'yesterday',
    ]);
  });

  test('Date keeps a listed session with no time yet in the group of the row above it', () => {
    view.setSessionView({ group: 'date' });
    const undated = row('undated');
    expect(flat(arrange([ROWS[0], ROWS[1], undated, ROWS[2]]))).toEqual([
      '# Today',
      'now',
      '# Yesterday',
      'yesterday',
      'undated',
      '# Last 7 days',
      'tuesday',
    ]);
  });

  test('State: the sessions whose agent runs now first, then the rest', () => {
    view.setSessionView({ group: 'state' });
    const live = row('tuesday', { last_ts: ago(50), state: 'live' });
    expect(flat(arrange([ROWS[0], ROWS[1], live]))).toEqual([
      '# Open now',
      'now',
      'tuesday',
      '# Earlier',
      'yesterday',
    ]);
  });
});

describe('the stored view', () => {
  test('is stored for bob only, and another user’s is never read', async () => {
    localStorage.setItem(view.SESSION_VIEW_STORAGE_KEY, JSON.stringify({ group: 'date' }));
    document.documentElement.dataset.ospreyStorageScope = 'bob';
    expect(view.readSessionView().group).toBe('none');

    view.setSessionView({ activity: '7d' });

    const stored = localStorage.getItem(`${view.SESSION_VIEW_STORAGE_KEY}--bob`);
    expect(JSON.parse(stored ?? 'null')).toMatchObject({ activity: '7d' });
    vi.resetModules();
    const again = await import(`${JS}/sidebar-session-view.js`);
    expect(again.readSessionView().activity).toBe('7d');
  });

  test('an unknown stored value falls back to its default', () => {
    localStorage.setItem(
      view.SESSION_VIEW_STORAGE_KEY,
      JSON.stringify({ activity: 'forever', group: 'date', counts: 'no' })
    );
    expect(view.readSessionView()).toEqual({ ...view.DEFAULT_VIEW, group: 'date' });
  });
});

describe('the view button', () => {
  test('shows a dot while the view differs, hides the counts when asked, and Reset clears both', () => {
    document.body.innerHTML = `<button id="${view.VIEW_BUTTON_ID}"></button>`;
    const onChange = vi.fn();
    view.watchViewButton(new AbortController().signal, onChange);
    const button = /** @type {HTMLElement} */ (document.getElementById(view.VIEW_BUTTON_ID));
    expect(button.hasAttribute('data-changed')).toBe(false);

    view.setSessionView({ counts: false });
    expect(button.hasAttribute('data-changed')).toBe(true);
    expect(document.documentElement.hasAttribute('data-session-counts-off')).toBe(true);
    expect(onChange).toHaveBeenCalledOnce();

    const reset = view.viewMenuItems().find((i) => i.label === 'Reset to defaults');
    reset?.run?.();
    expect(button.hasAttribute('data-changed')).toBe(false);
    expect(document.documentElement.hasAttribute('data-session-counts-off')).toBe(false);
  });

  test('each choice row names its current value', () => {
    view.setSessionView({ activity: '30d', group: 'state' });
    const values = view
      .viewMenuItems()
      .filter((i) => i.submenu)
      .map((i) => [i.label, i.value]);
    expect(values).toEqual([
      ['Last activity', '30 days'],
      ['Group by', 'State'],
      ['Artifacts by', 'Activity'],
    ]);
  });
});
