// @ts-check
/**
 * The session tree's peek (sidebar-session-peek.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-session-peek.test.mjs
 *
 * A folded session row, hovered for a moment, shows that session's artifacts
 * beside the sidebar. These tests pin when it opens (folded rows only, after
 * the wait, once there is something to list), that the pointer can move into
 * it, when it closes, and what its clicks do. The rows are the tree's own
 * builders, real here; the listing is a stub.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  PEEK_CLOSE_MS,
  PEEK_OPEN_MS,
  PEEK_ROWS,
  createSessionPeek,
  peekEntries,
} from '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-session-peek.js';

/** @param {string} id @param {object} [extra] */
const art = (id, extra = {}) => ({
  id,
  title: `Title ${id}`,
  timestamp: '2026-10-30T10:00:00Z',
  form: 'plots',
  ...extra,
});

/** @type {Record<string, {loaded: boolean, entries: any[]}>} */
let listings;
/** @type {ReturnType<typeof createSessionPeek>} */
let peek;
/** @type {AbortController} */
let abort;
const onPick = vi.fn();
const onMore = vi.fn();

/** @param {string} id @param {boolean} open */
function sessionRow(id, open) {
  const row = document.createElement('div');
  row.className = 'osprey-tree-session osprey-tree-d0';
  row.dataset.sessionId = id;
  row.setAttribute('aria-expanded', String(open));
  const title = document.createElement('span');
  title.className = 'osprey-tree-title';
  title.textContent = `Session ${id}`;
  row.append(title);
  return row;
}

function skeleton() {
  document.body.innerHTML = `
    <nav id="sidebar"><div data-session-tree id="tree"></div></nav>
    <main id="elsewhere">elsewhere</main>`;
  document.getElementById('tree')?.append(sessionRow('s1', true), sessionRow('s2', false));
}

const peekEl = () =>
  /** @type {HTMLElement | null} */ (document.getElementById('sidebar-session-peek'));
const isOpen = () => peekEl()?.hidden === false;
const titleOf = (/** @type {string} */ id) =>
  /** @type {HTMLElement} */ (
    document.querySelector(`[data-session-id="${id}"] .osprey-tree-title`)
  );
/** @param {Element} el */
const over = (el) => el.dispatchEvent(new Event('pointerover', { bubbles: true }));

beforeEach(() => {
  vi.useFakeTimers();
  skeleton();
  listings = {
    s1: { loaded: true, entries: [art('a1')] },
    s2: {
      loaded: true,
      entries: [
        art('b1'),
        art('run1', { form: 'runs', exec_id: 'x1', run_part: 'result' }),
        art('out1', { exec_id: 'x1', run_part: 'output' }),
        art('part', { form: null }),
      ],
    },
  };
  onPick.mockClear();
  onMore.mockClear();
  peek = createSessionPeek({
    listing: (id) => listings[id] ?? null,
    total: (id) => (id === 's2' ? 5 : null),
    shownId: () => 'b1',
    onPick,
    onMore,
  });
  abort = new AbortController();
  peek.watch(abort.signal);
});

afterEach(() => {
  abort.abort();
  vi.useRealTimers();
  document.body.replaceChildren();
});

describe('what a peek lists', () => {
  test("a run once, by its head; never a run's outputs or parts", () => {
    expect(peekEntries(listings.s2.entries).map((e) => e.id)).toEqual(['b1', 'run1']);
  });

  test(`at most ${PEEK_ROWS} rows, then how many more the session holds`, () => {
    listings.s2.entries = Array.from({ length: 12 }, (_, i) => art(`m${i}`));
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    expect(peekEl()?.querySelectorAll('.osprey-list-row')).toHaveLength(PEEK_ROWS);
    expect(peekEl()?.querySelector('[data-peek-more]')?.textContent).toBe('4 more');
  });
});

describe('when the peek opens', () => {
  test('on a folded session, after the wait; the shown artifact is marked', () => {
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS - 1);
    expect(isOpen()).toBe(false);
    vi.advanceTimersByTime(1);
    expect(isOpen()).toBe(true);
    expect(peekEl()?.getAttribute('aria-label')).toBe('Artifacts of Session s2');
    expect(peekEl()?.querySelector('[data-id="b1"]')?.getAttribute('aria-selected')).toBe('true');
    expect(peekEl()?.querySelector('[data-peek-more]')?.textContent).toBe('3 more');
  });

  test('never on an open session: its artifacts are in the tree', () => {
    over(titleOf('s1'));
    vi.advanceTimersByTime(PEEK_OPEN_MS * 2);
    expect(isOpen()).toBe(false);
  });

  test('not while the listing loads, then at once when it lands; never for an empty session', () => {
    listings.s2 = { loaded: false, entries: [] };
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    expect(isOpen()).toBe(false);
    listings.s2 = { loaded: true, entries: [art('b1')] };
    peek.refresh();
    expect(isOpen()).toBe(true);

    peek.close();
    listings.s2 = { loaded: true, entries: [] };
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    expect(isOpen()).toBe(false);
  });

  test('a pointer that passes over without resting opens nothing', () => {
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS / 2);
    over(/** @type {Element} */ (document.getElementById('elsewhere')));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    expect(isOpen()).toBe(false);
  });
});

describe('when it stays and when it closes', () => {
  /** Open the peek of s2. */
  function openPeek() {
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    expect(isOpen()).toBe(true);
  }

  test('it stays open while the pointer moves from the row into it', () => {
    openPeek();
    over(/** @type {Element} */ (document.getElementById('sidebar')));
    vi.advanceTimersByTime(PEEK_CLOSE_MS / 2);
    over(/** @type {Element} */ (peekEl()?.querySelector('.osprey-list-row')));
    vi.advanceTimersByTime(PEEK_CLOSE_MS * 4);
    expect(isOpen()).toBe(true);
  });

  test('it closes when the pointer leaves the row and the peek', () => {
    openPeek();
    over(/** @type {Element} */ (document.getElementById('elsewhere')));
    vi.advanceTimersByTime(PEEK_CLOSE_MS);
    expect(isOpen()).toBe(false);
  });

  test('moving onto the next folded session opens its peek in place of the first', () => {
    document.getElementById('tree')?.append(sessionRow('s3', false));
    listings.s3 = { loaded: true, entries: [art('c1')] };
    openPeek();
    over(titleOf('s3'));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    expect(isOpen()).toBe(true);
    expect(peekEl()?.getAttribute('aria-label')).toBe('Artifacts of Session s3');
  });

  test('Escape, a press outside, and the row unfolding each close it', () => {
    openPeek();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(isOpen()).toBe(false);

    openPeek();
    document
      .getElementById('elsewhere')
      ?.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    expect(isOpen()).toBe(false);

    openPeek();
    document.querySelector('[data-session-id="s2"]')?.setAttribute('aria-expanded', 'true');
    peek.refresh();
    expect(isOpen()).toBe(false);
  });

  test('a press inside it keeps it until the click acts', () => {
    openPeek();
    peekEl()
      ?.querySelector('.osprey-list-row')
      ?.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    expect(isOpen()).toBe(true);
  });
});

describe('its clicks', () => {
  test('an artifact opens it, as the tree does, and the peek closes', () => {
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    const title = peekEl()?.querySelector('[data-id="run1"] .osprey-list-title');
    /** @type {HTMLElement} */ (title).click();
    expect(onPick).toHaveBeenCalledWith('run1');
    expect(isOpen()).toBe(false);
  });

  test('"more" switches to the session', () => {
    over(titleOf('s2'));
    vi.advanceTimersByTime(PEEK_OPEN_MS);
    const more = peekEl()?.querySelector('[data-peek-more]');
    /** @type {HTMLElement} */ (more).click();
    expect(onMore).toHaveBeenCalledWith('s2');
    expect(onPick).not.toHaveBeenCalled();
  });
});
