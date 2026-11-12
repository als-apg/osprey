// @ts-check
/**
 * Unit tests for the Activity page controller
 * (web_terminal/static/js/activity-log-page.js): what it reads, when it reads
 * again, and what a press does. The page markup is the body of the real
 * `session.html`; `fetch` and the timers are stubbed.
 *
 *   npx vitest run tests/interfaces/web_terminal/activity-log-page.test.mjs
 */
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { clearToasts } from '/design-system/js/toast.js';

import { mountActivityPage } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-page.js';
import { call, payload } from '../_support/activity-log.mjs';
import { sessionRow, shellMarkup, stubActivityFetch } from '../_support/activity-page.mjs';
import { byId, qs } from '../_support/dom.mjs';
import { stampFacilityZone } from '../_support/facility-zone.mjs';

const ID = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee';
const OTHER = '11111111-2222-4333-8444-555555555555';
const KEY = '99999999-8888-4777-8666-555555555555';
const PAGE = '/static/session.html';
const POLL_MS = 10_000;

const ARTIFACT = {
  id: 'art1',
  title: 'Orbit plot',
  call: 'r1',
  tool: 'Read',
  agent: 'main',
  ts: '2026-11-12T10:00:02+00:00',
  form: 'plot',
  form_label: 'Plot',
  type: 'plot',
  size: 50381,
};

/** @type {ReturnType<typeof stubActivityFetch>} */
let api;
/** @type {ReturnType<typeof mountActivityPage> | null} */
let page;

/** Let every pending read land and render. */
const settle = () => vi.advanceTimersByTimeAsync(0);

/**
 * Mount the page at *url* and wait for its first reading.
 * @param {{ url?: string, sessionKey?: string | null, sidebarOpen?: boolean }} [opts]
 */
async function mount({ url = PAGE, sessionKey = null, sidebarOpen = true } = {}) {
  window.history.replaceState({}, '', url);
  page = mountActivityPage({ sessionKey, sidebarOpen });
  await settle();
}

/** @param {string} selector */
const press = async (selector) => {
  qs(document, selector, HTMLElement).click();
  await settle();
};

/** @param {'visible' | 'hidden'} state */
function setVisibility(state) {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state });
  document.dispatchEvent(new Event('visibilitychange'));
}

const sidebarReads = () => api.urls().filter((u) => u.includes('/api/sessions'));

beforeEach(() => {
  vi.useFakeTimers();
  stampFacilityZone('UTC');
  document.body.innerHTML = shellMarkup();
  api = stubActivityFetch();
});

afterEach(() => {
  page?.stop();
  page = null;
  clearToasts();
  Reflect.deleteProperty(document, 'visibilityState');
  stampFacilityZone(null);
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.history.replaceState({}, '', '/');
  document.body.innerHTML = '';
});

describe('what the page reads', () => {
  test('the session its URL names', async () => {
    await mount({ sessionKey: KEY });
    expect(api.reads()).toEqual([`/api/session-activity?session_id=${KEY}`]);
  });

  test("the terminal's current session when the URL names none", async () => {
    await mount();
    expect(api.reads()).toEqual(['/api/session-activity']);
  });

  test('the transcript in the hash, over the session key', async () => {
    await mount({ url: `${PAGE}?session_id=${KEY}#${OTHER}/tools`, sessionKey: KEY });
    expect(api.reads()).toEqual([`/api/session-activity?transcript_id=${OTHER}`]);
    expect(qs(document, '#views [data-view="tools"]').getAttribute('aria-pressed')).toBe('true');
    expect(document.querySelector('.act-table')).not.toBeNull();
  });

  test('a hash that changes under the page', async () => {
    await mount({ sessionKey: KEY });
    window.history.replaceState({}, '', `${PAGE}#${OTHER}/safety`);
    window.dispatchEvent(new Event('hashchange'));
    await settle();
    expect(api.reads().at(-1)).toBe(`/api/session-activity?transcript_id=${OTHER}`);
    expect(qs(document, '#views [data-view="safety"]').getAttribute('aria-pressed')).toBe('true');
  });

  test('the session a host frame moves it to, dropping the hash selection', async () => {
    await mount({ url: `${PAGE}#${OTHER}/tools` });
    page?.follow(KEY);
    await settle();
    expect(api.reads().at(-1)).toBe(`/api/session-activity?session_id=${KEY}`);
    expect(window.location.hash).toBe('#/tools');
  });
});

describe('what the page shows', () => {
  test('the session, its counts and its timeline', async () => {
    await mount();
    expect(qs(document, '.act-head h1').textContent).toBe('Read the beam current');
    expect(document.querySelectorAll('.act-stat')).toHaveLength(8);
    expect(document.getElementById('row-r1')).not.toBeNull();
    expect(qs(document, '#views [data-view="timeline"]').getAttribute('aria-pressed')).toBe('true');
    expect(qs(document, '#views [data-view="tools"] .osprey-count').textContent).toBe('5');
    expect(qs(document, '#views [data-view="safety"] .osprey-count').textContent).toBe('1');
    expect(document.querySelector('#views [data-view="artifacts"] .osprey-count')).toBeNull();
    expect(byId('record-chip').textContent).toBe('Record complete');
    expect(byId('record-chip').classList.contains('osprey-chip--ok')).toBe(true);
    expect(byId('live').textContent).toBe('Last entry 10:02');
  });

  test('the view in the URL, so a reload lands on it', async () => {
    await mount();
    await press('#views [data-view="safety"]');
    expect(window.location.hash).toBe('#/safety');
    expect(document.querySelector('.act-cards')).not.toBeNull();
  });

  test('an answer as rendered markdown', async () => {
    vi.stubGlobal('marked', {
      parse: (/** @type {string} */ text) =>
        `<p>${text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')}</p>`,
    });
    vi.stubGlobal('DOMPurify', { sanitize: (/** @type {string} */ html) => html });
    await mount();
    expect(qs(document, '.act-md strong').textContent).toBe('401.2 mA');
  });

  test('an answer as plain text when the markdown libraries are missing', async () => {
    await mount();
    expect(qs(document, '.act-md').textContent).toBe('The beam current is **401.2 mA**.');
  });

  test('an incomplete record as a banner over the views', async () => {
    api.answers.activity = payload({}, { record: 'incomplete' });
    await mount();
    expect(qs(document, '.act-banner').textContent).toContain('Entries may be missing');
    expect(byId('record-chip').textContent).toBe('Record incomplete');
  });

  test('what the record holds, from the record chip', async () => {
    await mount();
    await press('#record-chip');
    const popover = qs(document, '.osprey-info-popover');
    expect(popover.textContent).toContain('Agent record');
    expect(popover.textContent).toContain(ID);
  });

  test('a session with no entry yet, without falling back to another', async () => {
    api.answers.activity = { session: null, nodes: [], artifacts: [], record: 'complete' };
    await mount({ sessionKey: KEY });
    expect(qs(document, '#inner .osprey-state').textContent).toContain(
      'No activity in this session yet'
    );
    expect(document.querySelector('.act-head')).toBeNull();
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(api.reads()).toEqual([
      `/api/session-activity?session_id=${KEY}`,
      `/api/session-activity?session_id=${KEY}`,
    ]);
  });

  test('the first entries of that session when they arrive', async () => {
    api.answers.activity = { session: null, nodes: [], artifacts: [], record: 'complete' };
    await mount({ sessionKey: KEY });
    api.answers.activity = payload({ state: 'live' });
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(qs(document, '.act-head h1').textContent).toBe('Read the beam current');
  });
});

describe('an unreadable record', () => {
  const UNAVAILABLE = { session: { id: ID }, nodes: [], artifacts: [], record: 'unavailable' };

  test('is said, with a way to try again', async () => {
    api.answers.activity = UNAVAILABLE;
    await mount();
    const state = qs(document, '#inner .osprey-state');
    expect(state.textContent).toContain('The agent record cannot be read');
    expect(document.querySelector('.act-head')).toBeNull();
    expect(document.querySelector('.act-stats')).toBeNull();
    expect(byId('record-chip').textContent).toBe('Record unavailable');

    api.answers.activity = payload();
    await press('#inner [data-act="refresh"]');
    expect(qs(document, '.act-head h1').textContent).toBe('Read the beam current');
  });

  test('is what a read that fails shows', async () => {
    api.answers.activity = new Error('HTTP 502');
    await mount();
    expect(qs(document, '#inner .osprey-state').textContent).toContain(
      'The agent record cannot be read'
    );
    expect(qs(document, '#inner [data-act="refresh"]').textContent).toBe('Try again');
  });

  test('keeps a live session polling until the record answers again', async () => {
    api.answers.activity = payload({ state: 'live' });
    await mount();
    api.answers.activity = new Error('HTTP 503');
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(document.querySelector('#inner .osprey-state')).not.toBeNull();
    api.answers.activity = payload({ state: 'live' });
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(document.getElementById('row-r1')).not.toBeNull();
  });

  test('keeps the session list', async () => {
    api.answers.activity = UNAVAILABLE;
    api.answers.sessions = { sessions: [sessionRow(ID, 'Read the beam current')] };
    await mount();
    expect(document.querySelectorAll('#sb [data-session]')).toHaveLength(1);
  });
});

describe('polling', () => {
  test('a session at rest is read once', async () => {
    await mount();
    await vi.advanceTimersByTimeAsync(3 * POLL_MS);
    expect(api.reads()).toHaveLength(1);
  });

  test('a live session is read every ten seconds', async () => {
    api.answers.activity = payload({ state: 'live' });
    await mount();
    await vi.advanceTimersByTimeAsync(POLL_MS - 1);
    expect(api.reads()).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(api.reads()).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(api.reads()).toHaveLength(3);
  });

  test('a hidden page is not read, and is read at once when shown again', async () => {
    api.answers.activity = payload({ state: 'live' });
    await mount();
    setVisibility('hidden');
    await vi.advanceTimersByTimeAsync(3 * POLL_MS);
    expect(api.reads()).toHaveLength(1);
    setVisibility('visible');
    await settle();
    expect(api.reads()).toHaveLength(2);
  });

  test('an unchanged answer leaves the page as it is', async () => {
    api.answers.activity = payload({ state: 'live' });
    await mount();
    await press('#row-r1');
    const row = byId('row-r1');
    const detail = qs(document, '.act-detail');
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(api.reads()).toHaveLength(2);
    expect(byId('row-r1')).toBe(row);
    expect(qs(document, '.act-detail')).toBe(detail);
  });

  test('a changed answer keeps an open row open', async () => {
    api.answers.activity = payload({ state: 'live' });
    await mount();
    await press('#row-r1');
    const next = payload({ state: 'live', entries: 32 });
    next.nodes.push(call('n9', { tool: 'Grep', name: 'Grep' }));
    api.answers.activity = next;
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(document.getElementById('row-n9')).not.toBeNull();
    expect(byId('row-r1').getAttribute('aria-expanded')).toBe('true');
    expect(document.querySelector('#row-r1 + .act-body-block .act-detail')).not.toBeNull();
  });

  test('the refresh button reads at once, and the session list with it', async () => {
    await mount();
    const lists = sidebarReads().length;
    await press('#refresh');
    expect(api.reads()).toHaveLength(2);
    expect(sidebarReads()).toHaveLength(lists + 1);
  });

  test('an answer overtaken by a newer read is dropped', async () => {
    api.answers.sessions = {
      sessions: [sessionRow(ID, 'Read the beam current'), sessionRow(OTHER, 'Orbit bump')],
    };
    await mount();
    /** @type {(value: { status: number, ok: boolean, json: () => Promise<unknown> }) => void} */
    let answerLate = () => {};
    api.fetchStub.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          answerLate = resolve;
        })
    );
    qs(document, '#refresh', HTMLElement).click();
    api.answers.activity = payload({ id: OTHER, title: 'Orbit bump' });
    await press(`#sb [data-session="${OTHER}"]`);
    answerLate({
      status: 200,
      ok: true,
      json: () => Promise.resolve(payload({ title: 'Stale reading' })),
    });
    await settle();
    expect(qs(document, '.act-head h1').textContent).toBe('Orbit bump');
  });
});

describe('the session list', () => {
  beforeEach(() => {
    api.answers.sessions = {
      sessions: [
        sessionRow(ID, 'Read the beam current'),
        sessionRow(OTHER, 'Orbit bump'),
        { ...sessionRow('', 'Not in the record yet'), transcript_id: null },
      ],
    };
  });

  test('lists the sessions of the record and marks the one shown', async () => {
    await mount();
    const rows = [...document.querySelectorAll('#sb [data-session]')];
    expect(rows.map((row) => row.getAttribute('data-session'))).toEqual([ID, OTHER]);
    expect(rows.map((row) => row.getAttribute('aria-selected'))).toEqual(['true', 'false']);
    expect(sidebarReads()).toEqual(['/api/sessions']);
  });

  test('a pick shows that session, names it in the URL and leaves the key behind', async () => {
    await mount({ url: `${PAGE}?session_id=${KEY}`, sessionKey: KEY });
    await press('#row-r1');
    api.answers.activity = payload({ id: OTHER, title: 'Orbit bump' });
    await press(`#sb [data-session="${OTHER}"]`);
    expect(api.reads().at(-1)).toBe(`/api/session-activity?transcript_id=${OTHER}`);
    expect(window.location.hash).toBe(`#${OTHER}/timeline`);
    expect(qs(document, '.act-head h1').textContent).toBe('Orbit bump');
    expect(qs(document, `#sb [data-session="${OTHER}"]`).getAttribute('aria-selected')).toBe(
      'true'
    );
    // What was open belonged to the session left behind.
    expect(byId('row-r1').getAttribute('aria-expanded')).toBe('false');
    await press('#refresh');
    expect(api.reads().at(-1)).toBe(`/api/session-activity?transcript_id=${OTHER}`);
  });

  test('is read again, once, when the session shown is not in it', async () => {
    api.answers.activity = payload({ id: KEY, state: 'live' });
    await mount();
    expect(sidebarReads()).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(2 * POLL_MS);
    expect(sidebarReads()).toHaveLength(2);
  });

  test('says so when the record holds no session', async () => {
    api.answers.sessions = { sessions: [] };
    await mount();
    expect(qs(document, '#sb .osprey-state').textContent).toContain('No sessions in the record');
  });

  test('hides and shows from its toolbar button', async () => {
    await mount();
    expect(byId('sb').dataset.open).toBe('true');
    await press('#sb-toggle');
    expect(byId('sb').dataset.open).toBe('false');
    expect(byId('sb-toggle').getAttribute('aria-pressed')).toBe('false');
    await press('#sb-toggle');
    expect(byId('sb').dataset.open).toBe('true');
  });

  test('starts hidden where the page is given no room for it', async () => {
    await mount({ sidebarOpen: false });
    expect(byId('sb').dataset.open).toBe('false');
    expect(byId('sb-toggle').getAttribute('aria-pressed')).toBe('false');
  });
});

describe('what a press does', () => {
  test('a row opens onto its detail and keeps the keyboard focus', async () => {
    await mount();
    byId('row-r1').focus();
    await press('#row-r1');
    expect(byId('row-r1').getAttribute('aria-expanded')).toBe('true');
    expect(document.querySelector('#row-r1 + .act-body-block .act-detail')).not.toBeNull();
    expect(document.activeElement).toBe(byId('row-r1'));
  });

  test('a subagent group folds', async () => {
    await mount();
    expect(document.getElementById('row-w1')).not.toBeNull();
    await press('[data-collapse="a1"]');
    expect(document.getElementById('row-w1')).toBeNull();
  });

  test('a filter narrows the timeline', async () => {
    await mount();
    await press('[data-filter="errors"]');
    expect([...document.querySelectorAll('.act-row[data-kind="tool"]')].map((r) => r.id)).toEqual([
      'row-f1',
    ]);
  });

  test('a count goes to the view that lists it', async () => {
    await mount();
    await press('.act-stat[data-go="tools:all"]');
    expect(document.querySelector('.act-table')).not.toBeNull();
    expect(window.location.hash).toBe('#/tools');
  });

  test('expand all opens every call and every fold', async () => {
    await mount();
    await press('[data-act="expand"]');
    const closed = [...document.querySelectorAll('#inner [aria-expanded="false"]')];
    expect(closed).toEqual([]);
    // r1, w1, f1 and o1; a subagent's own call is a group, not a detail.
    expect(document.querySelectorAll('.act-detail')).toHaveLength(4);
    await press('[data-act="collapse"]');
    expect(document.querySelectorAll('.act-detail')).toHaveLength(0);
  });

  test('typing searches the session', async () => {
    await mount();
    const q = byId('q', HTMLInputElement);
    q.value = 'DCCT timed out';
    q.dispatchEvent(new Event('input', { bubbles: true }));
    await vi.advanceTimersByTimeAsync(200);
    expect([...document.querySelectorAll('.act-row[data-kind="tool"]')].map((r) => r.id)).toEqual([
      'row-f1',
    ]);
    q.value = 'no such thing anywhere';
    q.dispatchEvent(new Event('input', { bubbles: true }));
    await vi.advanceTimersByTimeAsync(200);
    expect(qs(document, '#inner .osprey-state').textContent).toContain('Nothing in this session');
  });

  test('Enter on a table row opens it, and on a column sorts by it', async () => {
    await mount({ url: `${PAGE}#/tools` });
    const row = qs(document, 'tr.row[data-toggle="r1"]', HTMLElement);
    row.focus();
    row.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();
    expect(qs(document, 'tr.row[data-toggle="r1"]').getAttribute('aria-expanded')).toBe('true');
    expect(document.querySelector('tr.detail')).not.toBeNull();
    expect(document.activeElement).toBe(qs(document, 'tr.row[data-toggle="r1"]'));

    const head = qs(document, 'th[data-sort="tool"]', HTMLElement);
    head.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    await settle();
    expect(qs(document, 'th[data-sort="tool"]').getAttribute('aria-sort')).toBe('ascending');
  });

  test('a key on a button is left to the button', async () => {
    await mount();
    byId('row-r1').dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();
    expect(byId('row-r1').getAttribute('aria-expanded')).toBe('false');
  });

  test('Show in timeline lands on the call, open, inside its subagent', async () => {
    const scrolled = vi.spyOn(Element.prototype, 'scrollIntoView').mockImplementation(() => {});
    await mount();
    await press('[data-collapse="a1"]');
    await press('#views [data-view="safety"]');
    await press('.act-approval [data-jump="w1"]');
    const row = byId('row-w1');
    expect(window.location.hash).toBe('#/timeline');
    expect(row.getAttribute('aria-expanded')).toBe('true');
    expect(row.classList.contains('act-flash')).toBe(true);
    expect(scrolled.mock.contexts.at(-1)).toBe(row);
  });

  test('Show in timeline finds a call folded into a task list', async () => {
    vi.spyOn(Element.prototype, 'scrollIntoView').mockImplementation(() => {});
    const tasks = payload();
    tasks.nodes.push(
      call('t1', { tool: 'TaskCreate', name: 'TaskCreate' }),
      call('t2', { tool: 'TaskUpdate', name: 'TaskUpdate' })
    );
    tasks.artifacts = [{ ...ARTIFACT, call: 't2' }];
    api.answers.activity = tasks;
    await mount({ url: `${PAGE}#/artifacts` });
    await press('.act-art [data-jump="t2"]');
    expect(byId('row-t2').classList.contains('act-flash')).toBe(true);
  });

  test('a cut result loads whole', async () => {
    const cut = payload();
    cut.nodes.push(
      call('big', {
        result: { text: 'first part', size: 9000, cut: true, key: 'cc:alice:s:big/r' },
      })
    );
    api.answers.activity = cut;
    await mount();
    await press('#row-big');
    expect(qs(document, '#row-big + .act-body-block .act-result').textContent).toContain(
      'first part'
    );
    await press('[data-full="big"]');
    expect(api.urls().at(-1)).toBe('/api/session-activity/result?key=cc%3Aalice%3As%3Abig%2Fr');
    expect(qs(document, '#row-big + .act-body-block .act-result').textContent).toBe(
      'the whole result'
    );
    expect(document.querySelector('[data-full="big"]')).toBeNull();
  });

  test('a whole result that cannot be read is said', async () => {
    const cut = payload();
    cut.nodes.push(
      call('big', { result: { text: 'first part', size: 9000, cut: true, key: 'k:big' } })
    );
    api.answers.activity = cut;
    api.answers.result = new Error('HTTP 404');
    await mount();
    await press('#row-big');
    await press('[data-full="big"]');
    expect(qs(document, '.osprey-toast--error').textContent).toContain(
      'The full result could not be loaded'
    );
    expect(document.querySelector('[data-full="big"]')).not.toBeNull();
  });

  test('Copy link gives the link to the session shown, on the view shown', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    await mount({ url: `${PAGE}?session_id=${KEY}`, sessionKey: KEY });
    await press('#views [data-view="tools"]');
    await press('[data-act="copy-link"]');
    expect(writeText).toHaveBeenCalledWith(`${window.location.origin}${PAGE}#${ID}/tools`);
    expect(qs(document, '.osprey-toast').textContent).toContain('Link copied');
    Reflect.deleteProperty(navigator, 'clipboard');
  });

  test('Open on an artifact opens its file in a new tab', async () => {
    const opened = vi.spyOn(window, 'open').mockReturnValue(null);
    api.answers.activity = payload({}, { artifacts: [ARTIFACT] });
    await mount({ url: `${PAGE}#/artifacts` });
    expect(qs(document, '#views [data-view="artifacts"] .osprey-count').textContent).toBe('1');
    await press('[data-open-artifact="art1"]');
    expect(opened).toHaveBeenCalledWith('/panel/artifacts/files/art1/file', '_blank', 'noopener');
  });
});
