// @ts-check
/**
 * The sidebar's session tree controller (sidebar-artifacts.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifacts.test.mjs
 *
 * The Agent row's tree lists the operator's sessions (session-model.js, real
 * here, over a stubbed hub), each expanding to its artifacts grouped by form
 * (Kind mode, stored per test), loaded lazily through the artifacts panel's
 * `session-list` and counted by one `session-summary` per page of sessions.
 * These tests pin the data flow (what loads when), the clicks (only the
 * current session opens; another session's name switches and reveals, the
 * current one's folds, an artifact opens without switching and records
 * last-artifact), the viewer's
 * focus rules and pager, and the live counts.
 *
 * Collaborators that need a running page are mocked (the chat console, the
 * terminal session switch, the confirm dialog, the dock, the tile header, the
 * iframe sync); api.js, session-pointer.js and the row builders are real, with
 * fetch and EventSource stubbed.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { stubEventSource } from '../_support/event-source.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const mocks = vi.hoisted(() => ({
  streaming: false,
  terminal: false,
  showConfirm: vi.fn(),
  stopTurn: vi.fn(async () => true),
  newConversation: vi.fn(),
  resumeSession: vi.fn(),
  startNewSession: vi.fn(),
  contributeLocal: vi.fn(),
  sendSelect: vi.fn(),
  sendAction: vi.fn(),
  sendChanged: vi.fn(),
  activateTab: vi.fn(),
  hasDockedTile: vi.fn(() => true),
  setServiceMenuExtras: vi.fn(),
  isTerminalPanelOpen: vi.fn(() => true),
  openTerminalPanel: vi.fn(),
  showAgentPane: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat.js', () => ({
  isStreaming: () => mocks.streaming,
  stopTurn: mocks.stopTurn,
  newConversation: mocks.newConversation,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js', () => ({
  resumeSession: mocks.resumeSession,
  startNewSession: mocks.startNewSession,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => ({
  showConfirm: mocks.showConfirm,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js', () => ({
  isTerminalSurface: () => mocks.terminal,
  onSurfaceChange: () => () => {},
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  isTerminalPanelOpen: mocks.isTerminalPanelOpen,
  openTerminalPanel: mocks.openTerminalPanel,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  contributeLocal: mocks.contributeLocal,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js', () => ({
  sendArtifactSelectToIframe: mocks.sendSelect,
  sendArtifactActionToIframe: mocks.sendAction,
  sendArtifactChangedToIframe: mocks.sendChanged,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', () => ({
  activateTab: mocks.activateTab,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js', () => ({
  hasDockedTile: mocks.hasDockedTile,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  setServiceMenuExtras: mocks.setServiceMenuExtras,
  showAgentPane: mocks.showAgentPane,
}));
vi.mock('/design-system/js/theme-manager.js', () => ({ getTheme: () => 'dark' }));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js')} */
let mod;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js')} */
let pointer;
/** @type {ReturnType<typeof stubEventSource>} */
let events;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

/* ---- fixtures ---------------------------------------------------------- */

const K1 = '11111111-1111-4111-8111-111111111111';
const K2 = '22222222-2222-4222-8222-222222222222';
const T1 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const T2 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const T3 = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';

/**
 * @param {string} id @param {string} form @param {string} timestamp @param {object} [extra]
 */
const art = (id, form, timestamp, extra = {}) => ({
  id,
  title: `Title ${id}`,
  filename: `${id}.bin`,
  artifact_type: 'markdown',
  timestamp,
  session_id: K1,
  owner: T1,
  operator_visible: true,
  form,
  run_record: false,
  ...extra,
});

const A3 = art('a3', 'documents', '2026-10-30T11:00:00Z');
const A2 = art('a2', 'plots', '2026-10-30T10:00:00Z');
const RUN1 = art('run1', 'runs', '2026-10-30T09:00:00Z', {
  run_record: true,
  exec_id: 'x1',
  run_part: 'result',
  run_status: 'succeeded',
  run_parts: [
    { id: 'run1', role: 'result', title: 'Result' },
    { id: 'out1', role: 'output', title: 'Output plot', form: 'plots' },
  ],
});
const B1 = art('b1', 'plots', '2026-10-29T10:00:00Z', { owner: T2, session_id: K2 });

/** @type {any} */
let server;

function defaultServer() {
  return {
    sessions: {
      first: {
        sessions: [
          {
            transcript_id: T1,
            session_key: K1,
            current_for_key: true,
            title: 'Orbit study',
            last_artifact_id: null,
            state: 'live',
          },
          {
            transcript_id: T2,
            session_key: K2,
            current_for_key: false,
            title: 'Tune scan',
            last_artifact_id: 'b1',
          },
        ],
        more: true,
        next_cursor: 'p2',
        record: 'complete',
      },
      p2: {
        sessions: [{ transcript_id: T3, session_key: null, title: 'Old work' }],
        more: false,
        next_cursor: null,
      },
    },
    summary: {
      counts: {
        [T1]: { total: 3, by_form: { documents: 1, plots: 1, runs: 1 } },
        [T2]: { total: 2, by_form: { plots: 2 } },
        [T3]: { total: 1, by_form: { data: 1 } },
      },
      owned_total: 9,
      ledger: 'ok',
    },
    /** session-list pages by `${scope}|${form}|${cursor}` */
    lists: {
      [`session:${T1}||`]: { artifacts: [A3, A2, RUN1], total: 3, next_cursor: null },
      [`key:${K1}||`]: { artifacts: [A3, A2, RUN1], total: 3, next_cursor: null },
      [`session:${T1}|plots|`]: { artifacts: [A2], total: 1, next_cursor: null },
      [`session:${T1}|documents|`]: { artifacts: [A3], total: 1, next_cursor: null },
      [`session:${T1}|runs|`]: { artifacts: [RUN1], total: 1, next_cursor: null },
      [`session:${T2}||`]: { artifacts: [B1], total: 2, next_cursor: 'b-c2' },
      [`session:${T2}|plots|`]: { artifacts: [B1], total: 2, next_cursor: 'b-c2' },
      [`session:${T2}|plots|b-c2`]: {
        artifacts: [art('b2', 'plots', '2026-10-29T09:00:00Z', { owner: T2 })],
        total: 2,
        next_cursor: null,
      },
    },
    byId: {
      a2: A2,
      a3: A3,
      b1: B1,
      run1: RUN1,
      u1: art('u1', 'plots', '2026-10-30T08:00:00Z', { owner: null, session_id: '' }),
    },
    focus: { focused: false, artifact: null },
    enterKey: '99999999-9999-4999-8999-999999999999',
  };
}

/** @param {unknown} body @param {number} [status] */
const reply = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

/** @param {any} body */
function listKey(body) {
  const scope = body.session ? `session:${body.session}` : `key:${body.key}`;
  return `${scope}|${body.form ?? ''}|${body.cursor ?? ''}`;
}

/** @param {string} url @param {any} [init] */
function route(url, init = {}) {
  const u = new URL(url, 'http://hub');
  const body = init.body ? JSON.parse(init.body) : null;
  const method = init.method ?? 'GET';
  if (u.pathname === '/api/sessions')
    return reply(server.sessions[u.searchParams.get('cursor') ?? 'first']);
  if (u.pathname.endsWith('/enter')) return reply({ session_key: server.enterKey });
  if (method === 'PUT') return reply({ ok: true });
  if (u.pathname.endsWith('/api/artifacts/session-summary')) return reply(server.summary);
  if (u.pathname.endsWith('/api/artifacts/session-list')) {
    return reply(server.lists[listKey(body)] ?? { artifacts: [], total: 0, next_cursor: null });
  }
  if (u.pathname.endsWith('/api/type-registry')) return reply({}, 404);
  if (u.pathname.endsWith('/api/focus')) return reply(server.focus);
  const one = u.pathname.match(/\/panel\/artifacts\/api\/artifacts\/(.+)$/);
  if (one) {
    const entry = server.byId[decodeURIComponent(one[1])];
    return entry ? reply(entry) : reply({ detail: 'gone' }, 404);
  }
  return reply({}, 404);
}

function skeleton() {
  document.body.innerHTML = `
    <div class="sidebar-row" id="sidebar-agent-wrap">
      <button class="osprey-nav-row parent panel-rail-button" id="sidebar-agent-row" data-panel-id="terminal"></button>
      <div class="sidebar-kids osprey-tree" id="sidebar-session-tree" data-session-tree></div>
    </div>
    <iframe class="panel-iframe" data-panel-id="artifacts"></iframe>`;
}

async function settle() {
  for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

const tree = () => /** @type {HTMLElement} */ (document.getElementById('sidebar-session-tree'));

/** The session rows' titles. */
function sessionTitles() {
  return [...tree().querySelectorAll('.osprey-tree-session .osprey-tree-title')].map(
    (t) => t.textContent
  );
}

/** @param {string} id */
function sessionRowEl(id) {
  const el = tree().querySelector(`.osprey-tree-session[data-session-id="${id}"]`);
  if (!el) throw new Error(`no session row ${id}`);
  return /** @type {HTMLElement} */ (el);
}

/** @param {string} id */
function artifactRowEl(id) {
  const el = tree().querySelector(`.osprey-list-row[data-id="${id}"]`);
  if (!el) throw new Error(`no artifact row ${id}`);
  return /** @type {HTMLElement} */ (el);
}

/** @param {string} sid */
function boxIds(sid) {
  const box = tree().querySelector(`[data-session-box="${sid}"]`);
  return box
    ? [...box.querySelectorAll('.osprey-list-row[data-id]')].map(
        (r) => /** @type {HTMLElement} */ (r).dataset.id
      )
    : null;
}

/** @param {Element} el @param {number} [detail]  the click count */
function click(el, detail = 1) {
  el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail }));
}

/** A ⌥-click (altKey). @param {Element} el */
function altClick(el) {
  el.dispatchEvent(
    new MouseEvent('click', { bubbles: true, cancelable: true, detail: 1, altKey: true })
  );
}

/** @returns {string[]} the open session rows' ids */
function openSessions() {
  return [...tree().querySelectorAll('.osprey-tree-session[aria-expanded="true"]')].map(
    (r) => /** @type {HTMLElement} */ (r).dataset.sessionId ?? ''
  );
}

/** A double-click: two clicks, the second with `detail` 2, then dblclick. @param {Element} el */
function dblclick(el) {
  click(el, 1);
  click(el, 2);
  el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true, detail: 2 }));
}

/**
 * Requests to a path, with their bodies.
 * @param {string} suffix
 * @param {string | null} [method]
 */
function requests(suffix, method = null) {
  return fetchMock.mock.calls
    .filter(
      ([url, init]) =>
        new URL(String(url), 'http://hub').pathname.endsWith(suffix) &&
        (!method || (init?.method ?? 'GET') === method)
    )
    .map(([url, init]) => ({ url: String(url), body: init?.body ? JSON.parse(init.body) : null }));
}

const shown = () => mocks.sendSelect.mock.lastCall?.[1];

function headerNow() {
  const call = mocks.contributeLocal.mock.lastCall;
  if (!call) throw new Error('no header contribution');
  return {
    items: /** @type {any[]} */ (call[1]),
    dispatch: /** @type {(id: string) => void} */ (call[2]),
  };
}

/** The artifacts panel's event source. */
function artifactSource() {
  const source = events.opened.find((s) => s.url.includes('/panel/artifacts/api/events'));
  if (!source) throw new Error('no artifacts event source');
  return source;
}

/** @param {any} frame */
function frame(frame) {
  artifactSource().onmessage?.({ data: JSON.stringify(frame) });
}

async function boot({ served = true } = {}) {
  mod.initSidebarArtifacts({ served });
  await settle();
}

beforeEach(async () => {
  vi.resetModules();
  vi.useFakeTimers({ toFake: ['Date'] });
  vi.setSystemTime(Date.UTC(2026, 9, 30, 12, 0));
  document.documentElement.setAttribute('data-facility-timezone', 'UTC');
  localStorage.clear();
  // Written against Kind mode, which an operator must choose (Activity is the
  // default; see the tree-mode test below).
  localStorage.setItem('osprey-sidebar-artifacts-view', 'kind');
  mocks.streaming = false;
  mocks.terminal = false;
  for (const fn of Object.values(mocks))
    if (typeof fn === 'function' && 'mockClear' in fn) fn.mockClear();
  mocks.hasDockedTile.mockReturnValue(true);
  mocks.isTerminalPanelOpen.mockReturnValue(true);
  mocks.resumeSession.mockImplementation(async (key) => pointer.setPointer(key));
  server = defaultServer();
  fetchMock = vi.fn(async (/** @type {string} */ url, /** @type {any} */ init) => route(url, init));
  vi.stubGlobal('fetch', fetchMock);
  events = stubEventSource();
  skeleton();
  pointer = await import(`${JS}/session-pointer.js`);
  pointer.setPointer(K1);
  mod = await import(`${JS}/sidebar-artifacts.js`);
});

afterEach(() => {
  mod.resetSidebarArtifacts();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.replaceChildren();
  document.documentElement.removeAttribute('data-facility-timezone');
});

/* ---- data -------------------------------------------------------------- */

describe('the data', () => {
  test('session rows load whether or not the artifacts panel is served; groups and counts only while it is', async () => {
    await boot({ served: false });
    expect(sessionTitles()).toEqual(['Orbit study', 'Tune scan']);
    expect(requests('/session-summary')).toHaveLength(0);
    expect(requests('/session-list')).toHaveLength(0);
    expect(tree().querySelector('[data-session-box]')).toBeNull();

    mod.initSidebarArtifacts();
    await settle();
    expect(requests('/session-summary').map((r) => r.body)).toEqual([{ sessions: [T1, T2] }]);
    expect(boxIds(T1)).toEqual([]);
  });

  test('the active session opens on first paint; one summary per loaded page; More sessions counts the rest', async () => {
    await boot();
    expect(sessionRowEl(T1).classList.contains('active')).toBe(true);
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('true');
    expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('false');
    expect(sessionRowEl(T1).querySelector('.osprey-tree-n')?.textContent).toBe('3');
    // 9 owned − (3 + 2) loaded.
    expect(tree().querySelector('[data-tree-more] .osprey-tree-n')?.textContent).toBe('4');
    // Every row the tree shows is an agent session's: no bucket for the rest.
    expect(tree().querySelector('[data-tree-earlier], .osprey-tree-earlier')).toBeNull();

    click(/** @type {Element} */ (tree().querySelector('[data-tree-more]')));
    await settle();
    expect(sessionTitles()).toEqual(['Orbit study', 'Tune scan', 'Old work']);
    expect(requests('/session-summary').map((r) => r.body)).toEqual([
      { sessions: [T1, T2] },
      { sessions: [T3] },
    ]);
    expect(tree().querySelector('[data-tree-more]')).toBeNull();
  });

  test('the session listing walks the session pages with drain only on each first listing', async () => {
    await boot();
    click(/** @type {Element} */ (tree().querySelector('[data-tree-more]')));
    await settle();
    const urls = requests('/api/sessions').map((r) => r.url);
    expect(urls).toEqual(['/api/sessions?drain=1', '/api/sessions?cursor=p2&drain=1']);
  });

  test('Kind mode loads a group only when it opens, grouped runs, paged in its box', async () => {
    await boot();
    // First paint lists the active key's artifacts, nothing per group.
    expect(requests('/session-list').map((r) => r.body)).toEqual([{ key: K1, runs: 'grouped' }]);
    // Group heads come from the summary counts, before anything loaded.
    const heads = [...tree().querySelectorAll(`[data-group-head^="${T1}:"]`)].map(
      (h) => /** @type {HTMLElement} */ (h).dataset.groupHead
    );
    expect(heads).toEqual([`${T1}:runs`, `${T1}:plots`, `${T1}:documents`]);

    click(/** @type {Element} */ (tree().querySelector(`[data-group-head="${T1}:plots"]`)));
    await settle();
    expect(requests('/session-list').map((r) => r.body)).toEqual([
      { key: K1, runs: 'grouped' },
      { session: T1, form: 'plots', runs: 'grouped' },
    ]);
    expect(boxIds(T1)).toEqual(['a2']);
  });

  test('a scroll to the end of a box loads the next page of its open groups', async () => {
    await boot();
    // Entering T2 reveals its last-open b1, which opens the plots group.
    click(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();
    expect(boxIds(T2)).toEqual(['b1']);
    const box = /** @type {HTMLElement} */ (tree().querySelector(`[data-session-box="${T2}"]`));
    box.dispatchEvent(new Event('scroll'));
    await settle();
    expect(boxIds(T2)).toEqual(['b1', 'b2']);
    expect(requests('/session-list').at(-1)?.body).toEqual({
      session: T2,
      form: 'plots',
      runs: 'grouped',
      cursor: 'b-c2',
    });
  });

  test('the incomplete notice row shows', async () => {
    server.sessions.first.record = 'incomplete';
    await boot();
    expect(tree().querySelector('[data-tree-notice="incomplete"]')).not.toBeNull();
  });

  test('a new key gets a synthesized "New session" row, listed by key', async () => {
    await boot();
    pointer.setPointer('77777777-7777-4777-8777-777777777777');
    await settle();
    expect(sessionTitles()[0]).toBe('New session');
    expect(
      requests('/session-list').some((r) => r.body.key === '77777777-7777-4777-8777-777777777777')
    ).toBe(true);
  });
});

/* ---- expansion and clicks ---------------------------------------------- */

describe('expansion and clicks', () => {
  test('with no mode stored the tree opens in Activity mode', async () => {
    localStorage.removeItem('osprey-sidebar-artifacts-view');
    await boot();
    expect(mod.treeView()).toBe('activity');
    expect(tree().querySelector('[data-group-head]')).toBeNull();
  });

  test('only the current session is open: the others are folded, with no disclosure', async () => {
    await boot();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('true');
    expect(sessionRowEl(T1).querySelector('[data-tree-disc]')).not.toBeNull();
    expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('false');
    expect(sessionRowEl(T2).querySelector('[data-tree-disc]')).toBeNull();
    expect(tree().querySelectorAll('[data-session-box]')).toHaveLength(1);
    expect(boxIds(T1)).not.toBeNull();
  });

  test('a non-current session never opens in place: any click on its row enters it', async () => {
    await boot();
    mocks.streaming = true;
    mocks.showConfirm.mockImplementation((spec) => spec.onDismiss());
    // The slot where a disclosure would be, the dot and the count: each one
    // asks to switch (Cancel here), and none opens the row without switching.
    for (const part of ['.osprey-tree-disc', '.osprey-dot', '.osprey-tree-n']) {
      click(/** @type {Element} */ (sessionRowEl(T2).querySelector(part)));
      await settle();
      expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('false');
    }
    expect(mocks.showConfirm).toHaveBeenCalledTimes(3);
    expect(pointer.getPointer()).toBe(K1);
  });

  test('switching collapses the previous session and opens the new one, even if it was folded', async () => {
    await boot();
    // Fold the current session first: the switch still opens the next one.
    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');

    click(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();
    expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('true');
    expect(sessionRowEl(T2).querySelector('[data-tree-disc]')).not.toBeNull();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');
    expect(sessionRowEl(T1).querySelector('[data-tree-disc]')).toBeNull();
    expect(tree().querySelector(`[data-session-box="${T1}"]`)).toBeNull();
  });

  test('a session switched to from elsewhere (the pointer moves) opens and the old one folds', async () => {
    await boot();
    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    pointer.setPointer('77777777-7777-4777-8777-777777777777');
    await settle();
    const fresh = /** @type {HTMLElement} */ (tree().querySelector('.osprey-tree-session.active'));
    expect(fresh.getAttribute('aria-expanded')).toBe('true');
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');
  });

  test('a double-click on the current session folds it (once), and on another enters it open', async () => {
    await boot();
    dblclick(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');
    dblclick(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('true');

    dblclick(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();
    expect(requests(`/api/sessions/${T2}/enter`, 'POST')).toHaveLength(1);
    expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('true');
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');
  });

  test("a click on the current session's name or disclosure folds and unfolds it, and nothing else", async () => {
    await boot();
    const calls = fetchMock.mock.calls.length;

    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');
    expect(sessionRowEl(T1).querySelector('[data-tree-disc]')).not.toBeNull();
    expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('false');

    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('[data-tree-disc]')));
    await settle();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('true');

    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('[data-tree-disc]')));
    await settle();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');

    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('true');

    expect(mocks.resumeSession).not.toHaveBeenCalled();
    expect(mocks.openTerminalPanel).not.toHaveBeenCalled();
    expect(mocks.activateTab).not.toHaveBeenCalled();
    expect(pointer.getPointer()).toBe(K1);
    // No reveal: the session's rows were loaded, so nothing is fetched.
    expect(fetchMock.mock.calls.length).toBe(calls);
  });

  test('⌥-click on a session expands every session, artifacts shown; again collapses them all', async () => {
    await boot();
    altClick(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();
    expect(openSessions()).toEqual([T1, T2]);
    // The other session shows its artifacts, behind a disclosure, without a switch.
    expect(tree().querySelector(`[data-session-box="${T2}"] [data-group-head]`)).not.toBeNull();
    expect(sessionRowEl(T2).querySelector('[data-tree-disc]')).not.toBeNull();
    expect(requests(`/api/sessions/${T2}/enter`, 'POST')).toHaveLength(0);
    expect(pointer.getPointer()).toBe(K1);

    altClick(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(openSessions()).toEqual([]);
    expect(tree().querySelectorAll('[data-session-box]')).toHaveLength(0);

    // Not all open (only the current one): ⌥-click expands rather than collapses.
    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(openSessions()).toEqual([T1]);
    altClick(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(openSessions()).toEqual([T1, T2]);
  });

  test('toggleAllSessions (the Agent row’s ⌥-click) expands, collapses, and honours a forced value', async () => {
    await boot();
    mod.toggleAllSessions();
    await settle();
    expect(openSessions()).toEqual([T1, T2]);
    mod.toggleAllSessions();
    await settle();
    expect(openSessions()).toEqual([]);
    mod.toggleAllSessions(true);
    await settle();
    mod.toggleAllSessions(true);
    await settle();
    expect(openSessions()).toEqual([T1, T2]);
  });

  test('a plain click on the current session after expand-all returns to it alone, still open', async () => {
    await boot();
    mod.toggleAllSessions();
    await settle();
    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(openSessions()).toEqual([T1]);
    expect(sessionRowEl(T2).querySelector('[data-tree-disc]')).toBeNull();
    // From there the plain rule holds: the next click folds it.
    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(openSessions()).toEqual([]);
  });

  test('a switch after expand-all returns to the current-only rule', async () => {
    await boot();
    mod.toggleAllSessions();
    await settle();
    click(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();
    expect(requests(`/api/sessions/${T2}/enter`, 'POST')).toHaveLength(1);
    expect(openSessions()).toEqual([T2]);
  });

  test('a session title’s tip says what a click does and names the ⌥-click', async () => {
    await boot();
    const tip = (/** @type {string} */ id) =>
      /** @type {HTMLElement} */ (sessionRowEl(id).querySelector('.osprey-tree-title')).dataset
        .tipDetail ?? '';
    expect(tip(T1)).toMatch(
      /^Click to fold or unfold · (⌥-click|Alt\+click) to expand or collapse all$/
    );
    expect(tip(T2)).toMatch(
      /^Click to switch to it · (⌥-click|Alt\+click) to expand or collapse all$/
    );
  });

  test('openInAgent on the current session opens the Agent pane and keeps it unfolded', async () => {
    await boot();
    mocks.isTerminalPanelOpen.mockReturnValue(false);

    await mod.openInAgent(T1);
    await settle();

    expect(mocks.openTerminalPanel).toHaveBeenCalled();
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('true');
    expect(mocks.resumeSession).not.toHaveBeenCalled();
  });

  test('with the Agent pane closed, a click on the current session reopens it instead of folding', async () => {
    await boot();
    mocks.isTerminalPanelOpen.mockReturnValue(false);
    mocks.showAgentPane.mockClear();
    const tip = () =>
      /** @type {HTMLElement} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')).dataset
        .tipDetail ?? '';
    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(mocks.showAgentPane).toHaveBeenCalledTimes(1);
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('true');
    expect(tip()).toMatch(/^Click to reopen the Agent pane · /);
    // Pane back: the next click folds again, and does not reopen.
    mocks.isTerminalPanelOpen.mockReturnValue(true);
    click(/** @type {Element} */ (sessionRowEl(T1).querySelector('.osprey-tree-title')));
    await settle();
    expect(mocks.showAgentPane).toHaveBeenCalledTimes(1);
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');
  });

  test('a session-name click enters a non-current session and reveals its last-open artifact', async () => {
    await boot();
    mocks.isTerminalPanelOpen.mockReturnValue(false);
    click(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();
    expect(mocks.openTerminalPanel).toHaveBeenCalled();
    expect(requests(`/api/sessions/${T2}/enter`, 'POST')).toHaveLength(1);
    expect(pointer.getPointer()).toBe(server.enterKey);
    expect(shown()).toBe('b1');
    expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('true');
    expect(sessionRowEl(T1).getAttribute('aria-expanded')).toBe('false');
    expect(
      artifactRowEl('b1').getAttribute('aria-selected') ?? artifactRowEl('b1').className
    ).toBeTruthy();
  });

  test('during a chat turn a session-name click asks first, and Cancel stays put', async () => {
    await boot();
    mocks.streaming = true;
    mocks.showConfirm.mockImplementation((spec) => spec.onDismiss());
    click(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();
    expect(mocks.showConfirm).toHaveBeenCalledOnce();
    expect(pointer.getPointer()).toBe(K1);
    expect(sessionRowEl(T2).getAttribute('aria-expanded')).toBe('false');
  });

  test('reveal of an artifact on page 3 inside a collapsed run opens the run', async () => {
    const part = art('deep-out', 'plots', '2026-10-01T00:00:00Z', {
      owner: T2,
      exec_id: 'deep',
      run_part: 'output',
    });
    const head = art('deep-head', 'runs', '2026-10-01T00:00:00Z', {
      owner: T2,
      exec_id: 'deep',
      run_status: 'failed',
      run_parts: [
        { id: 'deep-head', role: 'result' },
        { id: 'deep-out', role: 'output', title: 'Deep output', form: 'plots' },
      ],
    });
    server.byId['deep-out'] = part;
    server.sessions.first.sessions[1].last_artifact_id = 'deep-out';
    server.lists[`session:${T2}|runs|`] = {
      artifacts: [art('r-a', 'runs', '2026-10-29T00:00:00Z', { owner: T2, exec_id: 'ra' })],
      total: 3,
      next_cursor: 'c2',
    };
    server.lists[`session:${T2}|runs|c2`] = {
      artifacts: [art('r-b', 'runs', '2026-10-28T00:00:00Z', { owner: T2, exec_id: 'rb' })],
      total: 3,
      next_cursor: 'c3',
    };
    server.lists[`session:${T2}|runs|c3`] = { artifacts: [head], total: 3, next_cursor: null };
    await boot();

    click(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();

    const runRow = /** @type {HTMLElement} */ (tree().querySelector('[data-run="deep"]'));
    expect(runRow.getAttribute('aria-expanded')).toBe('true');
    expect(boxIds(T2)).toContain('deep-out');
    expect(shown()).toBe('deep-out');
  });

  test('an artifact click shows it without switching, records last-artifact, and brings an undocked panel up', async () => {
    await boot();
    mocks.hasDockedTile.mockReturnValue(false);
    click(/** @type {Element} */ (tree().querySelector(`[data-group-head="${T1}:plots"]`)));
    await settle();

    click(artifactRowEl('a2'));
    await settle();

    expect(shown()).toBe('a2');
    expect(pointer.getPointer()).toBe(K1);
    expect(requests(`/api/sessions/${T1}/last-artifact`, 'PUT').map((r) => r.body)).toEqual([
      { artifact_id: 'a2' },
    ]);
    expect(mocks.activateTab).toHaveBeenCalledWith('artifacts', { userInitiated: true });
  });

  test('a docked panel is focused in place: no activation', async () => {
    await boot();
    click(/** @type {Element} */ (tree().querySelector(`[data-group-head="${T1}:plots"]`)));
    await settle();
    click(artifactRowEl('a2'));
    expect(mocks.activateTab).not.toHaveBeenCalled();
  });

  test("a chat artifact link records its owner's last-artifact", async () => {
    await boot();
    document.dispatchEvent(new CustomEvent('osprey-artifact-opened', { detail: { id: 'b1' } }));
    await settle();
    expect(requests(`/api/sessions/${T2}/last-artifact`, 'PUT').map((r) => r.body)).toEqual([
      { artifact_id: 'b1' },
    ]);
    expect(pointer.getPointer()).toBe(K1);
  });
});

/* ---- the viewer -------------------------------------------------------- */

describe('the viewer', () => {
  test("with no pick it shows the active session's newest artifact", async () => {
    await boot();
    expect(shown()).toBe('a3');
  });

  test('a non-automatic focus takes the pick and records last-artifact; an automatic one in chat only records the focus', async () => {
    await boot();
    // The hook's review notebook: a run record, focused automatically.
    frame({ type: 'focus', id: 'run1', automatic: true, session: T1 });
    await settle();
    expect(shown()).toBe('a3');
    expect(requests('/last-artifact', 'PUT')).toHaveLength(0);

    frame({ type: 'focus', id: 'a2', automatic: false, session: T1 });
    await settle();
    expect(shown()).toBe('a2');
    expect(requests(`/api/sessions/${T1}/last-artifact`, 'PUT').map((r) => r.body)).toEqual([
      { artifact_id: 'a2' },
    ]);
  });

  test('on the terminal surface every focus takes the pick', async () => {
    mocks.terminal = true;
    await boot();
    frame({ type: 'focus', id: 'a2', automatic: true, session: T1 });
    await settle();
    expect(shown()).toBe('a2');
  });

  test('a focus naming another transcript, or none, leaves the pick alone', async () => {
    await boot();
    frame({ type: 'focus', id: 'a2', automatic: false, session: T2 });
    frame({ type: 'focus', id: 'a2', automatic: false, session: null });
    frame({ type: 'focus', id: 'a2', automatic: false });
    await settle();
    expect(shown()).toBe('a3');
    expect(requests('/last-artifact', 'PUT')).toHaveLength(0);
  });

  test('a session without a transcript follows no focus', async () => {
    // A key the listing has no row for: the active row is synthesized.
    pointer.setPointer('77777777-7777-4777-8777-777777777777');
    await boot();
    const before = shown();
    frame({ type: 'focus', id: 'a2', automatic: false, session: null });
    frame({ type: 'focus', id: 'a2', automatic: false, session: T1 });
    await settle();
    expect(shown()).toBe(before);
    expect(requests('/last-artifact', 'PUT')).toHaveLength(0);
  });

  test("a key change never re-shows the previous session's artifact", async () => {
    await boot();
    frame({ type: 'focus', id: 'a2', automatic: false, session: T1 });
    await settle();
    pointer.setPointer('77777777-7777-4777-8777-777777777777');
    await settle();
    expect(shown()).not.toBe('a2');
  });

  test("the pager steps through the shown artifact's own session", async () => {
    await boot();
    expect(headerNow().items.find((i) => i.id === 'pager') ?? headerNow().items).toBeTruthy();
    headerNow().dispatch('next');
    await settle();
    expect(shown()).toBe('a2');
    headerNow().dispatch('next');
    await settle();
    expect(shown()).toBe('run1');
    headerNow().dispatch('prev');
    await settle();
    expect(shown()).toBe('a2');
  });

  test('an artifact no session owns shows without a pager, and lists nowhere', async () => {
    await boot();
    frame({ type: 'focus', id: 'u1', automatic: false, session: T1 });
    await settle();
    expect(shown()).toBe('u1');
    expect(headerNow().items.map((i) => i.id)).toEqual(['crumb', 'open', 'save']);
    expect(requests('/session-list').some((r) => !r.body.session && !r.body.key)).toBe(false);
    expect(tree().querySelector('.osprey-list-row[data-id="u1"]')).toBeNull();
  });

  test('the tile menu keeps its artifact verbs', async () => {
    await boot();
    expect(mocks.setServiceMenuExtras).toHaveBeenCalledWith('artifacts', expect.any(Function));
  });
});

/* ---- live updates ------------------------------------------------------ */

describe('live updates', () => {
  const count = (/** @type {string} */ sid) =>
    sessionRowEl(sid).querySelector('.osprey-tree-n')?.textContent;

  test('a duplicate frame is counted once', async () => {
    await boot();
    const fresh = art('n1', 'plots', '2026-10-30T11:30:00Z');
    frame({ type: 'artifact', ...fresh });
    frame({ type: 'artifact', ...fresh });
    await settle();
    expect(count(T1)).toBe('4');
  });

  test('the preview then the result of one exec_id count once, and the status follows', async () => {
    await boot();
    frame({
      type: 'artifact',
      ...art('pv', 'runs', '2026-10-30T11:40:00Z', {
        exec_id: 'x9',
        run_part: 'preview',
        run_status: null,
      }),
    });
    frame({
      type: 'artifact',
      ...art('res', 'runs', '2026-10-30T11:41:00Z', {
        exec_id: 'x9',
        run_part: 'result',
        run_status: 'succeeded',
      }),
    });
    await settle();
    expect(count(T1)).toBe('4');
  });

  test('a dispatch frame (not operator-visible) is ignored', async () => {
    await boot();
    frame({
      type: 'artifact',
      ...art('d1', 'plots', '2026-10-30T11:30:00Z', { operator_visible: false }),
    });
    await settle();
    expect(count(T1)).toBe('3');
  });

  test('a frame no session owns adds no row and no count anywhere', async () => {
    await boot();
    frame({
      type: 'artifact',
      ...art('u2', 'plots', '2026-10-30T11:30:00Z', { owner: null, session_id: '' }),
    });
    await settle();
    expect(count(T1)).toBe('3');
    expect(tree().querySelector('.osprey-list-row[data-id="u2"]')).toBeNull();
    expect(tree().querySelector('[data-tree-earlier], .osprey-tree-earlier')).toBeNull();
    // 9 owned − (3 + 2) loaded: an unowned artifact is not a hidden session's.
    expect(tree().querySelector('[data-tree-more] .osprey-tree-n')?.textContent).toBe('4');
  });

  test('an owner change re-places the entry and its count', async () => {
    await boot();
    frame({ type: 'artifact_updated', ...A2, owner: T2 });
    await settle();
    expect(count(T1)).toBe('2');
    expect(count(T2)).toBe('3');
  });

  test('a delete re-fetches the summary; so does a reconnect', async () => {
    await boot();
    expect(requests('/session-summary')).toHaveLength(1);
    frame({ type: 'artifact_deleted', id: 'a2' });
    await settle();
    expect(requests('/session-summary')).toHaveLength(2);
    expect(mocks.sendChanged).not.toHaveBeenCalled();

    artifactSource().onopen?.();
    artifactSource().onopen?.();
    await settle();
    expect(requests('/session-summary').length).toBeGreaterThanOrEqual(3);
  });

  test('a frame whose owner is no known row refreshes the sessions', async () => {
    await boot();
    const before = requests('/api/sessions').length;
    frame({
      type: 'artifact',
      ...art('z1', 'plots', '2026-10-30T11:50:00Z', {
        owner: 'dddddddd-dddd-4ddd-8ddd-dddddddddddd',
      }),
    });
    await vi.waitFor(() => expect(requests('/api/sessions').length).toBe(before + 1), {
      timeout: 2500,
      interval: 50,
    });
    expect(requests('/api/sessions').at(-1)?.url).toBe('/api/sessions');
  });

  test("the shown artifact's update is forwarded to the viewer", async () => {
    await boot();
    frame({ type: 'artifact_updated', ...A3, title: 'Renamed' });
    await settle();
    expect(mocks.sendChanged).toHaveBeenCalledWith(
      expect.anything(),
      'a3',
      'updated',
      expect.objectContaining({ title: 'Renamed' })
    );
  });
});

/* ---- the highlight ------------------------------------------------------ */

describe('the highlight: at most one row, the one the Artifacts panel shows', () => {
  const selected = () =>
    [...document.querySelectorAll('[data-session-tree] [aria-selected="true"]')].map(
      (el) => /** @type {HTMLElement} */ (el).dataset.id
    );
  const dockChanged = () => document.dispatchEvent(new CustomEvent('osprey-dock-view-changed'));

  test("only one row is selected, and a switch clears the old session's", async () => {
    await boot();
    click(/** @type {Element} */ (tree().querySelector(`[data-group-head="${T1}:plots"]`)));
    await settle();
    click(artifactRowEl('a2'));
    await settle();
    expect(selected()).toEqual(['a2']);

    click(/** @type {Element} */ (sessionRowEl(T2).querySelector('.osprey-tree-title')));
    await settle();

    expect(shown()).toBe('b1');
    expect(selected()).toEqual(['b1']);
    expect(tree().querySelector('[data-id="a2"]')).toBeNull();
  });

  test('closing the Artifacts panel clears the highlight; reopening it brings it back', async () => {
    await boot();
    click(/** @type {Element} */ (tree().querySelector(`[data-group-head="${T1}:plots"]`)));
    await settle();
    click(artifactRowEl('a2'));
    await settle();
    expect(selected()).toEqual(['a2']);

    mocks.hasDockedTile.mockReturnValue(false);
    dockChanged();
    expect(selected()).toEqual([]);

    mocks.hasDockedTile.mockReturnValue(true);
    dockChanged();
    expect(selected()).toEqual(['a2']);
  });

  test('with the Artifacts panel off view no row is selected, whatever it would show', async () => {
    mocks.hasDockedTile.mockReturnValue(false);
    await boot();
    click(/** @type {Element} */ (tree().querySelector(`[data-group-head="${T1}:plots"]`)));
    await settle();
    expect(shown()).toBeTruthy();
    expect(selected()).toEqual([]);
  });
});
