// @ts-check
/**
 * Page load resumes the operator's last-active session:
 *   npx vitest run tests/interfaces/web_terminal/page-load-resume.test.mjs
 *
 * Before either surface boots, app.js makes one bounded GET of
 * `/api/sessions/last-active` and points the tab at its key (the server wins
 * over this browser's pointer). On no key, an error or a timeout the stored
 * pointer stays, and with neither the surfaces mint a key as they always have.
 * The terminal and the chat then bind from the pointer once each, as before.
 *
 * Once the listing has the active row, the tree applies the page-load rule
 * (OR-3): with the Artifacts panel docked its last-open artifact is shown
 * there; otherwise the layout is left alone and no row is highlighted (the
 * tree highlights only what the panel shows). Page load never calls the open
 * rule (`activateTab`).
 *
 * The real terminal.js, chat.js, session-pointer.js, session-model.js and
 * sidebar-artifacts.js are mounted in app.js's boot order; xterm and the
 * WebSocket are _terminal-fake.mjs's, the chat client is a mock whose
 * `fetchHistory` counts the console's replays, and the panel collaborators
 * that need a running page are mocks.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { stubEventSource } from '../_support/event-source.mjs';
import { FakeWebSocket, stubTerminalGlobals } from './_terminal-fake.mjs';

const mocks = vi.hoisted(() => ({
  fetchHistory: vi.fn(),
  activateTab: vi.fn(),
  hasDockedTile: vi.fn(() => true),
  sendSelect: vi.fn(),
  contributeLocal: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js', () => ({
  fetchHistory: mocks.fetchHistory,
  sendPrompt: vi.fn(() => ({ abort: () => {}, aborted: false })),
  interrupt: vi.fn(async () => undefined),
  requestHandoff: vi.fn(async () => ({ state: 'chat', session_id: 'k' })),
  fetchCommands: vi.fn(async () => []),
  followChat: vi.fn(() => ({ close: () => {} })),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', () => ({
  activateTab: mocks.activateTab,
  initPanelManager: vi.fn(),
  navigateAndActivatePanel: vi.fn(),
  getPanelStandaloneUrl: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js', () => ({
  hasDockedTile: mocks.hasDockedTile,
}));
// The panel's frame as dock-iframe holds it: the skeleton's.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  managedIframe: (/** @type {string} */ id) => {
    const iframe = document.querySelector(`iframe.panel-iframe[data-panel-id="${id}"]`);
    return iframe ? { iframe, title: 'Artifacts' } : null;
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js', () => ({
  sendArtifactSelectToIframe: mocks.sendSelect,
  sendArtifactActionToIframe: vi.fn(),
  sendArtifactChangedToIframe: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  contributeLocal: mocks.contributeLocal,
  initHeaderContrib: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  setServiceMenuExtras: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  isTerminalPanelOpen: () => true,
  openTerminalPanel: vi.fn(),
  initDockWorkspace: vi.fn(),
}));

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const STORAGE_KEY = 'osprey-pty-session';

/* ---- fixtures ---------------------------------------------------------- */

const K1 = '11111111-1111-4111-8111-111111111111';
const K_LOCAL = '33333333-3333-4333-8333-333333333333';
const T1 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const T_OLD = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';

/** @param {string} id @param {string} form @param {string} timestamp @param {object} [extra] */
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
const OLD1 = art('old1', 'plots', '2026-10-29T10:00:00Z', { owner: T_OLD });

/** @type {any} */
let server;

function defaultServer() {
  return {
    /** The last-active answer: a body, an HTTP status, or 'hang'. */
    lastActive: /** @type {any} */ ({ session_key: K1, at: '2026-10-30T11:30:00Z' }),
    sessions: {
      sessions: [
        {
          transcript_id: T1,
          session_key: K1,
          current_for_key: true,
          title: 'Orbit study',
          last_artifact_id: 'a2',
          state: 'live',
        },
      ],
      more: false,
      next_cursor: null,
      record: 'complete',
    },
    summary: {
      counts: { [T1]: { total: 2, by_form: { documents: 1, plots: 1 } } },
      owned_total: 2,
      ledger: 'ok',
    },
    lists: {
      [`session:${T1}||`]: { artifacts: [A3, A2], total: 2, next_cursor: null },
      [`key:${K1}||`]: { artifacts: [A3, A2], total: 2, next_cursor: null },
      [`session:${T1}|plots|`]: { artifacts: [A2], total: 1, next_cursor: null },
      [`session:${T1}|documents|`]: { artifacts: [A3], total: 1, next_cursor: null },
    },
    byId: { a2: A2, a3: A3, old1: OLD1 },
  };
}

/** @param {unknown} body @param {number} [status] */
const reply = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

/** @param {any} body */
function listKey(body) {
  const scope = body.session ? `session:${body.session}` : body.key ? `key:${body.key}` : 'other';
  return `${scope}|${body.form ?? ''}|${body.cursor ?? ''}`;
}

/** @param {string} url @param {any} [init] */
function route(url, init = {}) {
  const u = new URL(url, 'http://hub');
  const body = init.body ? JSON.parse(init.body) : null;
  const method = init.method ?? 'GET';
  if (u.pathname === '/api/session') return reply({});
  if (u.pathname === '/api/sessions/last-active' && method === 'GET') {
    const answer = server.lastActive;
    if (answer === 'hang') return new Promise(() => {});
    if (typeof answer === 'number') return reply({ detail: 'down' }, answer);
    return reply(answer);
  }
  if (u.pathname === '/api/sessions') return reply(server.sessions);
  if (method === 'PUT') return reply({ ok: true });
  if (u.pathname.endsWith('/api/artifacts/session-summary')) return reply(server.summary);
  if (u.pathname.endsWith('/api/artifacts/session-list'))
    return reply(server.lists[listKey(body)] ?? { artifacts: [], total: 0, next_cursor: null });
  if (u.pathname.endsWith('/api/type-registry')) return reply({}, 404);
  if (u.pathname.endsWith('/api/focus')) return reply({ focused: false, artifact: null });
  const one = u.pathname.match(/\/panel\/artifacts\/api\/artifacts\/(.+)$/);
  if (one) {
    const entry = server.byId[decodeURIComponent(one[1])];
    return entry ? reply(entry) : reply({ detail: 'gone' }, 404);
  }
  return reply({}, 404);
}

/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js')} */
let pointer;
/** Every value the pointer was set to after the page loaded, in order. @type {(string|null)[]} */
let writes;

async function settle() {
  for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Load the page the way app.js boots it: the bounded last-active GET, then
 * the terminal, then the chat, then the session tree (served when `served`).
 * @param {{surface?: 'chat'|'terminal', served?: boolean, timeoutMs?: number}} [options]
 */
async function loadPage({ surface = 'chat', served = true, timeoutMs } = {}) {
  if (surface === 'terminal')
    document.documentElement.setAttribute('data-agent-surface', 'terminal');
  const app = await import(`${JS}/app.js`);
  const terminal = await import(`${JS}/terminal.js`);
  const chat = await import(`${JS}/chat.js`);
  const sidebar = await import(`${JS}/sidebar-artifacts.js`);
  pointer = await import(`${JS}/session-pointer.js`);
  writes = [];
  pointer.subscribe((key) => writes.push(key));
  await app.resumeLastActive(timeoutMs === undefined ? {} : { timeoutMs });
  terminal.initTerminal('terminal-container');
  chat.initChat('operator-container');
  const retryLoadReveal = app.initLoadReveal();
  sidebar.initSidebarArtifacts({ served: false });
  if (served) {
    sidebar.initSidebarArtifacts();
    retryLoadReveal();
  }
  await settle();
  return { app, terminal, chat, sidebar };
}

const tree = () => /** @type {HTMLElement} */ (document.getElementById('sidebar-session-tree'));

/** @param {string} id */
const artifactRow = (id) =>
  /** @type {HTMLElement | null} */ (tree().querySelector(`.osprey-list-row[data-id="${id}"]`));

/** @param {string} id */
const sessionRow = (id) =>
  /** @type {HTMLElement | null} */ (
    tree().querySelector(`.osprey-tree-session[data-session-id="${id}"]`)
  );

const shown = () => mocks.sendSelect.mock.lastCall?.[1];

/** GETs of a path. @param {string} path */
const gets = (path) =>
  fetchMock.mock.calls.filter(
    ([url, init]) =>
      new URL(String(url), 'http://hub').pathname === path && (init?.method ?? 'GET') === 'GET'
  );

beforeEach(() => {
  vi.resetModules();
  localStorage.clear();
  server = defaultServer();
  for (const fn of Object.values(mocks)) fn.mockClear();
  mocks.fetchHistory.mockImplementation(async () => []);
  mocks.hasDockedTile.mockImplementation(() => true);
  document.body.innerHTML = `
    <div><div id="terminal-container"></div></div>
    <div id="operator-container"></div>
    <div class="sidebar-row" id="sidebar-agent-wrap">
      <button class="osprey-nav-row parent panel-rail-button" id="sidebar-agent-row" data-panel-id="terminal"></button>
      <div class="sidebar-kids osprey-tree" id="sidebar-session-tree" data-session-tree></div>
    </div>
    <iframe class="panel-iframe" data-panel-id="artifacts"></iframe>`;
  // @ts-expect-error -- test stub, not a full FontFaceSet
  document.fonts = { ready: Promise.resolve() };
  stubTerminalGlobals();
  stubEventSource();
  fetchMock = vi.fn(async (url, init) => route(String(url), init));
  vi.stubGlobal('fetch', fetchMock);
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(async () => {
  const sidebar = await import(`${JS}/sidebar-artifacts.js`);
  sidebar.resetSidebarArtifacts();
  document.documentElement.removeAttribute('data-agent-surface');
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('the server wins', () => {
  test('a chat page binds the server key once, over the stored one, and shows its artifact in the docked panel', async () => {
    localStorage.setItem(STORAGE_KEY, K_LOCAL);

    await loadPage();

    expect(gets('/api/sessions/last-active')).toHaveLength(1);
    expect(pointer.getPointer()).toBe(K1);
    // Exactly one bind: one replay, of the server's key; the terminal does not connect.
    expect(mocks.fetchHistory).toHaveBeenCalledTimes(1);
    expect(mocks.fetchHistory).toHaveBeenCalledWith(K1);
    expect(FakeWebSocket.created).toBe(0);
    // One pointer write, the resume's; the surfaces bound from it and wrote nothing.
    expect(writes).toEqual([K1]);

    // OR-3, docked: the row's last-open artifact is shown in the panel.
    expect(shown()).toBe('a2');
    expect(sessionRow(T1)?.classList.contains('active')).toBe(true);
    expect(artifactRow('a2')).not.toBeNull();
    // Page load never calls the open rule.
    expect(mocks.activateTab).not.toHaveBeenCalled();
  });

  test('a terminal page resumes the server key over one socket and replays nothing', async () => {
    localStorage.setItem(STORAGE_KEY, K_LOCAL);

    await loadPage({ surface: 'terminal' });

    expect(pointer.getPointer()).toBe(K1);
    expect(FakeWebSocket.created).toBe(1);
    expect(/** @type {FakeWebSocket} */ (FakeWebSocket.last).url).toContain(
      `session_id=${K1}&mode=resume`
    );
    expect(mocks.fetchHistory).not.toHaveBeenCalled();
    expect(mocks.activateTab).not.toHaveBeenCalled();
  });
});

describe('OR-3 without a docked Artifacts panel', () => {
  test('the artifact is revealed in the tree, nothing is highlighted, the layout is left as it is', async () => {
    mocks.hasDockedTile.mockImplementation(() => false);

    await loadPage();

    expect(artifactRow('a2')).not.toBeNull();
    // No panel shows an artifact, so no row is highlighted.
    expect(tree().querySelectorAll('[aria-selected="true"]')).toHaveLength(0);
    // The layout is untouched: no panel opened or activated.
    expect(mocks.activateTab).not.toHaveBeenCalled();
    // The pick is not taken: the panel (wherever it is) keeps its usual rule.
    expect(shown()).not.toBe('a2');
  });

  test('an operator pick shows the artifact', async () => {
    mocks.hasDockedTile.mockImplementation(() => false);
    await loadPage();

    /** @type {HTMLElement} */ (artifactRow('a2')).dispatchEvent(
      new MouseEvent('click', { bubbles: true, cancelable: true })
    );
    await settle();

    expect(shown()).toBe('a2');
  });
});

describe('a key whose transcript moved after /clear', () => {
  test('the new transcript is the one that shows', async () => {
    server.sessions.sessions = [
      {
        transcript_id: T1,
        session_key: K1,
        current_for_key: true,
        title: 'After clear',
        last_artifact_id: 'a2',
      },
      {
        transcript_id: T_OLD,
        session_key: K1,
        current_for_key: false,
        title: 'Before clear',
        last_artifact_id: 'old1',
      },
    ];
    mocks.fetchHistory.mockImplementation(async (key) =>
      key === K1 ? /** @type {any} */ ([{ role: 'user', text: 'after clear' }]) : []
    );

    await loadPage();

    // One replay, keyed by the session key — the server answers it with the
    // transcript now current for that key.
    expect(mocks.fetchHistory).toHaveBeenCalledTimes(1);
    expect(mocks.fetchHistory).toHaveBeenCalledWith(K1);
    // The tree's active row is the new transcript, and its artifact is shown.
    expect(sessionRow(T1)?.classList.contains('active')).toBe(true);
    expect(sessionRow(T_OLD)?.classList.contains('active')).toBe(false);
    expect(shown()).toBe('a2');
    expect(shown()).not.toBe('old1');
  });
});

describe('fallbacks keep the stored pointer', () => {
  test('no last-active key: the stored pointer binds', async () => {
    localStorage.setItem(STORAGE_KEY, K1);
    server.lastActive = { session_key: null, at: null };

    await loadPage();

    expect(pointer.getPointer()).toBe(K1);
    expect(mocks.fetchHistory).toHaveBeenCalledTimes(1);
    expect(mocks.fetchHistory).toHaveBeenCalledWith(K1);
  });

  test('an error: the stored pointer binds', async () => {
    localStorage.setItem(STORAGE_KEY, K1);
    server.lastActive = 500;

    await loadPage();

    expect(pointer.getPointer()).toBe(K1);
    expect(mocks.fetchHistory).toHaveBeenCalledTimes(1);
    expect(mocks.fetchHistory).toHaveBeenCalledWith(K1);
  });

  test('a timeout: page load stops waiting and the stored pointer binds', async () => {
    localStorage.setItem(STORAGE_KEY, K1);
    server.lastActive = 'hang';

    const started = Date.now();
    await loadPage({ timeoutMs: 30 });

    expect(Date.now() - started).toBeLessThan(1000);
    expect(pointer.getPointer()).toBe(K1);
    expect(mocks.fetchHistory).toHaveBeenCalledTimes(1);
    expect(mocks.fetchHistory).toHaveBeenCalledWith(K1);
    // The abort reached the request.
    const [, init] = /** @type {any} */ (gets('/api/sessions/last-active')[0]);
    expect(init.signal.aborted).toBe(true);
  });

  test('the default bound is about 1.5 s', async () => {
    const app = await import(`${JS}/app.js`);
    expect(app.LAST_ACTIVE_TIMEOUT_MS).toBe(1500);
  });

  test('with neither a server key nor a stored one, the chat mints a key once', async () => {
    server.lastActive = { session_key: null, at: null };

    await loadPage();

    expect(writes).toHaveLength(1);
    expect(writes[0]).toMatch(/^[0-9a-f-]{36}$/);
    expect(pointer.getPointer()).toBe(writes[0]);
    // A minted key has nothing to replay.
    expect(mocks.fetchHistory).not.toHaveBeenCalled();
    expect(mocks.activateTab).not.toHaveBeenCalled();
  });
});

describe('logout', () => {
  test('a later login resumes the server last-active even though logout cleared the pointer', async () => {
    // Logout cleared the browser's pointer; the server kept the operator's own key.
    localStorage.removeItem(STORAGE_KEY);

    await loadPage();

    expect(pointer.getPointer()).toBe(K1);
    expect(mocks.fetchHistory).toHaveBeenCalledTimes(1);
    expect(mocks.fetchHistory).toHaveBeenCalledWith(K1);
  });
});
