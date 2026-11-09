// @ts-check
/**
 * Unit tests for the Agent tile's header session switcher (session-switcher.js):
 *   npx vitest run tests/interfaces/web_terminal/session-switcher.test.mjs
 *
 * The label on both surfaces shows the active session's title; the
 * shared popover lists the sessions, "+ New session", and the way to the other
 * surface. Picks go through the real session model's guarded `switchTo`, so a
 * pick while the chat streams asks first and the active row is a no-op. The
 * old controls — the terminal picker and the ⋯ "Earlier conversations" menu —
 * are gone.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';

const mocks = vi.hoisted(() => ({
  streaming: false,
  terminal: false,
  stopTurn: vi.fn(),
  newConversation: vi.fn(),
  startNewSession: vi.fn(async () => {}),
  resumeSession: vi.fn(),
  showConfirm: vi.fn(),
  showChatSurface: vi.fn(),
  showTerminalSurface: vi.fn(),
  /** @type {((data: any) => void)[]} */
  hubHandlers: [],
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat.js', () => ({
  isStreaming: () => mocks.streaming,
  stopTurn: mocks.stopTurn,
  newConversation: mocks.newConversation,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js', () => ({
  startNewSession: mocks.startNewSession,
  resumeSession: mocks.resumeSession,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => ({
  showConfirm: mocks.showConfirm,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js', () => ({
  isTerminalSurface: () => mocks.terminal,
  showChatSurface: mocks.showChatSurface,
  showTerminalSurface: mocks.showTerminalSurface,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js', () => ({
  onChat: () => () => {},
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', async (orig) => ({
  ...(await orig()),
  createEventSource: (/** @type {string} */ _url, /** @type {any} */ { onMessage }) => {
    mocks.hubHandlers.push(onMessage);
    return { stop() {} };
  },
}));

const STATIC = resolve(process.cwd(), 'src/osprey/interfaces/web_terminal/static');
/** @param {string} rel */
const source = (rel) => readFileSync(resolve(STATIC, rel), 'utf8');

const K1 = '11111111-1111-4111-8111-111111111111';
const KNEW = '99999999-9999-4999-8999-999999999999';
const T1 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const T2 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const MINTED = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js')} */
let model;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js')} */
let pointer;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-switcher.js')} */
let switcher;
/** @type {{url: string, method: string}[]} */
let calls;

const flush = async () => {
  for (let i = 0; i < 10; i++) await Promise.resolve();
  await vi.advanceTimersByTimeAsync(0);
};

beforeEach(async () => {
  vi.resetModules();
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
  vi.setSystemTime(new Date('2026-11-04T12:00:00Z'));
  localStorage.clear();
  document.body.replaceChildren();
  document.documentElement.removeAttribute('data-terminal');
  mocks.streaming = false;
  mocks.terminal = false;
  mocks.hubHandlers = [];
  for (const fn of [
    mocks.stopTurn,
    mocks.newConversation,
    mocks.startNewSession,
    mocks.resumeSession,
    mocks.showConfirm,
    mocks.showChatSurface,
    mocks.showTerminalSurface,
  ])
    fn.mockReset();
  mocks.resumeSession.mockImplementation(async (key) => pointer.setPointer(key));
  mocks.newConversation.mockImplementation(() => pointer.setPointer(KNEW));
  calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url, /** @type {any} */ init = {}) => {
      calls.push({ url, method: init.method ?? 'GET' });
      const parsed = new URL(url, 'http://hub');
      let body = {};
      if (parsed.pathname === '/api/sessions') {
        body = {
          sessions: [
            {
              transcript_id: T1,
              session_key: K1,
              current_for_key: true,
              title: 'Tune the orbit',
              state: 'live',
              message_count: 12,
              last_ts: '2026-11-04T11:55:00Z',
              last_artifact_id: null,
            },
            {
              transcript_id: T2,
              session_key: null,
              current_for_key: false,
              title: 'Injection check',
              state: 'idle',
              message_count: 1,
              last_ts: '2026-11-03T12:00:00Z',
              last_artifact_id: null,
            },
          ],
          more: false,
          next_cursor: null,
          record: 'complete',
        };
      } else if (parsed.pathname.endsWith('/enter')) {
        body = { session_key: MINTED };
      }
      return new Response(JSON.stringify(body), { status: 200 });
    })
  );
  pointer =
    await import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js');
  model = await import('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js');
  switcher =
    await import('../../../src/osprey/interfaces/web_terminal/static/js/session-switcher.js');
  pointer.setPointer(K1);
  await model.initSessionModel();
});

afterEach(() => {
  switcher.resetSessionSwitcher();
  model.resetSessionModel();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

/** @param {string} cls @param {string} [id] */
function mountLabel(cls, id) {
  const label = document.createElement('span');
  label.className = cls;
  if (id) label.id = id;
  label.textContent = 'Agent';
  document.body.append(label);
  switcher.attachSessionSwitcher(label);
  return label;
}

function menu() {
  return /** @type {HTMLElement} */ (document.getElementById('session-switcher-menu'));
}

/** @param {HTMLElement} label */
function open(label) {
  label.click();
  return menu();
}

/** @param {HTMLElement} pop */
const sessionRows = (pop) =>
  /** @type {HTMLElement[]} */ ([...pop.querySelectorAll('.session-switcher-row')]);

describe('the header label', () => {
  test('shows the active session title with no caret on both surfaces', () => {
    const chat = mountLabel('op-session-label');
    const term = mountLabel('terminal-label', 'terminal-label');
    for (const label of [chat, term]) {
      expect(label.querySelector('.session-switcher-title')?.textContent).toBe('Tune the orbit');
      expect(label.querySelector('.session-switcher-caret')).toBeNull();
      expect(label.getAttribute('aria-haspopup')).toBe('menu');
      expect(label.getAttribute('role')).toBe('button');
    }
  });

  test('follows session_named', () => {
    const label = mountLabel('op-session-label');
    for (const fn of mocks.hubHandlers)
      fn({ type: 'session_named', transcript_id: T1, title: 'Orbit feedback', naming: 'done' });
    expect(label.querySelector('.session-switcher-title')?.textContent).toBe('Orbit feedback');
  });

  test('follows a rename landing in the listing', async () => {
    const label = mountLabel('op-session-label');
    const fetchMock = /** @type {any} */ (globalThis.fetch);
    const base = fetchMock.getMockImplementation();
    fetchMock.mockImplementation(async (/** @type {string} */ url, /** @type {any} */ init) => {
      const res = await base(url, init);
      if (!url.includes('/api/sessions?') && !url.endsWith('/api/sessions')) return res;
      const body = await res.json();
      body.sessions[0].title = 'Renamed by hand';
      return new Response(JSON.stringify(body), { status: 200 });
    });
    model.refresh();
    await vi.advanceTimersByTimeAsync(2000);
    await flush();
    expect(label.querySelector('.session-switcher-title')?.textContent).toBe('Renamed by hand');
  });

  test('reads "New session" for a key the listing has not caught up with', async () => {
    const label = mountLabel('op-session-label');
    pointer.setPointer(KNEW);
    expect(label.querySelector('.session-switcher-title')?.textContent).toBe('New session');
  });

  test('a terminal label owned by the switcher keeps its title across setSessionLabel', async () => {
    const terminal =
      await import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js');
    const label = mountLabel('terminal-label', 'terminal-label');
    terminal.setSessionLabel(K1);
    expect(label.querySelector('.session-switcher-title')?.textContent).toBe('Tune the orbit');
    expect(label.dataset.sessionId).toBe(K1);
  });
});

describe('the popover', () => {
  test('lists the sessions with their facts, the active one checked, then + New session', () => {
    const pop = open(mountLabel('op-session-label'));
    expect(pop.hidden).toBe(false);
    const items = sessionRows(pop);
    expect(items.map((r) => r.querySelector('.session-switcher-row-title')?.textContent)).toEqual([
      'Tune the orbit',
      'Injection check',
    ]);
    expect(items.map((r) => r.getAttribute('aria-checked'))).toEqual(['true', 'false']);
    expect(items[0].querySelector('.osprey-popover-aside')?.textContent).toBe(
      'live · 5m ago · 12 msgs'
    );
    expect(items[1].querySelector('.osprey-popover-aside')?.textContent).toBe('1d ago · 1 msg');
    const add = /** @type {HTMLElement} */ (pop.querySelector('.session-switcher-new'));
    expect(add.textContent).toBe('+ New session');
    // "+ New session" follows the rows.
    expect(items[1].compareDocumentPosition(add) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  test('one shared popover for both labels', () => {
    const chat = mountLabel('op-session-label');
    const term = mountLabel('terminal-label', 'terminal-label');
    open(chat);
    open(term);
    expect(document.querySelectorAll('.session-switcher-menu')).toHaveLength(1);
  });

  test('footer: Open terminal in chat, Back to chat on the terminal', () => {
    const label = mountLabel('op-session-label');
    let pop = open(label);
    const footer = /** @type {HTMLElement} */ (pop.querySelector('.session-switcher-surface'));
    expect(footer.textContent).toBe('Open terminal');
    expect(footer.previousElementSibling?.classList.contains('osprey-popover-sep')).toBe(true);
    footer.click();
    expect(mocks.showTerminalSurface).toHaveBeenCalledTimes(1);
    expect(pop.hidden).toBe(true);

    mocks.terminal = true;
    pop = open(label);
    const back = /** @type {HTMLElement} */ (pop.querySelector('.session-switcher-surface'));
    expect(back.textContent).toBe('Back to chat');
    back.click();
    expect(mocks.showChatSurface).toHaveBeenCalledTimes(1);
  });

  test('no footer where no terminal is offered', () => {
    document.documentElement.setAttribute('data-terminal', 'off');
    const pop = open(mountLabel('op-session-label'));
    expect(pop.querySelector('.session-switcher-surface')).toBeNull();
    expect(pop.querySelector('.osprey-popover-sep')).toBeNull();
  });
});

describe('keyboard and inert titles', () => {
  test('Enter on the label opens the popover with focus on a row; Escape closes it and returns focus to the label', () => {
    const label = mountLabel('op-session-label');
    label.focus();
    label.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    const pop = menu();
    expect(pop.hidden).toBe(false);
    expect(pop.contains(document.activeElement)).toBe(true);
    /** @type {HTMLElement} */ (document.activeElement).dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })
    );
    expect(pop.hidden).toBe(true);
    expect(document.activeElement).toBe(label);
  });

  test('a session title carrying markup renders as text, never as elements', () => {
    const label = mountLabel('op-session-label');
    const hostile = '<img src=x onerror="alert(1)">';
    for (const fn of mocks.hubHandlers)
      fn({ type: 'session_named', transcript_id: T1, title: hostile, naming: 'done' });
    expect(label.querySelector('img')).toBeNull();
    expect(label.querySelector('.session-switcher-title')?.textContent).toBe(hostile);
    const pop = open(label);
    expect(pop.querySelector('img')).toBeNull();
    expect(sessionRows(pop)[0].querySelector('.session-switcher-row-title')?.textContent).toBe(
      hostile
    );
  });
});

describe('picks', () => {
  test('a pick enters the session through switchTo', async () => {
    const label = mountLabel('op-session-label');
    const pop = open(label);
    sessionRows(pop)[1].click();
    await flush();
    expect(pop.hidden).toBe(true);
    expect(calls.some((c) => c.method === 'POST' && c.url.includes(`/${T2}/enter`))).toBe(true);
    expect(mocks.resumeSession).toHaveBeenCalledWith(MINTED);
  });

  test('the active row is a no-op', async () => {
    const pop = open(mountLabel('op-session-label'));
    sessionRows(pop)[0].click();
    await flush();
    expect(mocks.resumeSession).not.toHaveBeenCalled();
    expect(mocks.showConfirm).not.toHaveBeenCalled();
    expect(calls.some((c) => c.method === 'POST')).toBe(false);
  });

  test('a pick while the chat streams asks first, and a cancel stays put', async () => {
    mocks.streaming = true;
    mocks.showConfirm.mockImplementation((spec) => spec.onDismiss());
    const pop = open(mountLabel('op-session-label'));
    sessionRows(pop)[1].click();
    await flush();
    expect(mocks.showConfirm).toHaveBeenCalledTimes(1);
    expect(mocks.showConfirm.mock.calls[0][0].confirmLabel).toBe('Stop & switch');
    expect(mocks.resumeSession).not.toHaveBeenCalled();
    expect(pointer.getPointer()).toBe(K1);
  });

  test('+ New session starts a new conversation', async () => {
    const pop = open(mountLabel('op-session-label'));
    /** @type {HTMLElement} */ (pop.querySelector('.session-switcher-new')).click();
    await flush();
    expect(mocks.newConversation).toHaveBeenCalledTimes(1);
    expect(pointer.getPointer()).toBe(KNEW);
  });

  test('+ New session while the chat streams asks first', async () => {
    mocks.streaming = true;
    mocks.showConfirm.mockImplementation((spec) => spec.onDismiss());
    const pop = open(mountLabel('op-session-label'));
    /** @type {HTMLElement} */ (pop.querySelector('.session-switcher-new')).click();
    await flush();
    expect(mocks.showConfirm).toHaveBeenCalledTimes(1);
    expect(mocks.newConversation).not.toHaveBeenCalled();
  });

  test('+ New session on the terminal starts a terminal session', async () => {
    mocks.terminal = true;
    const pop = open(mountLabel('terminal-label', 'terminal-label'));
    /** @type {HTMLElement} */ (pop.querySelector('.session-switcher-new')).click();
    await flush();
    expect(mocks.startNewSession).toHaveBeenCalledTimes(1);
  });
});

describe('the old controls are gone', () => {
  test('sessions.js exports no picker or ⋯ menu', () => {
    const js = source('js/sessions.js');
    expect(js).not.toMatch(/initSessionSelector|attachConversationMenu|Earlier conversations/);
    expect(js).toMatch(/export async function resumeSession/);
    expect(js).toMatch(/export async function startNewSession/);
  });

  test('app.js and chat.js no longer wire them', () => {
    expect(source('js/app.js')).not.toMatch(/initSessionSelector/);
    const chat = source('js/chat.js');
    expect(chat).not.toMatch(/attachConversationMenu|moreBtn/);
    expect(chat).toMatch(/attachSessionSwitcher\(label\)/);
    expect(source('js/app.js')).toMatch(
      /attachSessionSwitcher\(document\.getElementById\('terminal-label'\)\)/
    );
  });

  test('the markup has no picker container and no ⋯', () => {
    const html = source('index.html');
    expect(html).not.toMatch(/id="session-selector"/);
    expect(html).not.toMatch(/terminal-more-btn/);
    expect(html).not.toMatch(/op-session-more/);
    expect(html).toMatch(/id="terminal-label"/);
  });

  test('the chat console builds no ⋯ button', async () => {
    const { buildConsole } =
      await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-console.js');
    const handles = /** @type {any} */ (buildConsole());
    expect(handles.moreBtn).toBeUndefined();
    expect(handles.bar.querySelector('.op-session-more')).toBeNull();
    expect(handles.label.classList.contains('op-session-label')).toBe(true);
  });
});
