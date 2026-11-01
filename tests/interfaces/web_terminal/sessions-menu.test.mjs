// @ts-check
/**
 * Unit tests for the session picker in the chat (sessions.js):
 *   npx vitest run tests/interfaces/web_terminal/sessions-menu.test.mjs
 *
 * In chat the console is the surface that runs the session, and no
 * terminal socket is open. Picking a session there must move the session
 * pointer, which the chat subscribes to and replays, and must not start a
 * terminal, because a second surface on the session's agent would take the
 * conversation away from the chat. The terminal surface keeps its resume.
 *
 * The keyboard tests cover the picker's focus: the list sits at the end of the
 * body, so focus must move into it on open, and Escape must return it.
 *
 * The busy tests cover the chat's `busy` event: while a turn streams, a pick
 * would cut it off, so the picker is withheld in the chat only.
 *
 * terminal.js is xterm glue, so it is replaced with doubles; the pointer is the
 * real session-pointer.js, which is what the chat and the picker share.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const terminal = {
  startTerminal: vi.fn(),
  stopTerminal: vi.fn(),
  restartTerminal: vi.fn(),
  getCurrentSessionId: vi.fn(() => /** @type {string | null} */ (null)),
  notifySessionChange: vi.fn(),
  switchSession: vi.fn(/** @type {(id: string) => boolean} */ (() => false)),
  setSessionLabel: vi.fn(),
};

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  startTerminal: (/** @type {any[]} */ ...args) => terminal.startTerminal(...args),
  stopTerminal: () => terminal.stopTerminal(),
  restartTerminal: () => terminal.restartTerminal(),
  getCurrentSessionId: () => terminal.getCurrentSessionId(),
  notifySessionChange: (/** @type {any} */ id) => terminal.notifySessionChange(id),
  switchSession: (/** @type {any} */ id) => terminal.switchSession(id),
  setSessionLabel: (/** @type {any} */ id) => terminal.setSessionLabel(id),
}));

const FIRST = 'aaaaaaaa-1111-2222-3333-444444444444';
const SECOND = 'bbbbbbbb-5555-6666-7777-888888888888';
const NOW_ISO = new Date().toISOString();
const RECORDS = [
  { session_id: FIRST, last_modified: NOW_ISO, message_count: 7, first_message: 'Vacuum pressure' },
  {
    session_id: SECOND,
    last_modified: NOW_ISO,
    message_count: 2,
    first_message: 'Corrector limits',
  },
];

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js')} */
let sessions;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js')} */
let pointer;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js')} */
let bus;

/** Resolve after the queued work the picker's click handler starts. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** @param {{ sessions: typeof RECORDS }} body */
function stubSessions(body) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, statusText: 'OK', json: async () => body }))
  );
}

/** @param {'chat' | 'terminal'} surface */
function setSurface(surface) {
  if (surface === 'terminal')
    document.documentElement.setAttribute('data-agent-surface', 'terminal');
  else document.documentElement.removeAttribute('data-agent-surface');
}

async function openPicker() {
  const btn = /** @type {HTMLElement} */ (document.getElementById('session-picker-btn'));
  btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  await flush();
}

/** @param {string} sessionId */
function pickSession(sessionId) {
  const item = /** @type {HTMLElement} */ (
    document.querySelector(`.session-item[data-session-id="${sessionId}"]`)
  );
  item.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
}

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js')} */
let surface;

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  document.body.innerHTML = '<div id="session-selector"></div>';
  Object.values(terminal).forEach((fn) => fn.mockClear());
  terminal.getCurrentSessionId.mockReturnValue(null);
  sessions = await import(`${JS}/sessions.js`);
  pointer = await import(`${JS}/session-pointer.js`);
  bus = await import(`${JS}/chat-events.js`);
  surface = await import(`${JS}/agent-surface.js`);
  sessions.initSessionSelector('session-selector');
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.documentElement.removeAttribute('data-agent-surface');
});

describe('in chat, picking a session moves the pointer', () => {
  test('the chosen session becomes the tab pointer, and no terminal is started', async () => {
    setSurface('chat');
    stubSessions({ sessions: RECORDS });
    await openPicker();

    pickSession(SECOND);

    expect(pointer.getPointer()).toBe(SECOND);
    expect(terminal.startTerminal).not.toHaveBeenCalled();
    expect(terminal.stopTerminal).not.toHaveBeenCalled();
  });

  test('pointer subscribers hear the pick, which is how the chat rebinds and replays it', async () => {
    setSurface('chat');
    stubSessions({ sessions: RECORDS });
    /** @type {Array<string | null>} */
    const heard = [];
    pointer.subscribe((id) => heard.push(id));
    await openPicker();

    pickSession(SECOND);

    expect(heard).toEqual([SECOND]);
  });
});

describe('on the terminal surface, picking a session still resumes it there', () => {
  test('the terminal is resumed on the chosen session', async () => {
    setSurface('terminal');
    stubSessions({ sessions: RECORDS });
    await openPicker();
    vi.useFakeTimers();
    pickSession(SECOND);
    // resumeSession pauses between stopping the old socket and starting the new one.
    await vi.runAllTimersAsync();

    expect(terminal.stopTerminal).toHaveBeenCalled();
    expect(terminal.startTerminal).toHaveBeenCalledWith(SECOND, 'resume');
  });
});

describe('keyboard: the picker list takes focus and Escape returns it', () => {
  test('opening the picker focuses the current session, or the first one', async () => {
    setSurface('chat');
    terminal.getCurrentSessionId.mockReturnValue(SECOND);
    stubSessions({ sessions: RECORDS });
    await openPicker();

    expect(document.activeElement?.getAttribute('data-session-id')).toBe(SECOND);
  });

  test('with no current session, the first listed session takes focus', async () => {
    setSurface('chat');
    stubSessions({ sessions: RECORDS });
    await openPicker();

    expect(document.activeElement?.getAttribute('data-session-id')).toBe(FIRST);
  });

  test('Escape closes the picker and returns focus to its button', async () => {
    setSurface('chat');
    stubSessions({ sessions: RECORDS });
    await openPicker();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect(document.getElementById('session-dropdown')).toBeNull();
    expect(document.activeElement?.id).toBe('session-picker-btn');
  });
});

describe('the picker is withheld while the chat is busy', () => {
  const pickerButton = () =>
    /** @type {HTMLButtonElement} */ (document.getElementById('session-picker-btn'));

  test('a turn that starts with the list open closes it and disables the button until the turn ends', async () => {
    setSurface('chat');
    stubSessions({ sessions: RECORDS });
    await openPicker();
    expect(document.getElementById('session-dropdown')).not.toBeNull();

    bus.emitChat('busy', { busy: true });

    expect(pickerButton().disabled).toBe(true);
    expect(document.getElementById('session-dropdown')).toBeNull();
    expect(pickerButton().getAttribute('aria-expanded')).toBe('false');

    bus.emitChat('busy', { busy: false });

    expect(pickerButton().disabled).toBe(false);
  });

  test('the terminal keeps its picker: the busy console is hidden there', () => {
    setSurface('terminal');
    bus.emitChat('busy', { busy: true });
    expect(pickerButton().disabled).toBe(false);
  });

  test('opening the terminal frees the picker a busy console had withheld', () => {
    setSurface('chat');
    bus.emitChat('busy', { busy: true });
    expect(pickerButton().disabled).toBe(true);

    surface.showTerminalSurface();

    expect(pickerButton().disabled).toBe(false);
  });

  test('a picker built after the chat went busy starts withheld', async () => {
    setSurface('chat');
    bus.emitChat('busy', { busy: true });
    sessions.initSessionSelector('session-selector');
    expect(pickerButton().disabled).toBe(true);
  });
});

describe("the Agent bar's Earlier conversations menu", () => {
  /** @type {HTMLButtonElement} */
  let opener;

  beforeEach(() => {
    setSurface('chat');
    opener = document.createElement('button');
    opener.className = 'op-session-more';
    document.body.append(opener);
    sessions.attachConversationMenu(opener);
  });

  const menu = () =>
    /** @type {HTMLElement} */ (document.getElementById('agent-conversations-menu'));
  const rows = () => [...menu().querySelectorAll('[role="menuitemradio"]')];

  async function openMenu() {
    opener.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    await flush();
  }

  test('the ⋯ opens a body-level popover listing the sessions, the current one checked', async () => {
    terminal.getCurrentSessionId.mockReturnValue(SECOND);
    stubSessions({ sessions: RECORDS });
    expect(menu().hidden).toBe(true);

    await openMenu();

    expect(menu().hidden).toBe(false);
    expect(menu().parentElement).toBe(document.body);
    expect(menu().classList.contains('osprey-popover')).toBe(true);
    expect(menu().querySelector('.osprey-popover-title')?.textContent).toBe(
      'Earlier conversations'
    );
    expect(rows().map((r) => r.querySelector('.op-conversation-title')?.textContent)).toEqual([
      'Vacuum pressure',
      'Corrector limits',
    ]);
    expect(rows().map((r) => r.getAttribute('aria-checked'))).toEqual(['false', 'true']);
    expect(opener.getAttribute('aria-expanded')).toBe('true');
  });

  test('picking a conversation moves the pointer and closes the menu; no terminal starts', async () => {
    stubSessions({ sessions: RECORDS });
    await openMenu();

    /** @type {HTMLElement} */ (rows()[1]).click();

    expect(pointer.getPointer()).toBe(SECOND);
    expect(menu().hidden).toBe(true);
    expect(terminal.startTerminal).not.toHaveBeenCalled();
  });

  test('a second press on the ⋯ closes the open menu', async () => {
    stubSessions({ sessions: RECORDS });
    await openMenu();
    await openMenu();
    expect(menu().hidden).toBe(true);
  });

  test('with no sessions the menu says so', async () => {
    stubSessions({ sessions: [] });
    await openMenu();
    expect(rows()).toHaveLength(0);
    expect(menu().querySelector('.osprey-list-empty')?.textContent).toBe(
      'No earlier conversations'
    );
  });

  test('a busy chat closes the open menu; reopened, it offers only the terminal', async () => {
    stubSessions({ sessions: RECORDS });
    await openMenu();

    bus.emitChat('busy', { busy: true });

    expect(menu().hidden).toBe(true);
    expect(opener.disabled).toBe(false);
    opener.click();
    await flush();
    expect(menu().hidden).toBe(false);
    expect(menu().querySelectorAll('[data-session-id]')).toHaveLength(0);
    expect(menu().textContent).toBe('Open terminal');
  });

  test('a turn that starts while the list loads leaves only the terminal row', async () => {
    stubSessions({ sessions: RECORDS });
    opener.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    bus.emitChat('busy', { busy: true });
    await flush();
    expect(menu().querySelectorAll('[data-session-id]')).toHaveLength(0);
  });

  test('with no terminal offered a busy chat withholds the ⋯ outright', () => {
    document.documentElement.setAttribute('data-terminal', 'off');
    bus.emitChat('busy', { busy: true });
    expect(opener.disabled).toBe(true);
    bus.emitChat('busy', { busy: false });
    expect(opener.disabled).toBe(false);
    document.documentElement.removeAttribute('data-terminal');
  });
});

describe("the ⋯ menu's way between surfaces", () => {
  /** @type {HTMLButtonElement} */
  let opener;

  const row = () =>
    /** @type {HTMLButtonElement | null} */ (document.querySelector('.op-surface-row'));

  /** Open the menu and wait for it to render. */
  async function open() {
    stubSessions({ sessions: RECORDS });
    opener.click();
    await vi.waitFor(() =>
      expect(document.getElementById('agent-conversations-menu')?.hidden).toBe(false)
    );
  }

  beforeEach(() => {
    opener = document.createElement('button');
    document.body.append(opener);
    sessions.attachConversationMenu(opener);
  });

  afterEach(() => {
    document.documentElement.removeAttribute('data-terminal');
    document.getElementById('agent-conversations-menu')?.remove();
  });

  test('in chat it is the last row, quiet and under a separator, reading "Open terminal"', async () => {
    await open();
    const menu = /** @type {HTMLElement} */ (document.getElementById('agent-conversations-menu'));
    expect(menu.lastElementChild).toBe(row());
    expect(row()?.textContent).toBe('Open terminal');
    expect(row()?.previousElementSibling?.getAttribute('role')).toBe('separator');
  });

  test('picking it opens the terminal and closes the menu', async () => {
    await open();
    row()?.click();
    expect(document.documentElement.getAttribute('data-agent-surface')).toBe('terminal');
    expect(document.getElementById('agent-conversations-menu')?.hidden).toBe(true);
  });

  test('on the terminal it reads "Back to chat", alone, and picking it goes back', async () => {
    setSurface('terminal');
    const header = document.createElement('button');
    document.body.append(header);
    sessions.attachConversationMenu(header);
    header.click();
    await vi.waitFor(() => expect(row()?.textContent).toBe('Back to chat'));
    const menu = /** @type {HTMLElement} */ (document.getElementById('agent-conversations-menu'));
    expect(menu.children).toHaveLength(1);

    row()?.click();

    expect(document.documentElement.hasAttribute('data-agent-surface')).toBe(false);
  });

  test('it is absent under web.terminal: false', async () => {
    document.documentElement.setAttribute('data-terminal', 'off');
    await open();
    expect(row()).toBeNull();
    expect(document.querySelector('[role="separator"]')).toBeNull();
  });
});
