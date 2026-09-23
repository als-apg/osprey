// @ts-check
/**
 * Unit tests for who writes the tab's session pointer through a page boot:
 *   npx vitest run tests/interfaces/web_terminal/session-pointer-boot.test.mjs
 *
 * The pointer is one slot with two writers — the terminal (terminal.js) and
 * the operator console (chat.js) — and both views read it. The surface that
 * owns the session is the one that names it: in the Expert view that is the
 * id the terminal's server confirms, in the Simple view it is the console's
 * own key, because the terminal never connects there. A boot that writes the
 * slot twice leaves the two views disagreeing for as long as a PTY takes to
 * spawn, and every reader in that window believes the first value.
 *
 * No other suite can hold this: chat.test.mjs replaces terminal.js with a
 * mock, and terminal-resume.test.mjs never mounts the console. This module
 * mounts both real modules into one document, in app.js's boot order
 * (initTerminal, then initChat), and records every pointer write.
 *
 * The xterm, WebSocket, fetch and font stubs are the ones
 * terminal-resume.test.mjs uses; the chat-client.js mock is the one
 * chat.test.mjs uses.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js')} */
let terminal;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat.js')} */
let chat;
/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js')} */
let pointer;

const STORAGE_KEY = 'osprey-pty-session';

/** Every value the pointer was set to, in order. @type {(string|null)[]} */
let writes;

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js', () => ({
  fetchHistory: vi.fn(async () => []),
  sendPrompt: vi.fn(() => ({ abort: () => {}, aborted: false })),
  interrupt: vi.fn(async () => undefined),
  requestHandoff: vi.fn(async () => ({ state: 'simple', session_id: 'k' })),
}));

/** Minimal fake xterm.js Terminal -- just enough surface for initTerminal(). */
class FakeTerminal {
  constructor() {
    this.cols = 80;
    this.rows = 24;
    this.options = {};
  }
  loadAddon() {}
  open() {}
  onData() {}
  onResize() {}
  write() {}
  reset() {}
  focus() {}
  attachCustomKeyEventHandler() {}
}

class FakeAddon {
  fit() {}
}

/** Minimal fake WebSocket that captures the most recently constructed instance. */
class FakeWebSocket {
  /** @param {string} url */
  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.CONNECTING;
    /** @type {any[]} */
    this.sent = [];
    /** @type {((ev?: any) => void)|null} */
    this.onopen = null;
    /** @type {((ev?: any) => void)|null} */
    this.onmessage = null;
    /** @type {((ev?: any) => void)|null} */
    this.onclose = null;
    FakeWebSocket.last = this;
  }
  /** @param {any} data */
  send(data) {
    this.sent.push(data);
  }
  close() {
    this.readyState = FakeWebSocket.CLOSED;
    if (this.onclose) this.onclose({});
  }
}
FakeWebSocket.CONNECTING = 0;
FakeWebSocket.OPEN = 1;
FakeWebSocket.CLOSING = 2;
FakeWebSocket.CLOSED = 3;
/** @type {FakeWebSocket|null} */
FakeWebSocket.last = null;

/** Flip the most recently created fake socket to OPEN and fire its onopen. */
function openSocket() {
  const ws = /** @type {FakeWebSocket} */ (FakeWebSocket.last);
  ws.readyState = FakeWebSocket.OPEN;
  if (!ws.onopen) throw new Error('onopen handler not set');
  ws.onopen();
}

/**
 * Deliver a JSON control message from the "server" on the current socket.
 * @param {any} msg
 */
function receive(msg) {
  const ws = /** @type {FakeWebSocket} */ (FakeWebSocket.last);
  if (!ws.onmessage) throw new Error('onmessage handler not set');
  ws.onmessage({ data: JSON.stringify(msg) });
}

/**
 * Load both surfaces into a fresh document in the given view.
 * @param {'expert'|'simple'} mode
 */
async function loadPage(mode) {
  document.documentElement.setAttribute('data-ui-mode', mode);
  terminal = await import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js');
  chat = await import('../../../src/osprey/interfaces/web_terminal/static/js/chat.js');
  pointer = await import(
    '../../../src/osprey/interfaces/web_terminal/static/js/session-pointer.js'
  );
  writes = [];
  pointer.subscribe((key) => writes.push(key));
}

beforeEach(() => {
  vi.resetModules();
  localStorage.clear();

  document.body.innerHTML =
    '<div><div id="terminal-container"></div></div><div id="operator-container"></div>';
  // @ts-expect-error -- test stub, not a full FontFaceSet
  document.fonts = { ready: Promise.resolve() };

  vi.stubGlobal('Terminal', FakeTerminal);
  vi.stubGlobal('FitAddon', { FitAddon: FakeAddon });
  vi.stubGlobal('WebLinksAddon', { WebLinksAddon: class {} });
  vi.stubGlobal('ClipboardAddon', { ClipboardAddon: class {}, Base64: class {} });
  vi.stubGlobal('WebSocket', FakeWebSocket);
  vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
    if (url !== '/api/session') throw new Error(`unstubbed fetch: ${url}`);
    return { ok: true, status: 200, json: async () => ({}) };
  }));
  vi.spyOn(console, 'error').mockImplementation(() => {});

  FakeWebSocket.last = null;
});

afterEach(() => {
  document.documentElement.removeAttribute('data-ui-mode');
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

test('an Expert boot with nothing stored writes the pointer once, with the id the server confirmed', async () => {
  await loadPage('expert');

  terminal.initTerminal('terminal-container');
  chat.initChat('operator-container');

  // No surface answers for a session that does not exist yet.
  expect(writes).toEqual([]);
  expect(pointer.getPointer()).toBeNull();
  expect(terminal.getCurrentSessionId()).toBeNull();

  openSocket();
  receive({ type: 'session_info', session_id: 'server-minted-id' });

  expect(writes).toEqual(['server-minted-id']);
  expect(pointer.getPointer()).toBe('server-minted-id');
});

test('an Expert boot on a kept-warm key writes nothing', async () => {
  localStorage.setItem(STORAGE_KEY, 'kept-warm-id');
  await loadPage('expert');

  terminal.initTerminal('terminal-container');
  expect(/** @type {FakeWebSocket} */ (FakeWebSocket.last).url).toContain(
    'session_id=kept-warm-id&mode=resume'
  );

  chat.initChat('operator-container');
  expect(writes).toEqual([]);

  openSocket();
  expect(writes).toEqual([]);

  receive({ type: 'session_info', session_id: 'kept-warm-id' });
  expect(writes).toEqual([]);
  expect(pointer.getPointer()).toBe('kept-warm-id');
});

test('a Simple boot with nothing stored writes the pointer once, with the key the console minted', async () => {
  await loadPage('simple');

  terminal.initTerminal('terminal-container');
  // The terminal does not connect in the Simple view.
  expect(FakeWebSocket.last).toBeNull();

  chat.initChat('operator-container');

  expect(writes).toHaveLength(1);
  const [minted] = writes;
  expect(minted).toMatch(/^[0-9a-f-]{36}$/);
  // The console's write is the only thing the terminal has to fall back to here.
  expect(terminal.getCurrentSessionId()).toBe(minted);
});
