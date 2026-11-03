// @ts-check
/**
 * The xterm.js and WebSocket stand-ins the terminal.js suites drive.
 *
 * The resume, hand-off, copy and session-pointer boot suites each carried the
 * same fakes; this module centralizes them. xterm.js (Terminal/FitAddon/
 * WebLinksAddon/ClipboardAddon) is loaded as a global in the real app
 * (vendored <script> tags, not a module), and WebSocket is replaced with a
 * controllable fake rather than happy-dom's real one, which dials out.
 *
 * `stubTerminalGlobals` installs all of them and forgets the previous test's
 * instances; `stubSessionProbe` answers the `/api/session` probe api.js makes
 * whenever a channel closes (to tell an expired session from a dropped
 * connection) with the signed-in status, and fails loudly on any other URL —
 * a dependency the suite has not declared. `vi.unstubAllGlobals()` undoes both.
 */

import { vi } from 'vitest';

/** Fake xterm.js Terminal: records its options, addons, input and output handlers. */
export class FakeTerminal {
  /** @param {any} [options] */
  constructor(options) {
    this.cols = 80;
    this.rows = 24;
    this.options = options ?? {};
    /** Everything written to the screen, concatenated. */
    this.written = '';
    /** @type {((data: string) => void)|null} */
    this.dataHandler = null;
    /** @type {((ev: KeyboardEvent) => boolean)|null} */
    this.keyHandler = null;
    /** What `getSelection()` answers; `''` is no selection. */
    this.selection = '';
    /** @type {any[]} */
    this.addons = [];
    FakeTerminal.last = this;
  }
  /** @param {any} addon */
  loadAddon(addon) {
    this.addons.push(addon);
  }
  open() {}
  /** @param {(data: string) => void} fn */
  onData(fn) {
    this.dataHandler = fn;
  }
  onResize() {}
  /** @param {string} text */
  write(text) {
    this.written += text;
  }
  reset() {}
  focus() {}
  /** @param {(ev: KeyboardEvent) => boolean} handler */
  attachCustomKeyEventHandler(handler) {
    this.keyHandler = handler;
  }
  hasSelection() {
    return this.selection !== '';
  }
  getSelection() {
    return this.selection;
  }
}
/** @type {FakeTerminal|null} */
FakeTerminal.last = null;

export class FakeAddon {
  fit() {}
}

/** Fake WebSocket that captures the most recently constructed instance. */
export class FakeWebSocket {
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
    FakeWebSocket.created += 1;
  }
  /** @param {any} data */
  send(data) {
    this.sent.push(data);
  }
  close() {
    this.readyState = FakeWebSocket.CLOSED;
    if (this.onclose) this.onclose({ code: 1000, reason: '' });
  }
}
FakeWebSocket.CONNECTING = 0;
FakeWebSocket.OPEN = 1;
FakeWebSocket.CLOSING = 2;
FakeWebSocket.CLOSED = 3;
/** @type {FakeWebSocket|null} */
FakeWebSocket.last = null;
/** How many have been constructed — a reconnect is a new one. */
FakeWebSocket.created = 0;

/**
 * Install the xterm.js globals and the fake WebSocket, with no instance left over.
 * @param {{ ClipboardAddon?: any, Base64?: any }} [clipboard] The clipboard
 *   addon's constructors; inert classes by default.
 */
export function stubTerminalGlobals(clipboard = {}) {
  vi.stubGlobal('Terminal', FakeTerminal);
  vi.stubGlobal('FitAddon', { FitAddon: FakeAddon });
  vi.stubGlobal('WebLinksAddon', { WebLinksAddon: class {} });
  vi.stubGlobal('ClipboardAddon', {
    ClipboardAddon: clipboard.ClipboardAddon ?? class {},
    Base64: clipboard.Base64 ?? class {},
  });
  vi.stubGlobal('WebSocket', FakeWebSocket);
  FakeTerminal.last = null;
  FakeWebSocket.last = null;
  FakeWebSocket.created = 0;
}

/** Answer api.js's `/api/session` probe as signed in; any other fetch throws. */
export function stubSessionProbe() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      if (url !== '/api/session') throw new Error(`unstubbed fetch: ${url}`);
      return { ok: true, status: 200, json: async () => ({}) };
    })
  );
}

/** Type into the terminal as the operator would. @param {string} data */
export function press(data) {
  const t = /** @type {FakeTerminal} */ (FakeTerminal.last);
  if (!t.dataHandler) throw new Error('onData handler not set');
  t.dataHandler(data);
}

/** Flip the most recently created fake socket to OPEN and fire its onopen. */
export function openSocket() {
  const ws = /** @type {FakeWebSocket} */ (FakeWebSocket.last);
  ws.readyState = FakeWebSocket.OPEN;
  if (!ws.onopen) throw new Error('onopen handler not set');
  ws.onopen();
}

/**
 * Deliver a JSON control message from the "server" on the current socket.
 * @param {any} msg
 */
export function receive(msg) {
  const ws = /** @type {FakeWebSocket} */ (FakeWebSocket.last);
  if (!ws.onmessage) throw new Error('onmessage handler not set');
  ws.onmessage({ data: JSON.stringify(msg) });
}

/**
 * Close the current socket the way a server refusal does.
 * @param {number} code
 * @param {string} [reason]
 */
export function refuse(code, reason = '') {
  const ws = /** @type {FakeWebSocket} */ (FakeWebSocket.last);
  ws.readyState = FakeWebSocket.CLOSED;
  if (!ws.onclose) throw new Error('onclose handler not set');
  ws.onclose({ code, reason });
}
