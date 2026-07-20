// @ts-check
/**
 * Unit tests for the Web Terminal's stale-resume self-correction (terminal.js):
 *   npx vitest run tests/interfaces/web_terminal/terminal-resume.test.mjs
 *
 * routes/websocket.py now emits a `session_info` confirmation on the
 * mode=resume connect path too (previously only mode=new got one — see the
 * RESUME_LIVENESS_TIMEOUT_MS comment in terminal.js). This file covers what
 * terminal.js does with that confirmation when it disagrees with the
 * requested --resume-id: a server that silently starts a fresh PTY on a
 * stale/dead id is now detectable, so localStorage['osprey-pty-session']
 * must be corrected rather than left pointing at the dead id forever. The
 * matching-id (normal resume) and new-session paths are asserted unchanged.
 *
 * Module isolation: term/wsConnection/currentSessionId/autoResumeFailoverId
 * are module-private state with no reset API, so each test gets a fresh
 * module instance via vi.resetModules() + dynamic import (same pattern as
 * api.test.mjs / sessions.test.mjs).
 *
 * xterm.js (Terminal/FitAddon/WebLinksAddon) is loaded as a global in the
 * real app (vendored <script> tags, not a module) and WebSocket is stubbed
 * with a controllable fake rather than happy-dom's real one (same reasoning
 * as api.test.mjs stubbing EventSource: the real thing dials out).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js')} */
let terminal;

const STORAGE_KEY = 'osprey-pty-session';

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
}

class FakeAddon {
  fit() {}
}

/**
 * Minimal fake WebSocket. happy-dom does implement a real WebSocket, but it
 * dials out for real; this captures the most recently constructed instance
 * so a test can drive onopen/onmessage directly instead.
 */
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

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();

  document.body.innerHTML = '<div><div id="terminal-container"></div></div>';
  // happy-dom does not implement document.fonts; initTerminal() awaits its
  // `ready` promise to re-fit after web fonts load.
  // @ts-expect-error -- test stub, not a full FontFaceSet
  document.fonts = { ready: Promise.resolve() };

  vi.stubGlobal('Terminal', FakeTerminal);
  vi.stubGlobal('FitAddon', { FitAddon: FakeAddon });
  vi.stubGlobal('WebLinksAddon', { WebLinksAddon: class {} });
  vi.stubGlobal('WebSocket', FakeWebSocket);
  // xtermPalette() logs a console.error when the CSS custom properties it
  // reads are absent, which they are in this bare happy-dom document — this
  // suite doesn't touch theming and the noise isn't a signal here.
  vi.spyOn(console, 'error').mockImplementation(() => {});

  FakeWebSocket.last = null;
  terminal = await import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js');
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('resume confirmation: stale id self-correction', () => {
  test('matching confirmed id: storage keeps the resumed id, currentSessionId updates', () => {
    localStorage.setItem(STORAGE_KEY, 'requested-id');

    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'session_info', session_id: 'requested-id' });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('requested-id');
    expect(terminal.getCurrentSessionId()).toBe('requested-id');
  });

  test('mismatched confirmed id: storage is cleared, currentSessionId reflects the actually-attached id', () => {
    localStorage.setItem(STORAGE_KEY, 'stale-id');

    terminal.initTerminal('terminal-container');
    openSocket();
    // Server silently started a fresh PTY instead of erroring on the dead id.
    receive({ type: 'session_info', session_id: 'fresh-id' });

    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(terminal.getCurrentSessionId()).toBe('fresh-id');
  });

  test('new-session path (no prior --resume-id) is unaffected: confirmation always persists', () => {
    // No stored session -> initTerminal() takes the mode='new' branch.
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'session_info', session_id: 'brand-new-id' });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('brand-new-id');
    expect(terminal.getCurrentSessionId()).toBe('brand-new-id');
  });

  test('explicit resume (e.g. sessions.js resumeSession), not just auto-resume-on-load, also self-corrects', () => {
    // No stored session, so initTerminal() starts fresh; this exercises a
    // resume triggered independently of the page-load auto-resume path.
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'session_info', session_id: 'first-id' });
    terminal.stopTerminal();

    terminal.startTerminal('stale-explicit-id', 'resume');
    openSocket();
    receive({ type: 'session_info', session_id: 'fresh-explicit-id' });

    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(terminal.getCurrentSessionId()).toBe('fresh-explicit-id');
  });
});
