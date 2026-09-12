// @ts-check
/**
 * Unit tests for the Web Terminal's side of the session hand-off (terminal.js):
 *   npx vitest run tests/interfaces/web_terminal/terminal-handoff.test.mjs
 *
 * Both views are windows onto one session, and only one of them may run its
 * agent at a time. Taking it over is therefore a negotiation rather than a
 * connect: the server answers `handoff_pending` while the outgoing agent
 * finishes its turn (never ended on a clock), `session_info` once this
 * terminal holds it, or a refusal close code when it cannot have it at all.
 * This file covers what the operator sees for each of those, and the two
 * places the Simple view changes what the terminal may do on its own.
 *
 * Module isolation and the xterm/WebSocket fakes follow terminal-resume.test.mjs
 * — module-private state has no reset API, so each test gets a fresh instance
 * via vi.resetModules() + dynamic import.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { HANDOFF_RESTART_BUDGET_MS } from '../../../src/osprey/interfaces/web_terminal/static/js/terminal-handoff.js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js')} */
let terminal;

const STORAGE_KEY = 'osprey-pty-session';

/** Close codes the server refuses a hand-off with (api.js). */
const WS_CLOSE_SESSION_ATTACHED = 4409;
const WS_CLOSE_OUTGOING_RUNNING = 4503;

/** Minimal fake xterm.js Terminal -- just enough surface for initTerminal(). */
class FakeTerminal {
  constructor() {
    this.cols = 80;
    this.rows = 24;
    this.options = {};
    /** Everything written to the screen, concatenated. */
    this.written = '';
    /** @type {((data: string) => void)|null} */
    this.dataHandler = null;
    FakeTerminal.last = this;
  }
  loadAddon() {}
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
  attachCustomKeyEventHandler() {}
}
/** @type {FakeTerminal|null} */
FakeTerminal.last = null;

class FakeAddon {
  fit() {}
}

/** Minimal fake WebSocket; happy-dom's own implementation dials out for real. */
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
 * Close the current socket the way a server refusal does.
 * @param {number} code
 * @param {string} [reason]
 */
function refuse(code, reason = '') {
  const ws = /** @type {FakeWebSocket} */ (FakeWebSocket.last);
  ws.readyState = FakeWebSocket.CLOSED;
  if (!ws.onclose) throw new Error('onclose handler not set');
  ws.onclose({ code, reason });
}

/** The overlay's single line of copy, or null when no overlay is up. */
function overlayText() {
  const message = document.querySelector('.terminal-handoff-message');
  return message ? message.textContent : null;
}

/** The overlay's action button, or null when no overlay is up. */
function overlayAction() {
  return /** @type {HTMLButtonElement|null} */ (
    document.querySelector('.terminal-handoff-action')
  );
}

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();

  // The real card: the overlay mounts on `.terminal-body`, which is also the
  // element initTerminal() observes for resizes.
  document.body.innerHTML =
    '<div class="terminal-card"><div class="terminal-body" id="terminal-body">' +
    '<div id="terminal-container"></div></div></div>';
  // happy-dom does not implement document.fonts; initTerminal() awaits its
  // `ready` promise to re-fit after web fonts load.
  // @ts-expect-error -- test stub, not a full FontFaceSet
  document.fonts = { ready: Promise.resolve() };

  vi.stubGlobal('Terminal', FakeTerminal);
  vi.stubGlobal('FitAddon', { FitAddon: FakeAddon });
  vi.stubGlobal('WebLinksAddon', { WebLinksAddon: class {} });
  vi.stubGlobal('ClipboardAddon', { ClipboardAddon: class {}, Base64: class {} });
  vi.stubGlobal('WebSocket', FakeWebSocket);
  // api.js probes `/api/session` whenever a channel closes, to tell an expired
  // session from a dropped connection. Nothing serves this environment, so the
  // probe is answered here with the signed-in status every test below assumes.
  // Any other URL is a dependency this file has not declared, and fails loudly.
  vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
    if (url !== '/api/session') throw new Error(`unstubbed fetch: ${url}`);
    return { ok: true, status: 200, json: async () => ({}) };
  }));
  // xtermPalette() logs a console.error when the CSS custom properties it
  // reads are absent, which they are in this bare happy-dom document.
  vi.spyOn(console, 'error').mockImplementation(() => {});

  FakeWebSocket.last = null;
  FakeWebSocket.created = 0;
  terminal = await import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js');
});

afterEach(() => {
  document.documentElement.removeAttribute('data-ui-mode');
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('the Simple view starts nothing', () => {
  // One live agent per session. In Simple view the chat is holding it and the
  // terminal is off screen, so a terminal that connected here would take the
  // conversation away from the view the operator is looking at.

  test('initTerminal does not open a connection', () => {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    document.documentElement.setAttribute('data-ui-mode', 'simple');

    terminal.initTerminal('terminal-container');

    expect(FakeWebSocket.created).toBe(0);
  });

  test('the terminal itself is still built, so the flip has something to attach', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');

    terminal.initTerminal('terminal-container');

    expect(terminal.getTerminalInstance()).not.toBeNull();
    expect(terminal.getTerminalDimensions()).toEqual({ cols: 80, rows: 24 });
  });

  test('Expert view still connects on load', () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');

    terminal.initTerminal('terminal-container');

    expect(FakeWebSocket.created).toBe(1);
  });
});

describe('startExpert: taking the session over', () => {
  test('resumes the key the tab is pointed at', () => {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    terminal.initTerminal('terminal-container');

    terminal.startExpert();

    const url = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(url).toContain('session_id=shared-key');
    expect(url).toContain('mode=resume');
    expect(url).not.toContain('interrupt=');
  });

  test('starts an ordinary session when no key is stored', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    terminal.initTerminal('terminal-container');

    terminal.startExpert();

    const url = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(url).not.toContain('session_id=');
    expect(url).not.toContain('mode=resume');
  });

  test('an early exit does not drop the shared key', () => {
    // The page-load auto-resume treats an early exit as a dead id and forgets
    // it. A flip must not: the key is the chat's too, and an exit here is a
    // hand-off that did not complete.
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    terminal.initTerminal('terminal-container');
    terminal.startExpert();
    openSocket();
    const socketsBefore = FakeWebSocket.created;

    receive({ type: 'exit', code: 1 });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('shared-key');
    expect(FakeWebSocket.created).toBe(socketsBefore);
  });
});

describe('startExpert settles when the acquire has an answer', () => {
  // app.js serialises the flip on this promise: a Simple → Expert → Simple
  // round trip must not ask for the session again until the server has seen
  // the Expert channel resolve, one way or the other.

  /** Reach the flip's starting point: Simple view, a key stored, no socket. */
  function flipToExpert() {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    terminal.initTerminal('terminal-container');
    return terminal.startExpert();
  }

  test('a pending hand-off does not settle it', async () => {
    const settled = flipToExpert();
    openSocket();
    receive({ type: 'handoff_pending', busy: true });

    const race = await Promise.race([
      settled.then(() => 'settled'),
      Promise.resolve('pending'),
    ]);
    expect(race).toBe('pending');

    // Leave nothing dangling for the next test.
    receive({ type: 'session_info', session_id: 'shared-key' });
    await settled;
  });

  test('session_info settles it', async () => {
    const settled = flipToExpert();
    openSocket();
    receive({ type: 'session_info', session_id: 'shared-key' });

    await expect(settled).resolves.toBeUndefined();
  });

  test('a 4409 refusal settles it, with the notice up', async () => {
    const settled = flipToExpert();
    openSocket();
    refuse(WS_CLOSE_SESSION_ATTACHED);

    await expect(settled).resolves.toBeUndefined();
    expect(overlayText()).toBe('This session is in use in another tab or view.');
  });

  test('a 4503 refusal settles it', async () => {
    const settled = flipToExpert();
    openSocket();
    refuse(WS_CLOSE_OUTGOING_RUNNING);

    await expect(settled).resolves.toBeUndefined();
  });

  test('a close with no answer at all settles it', async () => {
    vi.useFakeTimers();
    try {
      const settled = flipToExpert();
      openSocket();
      refuse(1006);

      await expect(settled).resolves.toBeUndefined();
    } finally {
      vi.useRealTimers();
    }
  });

  test('an error frame settles it', async () => {
    const settled = flipToExpert();
    openSocket();
    receive({ type: 'error', message: 'no capacity' });

    await expect(settled).resolves.toBeUndefined();
  });

  test('a second call while a connection exists settles at once', async () => {
    const settled = flipToExpert();
    openSocket();
    receive({ type: 'session_info', session_id: 'shared-key' });
    await settled;

    await expect(terminal.startExpert()).resolves.toBeUndefined();
  });

  test('nothing here ever rejects', async () => {
    // The flip chain does not catch, so a rejection would break the round
    // trip rather than the connection.
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    terminal.initTerminal('terminal-container');

    const outcomes = [];
    for (const answer of [
      () => receive({ type: 'session_info', session_id: 'shared-key' }),
      () => refuse(WS_CLOSE_SESSION_ATTACHED),
      () => receive({ type: 'error', message: 'no capacity' }),
    ]) {
      localStorage.setItem(STORAGE_KEY, 'shared-key');
      const settled = terminal.startExpert();
      openSocket();
      answer();
      outcomes.push(await settled.then(() => 'resolved', () => 'rejected'));
      terminal.stopTerminal();
    }

    expect(outcomes).toEqual(['resolved', 'resolved', 'resolved']);
  });
});

describe('handoff_pending: the transitional state', () => {
  /** Flip to Expert on a stored key and reach the pending state. */
  function pending() {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'handoff_pending', busy: true });
  }

  test('says what is happening and how long it has been', () => {
    pending();

    expect(overlayText()).toBe('Finishing in the other view · 0:00');
  });

  test('the elapsed wait counts up', () => {
    vi.useFakeTimers();
    try {
      pending();
      vi.advanceTimersByTime(65_000);

      expect(overlayText()).toBe('Finishing in the other view · 1:05');
    } finally {
      vi.useRealTimers();
    }
  });

  test('the ticking counter is not re-announced every second', () => {
    pending();

    const elapsed = document.querySelector('.terminal-handoff-elapsed');
    expect(elapsed?.getAttribute('aria-live')).toBe('off');
  });

  test('the way out of an unbounded wait is offered', () => {
    pending();

    const action = overlayAction();
    expect(action).not.toBeNull();
    expect(action?.hidden).toBe(false);
    expect(action?.textContent).toBe('Stop and switch now');
  });

  test('session_info clears it', () => {
    pending();

    receive({ type: 'session_info', session_id: 'shared-key' });

    expect(document.querySelector('.terminal-handoff')).toBeNull();
  });

  test('the elapsed timer stops with the overlay', () => {
    vi.useFakeTimers();
    try {
      pending();
      receive({ type: 'session_info', session_id: 'shared-key' });
      vi.advanceTimersByTime(60_000);

      expect(document.querySelector('.terminal-handoff')).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  test('flipping back to Simple clears it', () => {
    pending();

    terminal.stopTerminal();

    expect(document.querySelector('.terminal-handoff')).toBeNull();
  });

  test('session_switched clears it too, since that is how a switch reports success', () => {
    // Switching onto a key the chat holds is negotiated like any other
    // hand-off, but the server answers the completed switch with
    // `session_switched` rather than `session_info`.
    pending();

    receive({ type: 'session_switched', session_id: 'other-key' });

    expect(document.querySelector('.terminal-handoff')).toBeNull();
  });

  test('an error frame replaces it rather than arriving underneath it', () => {
    pending();

    receive({ type: 'error', message: 'no capacity' });

    expect(document.querySelector('.terminal-handoff')).toBeNull();
    expect(/** @type {FakeTerminal} */ (FakeTerminal.last).written).toContain('no capacity');
  });

  test('it disarms the page-load failover, so the outgoing agent\'s exit is not read as a dead key', () => {
    // This connection IS the page-load auto-resume, so the failover is armed:
    // an early exit would ordinarily mean "that id is gone". A handoff_pending
    // says the opposite — the key is alive in the other view — and the exit
    // that follows is the outgoing agent being handed over, not a dead id.
    pending();
    const socketsBefore = FakeWebSocket.created;

    receive({ type: 'exit', code: 0 });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('shared-key');
    expect(terminal.getCurrentSessionId()).toBe('shared-key');
    expect(FakeWebSocket.created).toBe(socketsBefore);
  });
});

describe('handoff_pending on an idle chat: a restart, not a wait', () => {
  // The server says whether the chat holding the key is mid-turn. When it is
  // not, nothing is being finished anywhere: the session's agent is stopped
  // in the other view and started in this one, which takes about a second.
  // Showing that as a wait with a clock and a way out would be untrue.

  /** Flip to Expert on a stored key; the server reports the chat idle. */
  function idle() {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'handoff_pending', busy: false });
  }

  test('says the agent is restarting here, with no clock and no way out', () => {
    idle();

    expect(overlayText()).toBe('Restarting the agent in this view…');
    expect(document.querySelector('.terminal-handoff-elapsed')).toBeNull();
    expect(overlayAction()?.hidden).toBe(true);
  });

  test('a frame that says nothing about the turn is read as idle', () => {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'handoff_pending' });

    expect(overlayText()).toBe('Restarting the agent in this view…');
  });

  test('a restart that outlasts its budget becomes the wait, clocked from the flip', () => {
    vi.useFakeTimers();
    try {
      idle();
      vi.advanceTimersByTime(HANDOFF_RESTART_BUDGET_MS - 1);
      expect(overlayText()).toBe('Restarting the agent in this view…');

      vi.advanceTimersByTime(1);

      expect(overlayText()).toBe('Finishing in the other view · 0:04');
      expect(overlayAction()?.hidden).toBe(false);
      expect(overlayAction()?.textContent).toBe('Stop and switch now');
    } finally {
      vi.useRealTimers();
    }
  });

  test('session_info in budget clears it, and no wait appears afterwards', () => {
    vi.useFakeTimers();
    try {
      idle();
      receive({ type: 'session_info', session_id: 'shared-key' });
      vi.advanceTimersByTime(HANDOFF_RESTART_BUDGET_MS * 2);

      expect(document.querySelector('.terminal-handoff')).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  test('a busy frame after the wait is up keeps the wait, and so does an idle one', () => {
    // An interrupt's reconnect brings a second frame. The chat it cut short
    // may report idle by then; the operator is already watching a wait, and
    // that state never steps back to a restart.
    vi.useFakeTimers();
    try {
      localStorage.setItem(STORAGE_KEY, 'shared-key');
      terminal.initTerminal('terminal-container');
      openSocket();
      receive({ type: 'handoff_pending', busy: true });
      vi.advanceTimersByTime(3_000);
      /** @type {HTMLButtonElement} */ (overlayAction()).click();
      openSocket();
      receive({ type: 'handoff_pending', busy: false });

      expect(overlayText()).toBe('Finishing in the other view · 0:03');
      expect(overlayAction()?.hidden).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('"Stop and switch now"', () => {
  /** Reach the pending state and press the button. */
  function interrupt() {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'handoff_pending', busy: true });
    /** @type {HTMLButtonElement} */ (overlayAction()).click();
  }

  test('asks for the same session again, with the interrupt', () => {
    interrupt();

    const url = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(url).toContain('session_id=shared-key');
    expect(url).toContain('mode=resume');
    expect(url).toContain('interrupt=1');
  });

  test('the refused wrapper is not reused: the retry is a new socket', () => {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'handoff_pending', busy: true });
    const socketsBefore = FakeWebSocket.created;

    /** @type {HTMLButtonElement} */ (overlayAction()).click();

    expect(FakeWebSocket.created).toBe(socketsBefore + 1);
  });

  test('it cannot be pressed twice while the first attempt is in flight', () => {
    interrupt();

    expect(overlayAction()?.disabled).toBe(true);
  });

  test('the wait the operator is watching carries across the reconnect', () => {
    vi.useFakeTimers();
    try {
      localStorage.setItem(STORAGE_KEY, 'shared-key');
      terminal.initTerminal('terminal-container');
      openSocket();
      receive({ type: 'handoff_pending', busy: true });
      vi.advanceTimersByTime(30_000);
      /** @type {HTMLButtonElement} */ (overlayAction()).click();
      openSocket();
      receive({ type: 'handoff_pending', busy: true });

      // 0:30 and counting, not back to 0:00.
      expect(overlayText()).toBe('Finishing in the other view · 0:30');
      expect(overlayAction()?.disabled).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('stopTerminal reports when the channel is actually gone', () => {
  // The flip to Simple drops this socket and then asks the server for the
  // session over HTTP. Asking while the Expert channel is still attached is
  // refused with the one refusal that has no retry, so the flip waits for the
  // close rather than racing the server's grace window.

  test('resolves once the dropped socket reports its close', async () => {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();

    await expect(terminal.stopTerminal()).resolves.toBeUndefined();
    expect(/** @type {FakeWebSocket} */ (FakeWebSocket.last).readyState).toBe(
      FakeWebSocket.CLOSED,
    );
  });

  test('resolves at once when there is nothing open to drop', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    terminal.initTerminal('terminal-container');

    await expect(terminal.stopTerminal()).resolves.toBeUndefined();
    expect(FakeWebSocket.created).toBe(0);
  });

  test('a socket that never reports a close cannot hang the flip', async () => {
    vi.useFakeTimers();
    try {
      localStorage.setItem(STORAGE_KEY, 'shared-key');
      terminal.initTerminal('terminal-container');
      openSocket();
      // Closing this one is silent: no close event ever arrives.
      /** @type {FakeWebSocket} */ (FakeWebSocket.last).close = () => {};

      const stopped = terminal.stopTerminal();
      let settled = false;
      stopped.then(() => {
        settled = true;
      });

      await vi.advanceTimersByTimeAsync(1000);
      expect(settled).toBe(false);

      await vi.advanceTimersByTimeAsync(1000);
      await expect(stopped).resolves.toBeUndefined();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('a refused connection', () => {
  /**
   * Flip to Expert on a stored key and have the server refuse.
   * @param {number} code
   */
  function refused(code) {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();
    refuse(code, 'session_attached_elsewhere');
  }

  test('4409 says the session is in use elsewhere', () => {
    refused(WS_CLOSE_SESSION_ATTACHED);

    expect(overlayText()).toBe('This session is in use in another tab or view.');
  });

  test('4409 offers nothing to retry, because retrying is not the answer', () => {
    refused(WS_CLOSE_SESSION_ATTACHED);

    expect(overlayAction()?.hidden).toBe(true);
  });

  test('4503 says the previous agent is still shutting down, and offers the retry', () => {
    refused(WS_CLOSE_OUTGOING_RUNNING);

    expect(overlayText()).toBe('The previous agent is still shutting down.');
    const action = overlayAction();
    expect(action?.hidden).toBe(false);
    expect(action?.textContent).toBe('Retry');
  });

  test('the retry builds a new connection and clears the notice', () => {
    refused(WS_CLOSE_OUTGOING_RUNNING);
    const socketsBefore = FakeWebSocket.created;

    /** @type {HTMLButtonElement} */ (overlayAction()).click();

    expect(FakeWebSocket.created).toBe(socketsBefore + 1);
    expect(document.querySelector('.terminal-handoff')).toBeNull();
    expect(/** @type {FakeWebSocket} */ (FakeWebSocket.last).url).toContain(
      'session_id=shared-key',
    );
  });

  test('the spent wrapper is released, so a later start is not swallowed', () => {
    // createWebSocket stops reconnecting after a refusal. Holding that dead
    // wrapper would make every later startTerminal() return early on the
    // "already connected" guard.
    refused(WS_CLOSE_SESSION_ATTACHED);
    const socketsBefore = FakeWebSocket.created;

    terminal.startTerminal();

    expect(FakeWebSocket.created).toBe(socketsBefore + 1);
  });

  test('the session key survives: the refusal is proof it is alive elsewhere', () => {
    refused(WS_CLOSE_SESSION_ATTACHED);

    expect(localStorage.getItem(STORAGE_KEY)).toBe('shared-key');
    expect(terminal.getCurrentSessionId()).toBe('shared-key');
  });

  test('a refusal on an abandoned socket does not paint over the live one', () => {
    // "Stop and switch now" leaves the first socket behind. Its close can
    // arrive after the reconnect is already waiting on its own hand-off, and
    // it says nothing about the connection now in place.
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    const abandoned = /** @type {FakeWebSocket} */ (FakeWebSocket.last);
    openSocket();
    receive({ type: 'handoff_pending', busy: true });
    /** @type {HTMLButtonElement} */ (overlayAction()).click();
    openSocket();
    receive({ type: 'handoff_pending', busy: true });

    // The abandoned socket's refusal lands late.
    /** @type {(ev: any) => void} */ (abandoned.onclose)({
      code: WS_CLOSE_SESSION_ATTACHED,
      reason: '',
    });

    expect(overlayText()).toBe('Finishing in the other view · 0:00');
  });

  test('a refusal after a pending wait replaces the transitional state', () => {
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'handoff_pending', busy: true });

    refuse(WS_CLOSE_OUTGOING_RUNNING);

    expect(overlayText()).toBe('The previous agent is still shutting down.');
  });

  test('an ordinary close is not a refusal and shows nothing', () => {
    // Fake timers so the backoff reconnect this close schedules — the whole
    // point of it not being a refusal — is discarded with them.
    vi.useFakeTimers();
    try {
      localStorage.setItem(STORAGE_KEY, 'shared-key');
      terminal.initTerminal('terminal-container');
      openSocket();

      refuse(1006);

      expect(document.querySelector('.terminal-handoff')).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });
});
