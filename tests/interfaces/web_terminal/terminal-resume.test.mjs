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

import {
  FakeTerminal,
  FakeWebSocket,
  openSocket,
  press,
  receive,
  stubSessionProbe,
  stubTerminalGlobals,
} from './_terminal-fake.mjs';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js')} */
let terminal;

const STORAGE_KEY = 'osprey-pty-session';

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();

  document.body.innerHTML = '<div><div id="terminal-container"></div></div>';
  // happy-dom does not implement document.fonts; initTerminal() awaits its
  // `ready` promise to re-fit after web fonts load.
  // @ts-expect-error -- test stub, not a full FontFaceSet
  document.fonts = { ready: Promise.resolve() };

  stubTerminalGlobals();
  stubSessionProbe();
  // xtermPalette() logs a console.error when the CSS custom properties it
  // reads are absent, which they are in this bare happy-dom document — this
  // suite doesn't touch theming and the noise isn't a signal here.
  vi.spyOn(console, 'error').mockImplementation(() => {});

  terminal = await import('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js');
});

afterEach(() => {
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  document.documentElement.removeAttribute('data-ui-mode');
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('per-user pointer scope', () => {
  // On a multi-user mount every `/u/<user>/` shares one origin, so the bare
  // key is a slot any user may have written. The server stamps the user
  // on <html>; auto-resume must read only that user's key — a stale id in
  // the shared slot would attach this terminal to a PTY that is not theirs.
  // (The write and clear halves are pinned in session-pointer.test.mjs.)
  const SCOPE_ATTR = 'data-osprey-storage-scope';

  test("a bare id left by another user does not become bob's auto-resume", () => {
    localStorage.setItem(STORAGE_KEY, 'someone-elses-session');
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

    terminal.initTerminal('terminal-container');

    const url = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(url).not.toContain('mode=resume');
    expect(url).not.toContain('session_id=');
  });

  test("bob's own scoped id is what auto-resume asks for", () => {
    localStorage.setItem(STORAGE_KEY, 'someone-elses-session');
    localStorage.setItem(`${STORAGE_KEY}--bob`, 'bobs-session');
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');

    terminal.initTerminal('terminal-container');

    const url = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(url).toContain('mode=resume');
    expect(url).toContain('session_id=bobs-session');
  });
});

describe('the session key outlives the connection', () => {
  // A view flip tears this terminal's socket down so the other surface can
  // take the session over. Both halves of the pointer have to survive that:
  // the stored key, so a reload still resumes it, and the answer to "which
  // session is this card on?", which the chat asks while nothing is attached.

  test('stopTerminal leaves the stored key and the reported id in place', () => {
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'session_info', session_id: 'session-k' });

    terminal.stopTerminal();

    expect(localStorage.getItem(STORAGE_KEY)).toBe('session-k');
    expect(terminal.getCurrentSessionId()).toBe('session-k');
  });

  test('the stored key answers before anything has connected', () => {
    // Page load with a kept-warm session: the chat can ask for the key before
    // the terminal's own confirmation has arrived.
    localStorage.setItem(STORAGE_KEY, 'session-k');

    terminal.initTerminal('terminal-container');

    expect(terminal.getCurrentSessionId()).toBe('session-k');
  });
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

describe('auto-resume failover window', () => {
  // The confirmation above is the tidy signal; these cover the untidy one.
  // When `claude --resume <dead-id>` cannot find the conversation it just
  // exits, and the resulting 'exit' frame is the ONLY thing that drops the
  // dead id from storage. How long that stays true is the window under test:
  // the CLI takes roughly two seconds to start up and fail, so a window that
  // closed at two seconds was racing the very event it existed to catch.

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  /**
   * Auto-resume `id` from storage and open the socket.
   * @param {string} id
   */
  function autoResume(id) {
    localStorage.setItem(STORAGE_KEY, id);
    terminal.initTerminal('terminal-container');
    openSocket();
  }

  test('exit after the notify timer but inside the window fails over to a fresh session', () => {
    autoResume('dead-id');

    // Past the 2s panel-notify timer, well inside the 10s failover window —
    // the interval a failed resume actually lands in.
    vi.advanceTimersByTime(5000);
    const socketsBefore = FakeWebSocket.created;
    receive({ type: 'exit', code: 1 });

    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(terminal.getCurrentSessionId()).toBeNull();
    // Failover reconnects, and does so without a session_id — a fresh session.
    expect(FakeWebSocket.created).toBeGreaterThan(socketsBefore);
    const url = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(url).not.toContain('session_id=');
  });

  test('exit after the window closes is left alone', () => {
    autoResume('live-id');

    // Past RESUME_FAILOVER_WINDOW_MS: a resume this old did work, and the exit
    // is the operator ending their own session. Dropping the id here would
    // discard a session they may well want to resume.
    vi.advanceTimersByTime(11000);
    receive({ type: 'exit', code: 0 });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('live-id');
  });

  test('a confirmation disarms the failover, so a later exit is left alone', () => {
    autoResume('live-id');

    receive({ type: 'session_info', session_id: 'live-id' });
    vi.advanceTimersByTime(3000);
    receive({ type: 'exit', code: 0 });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('live-id');
  });
});

describe('transcript missing: a new session, said out loud, never a dead PTY', () => {
  // routes/websocket.py refuses to resume an id with no live PTY, no chat and no
  // transcript, and says so with a `transcript_missing` frame (before spawning,
  // or after a `--resume` child reports `No conversation found` and exits). The
  // usual cause is a terminal opened and never prompted, since the CLI writes
  // the transcript on the first prompt, whose PTY did not survive a restart:
  // nothing was lost. So the client drops the stale pointer, says what happened
  // and starts a session; there is nothing for the operator to decide.

  test('auto-resume: a new session starts with no key press', () => {
    localStorage.setItem(STORAGE_KEY, 'gone-id');
    terminal.initTerminal('terminal-container');
    openSocket();
    const socketsBefore = FakeWebSocket.created;
    const refused = /** @type {FakeWebSocket} */ (FakeWebSocket.last);

    receive({ type: 'transcript_missing', session_id: 'gone-id' });

    expect(refused.readyState).toBe(FakeWebSocket.CLOSED);
    expect(FakeWebSocket.created).toBe(socketsBefore + 1);
    const url = /** @type {FakeWebSocket} */ (FakeWebSocket.last).url;
    expect(url).not.toContain('session_id=');
    expect(url).not.toContain('mode=resume');
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  test('the state is said in the terminal, and asks nothing', () => {
    localStorage.setItem(STORAGE_KEY, 'gone-id');
    terminal.initTerminal('terminal-container');
    openSocket();

    receive({ type: 'transcript_missing', session_id: 'gone-id' });
    openSocket();

    const written = /** @type {FakeTerminal} */ (FakeTerminal.last).written;
    expect(written).toContain('gone-id');
    expect(written).toMatch(/transcript/i);
    expect(written).not.toMatch(/Enter/);
    expect(written).not.toContain('[Process exited');
  });

  test('the note outlives the reset the new connection opens with', () => {
    localStorage.setItem(STORAGE_KEY, 'gone-id');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'transcript_missing', session_id: 'gone-id' });
    const fake = /** @type {FakeTerminal} */ (FakeTerminal.last);
    const reset = vi.spyOn(fake, 'reset').mockImplementation(() => {
      fake.written = '';
    });

    openSocket();

    expect(reset).toHaveBeenCalled();
    expect(fake.written).toContain('gone-id');
  });

  test('the note is said once, not on every later connection', () => {
    localStorage.setItem(STORAGE_KEY, 'gone-id');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'transcript_missing', session_id: 'gone-id' });
    openSocket();
    const fake = /** @type {FakeTerminal} */ (FakeTerminal.last);
    fake.written = '';

    terminal.stopTerminal();
    terminal.startTerminal();
    openSocket();

    expect(fake.written).not.toContain('gone-id');
  });

  test('keystrokes reach the new session, and Enter starts nothing else', () => {
    localStorage.setItem(STORAGE_KEY, 'gone-id');
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'transcript_missing', session_id: 'gone-id' });
    openSocket();
    const socketsBefore = FakeWebSocket.created;

    press('ls\r');

    expect(FakeWebSocket.created).toBe(socketsBefore);
    expect(/** @type {FakeWebSocket} */ (FakeWebSocket.last).sent).toContain('ls\r');
  });

  test('an explicit resume (sessions.js resumeSession) gets the same answer', () => {
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'session_info', session_id: 'first-id' });
    terminal.stopTerminal();

    terminal.startTerminal('gone-explicit-id', 'resume');
    openSocket();
    const socketsBefore = FakeWebSocket.created;
    receive({ type: 'transcript_missing', session_id: 'gone-explicit-id' });

    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(FakeWebSocket.created).toBe(socketsBefore + 1);
    expect(/** @type {FakeWebSocket} */ (FakeWebSocket.last).url).not.toContain('session_id=');
  });

  test('a reconnect refused after a restart gets the same answer', () => {
    // A terminal left open across a restart reconnects to its own session URL.
    terminal.initTerminal('terminal-container');
    openSocket();
    receive({ type: 'session_info', session_id: 'never-prompted' });
    terminal.stopTerminal();
    terminal.startTerminal('never-prompted', 'resume');
    openSocket();
    const socketsBefore = FakeWebSocket.created;

    receive({ type: 'transcript_missing', session_id: 'never-prompted' });

    expect(FakeWebSocket.created).toBe(socketsBefore + 1);
    expect(terminal.getCurrentSessionId()).toBeNull();
  });

  test.each([['expert'], ['simple']])(
    'a refused switch leaves the live session untouched (%s view)',
    (mode) => {
      // The server answers a switch_session to a missing id with the same frame
      // but keeps the current PTY attached; the operator is still on it, in
      // either view. The Simple view only reaches a connected terminal through
      // the flip, so it connects the way the flip does.
      document.documentElement.setAttribute('data-ui-mode', mode);
      terminal.initTerminal('terminal-container');
      if (mode === 'simple') terminal.startExpert();
      openSocket();
      receive({ type: 'session_info', session_id: 'live-id' });
      const socket = /** @type {FakeWebSocket} */ (FakeWebSocket.last);

      receive({ type: 'transcript_missing', session_id: 'other-id' });

      expect(localStorage.getItem(STORAGE_KEY)).toBe('live-id');
      expect(terminal.getCurrentSessionId()).toBe('live-id');
      expect(socket.readyState).not.toBe(FakeWebSocket.CLOSED);
      expect(/** @type {FakeTerminal} */ (FakeTerminal.last).written).toContain('other-id');
      // Enter is ordinary input here, forwarded to the live PTY.
      press('\r');
      expect(socket.sent).toContain('\r');
    }
  );
});

describe('a refused own-resume in Simple view', () => {
  // The block above pins the Expert answer to this same frame: say what
  // happened and start a new session. Simple view hides the terminal entirely,
  // so neither half of that answer applies. The note would land in a window
  // nobody can see, and nothing is started: the chat is the surface there, it is
  // bound to the same key, and a hidden terminal spawning the session's agent
  // would take the conversation away from it. The pointer is that shared key,
  // so it survives too — clearing it is how this client says "this
  // conversation is gone", which is not what a terminal that never got to
  // attach knows.
  //
  // The terminal only reaches this state in Simple view by being flipped
  // mid-connect, so these start it the way the flip does (startExpert), not
  // through initTerminal — which no longer connects in Simple view at all.

  test('the shared pointer survives, because the chat is bound to it', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    terminal.startExpert();
    openSocket();

    receive({ type: 'transcript_missing', session_id: 'shared-key' });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('shared-key');
    expect(terminal.getCurrentSessionId()).toBe('shared-key');
  });

  test('nothing is started in the hidden terminal, and nothing is asked of the operator', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    localStorage.setItem(STORAGE_KEY, 'shared-key');
    terminal.initTerminal('terminal-container');
    terminal.startExpert();
    openSocket();
    const socketsBefore = FakeWebSocket.created;

    receive({ type: 'transcript_missing', session_id: 'shared-key' });

    expect(FakeWebSocket.created).toBe(socketsBefore);
    // Nothing was asked of the operator, so nothing claims to have been.
    expect(/** @type {FakeTerminal} */ (FakeTerminal.last).written).not.toMatch(/Press Enter/);
  });

  test('a pointer that has moved on is left alone too', () => {
    // The chat minted a new key while this connect was in flight. The refused
    // id is not the one in storage, so there is nothing here to correct.
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    localStorage.setItem(STORAGE_KEY, 'old-key');
    terminal.initTerminal('terminal-container');
    terminal.startExpert();
    openSocket();
    localStorage.setItem(STORAGE_KEY, 'newer-key');

    receive({ type: 'transcript_missing', session_id: 'old-key' });

    expect(localStorage.getItem(STORAGE_KEY)).toBe('newer-key');
  });
});

describe('the header label', () => {
  test('reads Agent, with the session id as its tooltip and data attribute', () => {
    const label = document.createElement('span');
    label.id = 'terminal-label';
    document.body.append(label);

    terminal.setSessionLabel('abcdef12-3456-4789-8abc-def012345678');
    expect(label.textContent).toBe('Agent');
    expect(label.title).toBe('Session abcdef12-3456-4789-8abc-def012345678');
    expect(label.dataset.sessionId).toBe('abcdef12-3456-4789-8abc-def012345678');

    terminal.setSessionLabel(null);
    expect(label.textContent).toBe('Agent');
    expect(label.hasAttribute('title')).toBe(false);
    expect(label.hasAttribute('data-session-id')).toBe(false);
  });
});
