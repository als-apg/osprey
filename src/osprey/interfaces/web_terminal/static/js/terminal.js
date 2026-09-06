/* OSPREY Web Terminal — Terminal Module */

import { createWebSocket, wsUrl, withPrefix } from './api.js';
import { clearPointer, getPointer, setPointer } from './session-pointer.js';
import {
  hideHandoffOverlay,
  showHandoffPending,
  showHandoffRefused,
} from './terminal-handoff.js';
import { subscribe, xtermPalette } from '/design-system/js/theme-manager.js';

/** @type {any} */
let term = null;
/** @type {any} */
let fitAddon = null;
/** @type {ReturnType<typeof createWebSocket>|null} */
let wsConnection = null;
let hasConnectedBefore = false;
/** @type {string|null} */
let currentSessionId = null;

// Every connection gets a 'session_info' confirmation from
// routes/websocket.py carrying the id ACTUALLY attached. A resume of an id
// with no live PTY and no transcript on disk gets 'transcript_missing'
// instead, and nothing is spawned for it — see that branch in onMessage
// below, which renders the state and arms Enter to start fresh. The
// 'session_info' branch treats the confirmation as ground truth: a mismatch
// clears the stored id (the isStaleResumeMismatch check there) instead of
// leaving a dead id in localStorage. The two timers in startTerminal() and
// the 'exit' branch stay as an independent signal for a child that dies
// early for any other reason.
const RESUME_LIVENESS_TIMEOUT_MS = 2000;

// How long an 'exit' still counts as "that resume failed". Wider than
// RESUME_LIVENESS_TIMEOUT_MS on purpose: telling the panels which session
// they are on can be answered the moment the socket looks alive, but ruling
// out a failed resume cannot be answered until the CLI has had time to start,
// fail to find the conversation, and exit — around two seconds unloaded, more
// in a container. Kept under routes/websocket.py's discovery window so the
// failover resolves before the server's fallback confirmation arrives. See
// the two timers in startTerminal().
const RESUME_FAILOVER_WINDOW_MS = 10000;

// Session ID we auto-resumed on page load, if any. Armed by initTerminal()
// right before the auto-resume startTerminal() call; disarmed by whichever
// arrives first: a 'session_info' confirmation (see onMessage below), an
// 'exit' handled by falling back to a fresh session, or the failover window
// closing (RESUME_FAILOVER_WINDOW_MS after connecting, if neither of those
// happened — see startTerminal()). One-shot: only ever set for the initial
// page-load resume, never for other resume call sites (e.g. sessions.js's
// resumeSession), so explicit user-driven resumes are unaffected by this
// fallback.
/** @type {string|null} */
let autoResumeFailoverId = null;

// Set by the 'transcript_missing' branch in onMessage once this connection's
// own resume has been refused and the socket dropped: the terminal shows the
// state and the next Enter starts a fresh session. Every startTerminal()
// disarms it, so a fresh start from the picker or the palette wins too.
let freshStartArmed = false;

/**
 * Read the tab's session pointer. Returns null if none is stored.
 * @returns {string|null}
 */
function loadStoredSessionId() {
  return getPointer();
}

/**
 * Point the tab at the session this terminal is on, so a later page load —
 * and the chat, which reads the same pointer — resumes it.
 * @param {string} sessionId
 */
function storeSessionId(sessionId) {
  setPointer(sessionId);
}

/**
 * Clear the tab's session pointer (e.g. once a resume attempt turns out to
 * target a dead/expired session, or on logout — see app.js's
 * initLogoutButton, which clears the client's pointer before navigating so
 * the next page load's initTerminal() has nothing to auto-resume).
 */
export function clearStoredSessionId() {
  clearPointer();
}

/**
 * Put `text` on the system clipboard on behalf of the agent.
 *
 * The agent's full-screen renderer owns the mouse: a plain click-drag is its
 * own text selection, which it finishes by asking the terminal to copy the
 * text (OSC 52) — the same thing it does in a desktop terminal, where the
 * copy happens through `pbcopy`/`xclip` and the user never notices that
 * "select" already meant "copy". The clipboard addon routes that request
 * here.
 *
 * Browsers only expose the async clipboard API on secure pages (https or
 * localhost). Elsewhere the legacy `execCommand('copy')` path still works in
 * Chromium and Firefox while the drag's user activation is fresh, which it
 * is: the request is one PTY round-trip behind the mouse-up.
 *
 * @param {string} text
 * @returns {Promise<void>}
 */
export async function writeClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const scratch = document.createElement('textarea');
  scratch.value = text;
  scratch.setAttribute('readonly', '');
  scratch.style.position = 'fixed';
  scratch.style.opacity = '0';
  document.body.appendChild(scratch);
  scratch.select();
  let copied = false;
  try {
    copied = document.execCommand('copy');
  } finally {
    scratch.remove();
  }
  if (!copied) {
    throw new Error('clipboard write refused: the page is not a secure context and the legacy copy command was rejected');
  }
}

/**
 * The clipboard the agent is allowed to see: write-only.
 *
 * OSC 52 can also *query* the clipboard. Nothing the agent does needs that,
 * and granting it would let any program in the terminal pull whatever the
 * user last copied — so reads always answer empty, without touching the
 * browser API (which would prompt for permission).
 *
 * @returns {{ readText: () => string, writeText: (selection: string, text: string) => Promise<void> }}
 */
export function agentClipboardProvider() {
  return {
    readText: () => '',
    writeText: (_selection, text) =>
      writeClipboard(text).catch((err) => {
        console.warn('The agent asked to copy text but the browser refused:', err);
      }),
  };
}

/**
 * Claim Ctrl+Shift+C as "copy the selection" before xterm sees it.
 *
 * This is for the terminal's OWN selection — the one a modifier-drag makes
 * (see `macOptionClickForcesSelection` below). Plain Ctrl+C must keep
 * interrupting the agent, and on macOS Cmd+C is the browser's own copy
 * (xterm leaves Meta chords to it), so the only chord that needs claiming is
 * Ctrl+Shift+C — the convention every desktop terminal uses. It is swallowed
 * whether or not anything is selected: xterm would otherwise encode it as ^C
 * and interrupt the agent.
 *
 * Returning `false` tells xterm the event is handled; anything else returns
 * `true` and is processed as usual.
 *
 * @param {KeyboardEvent} event
 * @returns {boolean}
 */
function copyKeyHandler(event) {
  if (event.type !== 'keydown') return true;
  if (!(event.ctrlKey && event.shiftKey && !event.metaKey && !event.altKey)) return true;
  if (event.key !== 'c' && event.key !== 'C') return true;

  if (term && term.hasSelection()) {
    // A refusal surfaces in the console rather than as a pasted ^C.
    writeClipboard(term.getSelection()).catch((err) => {
      console.error('Could not copy the terminal selection:', err);
    });
  }
  return false;
}

/**
 * Initialize xterm.js terminal in the given container.
 * @param {string} containerId
 */
export function initTerminal(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  term = new Terminal({
    scrollback: 10000,
    cursorBlink: true,
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 14,
    lineHeight: 1.2,
    theme: xtermPalette(),
    // The agent switches mouse tracking on (DECSET 1000/1002/1003/1006): a
    // plain click-drag is ITS selection, copied through the clipboard addon
    // below. For the rare raw-screen grab (agent hung, text it won't select)
    // xterm lets a modifier force its own selection — Shift on Linux/Windows
    // unconditionally, Option on macOS only with this flag.
    macOptionClickForcesSelection: true,
  });

  // Live theme switching: re-read the palette from computed style on every
  // apply (see theme-manager.js's hidden-iframe protocol for why this is
  // never deduped on an unchanged theme id).
  subscribe(() => {
    if (term) term.options.theme = xtermPalette();
  });

  fitAddon = new FitAddon.FitAddon();
  const webLinksAddon = new WebLinksAddon.WebLinksAddon();
  // Honour the agent's "copy this" requests (OSC 52) — without this addon
  // xterm drops them and "select" silently stops meaning "copy". The shipped
  // 0.1.0 constructor is (base64Codec, provider) — its typings claim
  // (provider) alone, and passing the provider first makes every request
  // hang in the codec slot.
  const clipboardAddon = new ClipboardAddon.ClipboardAddon(
    new ClipboardAddon.Base64(),
    agentClipboardProvider(),
  );

  term.loadAddon(fitAddon);
  term.loadAddon(webLinksAddon);
  term.loadAddon(clipboardAddon);
  term.attachCustomKeyEventHandler(copyKeyHandler);
  term.open(container);

  // Initial fit — run once now and again after fonts finish loading, since
  // FitAddon measures character cell size using the current font metrics.  If
  // JetBrains Mono hasn't loaded yet the first fit() uses fallback metrics.
  requestAnimationFrame(() => fitAddon.fit());
  document.fonts.ready.then(() => fitAddon.fit());

  // Forward keystrokes to WebSocket — unless the terminal is showing a
  // refused resume, where Enter is the operator's deliberate fresh start and
  // there is no PTY for anything else to reach.
  term.onData((/** @type {string} */ data) => {
    if (freshStartArmed) {
      if (data.includes('\r')) startTerminal();
      return;
    }
    if (wsConnection) wsConnection.send(data);
  });

  // Forward resize events to the PTY via WebSocket
  term.onResize((/** @type {{ cols: number, rows: number }} */ { cols, rows }) => {
    if (wsConnection) {
      wsConnection.send(JSON.stringify({ type: 'resize', cols, rows }));
    }
  });

  // Resize handling — use BOTH window listener and ResizeObserver.
  // Window listener: catches browser window resize (proven approach).
  // ResizeObserver: catches panel drags, iframe loads, layout shifts.
  function doFit() {
    if (!fitAddon) return;
    try {
      fitAddon.fit();
    } catch {
      // Ignore — can happen during teardown
    }
  }

  window.addEventListener('resize', () => doFit());

  let lastObsW = 0, lastObsH = 0;
  const resizeObserver = new ResizeObserver((entries) => {
    const { width, height } = entries[0].contentRect;
    if (Math.abs(width - lastObsW) < 2 && Math.abs(height - lastObsH) < 2) return;
    lastObsW = width;
    lastObsH = height;
    requestAnimationFrame(() => doFit());
  });
  resizeObserver.observe(/** @type {Element} */ (container.parentElement));

  // In the Simple view the chat holds the session and the terminal is not on
  // screen. Only one surface may run the session's agent at a time, so a
  // hidden terminal that connected here would take the conversation away from
  // the view the operator is actually looking at. The flip to Expert starts
  // it instead — see startExpert().
  if (isSimpleView()) return;

  // Start the PTY WebSocket connection. If a session was kept warm from a
  // previous page load (e.g. a logout -> landing page -> return round
  // trip), resume it via the existing server mode=resume path instead of
  // starting a new one. A failed resume (dead/expired id) is detected and
  // falls back to a fresh session — see the 'exit' handling below.
  const storedSessionId = loadStoredSessionId();
  if (storedSessionId) {
    autoResumeFailoverId = storedSessionId;
    startTerminal(storedSessionId, 'resume');
  } else {
    startTerminal();
  }
}

/**
 * Start (or restart) the PTY WebSocket connection.
 *
 * @param {string|null} sessionId - Session UUID to resume. Null for new session.
 * @param {'new'|'resume'} mode - Whether to start a new session or resume.
 * @param {{ interrupt?: boolean }} [options] - `interrupt` ends the other
 *   view's running turn instead of waiting for it ("Stop and switch now").
 *   Only meaningful on a resume.
 * @returns {Promise<void>} Settles once this attempt has an answer — attached
 *   (`session_info`), refused, errored, or closed without one — and never
 *   rejects; a caller sequencing a hand-off awaits it so the next surface
 *   cannot ask for the session before the server has seen this one let go.
 */
export function startTerminal(sessionId = null, mode = 'new', { interrupt = false } = {}) {
  if (wsConnection) return Promise.resolve();
  if (!term) return Promise.resolve();
  freshStartArmed = false;

  // Resolved by whichever answer arrives first. Resolving is idempotent, so
  // every settling point below can call it without checking the others.
  let settle = () => {};
  /** @type {Promise<void>} */
  const settled = new Promise((resolve) => {
    settle = resolve;
  });

  let url = wsUrl('/ws/terminal'); // wsUrl prefixes this internally
  // Is this specifically the page-load auto-resume attempt (as opposed to
  // e.g. an explicit resume from sessions.js)? Captured now, before any
  // async work, since autoResumeFailoverId can change out from under us.
  const isAutoResumeAttempt = mode === 'resume' && sessionId != null && sessionId === autoResumeFailoverId;
  if (mode === 'resume' && sessionId) {
    url += `?session_id=${encodeURIComponent(sessionId)}&mode=resume`;
    if (interrupt) url += '&interrupt=1';
    currentSessionId = sessionId;
    // Persist optimistically so a reload mid-connect resumes the same id;
    // the server's answer corrects it — 'session_info' with another id, or
    // 'transcript_missing' — and the 'exit' fallback below covers the rest.
    storeSessionId(sessionId);
  }

  // Declared before the call so the handlers below can compare against it;
  // they only ever run once `createWebSocket` has returned.
  /** @type {ReturnType<typeof createWebSocket>|null} */
  let socket = null;
  socket = createWebSocket(url, {
    onOpen() {
      // On reconnection (server restart), reset terminal to avoid
      // garbled output from old session mixed with new.
      if (hasConnectedBefore) {
        term.reset();
      }
      hasConnectedBefore = true;

      setConnectionIndicator(true);

      // Send initial size FIRST — the server waits for this before
      // spawning the PTY, so the shell starts with correct dimensions.
      fitAddon.fit();
      /** @type {NonNullable<typeof wsConnection>} */ (wsConnection).send(JSON.stringify({
        type: 'resize',
        cols: term.cols,
        rows: term.rows,
      }));
    },
    onMessage(e) {
      if (typeof e.data === 'string') {
        // JSON control message
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'exit') {
            term.write(`\r\n\x1b[33m[Process exited with code ${msg.code}]\x1b[0m\r\n`);
            const led = document.getElementById('session-led');
            if (led) led.classList.remove('active');

            // If this was our page-load auto-resume attempt, treat the exit
            // as the resume having failed: drop the dead id and fall back to
            // a fresh session so the user isn't stuck reconnecting to it
            // forever. A missing transcript no longer arrives this way (the
            // server answers 'transcript_missing' and spawns nothing); this
            // covers a child that dies early for any other reason. It stops
            // counting as a resume failure after RESUME_FAILOVER_WINDOW_MS
            // (see startTerminal()), past which an exit is the operator
            // ending a session that did resume.
            if (autoResumeFailoverId) {
              autoResumeFailoverId = null;
              clearStoredSessionId();
              currentSessionId = null;
              stopTerminal();
              startTerminal();
            }
          } else if (msg.type === 'handoff_pending') {
            // The key is alive in the other view — the server just said so by
            // negotiating for it — so the page-load failover has its answer
            // and must not fire. Left armed, the exit that ends the outgoing
            // agent (or the one an interrupt forces) would read as a dead id
            // and drop the key both views are on.
            autoResumeFailoverId = null;
            // The other view still holds the session and its agent is mid-turn.
            // The server waits for that turn to end rather than killing it, so
            // this connection has no bound on how long it sits here — show the
            // wait, and the way to end it: ask for the same session again with
            // `interrupt=1`, which ends that turn instead of waiting it out.
            showHandoffPending(() => {
              dropConnection();
              startExpert({ interrupt: true });
            });
          } else if (msg.type === 'session_info') {
            // On resume, msg.session_id is the id ACTUALLY attached, which
            // may differ from the stale id we asked for (the server
            // silently starts a fresh PTY rather than erroring — see the
            // module comment above). This confirmation is ground truth: a
            // mismatch means the requested id is dead, so drop it from
            // storage rather than persisting an id nobody asked to resume
            // — the next page load then starts clean instead of silently
            // chaining onto an unrequested session. A match (or the
            // new-session path, where there is no request to compare
            // against) persists normally.
            const isStaleResumeMismatch =
              mode === 'resume' && sessionId != null && msg.session_id !== sessionId;
            currentSessionId = msg.session_id;
            autoResumeFailoverId = null;
            // Whatever the connection was waiting on is over: this terminal is
            // attached to the session.
            hideHandoffOverlay();
            settle();
            if (isStaleResumeMismatch) {
              clearStoredSessionId();
            } else {
              storeSessionId(msg.session_id);
            }
            setSessionLabel(msg.session_id);
            notifySessionChange(msg.session_id);
          } else if (msg.type === 'transcript_missing') {
            // The server refused to resume msg.session_id: no live PTY and no
            // transcript on disk, so nothing was spawned (or a --resume child
            // reported the same and quit). Not a silent fresh session — the
            // operator picked this chat, and starting another under them
            // would hide that it is gone. Say so, and make the next move
            // theirs.
            autoResumeFailoverId = null;
            if (msg.session_id === currentSessionId) {
              // This connection's own resume. The pointer is the session key
              // BOTH views run on, so clearing it says "this conversation is
              // gone" to the chat as well. Say that only about the key the
              // pointer actually holds, and only when the chat is not the view
              // bound to it: in Simple view the terminal is hidden and the chat
              // is the surface on that key. Drop the socket either way, so the
              // wrapper does not reconnect to the same refused URL.
              if (getPointer() === msg.session_id && !isSimpleView()) {
                clearStoredSessionId();
              }
              currentSessionId = null;
              stopTerminal();
              setSessionLabel(null);
              // Expert view can show what happened, so it does, and arms Enter
              // to make the next move the operator's. Simple view hides the
              // terminal, so both halves of that answer would land in a window
              // nobody can see or reach — and nothing is started in its place
              // either: the chat is the surface there, and a hidden terminal
              // spawning the session's agent would take the conversation away
              // from it.
              if (!isSimpleView()) {
                term.write(
                  `\r\n\x1b[33mThe transcript for session ${msg.session_id} is missing, so it cannot be resumed.\x1b[0m\r\n` +
                  'Press Enter to start a new session.\r\n'
                );
                freshStartArmed = true;
              }
            } else {
              // A refused switch_session: the server kept the current PTY
              // attached, so the operator is still on a live session.
              term.write(
                `\r\n\x1b[33mThe transcript for session ${msg.session_id} is missing, so it cannot be switched to.\x1b[0m\r\n`
              );
            }
          } else if (msg.type === 'session_switched') {
            term.reset();
            currentSessionId = msg.session_id;
            autoResumeFailoverId = null;
            // A switch onto a key the chat holds is negotiated like any other
            // hand-off, and this frame — not `session_info` — is how a switch
            // reports success. It is the end of the wait either way.
            hideHandoffOverlay();
            storeSessionId(msg.session_id);
            setSessionLabel(msg.session_id);
            notifySessionChange(msg.session_id);
            // Update reconnect URL so auto-reconnect targets the correct session
            if (wsConnection) {
              wsConnection.setUrl(
                wsUrl(`/ws/terminal?session_id=${encodeURIComponent(msg.session_id)}&mode=resume`) // wsUrl() adds the prefix
              );
            }
          } else if (msg.type === 'error') {
            // Whatever the connection was waiting on has resolved into this,
            // so the transitional state goes before the message that replaces
            // it — an error under a "finishing in the other view" overlay is
            // unreadable and untrue.
            hideHandoffOverlay();
            settle();
            term.write(`\r\n\x1b[31m[Error: ${msg.message}]\x1b[0m\r\n`);
          }
          return;
        } catch {
          term.write(e.data);
        }
      } else {
        // Binary PTY output
        term.write(new Uint8Array(e.data));
      }
    },
    onClose() {
      setConnectionIndicator(false);
      // A close before any answer is itself the answer: this attempt is over,
      // whatever the wrapper does about reconnecting.
      settle();
      notifyClosed();
    },
    onRefused(code) {
      // The server declined to hand the session over. A refusal that lands
      // after this socket was abandoned — the operator pressed "Stop and
      // switch now", and the reconnect is already waiting on its own
      // hand-off — says nothing about the connection now in place, so it is
      // dropped rather than allowed to paint over that one's state.
      if (wsConnection !== socket) return;
      // This wrapper is spent: it will not reconnect, so the module must stop
      // holding it or the next start would see a live-looking connection and
      // return early. Every retry builds a new one — the button below, and
      // the one the `handoff_pending` branch above wires into startExpert().
      wsConnection = null;
      // Stated here rather than left to the close above, so the contract does
      // not depend on the order api.js fires the two in.
      settle();
      showHandoffRefused(code, () => startExpert());
    },
  });
  wsConnection = socket;

  // Two jobs on two clocks, deliberately separated — they answer different
  // questions and a single timer served the shorter one at the longer one's
  // expense. Both are guarded by identity (`wsConnection === socket`) so a
  // connection torn down or replaced in the meantime can't spuriously fire.
  if (isAutoResumeAttempt) {
    // "Which session are the panels looking at?" — answer as soon as the
    // connection looks alive. Reached only when no session_info has arrived
    // yet: for a stale id the server is still off discovering what the CLI
    // actually started (routes/websocket.py), and that can outlast this
    // timer, so this is the one place panel iframes learn the resumed id when
    // the confirmation is slow. A confirmation that does arrive notifies with
    // the id the server actually attached rather than the one we
    // optimistically asked for.
    setTimeout(() => {
      if (autoResumeFailoverId === sessionId && wsConnection === socket) {
        notifySessionChange(/** @type {string} */ (sessionId));
      }
    }, RESUME_LIVENESS_TIMEOUT_MS);

    // "Did the resume fail?" — keep listening for the 'exit' that says so
    // until a failed resume could no longer plausibly be the cause. There is
    // no positive "resume succeeded" message, so absence-of-failure-within-a-
    // window is the best available signal, and the window has to be wider
    // than the CLI takes to start up, fail to find the conversation, and
    // exit. Measured at roughly two seconds on an idle laptop and slower in a
    // container — so the notify timer above is far too tight to double as
    // this one. Disarming early is not harmless: for an early exit with no
    // 'transcript_missing' verdict, the 'exit' branch in onMessage is what
    // drops the dead id from storage, and a tab that misses it prints
    // "[Process exited]" and stays there, resuming the same dead id on every
    // reload.
    setTimeout(() => {
      if (autoResumeFailoverId === sessionId && wsConnection === socket) {
        autoResumeFailoverId = null;
      }
    }, RESUME_FAILOVER_WINDOW_MS);
  }

  return settled;
}

/**
 * Take the session over for the Expert view.
 *
 * The flip to Expert calls this, and so does everything that retries a
 * hand-off. It resumes the key the tab is pointed at — the same key the
 * Simple view was on — so the conversation continues in the terminal rather
 * than a second one starting beside it. With no key stored there is nothing
 * to take over, so it starts a session the ordinary way.
 *
 * Deliberately not armed for auto-resume failover: the key is shared with the
 * chat, and an early exit here is a hand-off that did not complete, not a
 * dead session id to drop.
 *
 * @param {{ interrupt?: boolean }} [options] - `interrupt` ends the other
 *   view's running turn instead of waiting for it.
 * @returns {Promise<void>} Settles once the Expert acquire has an answer —
 *   attached, refused, errored, or closed without one — and never rejects.
 *   A flip chain awaits it so a Simple → Expert → Simple round trip cannot
 *   ask for the session again before the server has seen this channel go.
 */
export function startExpert({ interrupt = false } = {}) {
  const key = loadStoredSessionId();
  return key ? startTerminal(key, 'resume', { interrupt }) : startTerminal();
}

/**
 * Restart the terminal session with immediate visual feedback.
 * Clears the screen and shows a "Restarting..." message while the
 * backend restart endpoint is called, then reconnects.
 */
export async function restartTerminal() {
  // Immediate visual feedback: tear down old connection and clear screen
  stopTerminal();
  if (term) {
    term.reset();
    term.write('\x1b[90mRestarting session\u2026\x1b[0m\r\n');
  }

  // Hit the restart endpoint (kill old PTY on backend). Prefix-aware so it
  // reaches this container under /u/<user>/ in multi-user deployments.
  await fetch(withPrefix('/api/terminal/restart'), { method: 'POST' });
}

/**
 * Drop the PTY WebSocket connection and dim the indicators, leaving any
 * hand-off overlay standing.
 *
 * Split out of {@link stopTerminal} for the one caller that stops the socket
 * *because* a hand-off is under way ("Stop and switch now"): it reconnects
 * immediately and the overlay it is watching must not blink out and restart
 * its clock in between.
 */
function dropConnection() {
  if (wsConnection) {
    wsConnection.stop();
    wsConnection = null;
  }

  setConnectionIndicator(false);
}

// How long a caller sequencing a flip waits for the dropped socket's close
// event before going on without it. The server holds a short grace for an
// attached channel to let go, and the next surface asking for the session
// inside that window is refused with no retry — so waiting for the close is
// worth doing, and waiting forever on a socket that never reports one is not.
const CLOSE_ACK_TIMEOUT_MS = 2000;

/** Resolvers waiting for a dropped socket's close event. */
/** @type {(() => void)[]} */
let closeWaiters = [];

/** Release everything waiting on a close. */
function notifyClosed() {
  const waiting = closeWaiters;
  closeWaiters = [];
  for (const resolve of waiting) resolve();
}

/**
 * Stop the PTY WebSocket connection.
 *
 * This is also the teardown half of a view flip: the session key survives it
 * (see {@link getCurrentSessionId}), so nothing here touches the pointer.
 *
 * The socket is dropped synchronously, as it always was — callers that do not
 * care may ignore the return value.
 *
 * @returns {Promise<void>} Settles when the dropped socket's close event has
 *   fired, at once when there was nothing open to drop, and after a short
 *   timeout when the close never arrives; never rejects.
 */
export function stopTerminal() {
  const socket = wsConnection?.ws;
  // Registered before the drop, because a close can be reported synchronously.
  const closed = socket && socket.readyState !== WebSocket.CLOSED
    ? /** @type {Promise<void>} */ (new Promise((resolve) => {
      const timer = setTimeout(resolve, CLOSE_ACK_TIMEOUT_MS);
      closeWaiters.push(() => {
        clearTimeout(timer);
        resolve();
      });
    }))
    : Promise.resolve();

  dropConnection();
  hideHandoffOverlay();
  return closed;
}

/**
 * Whether this page is currently showing the Simple view. The server stamps
 * the mode on `<html>` and app.js flips it live, so this is the one question
 * every "may the terminal act on its own here?" decision asks.
 * @returns {boolean}
 */
function isSimpleView() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple';
}

/**
 * Single owner of the connection-status indicators: the header LED
 * (`#session-led`) and the terminal-body glow track PTY WebSocket
 * connectedness together. (The process-exit branch in onMessage
 * deliberately dims only the LED — the chrome keeps its glow until the
 * connection itself closes.)
 * @param {boolean} connected
 */
function setConnectionIndicator(connected) {
  const led = document.getElementById('session-led');
  if (led) led.classList.toggle('active', connected);
  const body = document.querySelector('.terminal-body');
  if (body) body.classList.toggle('active', connected);
}

/**
 * Single owner of the header session label (`#terminal-label`) — exported
 * so sessions.js updates it through terminal.js instead of reaching into
 * terminal-owned DOM. Pass null for the generic placeholder shown before a
 * session id is known.
 * @param {string|null} sessionId
 */
export function setSessionLabel(sessionId) {
  const label = document.getElementById('terminal-label');
  if (label) label.textContent = sessionId ? `Session ${sessionId.slice(0, 8)}` : 'Session';
}

/**
 * Switch to a different Claude session over the existing WebSocket.
 * Returns true if the switch message was sent (fast path), false if
 * no WebSocket is available (caller should use the cold fallback).
 *
 * @param {string} sessionId - Target session UUID.
 * @returns {boolean}
 */
export function switchSession(sessionId) {
  if (!wsConnection) return false;
  if (sessionId === currentSessionId) return true;
  wsConnection.send(JSON.stringify({ type: 'switch_session', session_id: sessionId }));
  return true;
}

/**
 * The session key this card is on. Tearing the socket down no longer forgets
 * it: a view flip stops this terminal so the other surface can take the
 * session over, and the key has to outlive the process that was serving it.
 * Falls back to the stored pointer for the window between page load and the
 * first connect, where the chat may ask before any confirmation has arrived.
 * @returns {string|null}
 */
export function getCurrentSessionId() {
  return currentSessionId ?? loadStoredSessionId();
}

/**
 * Re-fit the terminal (call after panel resize).
 */
export function fitTerminal() {
  if (fitAddon) {
    requestAnimationFrame(() => fitAddon.fit());
  }
}

/**
 * Focus the terminal.
 */
export function focusTerminal() {
  if (term) term.focus();
}

/**
 * Paste text into the terminal (sends to PTY via WebSocket).
 * Used by the postMessage bridge to receive text from embedded iframes.
 */
export function pasteToTerminal(/** @type {string} */ text) {
  if (wsConnection && text) {
    wsConnection.send(text);
  }
}

/**
 * In-page listeners for "which session is this card on?" — the same question
 * the panel iframes get answered by postMessage below, asked by hub modules
 * that live in this document (the control-target chip). Kept as one list rather
 * than a DOM event so the answer travels through exactly the seam every id
 * change already funnels into.
 * @type {((sessionId: string) => void)[]}
 */
const sessionChangeListeners = [];

/**
 * Subscribe to active-session changes. The callback fires for every id the
 * card settles on — a new session's `session_info`, a resume confirmation, a
 * session switch — but not for the current id at subscribe time; callers that
 * need the id now read {@link getCurrentSessionId}.
 * @param {(sessionId: string) => void} fn
 */
export function onSessionChange(fn) {
  sessionChangeListeners.push(fn);
}

/**
 * Notify all panel iframes that the active session has changed.
 * @param {string} sessionId - The new session UUID.
 */
export function notifySessionChange(sessionId) {
  document.querySelectorAll('.panel-iframe').forEach(iframe => {
    try {
      /** @type {Window} */ (/** @type {HTMLIFrameElement} */ (iframe).contentWindow).postMessage(
        { type: 'osprey-session-change', session_id: sessionId },
        window.location.origin
      );
    } catch { /* cross-origin — ignore */ }
  });
  // One listener throwing must not cost the others their notification.
  for (const fn of sessionChangeListeners) {
    try {
      fn(sessionId);
    } catch (err) {
      console.error('osprey web_terminal: a session-change listener threw', err);
    }
  }
}

/**
 * The fitted size xterm settled on, or null before `initTerminal()` has run.
 * Nothing in the page reads it today (the terminal-size bar item that did is
 * retired); the `window` seam below is its one consumer.
 */
export function getTerminalDimensions() {
  if (!term) return null;
  return { cols: term.cols, rows: term.rows };
}

/**
 * The same getter, published on `window` — a DELIBERATE new production global,
 * and the only one this module adds. It exists for exactly one caller, and the
 * constraint that forces it is worth stating rather than rediscovering:
 *
 *   `docs/screenshots/contact_sheet.py` reads the fitted COLUMN count before
 *   every capture, to prove the demo transcript will not wrap at the terminal's
 *   real width. It used to scrape the `#term-dims` status-bar readout. That
 *   element is gone: the size is now a bar ITEM the operator can move, fold or
 *   remove, so any DOM scrape would become a guard that silently reads nothing
 *   and passes. Playwright evaluates in page scope and cannot import an ES
 *   module, so the getter has to be reachable from `window` or not at all.
 *
 * It is a TEST SEAM, not a second public API: nothing in the page may read it.
 * Anything running inside the module graph imports `getTerminalDimensions`
 * above — reaching through `window` for a value that can be imported is
 * strictly worse, and would let this global grow readers that pin it in place.
 *
 * `Object.assign` rather than a bare property write because the property is
 * not on the DOM `Window` type and this file carries no ambient declaration.
 */
Object.assign(window, { __OSPREY_TERMINAL_DIMS__: getTerminalDimensions });

/**
 * The live xterm instance, or null before `initTerminal()` has run (or after a
 * failed init — the boot guards that call).
 *
 * The one reader is the feedback dialog, which walks `term.buffer` for the
 * scrollback it attaches (scrollback-capture.js). Handed out rather than
 * wrapped because that walk is a read of the buffer's own API and belongs with
 * the module that defines the capture rules, not here — and the caller
 * duck-types what it needs, so a null is a legitimate answer it already
 * handles.
 */
export function getTerminalInstance() {
  return term;
}
