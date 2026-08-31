/* OSPREY Web Terminal — Terminal Module */

import { createWebSocket, wsUrl, withPrefix } from './api.js';
import { subscribe, xtermPalette } from '/design-system/js/theme-manager.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** @type {any} */
let term = null;
/** @type {any} */
let fitAddon = null;
/** @type {ReturnType<typeof createWebSocket>|null} */
let wsConnection = null;
let hasConnectedBefore = false;
/** @type {string|null} */
let currentSessionId = null;

// localStorage key for persisting the active PTY session ID across page
// loads, so a kept-warm session survives a logout -> landing page ->
// return round trip.
//
// Per persona, not per origin. localStorage is origin-scoped, so on a
// multi-user mount (`/u/alice/`, `/u/bob/`) a bare key would be one shared
// pointer — and a session id is not a preference that can be shared: replaying
// one persona's id would attach another persona's terminal to a PTY that is
// not theirs. Every read, write and clear resolves the key through
// ptySessionStorageKey() below.
const PTY_SESSION_STORAGE_KEY_BASE = 'osprey-pty-session';

/**
 * This document's PTY-pointer key. Resolved per call rather than once at
 * module load: the scope lives on the served document, and reading it at the
 * point of use is what keeps the key honest.
 * @returns {string}
 */
function ptySessionStorageKey() {
  return scopedStorageKey(PTY_SESSION_STORAGE_KEY_BASE);
}

// A resume connection gets a 'session_info' confirmation too —
// routes/websocket.py discovers the attached id on the stale-resume path (a
// new session needs no discovery: the server dictates its id and confirms it
// at once) — carrying the id ACTUALLY attached, which may differ from the
// requested --resume-id if that id was stale/dead and the server silently
// started a fresh PTY instead of erroring. The 'session_info' branch in
// onMessage below treats that confirmation as ground truth: a mismatch
// clears the stored id (see the isStaleResumeMismatch check there) instead
// of leaving a dead id in localStorage forever. Confirmation can still take
// a while to arrive for a genuinely stale id (routes/websocket.py falls
// back to its full discovery-poll timeout), so we don't rely on it alone
// for fast failure detection — see the isAutoResumeAttempt handling in
// startTerminal() and the 'exit' branch in onMessage below, which remain in
// place as a faster, independent failure signal.
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

/**
 * Read the persisted PTY session ID. Returns null if none is stored or if
 * localStorage is unavailable (e.g. private browsing).
 * @returns {string|null}
 */
function loadStoredSessionId() {
  try {
    return localStorage.getItem(ptySessionStorageKey());
  } catch {
    return null;
  }
}

/**
 * Persist the active PTY session ID so a later page load can resume it.
 * @param {string} sessionId
 */
function storeSessionId(sessionId) {
  try {
    localStorage.setItem(ptySessionStorageKey(), sessionId);
  } catch {
    // Ignore — private browsing / storage disabled. Persistence is a
    // convenience; the terminal still works without it.
  }
}

/**
 * Clear the persisted PTY session ID (e.g. once a resume attempt turns out
 * to target a dead/expired session, or on logout — see app.js's
 * initLogoutButton, which clears the client's pointer before navigating so
 * the next page load's initTerminal() has nothing to auto-resume).
 */
export function clearStoredSessionId() {
  try {
    localStorage.removeItem(ptySessionStorageKey());
  } catch {
    // Ignore — see storeSessionId().
  }
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

  // Forward keystrokes to WebSocket
  term.onData((/** @type {string} */ data) => {
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
 */
export function startTerminal(sessionId = null, mode = 'new') {
  if (wsConnection) return;
  if (!term) return;

  let url = wsUrl('/ws/terminal'); // wsUrl prefixes this internally
  // Is this specifically the page-load auto-resume attempt (as opposed to
  // e.g. an explicit resume from sessions.js)? Captured now, before any
  // async work, since autoResumeFailoverId can change out from under us.
  const isAutoResumeAttempt = mode === 'resume' && sessionId != null && sessionId === autoResumeFailoverId;
  if (mode === 'resume' && sessionId) {
    url += `?session_id=${encodeURIComponent(sessionId)}&mode=resume`;
    currentSessionId = sessionId;
    // Persist optimistically — the server sends no confirmation for a
    // resume connection (session_info is only emitted for new sessions),
    // so this is corrected by the 'exit' fallback below if it turns out
    // to be wrong.
    storeSessionId(sessionId);
  }

  const socket = createWebSocket(url, {
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
            // forever. The server does confirm a resume, but only once it has
            // finished discovering what the CLI actually attached to, which
            // can take far longer than the CLI takes to fail — so the PTY
            // exiting is the timely signal, and this is the ONLY place a dead
            // id leaves storage. It stops counting as a resume failure after
            // RESUME_FAILOVER_WINDOW_MS (see startTerminal()), past which an
            // exit is the operator ending a session that did resume.
            if (autoResumeFailoverId) {
              autoResumeFailoverId = null;
              clearStoredSessionId();
              currentSessionId = null;
              stopTerminal();
              startTerminal();
            }
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
            if (isStaleResumeMismatch) {
              clearStoredSessionId();
            } else {
              storeSessionId(msg.session_id);
            }
            setSessionLabel(msg.session_id);
            notifySessionChange(msg.session_id);
          } else if (msg.type === 'session_switched') {
            term.reset();
            currentSessionId = msg.session_id;
            autoResumeFailoverId = null;
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
    // this one. Disarming early is not harmless: the 'exit' branch in
    // onMessage is the only thing that drops a dead id from storage, and a
    // tab that misses it prints "[Process exited]" and stays there, resuming
    // the same dead id on every reload.
    setTimeout(() => {
      if (autoResumeFailoverId === sessionId && wsConnection === socket) {
        autoResumeFailoverId = null;
      }
    }, RESUME_FAILOVER_WINDOW_MS);
  }
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
 * Stop the PTY WebSocket connection.
 */
export function stopTerminal() {
  if (wsConnection) {
    wsConnection.stop();
    wsConnection = null;
  }

  currentSessionId = null;

  setConnectionIndicator(false);
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
 * Get the current Claude Code session ID.
 */
export function getCurrentSessionId() {
  return currentSessionId;
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
 * Get current terminal dimensions.
 */
export function getTerminalDimensions() {
  if (!term) return null;
  return { cols: term.cols, rows: term.rows };
}

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
