/* OSPREY Web Terminal — Terminal Module */

import { createWebSocket, withPrefix, wsUrl } from './api.js';
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

// localStorage key for persisting the active PTY session ID across page
// loads, so a kept-warm session survives a logout -> landing page ->
// return round trip. Scoped to the terminal origin, same style as the
// welcome-modal's 'osprey-server-session' key in app.js.
const PTY_SESSION_STORAGE_KEY = 'osprey-pty-session';

// A resume connection now gets a 'session_info' confirmation too —
// routes/websocket.py runs session discovery on mode=resume as well as
// mode=new — carrying the id ACTUALLY attached, which may differ from the
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

// Session ID we auto-resumed on page load, if any. Armed by initTerminal()
// right before the auto-resume startTerminal() call; disarmed by whichever
// arrives first: a 'session_info' confirmation (see onMessage below), the
// liveness timer (RESUME_LIVENESS_TIMEOUT_MS after connecting, if neither a
// confirmation nor an 'exit' arrived — see startTerminal()), or an 'exit'
// handled by falling back to a fresh session. One-shot:
// only ever set for the initial page-load resume, never for other resume
// call sites (e.g. sessions.js's resumeSession), so explicit user-driven
// resumes are unaffected by this fallback.
/** @type {string|null} */
let autoResumeFailoverId = null;

/**
 * Read the persisted PTY session ID. Returns null if none is stored or if
 * localStorage is unavailable (e.g. private browsing).
 * @returns {string|null}
 */
function loadStoredSessionId() {
  try {
    return localStorage.getItem(PTY_SESSION_STORAGE_KEY);
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
    localStorage.setItem(PTY_SESSION_STORAGE_KEY, sessionId);
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
    localStorage.removeItem(PTY_SESSION_STORAGE_KEY);
  } catch {
    // Ignore — see storeSessionId().
  }
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
  });

  // Live theme switching: re-read the palette from computed style on every
  // apply (see theme-manager.js's hidden-iframe protocol for why this is
  // never deduped on an unchanged theme id).
  subscribe(() => {
    if (term) term.options.theme = xtermPalette();
  });

  fitAddon = new FitAddon.FitAddon();
  const webLinksAddon = new WebLinksAddon.WebLinksAddon();

  term.loadAddon(fitAddon);
  term.loadAddon(webLinksAddon);
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

      // Update session LED
      const led = document.getElementById('session-led');
      if (led) led.classList.add('active');

      // Activate terminal body glow
      const body = document.querySelector('.terminal-body');
      if (body) body.classList.add('active');

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

            // If this was our page-load auto-resume attempt, the server
            // has no way to report a resume failure other than letting
            // the PTY exit (resume connections get no session_info
            // confirmation). Treat it as a stale/expired session: drop
            // the dead id and fall back to a fresh one so the user isn't
            // stuck reconnecting to it forever.
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
            const label = document.getElementById('terminal-label');
            if (label) label.textContent = `Session ${msg.session_id.slice(0, 8)}`;
            notifySessionChange(msg.session_id);
          } else if (msg.type === 'session_switched') {
            term.reset();
            currentSessionId = msg.session_id;
            autoResumeFailoverId = null;
            storeSessionId(msg.session_id);
            const label = document.getElementById('terminal-label');
            if (label) label.textContent = `Session ${msg.session_id.slice(0, 8)}`;
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
      const led = document.getElementById('session-led');
      if (led) led.classList.remove('active');
      const body = document.querySelector('.terminal-body');
      if (body) body.classList.remove('active');
    },
  });
  wsConnection = socket;

  // Disarm the auto-resume failover if this is that attempt and it
  // survives briefly without an 'exit'. There is no positive "resume
  // succeeded" message (see RESUME_LIVENESS_TIMEOUT_MS comment above), so
  // absence-of-failure-within-a-window is the best available signal: a
  // genuinely stale/expired --resume id causes claude to exit almost
  // immediately, while a live resumed shell just keeps running. Guarded by
  // identity (`wsConnection === socket`) so a connection torn down or
  // replaced in the meantime can't spuriously disarm/notify.
  if (isAutoResumeAttempt) {
    setTimeout(() => {
      if (autoResumeFailoverId === sessionId && wsConnection === socket) {
        autoResumeFailoverId = null;
        // The resume never gets a session_info message (server-side —
        // discovery only runs for new sessions), so this is the only
        // place panel iframes learn the resumed session id.
        notifySessionChange(/** @type {string} */ (sessionId));
      }
    }, RESUME_LIVENESS_TIMEOUT_MS);
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

  const led = document.getElementById('session-led');
  if (led) led.classList.remove('active');
  const body = document.querySelector('.terminal-body');
  if (body) body.classList.remove('active');
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
}

/**
 * Get current terminal dimensions.
 */
export function getTerminalDimensions() {
  if (!term) return null;
  return { cols: term.cols, rows: term.rows };
}
