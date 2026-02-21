/* OSPREY Web Terminal — Terminal Module */

import { createWebSocket } from './api.js';
import { getXtermPalette, setTerminalRef } from './theme.js';

let term = null;
let fitAddon = null;
let wsConnection = null;
let hasConnectedBefore = false;
let currentSessionId = null;

/**
 * Initialize xterm.js terminal in the given container.
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
    theme: getXtermPalette(),
  });

  // Register terminal reference for live theme switching
  setTerminalRef(term);

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
  term.onData((data) => {
    if (wsConnection) wsConnection.send(data);
  });

  // Forward resize events to the PTY via WebSocket
  term.onResize(({ cols, rows }) => {
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

  const resizeObserver = new ResizeObserver(() => {
    requestAnimationFrame(() => doFit());
  });
  resizeObserver.observe(container.parentElement);

  // Start the PTY WebSocket connection
  startTerminal();
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

  let wsUrl = `ws://${location.host}/ws/terminal`;
  if (mode === 'resume' && sessionId) {
    wsUrl += `?session_id=${encodeURIComponent(sessionId)}&mode=resume`;
    currentSessionId = sessionId;
  }

  wsConnection = createWebSocket(wsUrl, {
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
      wsConnection.send(JSON.stringify({
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
          } else if (msg.type === 'session_info') {
            currentSessionId = msg.session_id;
            const label = document.getElementById('terminal-label');
            if (label) label.textContent = `Session ${msg.session_id.slice(0, 8)}`;
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

  // Hit the restart endpoint (kill old PTY on backend)
  await fetch('/api/terminal/restart', { method: 'POST' });
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
export function pasteToTerminal(text) {
  if (wsConnection && text) {
    wsConnection.send(text);
  }
}

/**
 * Get current terminal dimensions.
 */
export function getTerminalDimensions() {
  if (!term) return null;
  return { cols: term.cols, rows: term.rows };
}
