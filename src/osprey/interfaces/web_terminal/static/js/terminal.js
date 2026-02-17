/* OSPREY Web Terminal — Terminal Module */

import { createWebSocket } from './api.js';

let term = null;
let fitAddon = null;
let wsConnection = null;
let hasConnectedBefore = false;

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
    theme: {
      background: '#050a10',
      foreground: '#c8d6e5',
      cursor: '#4fd1c5',
      cursorAccent: '#050a10',
      selectionBackground: 'rgba(79, 209, 197, 0.25)',
      black: '#1a2332',
      red: '#ef4444',
      green: '#22c55e',
      yellow: '#f59e0b',
      blue: '#3b82f6',
      magenta: '#a855f7',
      cyan: '#4fd1c5',
      white: '#e2e8f0',
      brightBlack: '#64748b',
      brightRed: '#f87171',
      brightGreen: '#4ade80',
      brightYellow: '#fbbf24',
      brightBlue: '#60a5fa',
      brightMagenta: '#c084fc',
      brightCyan: '#67e8f9',
      brightWhite: '#f8fafc',
    },
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
 * No-op if already connected.
 */
export function startTerminal() {
  if (wsConnection) return;
  if (!term) return;

  const wsUrl = `ws://${location.host}/ws/terminal`;

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
          }
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
 * Clears the screen and shows a "Restarting…" message while the
 * backend restart endpoint is called, then reconnects.
 */
export async function restartTerminal() {
  // Immediate visual feedback: tear down old connection and clear screen
  stopTerminal();
  if (term) {
    term.reset();
    term.write('\x1b[90mRestarting session…\x1b[0m\r\n');
  }

  // Hit the restart endpoint (kill old PTY on backend)
  await fetch('/api/terminal/restart', { method: 'POST' });

  // Reconnect — new WebSocket triggers a fresh PTY spawn
  startTerminal();
}

/**
 * Stop the PTY WebSocket connection.
 */
export function stopTerminal() {
  if (wsConnection) {
    wsConnection.stop();
    wsConnection = null;
  }

  const led = document.getElementById('session-led');
  if (led) led.classList.remove('active');
  const body = document.querySelector('.terminal-body');
  if (body) body.classList.remove('active');
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
 * Get current terminal dimensions.
 */
export function getTerminalDimensions() {
  if (!term) return null;
  return { cols: term.cols, rows: term.rows };
}
