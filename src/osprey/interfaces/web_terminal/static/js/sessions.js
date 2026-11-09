/* OSPREY Web Terminal — Session moves
 *
 * The two low-level moves the session model (`session-model.js`) drives: resume
 * a session by key, and start a new one. Operators pick sessions in the header
 * switcher (`session-switcher.js`), whose picks go through the model's guarded
 * `switchTo`. */

import {
  stopTerminal,
  startTerminal,
  restartTerminal,
  notifySessionChange,
  switchSession,
  setSessionLabel,
} from './terminal.js';
import { isTerminalSurface } from './agent-surface.js';
import { setPointer } from './session-pointer.js';

/**
 * Resume a session by ID.
 *
 * Fast path: send switch_session over the existing WebSocket (near-instant
 * for warm sessions). Cold fallback: full stop/start cycle if no WS is open.
 * @param {string} sessionId
 */
export async function resumeSession(sessionId) {
  // Chat has no terminal socket: the console is the surface running the session.
  // Moving the pointer is the whole switch, because the console is subscribed to
  // it and replays the key. Starting a terminal here would put a second surface
  // on the session's agent.
  if (!isTerminalSurface()) {
    setPointer(sessionId);
    return;
  }
  if (switchSession(sessionId)) {
    // Fast path — server handles everything.
    // terminal.js onMessage updates UI on session_switched.
    return;
  }

  // Cold fallback — no WS open, do full reconnect
  stopTerminal();
  setSessionLabel(sessionId);
  await new Promise((r) => setTimeout(r, 100));
  startTerminal(sessionId, 'resume');
  notifySessionChange(sessionId);
}

/**
 * Start a brand new session.
 */
export async function startNewSession() {
  await restartTerminal();
  setSessionLabel(null);
  startTerminal();
}
