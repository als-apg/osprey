// @ts-check
/* OSPREY Web Terminal — the session rows' cues
 *
 * What the session tree (sidebar-artifacts.js) shows about the sessions
 * beyond their names and artifacts, and the Sessions row's "+":
 *
 *   - BUSY. The chat bus's `busy` (a turn streams or a hand-off runs) says
 *     the agent is working; the tree marks the active session, whose accent
 *     bar breathes.
 *   - SELECTING. A click on another session moves the accent to it at once;
 *     {@link SessionCues.selecting} holds it there until the switch has
 *     happened or did not.
 *   - THE AGENT PANE. Opening or closing it (`osprey-agent-pane`,
 *     dock-workspace.js) redraws the tree, since the active session's click
 *     and its tip depend on it.
 *   - "+". A new session, as the row's menu starts one (`switchTo({new:
 *     true})`: the chat's New conversation, or a new terminal session), with
 *     the Agent pane opened first.
 *   - THE PEEK. A folded session's artifacts on hover (sidebar-session-peek.js).
 *
 * @module sidebar-session-cues
 */

import { onChat } from './chat-events.js';
import { isTerminalPanelOpen, openTerminalPanel } from './dock-workspace.js';
import * as model from './session-model.js';
import { createSessionPeek } from './sidebar-session-peek.js';

/** The Sessions row's "+" (index.html). */
export const NEW_SESSION_ID = 'sidebar-new-session';

/** Start a new session, the Agent pane opened first. */
export function newSession() {
  if (!isTerminalPanelOpen()) openTerminalPanel();
  void model.switchTo({ new: true });
}

/**
 * @typedef {object} SessionCues
 * @property {(signal: AbortSignal) => void} watch  wire the cues until `signal` aborts
 * @property {() => boolean} busy  whether the agent is working
 * @property {() => string | null} selecting  the session being switched to
 * @property {(treeId: string, run: () => Promise<boolean>) => Promise<boolean>} select
 *   hold the accent on `treeId` while `run` switches to it; resolves what `run` did
 * @property {() => void} refresh  redraw the peek after the tree redrew
 */

/**
 * @param {{render: () => void, peek: import('./sidebar-session-peek.js').PeekDeps}} deps
 * @returns {SessionCues}
 */
export function createSessionCues({ render, peek: peekDeps }) {
  let busy = false;
  /** @type {string | null} */
  let selecting = null;
  const peek = createSessionPeek(peekDeps);

  return {
    watch(signal) {
      document.addEventListener('osprey-agent-pane', render, { signal });
      document.getElementById(NEW_SESSION_ID)?.addEventListener('click', newSession, { signal });
      const off = onChat('busy', (detail) => {
        if (detail.busy === busy) return;
        busy = detail.busy;
        render();
      });
      signal.addEventListener('abort', () => {
        off();
        busy = false;
        selecting = null;
      });
      peek.watch(signal);
    },
    busy: () => busy,
    selecting: () => selecting,
    async select(treeId, run) {
      selecting = treeId;
      render();
      try {
        return await run();
      } finally {
        if (selecting === treeId) selecting = null;
      }
    },
    refresh: () => peek.refresh(),
  };
}
