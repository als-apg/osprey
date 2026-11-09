// @ts-check
/* OSPREY Web Terminal — Transcript-Move Relay
 *
 * The hub-role fan-out for a transcript move on the shown session. Extracted
 * from panel-manager, which passes its panel states in, so that module stays
 * within its size budget; the single-target sends it fans out live in
 * panel-iframe-sync.js.
 */

import { sendSessionToIframe } from './panel-iframe-sync.js';
import { getCurrentSessionId } from './terminal.js';

/**
 * Relay a transcript move on the shown session to every embedded panel. A
 * `/clear` moves a session key onto a new transcript; the hub's
 * `session_transcript` frame for it reaches only this page's own stream
 * (session-model.js), so a panel scoped to that key would keep the old
 * transcript's view. When the frame names the key the panels show, every panel
 * iframe, in a tile or in a window of its own, is told that same key again
 * (`osprey-session-change`), and each re-reads its view for it.
 *
 * Unlike terminal.js's notifySessionChange this fires no session-change
 * listener: the key did not change, so the first-contact reset those listeners
 * drive must not run.
 * @param {string} sessionKey - The key whose transcript moved.
 * @param {Iterable<{iframe: HTMLIFrameElement | null}>} panels - Every panel's state.
 */
export function relayTranscriptMove(sessionKey, panels) {
  if (!sessionKey || sessionKey !== getCurrentSessionId()) return;
  for (const state of panels) sendSessionToIframe(state.iframe);
}
