/* OSPREY Web Terminal — Agent attention on the panel rail.

   The two ways a rail entry signals agent activity: a transient glow for
   "the agent just touched this", and a persistent badge for "this panel has
   activity the operator has not seen". Everything here anchors on the rail
   element bound once via initAgentAttention().

   A badge must outlive the page: the server's history ring is re-read on every
   SSE open (restoreAgentBadges) and any panel activity the operator has not
   seen re-badges its entry. "Seen" is an ACKNOWLEDGMENT — the server ts of the
   newest badge the operator cleared by surfacing that panel, kept per panel in
   localStorage under `agent-ack:<panelId>`.

   Only SERVER timestamps are ever stored or compared here. The ring's `ts` is
   the web-terminal process's clock; a browser clock skewed ahead of it would
   permanently suppress real badges, and one skewed behind would resurrect
   cleared ones on every reconnect. So a badge whose frame carried no ts leaves
   the stored ack untouched rather than substituting Date.now(). */

import { fetchJSON } from './api.js';
import { getEntry, setEntryAttention } from './panel-rail.js';
import { flashElement } from '/design-system/js/highlight.js';

/** @type {HTMLElement} */
let railEl = /** @type {HTMLElement} */ (/** @type {unknown} */ (null));

/**
 * Bind the rail element every glow/badge call anchors on. Call once, as soon
 * as the rail exists and before any SSE frame can arrive.
 * @param {HTMLElement} el
 */
export function initAgentAttention(el) { railEl = el; }

/**
 * Transient agent glow on a rail entry — flash only, never the persistent
 * badge (that is agent_activity's job via badgePanelActivity). No-op for ids
 * without an entry.
 * @param {string} panelId
 */
export function flashAgentGlow(panelId) { const entry = getEntry(railEl, panelId); if (entry) flashElement(entry); }

/** Newest badge-causing server ts seen this page lifetime, per panel.
 *  @type {Map<string, number>} */
const badgeTs = new Map();

const ACK_KEY_PREFIX = 'agent-ack:';

/**
 * Record a badge's server ts, keeping the newest. The history ring replays a
 * panel's older events alongside its newest, so this must not walk backwards.
 * @param {string} panelId
 * @param {number} [ts]
 */
function noteBadgeTs(panelId, ts) {
  if (typeof ts !== 'number') return;
  const prev = badgeTs.get(panelId);
  if (prev === undefined || ts > prev) badgeTs.set(panelId, ts);
}

/**
 * The acknowledged server ts for a panel. A panel that was never acknowledged
 * (and a storage read that is denied or corrupt) has seen nothing, so it reads
 * as -Infinity: an unknown ack RESTORES a badge, it never suppresses one.
 * @param {string} panelId
 * @returns {number}
 */
function ackedTs(panelId) {
  let raw = null;
  try { raw = localStorage.getItem(ACK_KEY_PREFIX + panelId); } catch { return -Infinity; }
  const ts = raw === null ? NaN : Number(raw);
  return Number.isFinite(ts) ? ts : -Infinity;
}

/**
 * Clear a panel's badge and acknowledge it up to that badge's own server ts,
 * so a reload does not bring it back. With no ts on record the badge is still
 * cleared but the stored ack is left exactly as it was (see the module note).
 * @param {string} panelId
 */
export function clearBadge(panelId) {
  setEntryAttention(railEl, panelId, false);
  const ts = badgeTs.get(panelId);
  if (ts === undefined) return;
  try { localStorage.setItem(ACK_KEY_PREFIX + panelId, String(ts)); } catch { /* storage denied */ }
}

/**
 * Badge a panel's rail entry for a live agent_activity frame, remembering the
 * badge's server ts so clearing it can acknowledge exactly this event and the
 * reload restore can skip it.
 * @param {string} panelId
 * @param {number} [ts]
 * @returns {boolean} true when the rail anchored the badge; false means the
 *   caller still owns the frame (no entry for this id).
 */
export function badgePanelActivity(panelId, ts) {
  if (!setEntryAttention(railEl, panelId, true, ts)) return false;
  noteBadgeTs(panelId, ts);
  return true;
}

/**
 * Re-badge rail entries for panel activity this operator has not acknowledged.
 * Runs after the membership resync on every SSE open — including the first, so
 * a reload restores badges — which is also why it runs after it: an entry that
 * the resync delta just added can then carry a badge.
 *
 * Only 'panel'-kind rows with a ts can badge anything; the strip owns the rest
 * and history there is its own concern (the seam is not replayed).
 */
export async function restoreAgentBadges() {
  let body = null;
  try { body = await fetchJSON('/api/agent-activity/recent'); } catch { return; }
  for (const ev of (body?.events || [])) {
    const target = ev?.target;
    if (target?.kind !== 'panel' || !target.panel || typeof ev.ts !== 'number') continue;
    if (ev.ts <= ackedTs(target.panel)) continue;
    badgePanelActivity(target.panel, ev.ts);
  }
}
