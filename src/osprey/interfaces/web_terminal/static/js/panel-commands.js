// @ts-check
/* OSPREY Web Terminal — Panel Commands
 *
 * Thin POST helpers for the panel visibility / registration endpoints, shared
 * by the tab strip's "×", the "+" add menu, and any other caller. Kept out of
 * panel-manager so that module stays focused on panel lifecycle and rendering.
 * Each issues its request and returns; the server's SSE echo drives the DOM, so
 * a human action and an agent MCP call are indistinguishable downstream.
 */

import { withPrefix } from './api.js';

/**
 * Show or hide a panel. Fire-and-forget: the panel_visibility SSE echo updates
 * every connected client's tab strip, so there is nothing to await here.
 * @param {string} panelId
 * @param {boolean} visible
 */
export function setPanelVisibility(panelId, visible) {
  fetch(withPrefix('/api/panel-visibility'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ panel: panelId, visible }),
  }).catch(() => {});
}

/**
 * Report a user-initiated tab switch so the server mirrors the active panel
 * (and does not echo a focus event back). Fire-and-forget.
 * @param {string} panelId
 */
export function setPanelFocus(panelId) {
  fetch(withPrefix('/api/panel-focus'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ panel: panelId }),
  }).catch(() => {});
}

/**
 * Report which service tiles this client currently has on screen, in spatial
 * reading order. A REPORT rather than a command: the server records it as the
 * workspace's `open_tiles` for the agent to read and broadcasts nothing, so one
 * operator's tile gestures never rearrange another's workspace. An empty list is
 * a meaningful report — every service tile closed — so a caller that cannot
 * determine occupancy must skip the call rather than send `[]`.
 *
 * Unlike the fire-and-forget commands above this resolves with the server's
 * acknowledgement, because the caller's dedupe baseline must track what the
 * server actually holds: a report the server never received must not be
 * remembered as sent, or the layout it described would be deduped away forever.
 * Never rejects — a failed or rejected report resolves null, which callers read
 * as "baseline unchanged, report again next time".
 * @param {string[]} tiles  service panel ids, left-to-right
 * @param {boolean} dock    whether this client has a dock shell; a false report
 *   carries no meaningful tile order, only the fact that a client is watching
 * @returns {Promise<{status: string, tiles: string[], dock: boolean,
 *   updated: boolean} | null>}  the acknowledgement — `updated: false` marks a
 *   server-side deduped no-op, which is still an acknowledgement — or null when
 *   the report did not land.
 */
export async function reportPanelLayout(tiles, dock) {
  try {
    const resp = await fetch(withPrefix('/api/panel-layout'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tiles, dock }),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

/**
 * Request a whole-workspace tile arrangement: exactly these service tiles open,
 * left to right. Fire-and-forget like the commands above — the server validates
 * the request and broadcasts one panel_arrange frame, and this client applies
 * the arrangement from that echo just like every other client, so a human
 * "Layouts" click and an agent arrange_workspace call are one operation.
 *
 * Pass exactly one of `tiles` and `preset`; a `preset` is resolved to its
 * members server-side and additionally prunes rail membership to them.
 * @param {{tiles?: string[], preset?: string, focus?: string}} request
 */
export function arrangePanels(request) {
  fetch(withPrefix('/api/panel-arrange'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  }).catch(() => {});
}

/**
 * Register a runtime URL panel. Returns the outcome so the caller (the "+" menu)
 * can surface the server's rejection reason (registration disabled, host not in
 * the allowlist, SSRF-blocked). The panel_register SSE echo adds the tab on
 * success; matching the agent-driven path, the new tab is not auto-activated.
 * @param {{id: string, label: string, url: string}} fields
 * @returns {Promise<{ok: boolean, error?: string}>}
 */
export async function registerUrlPanel({ id, label, url }) {
  try {
    const resp = await fetch(withPrefix('/api/panels/register'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, label, url, path: '/' }),
    });
    if (resp.ok) return { ok: true };
    let detail = `Could not add panel (${resp.status})`;
    try {
      const body = await resp.json();
      if (body && typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* non-JSON error body — keep the generic status message */
    }
    return { ok: false, error: detail };
  } catch {
    return { ok: false, error: 'Could not reach the server' };
  }
}
