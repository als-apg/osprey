// @ts-check
/* OSPREY Web Terminal — Panel Commands
 *
 * Thin POST helpers for the panel endpoints, shared by the tab strip's "×",
 * the "+" add menu, the Layouts presets, the dock op runner and any other
 * caller. Kept out of panel-manager so that module stays focused on panel
 * lifecycle and rendering. Rail commands issue their request and the server's
 * SSE echo drives the DOM; dock ops go through the same `/api/panel-op/*`
 * routes an agent uses, so a human action and an agent MCP call are
 * indistinguishable downstream.
 */

import { withPrefix } from './api.js';

/**
 * Show or hide a panel. The panel_visibility SSE echo updates every connected
 * client's rail, so the DOM never waits on this call and most callers drop the
 * result. It resolves whether the server accepted the change, for the caller
 * that offers an Undo: a removal that never happened has nothing to take back.
 * Never rejects.
 *
 * `index` places an ADDED panel at that position among the rail's panel
 * entries (the Agent entry not counted) — how an Undo puts a removed entry
 * back where the operator saw it. Without one an addition goes to the end. The
 * server ignores it when hiding and when the panel is already on the rail.
 * @param {string} panelId
 * @param {boolean} visible
 * @param {number} [index]
 * @returns {Promise<boolean>}
 */
export async function setPanelVisibility(panelId, visible, index) {
  const body =
    index === undefined ? { panel: panelId, visible } : { panel: panelId, visible, index };
  try {
    const resp = await fetch(withPrefix('/api/panel-visibility'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

/**
 * Move a rail member directly above `beforeId`, or to the foot of the rail when
 * null. Like a membership change, the rail itself only moves on the server's
 * `rail_order` echo, so every client moves together.
 * @param {string} panelId
 * @param {string | null} beforeId
 * @returns {Promise<boolean>} whether the server accepted the move
 */
export async function setRailOrder(panelId, beforeId) {
  try {
    const resp = await fetch(withPrefix('/api/rail-order'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ panel: panelId, before: beforeId }),
    });
    return resp.ok;
  } catch {
    return false;
  }
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
 * POST a JSON body and resolve the parsed answer, or null when the request did
 * not land (network failure, non-2xx, non-JSON body). Never rejects.
 * @param {string} path
 * @param {unknown} body
 * @returns {Promise<any | null>}
 */
async function postJSON(path, body) {
  try {
    const resp = await fetch(withPrefix(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

/** A fresh request id for one dock op. */
function newRequestId() {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `r-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Request one dock op through the same route an agent uses
 * (`/api/panel-op/<route>`, e.g. `open-arrangement`). The server relays it to
 * the current dock client as a `panel_op` frame — possibly this page — whose
 * op runner applies it like any agent op, so a person's preset click and an
 * agent's arrange_workspace are one operation. Resolves the server's answer
 * (`{status: 'ok', layout, rev}`), or null for a refusal or a request that did
 * not land. Never rejects.
 * @param {string} route  the op route segment (`drop`, `open-arrangement`, ...)
 * @param {Record<string, unknown>} body  the op's arguments, without `request_id`
 * @returns {Promise<{status: string, layout?: unknown, rev?: number} | null>}
 */
export function requestPanelOp(route, body) {
  return postJSON(`/api/panel-op/${route}`, { request_id: newRequestId(), ...body });
}

/**
 * Answer one `panel_op` frame: the layout the op left, or a named refusal.
 * Resolves the server's acknowledgement — `rev` is the dock's new ack baseline
 * — or null when the answer did not land. Never rejects.
 * @param {{request_id: string, client_id: string, layout?: unknown, popouts?: string[],
 *   refusal?: string, detail?: string}} answer
 * @returns {Promise<{status: string, layout?: unknown, popouts?: string[], rev?: number} | null>}
 */
export function postPanelOpResult(answer) {
  return postJSON('/api/panel-op-result', answer);
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
