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
