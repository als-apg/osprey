// @ts-check
/* OSPREY Web Terminal — Panel Iframe State Sync
 *
 * The per-iframe "push host state into a panel" seam: theme, active session,
 * and UI mode, each serialized to a panel iframe via postMessage. Extracted
 * from panel-manager so that module stays focused on panel lifecycle, rail
 * rendering, and health — these three helpers share one shape (take an
 * iframe, guard its contentWindow, swallow the cross-origin throw) and one
 * concern (host → follower messaging), and depend on none of panel-manager's
 * private panel state, so they live cleanly on their own.
 *
 * The hub-role fan-outs that iterate every panel (theme-manager's own theme
 * broadcast; panel-manager's broadcastMode) stay with their owners — this
 * module is only the single-iframe trigger points (iframe creation and tab
 * activation) those fan-outs and panel-manager call into.
 */

import { getTheme } from '/design-system/js/theme-manager.js';
import { getCurrentSessionId } from './terminal.js';

/**
 * The shared single-iframe send: guard a missing contentWindow (a not-yet-loaded
 * or detached iframe) and swallow the cross-origin postMessage throw. Every
 * host → follower message below funnels through here.
 * @param {HTMLIFrameElement | null} iframe
 * @param {object} message
 */
function postToIframe(iframe, message) {
  if (!iframe?.contentWindow) return;
  try {
    iframe.contentWindow.postMessage(message, window.location.origin);
  } catch { /* cross-origin */ }
}

/**
 * Send the hub's current theme to one iframe. Always sends — never
 * conditioned on whether the id differs from the last send — because a
 * hidden iframe can read empty custom properties on Firefox and needs a
 * fresh, unconditional resend once it's visible again (theme-manager's
 * hidden-iframe protocol; see that module's docstring). Broadcasting to
 * every iframe on every hub theme change is theme-manager's own job (hub
 * role); this is only the two per-iframe trigger points panel-manager
 * owns: iframe creation and tab activation.
 *
 * 'osprey-theme-change' is the one message type theme-manager's follower
 * role (and every embedded interface) actually listens for.
 * @param {HTMLIFrameElement | null} iframe
 */
export function sendThemeToIframe(iframe) {
  postToIframe(iframe, { type: 'osprey-theme-change', theme: getTheme() });
}

/**
 * Send the hub's active session id to one iframe — the twin of
 * sendThemeToIframe(), owned by the same two per-iframe trigger points
 * (iframe creation and tab activation). Guards on a missing contentWindow
 * (a not-yet-loaded or detached iframe) and on there being no active
 * session yet, and swallows the cross-origin postMessage throw the same
 * way its theme twin does.
 *
 * 'osprey-session-change' is the message type embedded interfaces listen
 * for to scope their view to the hub's active session.
 * @param {HTMLIFrameElement | null} iframe
 */
export function sendSessionToIframe(iframe) {
  const sid = getCurrentSessionId();
  if (!sid) return;
  postToIframe(iframe, { type: 'osprey-session-change', session_id: sid });
}

// ---- UI Mode Sync ----
//
// The mode axis (expert|simple) has no manager module the way the theme axis
// does — mode-boot.js resolves it pre-paint and the live value simply lives on
// <html data-ui-mode>. Read it straight from there, defaulting to 'expert' so
// the broadcast payload is never the empty/missing string consumers ignore.

/** @returns {string} */
export function getMode() {
  return document.documentElement.getAttribute('data-ui-mode') || 'expert';
}

/**
 * Send the hub's current UI mode to one iframe — the mode-axis twin of
 * sendThemeToIframe(), owned by the same two per-iframe trigger points (iframe
 * creation and tab activation). Always sends unconditionally for the same
 * hidden-iframe reason its theme twin does, and swallows the cross-origin
 * postMessage throw the same way.
 *
 * 'osprey-mode-change' is the message type embedded interfaces listen for to
 * flip their layout between Expert and Simple.
 * @param {HTMLIFrameElement | null} iframe
 */
export function sendModeToIframe(iframe) {
  postToIframe(iframe, { type: 'osprey-mode-change', mode: getMode() });
}
