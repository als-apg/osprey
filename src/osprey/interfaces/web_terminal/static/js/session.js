// @ts-check
/* OSPREY Web Terminal — Activity page: Entry Point
 *
 * Bootstraps session.html: theme and embedded-mode wiring, the page itself
 * (activity-log-page.js) on the session its own ?session_id= names, the
 * receiver that moves it when a host frame of the same origin names another
 * session (the receiver-rejects-foreign-origin contract is pinned by
 * test_contract_params.py), and the activity strip's SSE feed.
 *
 * @module session
 */

import { initTheme } from '/design-system/js/theme-manager.js';
import { applyEmbedded, isEmbedded } from '/design-system/js/frame-params.js';
import { createEventSource, pageEventsUrl } from './api.js';
import { bootActivityStrip } from './activity-strip.js';
import { AGENT_ACTIVITY_FRAME } from './activity-format.js';
import { isSessionKey, sessionIdFromQuery } from './activity-log-link.js';
import { mountActivityPage } from './activity-log-page.js';

/** @typedef {import('./panel-manager.js').AgentActivityEvent} AgentActivityEvent */

applyEmbedded();

// The page carries no theme control of its own: it shows the theme the
// terminal is in. Opened in its own tab it still runs theme-manager.js in the
// hub role, which is what reads the stored preference, follows the OS and
// honours ?theme=; broadcast is a structural no-op on a page with no iframes.
// Framed (?embedded=true) it is a follower instead: theme-boot.js already
// applied data-theme pre-paint, and this attaches the postMessage listener for
// the host's live broadcasts.
initTheme({ role: isEmbedded() ? 'follower' : 'hub' });

// Framed, the page is a narrow panel beside the terminal: the session list
// starts closed there and opens from its toolbar button.
const page = mountActivityPage({
  sessionKey: sessionIdFromQuery(window.location.search),
  sidebarOpen: !isEmbedded(),
});

// ---- Session-change receiver ----
window.addEventListener('message', (e) => {
  if (e.origin !== window.location.origin) return;
  if (!e.data) return;
  if (e.data.type === 'osprey-session-change') {
    page.follow(isSessionKey(e.data.session_id) ? e.data.session_id : null);
  }
});

// ---- Activity strip ----

/**
 * Feed the activity strip from the web terminal's SSE stream.
 *
 * The strip module self-registers on panel-manager's activity seam, but no
 * panel-manager runs on this page, so nothing would ever drive it here — this
 * is the session page's own subscription to the same `/api/files/events`
 * stream the terminal reads. Only `agent_activity` frames are forwarded; file
 * and panel frames on the shared stream belong to the terminal. A frame that
 * failed to parse arrives as the raw string (createEventSource's fallback) and
 * is ignored, as is one without a `target` — the same guard panel-sse's frame
 * dispatcher applies.
 *
 * Prefixing, backoff and reconnection are createEventSource's job.
 *
 * @param {{handleActivity: (frame: AgentActivityEvent) => void}} strip
 * @returns {{stop: () => void}}
 */
export function wireActivityStrip(strip) {
  return createEventSource(pageEventsUrl(), {
    onMessage: (data) => {
      if (!data || typeof data !== 'object') return;
      if (data.type !== AGENT_ACTIVITY_FRAME || !data.target) return;
      strip.handleActivity(data);
    },
  });
}

// The strip module boots itself on this page's mount too; bootActivityStrip
// is idempotent, so this reaches that same instance rather than binding a
// second strip to the shared mount.
const activityStrip = bootActivityStrip();
if (activityStrip) wireActivityStrip(activityStrip);
