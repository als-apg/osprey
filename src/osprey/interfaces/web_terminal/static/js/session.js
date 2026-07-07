// @ts-check
/* OSPREY Web Terminal — Session Activity Log: Entry Point
 *
 * Bootstraps session.html: theme + embedded-mode wiring, the same-origin
 * osprey-session-change receiver (the receiver-rejects-foreign-origin
 * contract is pinned by test_contract_params.py), the four-view nav, the
 * shared api/toast helpers the view renderers (session-views.js) depend
 * on, and the periodic refresh loop.
 *
 * @module session
 */

import { initTheme } from '/design-system/js/theme-manager.js';
import { applyEmbedded } from '/design-system/js/frame-params.js';
import '/design-system/js/components/osprey-theme-switcher.js';
import { renderAgents, renderToolLog, renderArtifacts, renderConversation } from './session-views.js';

/** @typedef {'agents'|'toollog'|'artifacts'|'conversation'} ViewName */
/**
 * @typedef {{
 *   agents: unknown,
 *   toollog: unknown,
 *   artifacts: unknown,
 *   conversation: unknown,
 * }} SessionCache
 */

applyEmbedded();

// Follower role: session.html is opened directly (new tab) or embedded
// read-only; it never persists a preference or broadcasts — it only
// applies whatever theme-boot.js already resolved pre-paint, a validated
// ?theme= query param, and whatever the hub broadcasts if this page is
// ever embedded. Manifest validation here is what closes the arbitrary
// data-theme injection hole the old hand-rolled ?theme= handling had.
//
// The osprey-theme-switcher.js side-effect import above registers the
// custom element before this call runs: module imports always evaluate
// before any of the importing module's own top-level statements, and a
// deferred `type="module"` script only runs after the DOM (including the
// <osprey-theme-switcher> tag in session.html's header) has been parsed
// and connected — so the button already exists by the time initTheme()
// wires its click handler.
initTheme({ role: 'follower' });

// ---- State ----
/** @type {ViewName} */
let activeView = 'agents';
/** @type {string|null} */
let currentSessionId = null;
/** @type {SessionCache} */
let cache = { agents: null, toollog: null, artifacts: null, conversation: null };

// ---- API ----
const FETCH_TIMEOUT = 4000;

/**
 * Fetch a same-origin JSON API endpoint, appending the current session id
 * (if any) as a query param. A 404 resolves to `null` (every view renderer
 * treats that as "no data yet", not an error); any other non-OK response
 * or network failure throws/rejects.
 *
 * @param {string} path
 * @returns {Promise<any>}
 */
async function apiFetch(path) {
  if (currentSessionId) {
    const sep = path.includes('?') ? '&' : '?';
    path = path + sep + 'session_id=' + encodeURIComponent(currentSessionId);
  }
  const resp = await fetch(path, { signal: AbortSignal.timeout(FETCH_TIMEOUT) });
  if (resp.status === 404) return null;
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json();
}

// ---- Toast ----
/** @type {ReturnType<typeof setTimeout>|undefined} */
let toastTimer;

/** @param {string} msg */
function showToast(msg) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 4000);
}

// ---- Nav ----
const navEl = document.getElementById('nav');
if (navEl) {
  navEl.addEventListener('click', (e) => {
    const target = /** @type {HTMLElement|null} */ (e.target);
    const pill = target ? target.closest('.pill') : null;
    if (!(pill instanceof HTMLElement)) return;
    const view = /** @type {ViewName|undefined} */ (pill.dataset.view);
    if (!view || view === activeView) return;
    activeView = view;
    const pills = /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll('.pill'));
    pills.forEach((p) => p.classList.toggle('active', p.dataset.view === view));
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + view));
    refreshActive();
  });
}

// ---- Refresh Logic ----
/**
 * @typedef {{
 *   apiFetch: (path: string) => Promise<any>,
 *   showToast: (msg: string) => void,
 *   cache: SessionCache,
 * }} RenderCtx
 */

/** @type {Record<ViewName, (ctx: RenderCtx) => Promise<void>>} */
const VIEWS = { agents: renderAgents, toollog: renderToolLog, artifacts: renderArtifacts, conversation: renderConversation };

async function refreshActive() {
  const dot = document.getElementById('refresh-dot');
  if (!dot) return;
  dot.classList.remove('pulse');
  void dot.offsetWidth;  // reflow
  dot.classList.add('pulse');
  try {
    await VIEWS[activeView]({ apiFetch, showToast, cache });
  } catch (err) {
    showToast('Refresh failed');
  }
}

// ---- Session-change receiver ----
window.addEventListener('message', (e) => {
  if (e.origin !== window.location.origin) return;
  if (!e.data) return;
  if (e.data.type === 'osprey-session-change') {
    currentSessionId = e.data.session_id || null;
    cache = { agents: null, toollog: null, artifacts: null, conversation: null };
    refreshActive();
  }
});

refreshActive();

setInterval(() => {
  if (document.hidden) return;
  refreshActive();
}, 12000);
