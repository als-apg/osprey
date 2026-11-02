// @ts-check
/* OSPREY Web Terminal — Service Panel Iframe Factory
 *
 * Builds a service panel's <iframe>, hands it to the dock adapter for
 * mounting/geometry, and wires its load-time theme/session sync plus the
 * host-resize forwarding. Extracted from panel-manager.js (which keeps the
 * panel state machine and remains the only caller) to keep that module under
 * the max-lines cap; the split boundary is "how an iframe element is built"
 * vs "when one is needed".
 */

import { sendThemeToIframe, sendSessionToIframe, buildEmbedSrc } from './panel-iframe-sync.js';
import { adoptIframe, setPanelReachable } from './dock-iframe.js';
import { clockTime } from './panel-rail.js';
import { withPrefix } from './api.js';

/**
 * The sandbox an OSPREY-shipped (first-party) panel is framed under — today's
 * attributes, unchanged: it is the deployment's own code and may use its
 * origin's APIs.
 */
export const FIRST_PARTY_SANDBOX =
  'allow-scripts allow-same-origin allow-popups allow-forms allow-modals allow-downloads';

/**
 * The sandbox an agent-writable panel (a discovered `<project>/panels` bundle or
 * a runtime registration) is framed under: no `allow-same-origin`, so it runs at
 * an opaque origin and the operator's cookie never rides its requests, and no
 * `allow-forms`, so it cannot submit a form to an operator route either.
 */
export const OPAQUE_SANDBOX = 'allow-scripts allow-popups allow-modals allow-downloads';

/**
 * Ask the hub whether a panel is opaque, and if so where to frame it.
 *
 * Whether a panel is opaque is decided by where its bytes come from, and only
 * the server knows that (a discovered bundle and a config-defined custom panel
 * look alike here), so the answer is the C-CAP mint: `200 {cap_url}` is an
 * opaque panel and the capability URL minted for this login, `404` is a
 * first-party panel. Anything else — a network failure, a 401/5xx, a body with
 * no `cap_url` — fails closed: the panel is treated as opaque but has no cap.
 * @param {string} panelId
 * @returns {Promise<{ opaque: boolean, capUrl: string | null }>}
 */
export async function resolvePanelFrame(panelId) {
  try {
    const resp = await fetch(withPrefix(`/api/panels/${encodeURIComponent(panelId)}/cap`), {
      cache: 'no-store',
    });
    if (resp.status === 404) return { opaque: false, capUrl: null };
    if (!resp.ok) return { opaque: true, capUrl: null };
    const body = await resp.json();
    const capUrl = typeof body?.cap_url === 'string' ? body.cap_url : null;
    return { opaque: true, capUrl };
  } catch {
    return { opaque: true, capUrl: null };
  }
}

/**
 * Rebase a panel URL onto its cap: a target under the panel's plain base
 * (`state.url`) keeps its remainder under `capUrl` (which ends in `/`), with no
 * doubled slash. A target outside that base stays as it is.
 * @param {string} targetUrl
 * @param {string} baseUrl
 * @param {string} capUrl
 * @returns {string}
 */
export function capTarget(targetUrl, baseUrl, capUrl) {
  const base = baseUrl.replace(/\/+$/, '');
  if (targetUrl !== base && !targetUrl.startsWith(`${base}/`)) return targetUrl;
  const rest = targetUrl.slice(base.length).replace(/^\/+/, '');
  return capUrl + rest;
}

/**
 * How long before its cap expires an open opaque panel re-mints it. The cap
 * lives CAP_TTL_SECONDS (12 h, panel_cap.py); renewing a few minutes early
 * keeps a panel left open loading through a valid cap without changing it.
 */
export const CAP_RENEW_MARGIN_MS = 5 * 60 * 1000;

/**
 * The shortest wait before a renewal. A cap that already reads as expired (a
 * client clock ahead of the hub's) renews after this, never in a tight loop.
 */
export const CAP_RENEW_FLOOR_MS = 30 * 1000;

/**
 * When a cap URL's cap expires, in epoch milliseconds, or null when the URL
 * carries no cap. The cap segment is `<expiry-hex>.<login-tag>.<signature>`
 * (panel_cap.mint_cap), so the expiry is readable here without asking the hub.
 * @param {string} capUrl
 * @returns {number | null}
 */
export function capExpiryMs(capUrl) {
  const m = /\/~c\/([0-9a-f]+)\./.exec(capUrl);
  return m ? parseInt(m[1], 16) * 1000 : null;
}

/**
 * Per-iframe framing record: `seq` orders resolutions so a late answer for a
 * superseded navigation is dropped, `timer` is the pending cap renewal.
 * @type {WeakMap<HTMLIFrameElement, { seq: number, timer: ReturnType<typeof setTimeout> | null }>}
 */
const frames = new WeakMap();

/**
 * Point a panel's iframe at `targetUrl` (a plain, already-prefixed panel URL).
 *
 * The one path every navigation takes — first load, a re-navigation, the
 * address a recovered panel was holding, and a cap renewal — so an opaque
 * panel is always framed through a cap freshly minted by
 * {@link resolvePanelFrame}, never at its bare URL. A catalog panel (one with a
 * `configEndpoint`) is first-party and framed at the plain URL with no request.
 * An opaque panel that got a cap renews it {@link CAP_RENEW_MARGIN_MS} before
 * it expires by framing the same target again; a frame that has left the
 * document stops renewing.
 * @param {PanelSpec | undefined} panel
 * @param {string} panelId
 * @param {HTMLIFrameElement} iframe
 * @param {string} targetUrl
 * @param {string} baseUrl                 the panel's plain base (`state.url`)
 */
export function framePanel(panel, panelId, iframe, targetUrl, baseUrl) {
  const rec = frames.get(iframe) ?? { seq: 0, timer: null };
  frames.set(iframe, rec);
  rec.seq += 1;
  if (rec.timer !== null) clearTimeout(rec.timer);
  rec.timer = null;

  if (panel?.configEndpoint) {
    iframe.sandbox = FIRST_PARTY_SANDBOX;
    iframe.src = buildEmbedSrc(targetUrl);
    return;
  }
  const seq = rec.seq;
  iframe.dataset.panelFrame = 'pending';
  resolvePanelFrame(panelId).then(({ opaque, capUrl }) => {
    if (rec.seq !== seq) return;
    if (!opaque) {
      iframe.sandbox = FIRST_PARTY_SANDBOX;
      iframe.dataset.panelFrame = 'first-party';
      iframe.src = buildEmbedSrc(targetUrl);
      return;
    }
    iframe.sandbox = OPAQUE_SANDBOX;
    iframe.dataset.panelFrame = 'opaque';
    iframe.src = buildEmbedSrc(capUrl ? capTarget(targetUrl, baseUrl, capUrl) : targetUrl);
    const expiry = capUrl ? capExpiryMs(capUrl) : null;
    if (expiry === null) return;
    const delay = Math.max(expiry - CAP_RENEW_MARGIN_MS - Date.now(), CAP_RENEW_FLOOR_MS);
    rec.timer = setTimeout(() => {
      rec.timer = null;
      if (rec.seq !== seq || !iframe.isConnected) return;
      framePanel(panel, panelId, iframe, targetUrl, baseUrl);
    }, delay);
  });
}

/**
 * How long a panel's page may take to load before its tile says the backend
 * is not answering. An iframe reports no error for a page that never arrives
 * (the request can sit in the browser's queue behind its per-host connection
 * cap), so without this the tile is a blank rectangle with no explanation.
 */
const LOAD_NOTICE_MS = 15000;

/**
 * Structural subset of panel-manager's records this factory needs. `state` is
 * mutated in place: `pendingUrl` is consumed and `iframe` is assigned.
 * @typedef {{ id: string, label?: string, path?: string, configEndpoint?: string | null }} PanelSpec
 * @typedef {{ url: string | null, pendingUrl?: string | null, iframe: HTMLIFrameElement | null }} PanelIframeState
 */

/**
 * Create and adopt the iframe for a service panel. No-op when the panel has no
 * resolved URL yet (state.url null). The caller owns deciding WHEN (first
 * activation, pin materialization); everything about the element itself lives
 * here.
 *
 * The src is set by {@link framePanel}: a catalog panel at once under
 * {@link FIRST_PARTY_SANDBOX}, any other panel under {@link OPAQUE_SANDBOX}
 * with no src until the hub has said what it is. The element itself exists
 * synchronously either way, so callers keep finding it in `state.iframe`.
 * @param {PanelSpec | undefined} panel   the PANELS registry entry (for path/label)
 * @param {string} panelId
 * @param {PanelIframeState} state        panel-manager's live state record
 * @param {HTMLElement} contentEl         resize-observed host (#panel-content)
 */
export function createPanelIframe(panel, panelId, state, contentEl) {
  if (!state.url) return;

  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  iframe.dataset.panelId = panelId;
  // Use pendingUrl (from a pre-iframe panel_focus navigation) if available,
  // otherwise base URL. For custom panels with a subpath (e.g. path:
  // "/panel/"), append it so the iframe loads the UI root, not the API root.
  const panelPath = panel?.path && panel.path !== '/' ? panel.path : '';
  const targetUrl = state.pendingUrl || state.url + panelPath;
  state.pendingUrl = null;
  // targetUrl is the already-prefixed, root-relative `state.url` (+ optional
  // subpath); buildEmbedSrc preserves it verbatim.
  // Restrictive until the hub has answered; the sandbox in force at the first
  // navigation is what the document gets, so it is set before any src.
  // framePanel loosens it for a panel the hub says is first-party.
  iframe.sandbox = OPAQUE_SANDBOX;
  framePanel(panel, panelId, iframe, targetUrl, state.url);

  // Cleared by the first load; the notice it raised, if any, goes with it.
  const loadStartedAt = Date.now();
  let stalled = false;
  const stallTimer = setTimeout(() => {
    stalled = true;
    setPanelReachable(panelId, false, clockTime(loadStartedAt));
  }, LOAD_NOTICE_MS);

  iframe.addEventListener('load', () => {
    // The about:blank an src-less iframe fires on attach is not the panel.
    if (iframe.dataset.panelFrame === 'pending') return;
    clearTimeout(stallTimer);
    if (stalled) {
      stalled = false;
      setPanelReachable(panelId, true);
    }
    iframe.classList.add('loaded');
    // Sync theme + session immediately so there's no flash of the wrong theme
    // and the embedded app scopes to the active session.
    sendThemeToIframe(iframe);
    sendSessionToIframe(iframe);
  });

  // Hand the element to the dock adapter, which mounts it in the overlay layer
  // (dockview shell present) or in #panel-content (fallback). The adapter owns
  // the iframe's geometry from here; panel-manager keeps owning its src, sandbox,
  // health and the theme/session postMessage protocol above.
  state.iframe = iframe;
  adoptIframe(panelId, iframe, { title: panel?.label || panelId });

  // Forward resize events to the iframe so embedded apps re-render. Observing
  // #panel-content covers the fallback (in-content) mount and the workspace-panel
  // resizes; the adapter additionally re-dispatches resize when it re-lays the
  // overlay iframe to a new dock rectangle.
  const observer = new ResizeObserver(() => {
    if (iframe.contentWindow) {
      try {
        iframe.contentWindow.dispatchEvent(new Event('resize'));
      } catch {
        // cross-origin — nothing we can do
      }
    }
  });
  observer.observe(contentEl);
}
