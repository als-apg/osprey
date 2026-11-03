// @ts-check
/* OSPREY Web Terminal — in-place navigation of a panel document
 *
 * A panel page that can route a url itself (CANOPY's viewer) says so with
 * `canopy-host-navigate-ready`; the hub then navigates it by message, so the
 * panel keeps its tabs and state, instead of reloading the iframe.
 *
 * Readiness is independent of event order: on every iframe load the hub
 * forgets the previous document's readiness and probes the new one; a panel
 * that handles `canopy-host-navigate` answers `canopy-host-navigate-ready` on
 * mount AND in reply to every probe. A ready that lands before the hub's load
 * handler is cleared by it, and the probe the same handler posts earns a
 * fresh ready.
 */

import { isNavigableUrl } from './panel-iframe-sync.js';

const NAVIGATE_MESSAGE = 'canopy-host-navigate';
const NAVIGATE_PROBE = 'canopy-host-navigate-probe';
// The hub sends only `canopy-host-*`; it also reads a viewer's old
// `osprey-navigate-ready` until canopy-screen-runtime 0.2.0 drops that name.
const NAVIGATE_READY = new Set(['canopy-host-navigate-ready', 'osprey-navigate-ready']);

/**
 * The part of a panel's state this module reads and writes.
 * @typedef {{ iframe: HTMLIFrameElement | null, inPlace?: boolean, pendingUrl?: string | null }} NavigableState
 */

/** Every panel state whose iframe is watched; the receiver looks the sender up here. */
const watched = new Set();
let wired = false;

/**
 * Hub-side receiver: mark the sending panel in-place navigable. Only a
 * same-origin message whose source is a watched panel's own iframe counts.
 */
function wireNavigateReady() {
  if (wired) return;
  wired = true;
  window.addEventListener('message', (e) => {
    if (e.origin !== window.location.origin) return;
    if (!e.data || !NAVIGATE_READY.has(e.data.type) || !e.source) return;
    for (const state of watched) {
      if (state.iframe?.contentWindow && state.iframe.contentWindow === e.source) {
        state.inPlace = true;
        return;
      }
    }
  });
}

/**
 * Probe every document `state`'s iframe loads, starting from not ready, and
 * take its ready messages from then on.
 * @param {NavigableState} state
 */
export function watchNavigateReady(state) {
  const iframe = state.iframe;
  if (!iframe) return;
  wireNavigateReady();
  watched.add(state);
  state.inPlace = false;
  iframe.addEventListener('load', () => probeNavigateReady(state, iframe));
}

/**
 * A fresh document in `iframe` is not yet known to handle
 * `canopy-host-navigate`: clear the flag and ask it.
 * @param {NavigableState} state
 * @param {HTMLIFrameElement} iframe
 */
export function probeNavigateReady(state, iframe) {
  if (state.iframe !== iframe) return;
  state.inPlace = false;
  if (!iframe.contentWindow) return;
  try {
    iframe.contentWindow.postMessage({ type: NAVIGATE_PROBE }, window.location.origin);
  } catch {
    /* cross-origin document — it cannot answer, so it stays src-navigated */
  }
}

/**
 * Navigate a ready document to `url` by message, and clear the panel's pending
 * url. The url must pass the guard a src would (buildEmbedSrc refuses what
 * isNavigableUrl refuses). Returns whether it was sent; on `false` the caller
 * loads the url as a src, and the new document has to announce itself again.
 * @param {NavigableState} state
 * @param {string} url
 * @returns {boolean}
 */
export function navigateInPlace(state, url) {
  if (state.inPlace && state.iframe?.contentWindow && isNavigableUrl(url)) {
    try {
      state.iframe.contentWindow.postMessage(
        { type: NAVIGATE_MESSAGE, url },
        window.location.origin
      );
      state.pendingUrl = null;
      return true;
    } catch {
      /* cross-origin document — fall back to a src load */
    }
  }
  state.inPlace = false;
  return false;
}
