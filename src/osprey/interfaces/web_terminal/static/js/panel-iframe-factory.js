// @ts-check
/* OSPREY Web Terminal — Service Panel Iframe Factory
 *
 * Builds a service panel's <iframe>, hands it to the dock adapter for
 * mounting/geometry, and wires its load-time theme/mode/session sync plus the
 * host-resize forwarding. Extracted from panel-manager.js (which keeps the
 * panel state machine and remains the only caller) to keep that module under
 * the max-lines cap; the split boundary is "how an iframe element is built"
 * vs "when one is needed".
 */

import { sendThemeToIframe, sendSessionToIframe, sendModeToIframe, buildEmbedSrc } from './panel-iframe-sync.js';
import { adoptIframe } from './dock-iframe.js';

/**
 * Structural subset of panel-manager's records this factory needs. `state` is
 * mutated in place: `pendingUrl` is consumed and `iframe` is assigned.
 * @typedef {{ id: string, label?: string, path?: string }} PanelSpec
 * @typedef {{ url: string | null, pendingUrl?: string | null, iframe: HTMLIFrameElement | null }} PanelIframeState
 */

/**
 * Create and adopt the iframe for a service panel. No-op when the panel has no
 * resolved URL yet (state.url null). The caller owns deciding WHEN (first
 * activation, pin materialization); everything about the element itself lives
 * here.
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
  const targetUrl = state.pendingUrl || (state.url + panelPath);
  state.pendingUrl = null;
  // targetUrl is the already-prefixed, root-relative `state.url` (+ optional
  // subpath); buildEmbedSrc preserves it verbatim.
  iframe.src = buildEmbedSrc(targetUrl);
  iframe.sandbox = 'allow-scripts allow-same-origin allow-popups allow-forms allow-modals';

  iframe.addEventListener('load', () => {
    iframe.classList.add('loaded');
    // Sync theme + mode + session immediately so there's no flash of the wrong
    // theme/mode and the embedded app scopes to the active session.
    sendThemeToIframe(iframe);
    sendModeToIframe(iframe);
    sendSessionToIframe(iframe);
  });

  // Hand the element to the dock adapter, which mounts it in the overlay layer
  // (dockview shell present) or in #panel-content (fallback). The adapter owns
  // the iframe's geometry from here; panel-manager keeps owning its src, sandbox,
  // health and the theme/mode/session postMessage protocol above.
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
