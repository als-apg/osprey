// @ts-check
/* OSPREY Web Terminal — the Artifacts panel as the session tree drives it
 *
 * The session tree (sidebar-artifacts.js) decides which artifact is shown;
 * this module carries that to the panel: the iframe is told which artifact
 * to show and when it changed, the tile menu gets the shown artifact's verbs,
 * and the panel is asked whether it is on view and brought up through the
 * open rule. It keeps no state: the tree passes what it shows.
 *
 * @module sidebar-artifact-viewer
 */

import * as iframeSync from './panel-iframe-sync.js';
import { activateTab } from './panel-manager.js';
import { hasDockedTile } from './panel-placement.js';
import { isDetached } from './dock-apply.js';
import { managedIframe } from './dock-iframe.js';
import { menuItems } from './artifacts-pane-header.js';

/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */

/** The panel id of the artifact gallery. */
export const ARTIFACTS_PANEL_ID = 'artifacts';

/**
 * The Artifacts panel's frame, wherever it is: Open in new window moves the
 * element into the pane's window, out of this document, and its ready message
 * reaches the hub from there.
 * @returns {HTMLIFrameElement | null}
 */
export function artifactsIframe() {
  return managedIframe(ARTIFACTS_PANEL_ID)?.iframe ?? null;
}

/**
 * Tell the viewer which artifact to show.
 * @param {string | null} shownId
 */
export function sendSelect(shownId) {
  const iframe = artifactsIframe();
  if (iframe) iframeSync.sendArtifactSelectToIframe(iframe, shownId);
}

/**
 * Tell the viewer the artifact it shows was updated or deleted.
 * @param {string | null} shownId
 * @param {'updated' | 'deleted'} change
 * @param {ArtifactEntry} [artifact]
 */
export function sendChanged(shownId, change, artifact) {
  const iframe = artifactsIframe();
  if (iframe && shownId) iframeSync.sendArtifactChangedToIframe(iframe, shownId, change, artifact);
}

/**
 * Whether a window message is the Artifacts iframe saying it is ready.
 * @param {MessageEvent} event
 */
export function isViewerReady(event) {
  if (event.origin !== window.location.origin) return false;
  if (event.data?.type !== 'osprey-artifact-ready') return false;
  const iframe = artifactsIframe();
  return !!iframe && event.source === iframe.contentWindow;
}

/**
 * The tile menu's rows for the shown artifact; none without one or a viewer.
 * @param {ArtifactEntry | null} shown
 */
export function viewerMenuItems(shown) {
  if (!shown || !artifactsIframe()) return [];
  return menuItems(shown, (action) =>
    iframeSync.sendArtifactActionToIframe(artifactsIframe(), shown.id, action)
  );
}

/** Whether the Artifacts panel is on view: docked in a tile, or in its own window. */
export const panelOnView = () =>
  hasDockedTile(ARTIFACTS_PANEL_ID) || isDetached(ARTIFACTS_PANEL_ID);

/** Show the Artifacts panel through the open rule: a docked one is focused in place. */
export function showArtifactsPanel() {
  if (!hasDockedTile(ARTIFACTS_PANEL_ID)) activateTab(ARTIFACTS_PANEL_ID, { userInitiated: true });
}
