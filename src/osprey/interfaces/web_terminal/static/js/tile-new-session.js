// @ts-check
/* OSPREY Web Terminal — Open in new session
 *
 * The docked tile menu's "Open in new session" row: it starts a new session
 * and opens the tile's pane in that session's workspace, beside the Agent
 * pane. A new session opens on the Agent pane alone (session-layout.js), and
 * `switchTo` takes no layout, so the pane is opened once the switch has
 * happened. The pane opens at its start page: its current page and zoom stay
 * with the session it was in. An artifact tile opens as a tile for the same
 * artifact; the new session does not own that artifact, so the tile shows its
 * origin notice (artifact-origin.js).
 *
 * The switch is the one every `+` makes (`switchTo({new: true})`). A live turn
 * asks first, and a declined question leaves everything as it was. A switch
 * that throws leaves the workspace unchanged and is logged the way the Agent
 * bar's `+` logs it.
 *
 * @module tile-new-session
 */

import { switchTo } from './session-model.js';
import { hasDockedTile, openPanelBeside } from './panel-placement.js';
import { artifactIdOf, artifactTileTitle, holdArtifactTile } from './artifact-tiles.js';
import { showUndoNotice } from './undo-notice.js';

/** @typedef {import('./panel-context-menu.js').MenuItem} MenuItem */

/** What the notice says when the new session has no room for the pane. */
export const OPEN_FAILED_MESSAGE = 'The new session could not open the pane.';

/**
 * Open a pane in the workspace of a new session.
 * @param {string} id  a service panel id, or an artifact tile's key
 * @param {Document} [noticeIn]  where a failed open says so
 * @returns {Promise<boolean>} whether the new session opened the pane
 */
export async function openInNewSession(id, noticeIn) {
  const artifactId = artifactIdOf(id);
  const title = artifactTileTitle(id) ?? '';
  try {
    if (!(await switchTo({ new: true }))) return false;
  } catch (err) {
    console.error('Failed to start new session:', err);
    return false;
  }
  try {
    if (artifactId !== null) holdArtifactTile({ id: artifactId, entry: null, title });
    openPanelBeside(id);
    if (hasDockedTile(id)) return true;
  } catch (err) {
    console.error('Failed to open the pane in the new session:', err);
  }
  showUndoNotice({ message: OPEN_FAILED_MESSAGE, doc: noticeIn });
  return false;
}

/**
 * The tile menu's row, for a docked service tile or artifact tile.
 * @param {string} id  a service panel id, or an artifact tile's key
 * @param {Document} [noticeIn]
 * @returns {MenuItem}
 */
export function openInNewSessionRow(id, noticeIn) {
  return {
    label: 'Open in new session',
    glyph: '⊕',
    run: () => void openInNewSession(id, noticeIn),
  };
}
