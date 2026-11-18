// @ts-check
/* OSPREY Web Terminal — Open in new session
 *
 * The "Open in new session" row of a docked tile's menu, of a service panel's
 * sidebar row and of an artifact row in the session tree: it starts a new
 * session and opens one pane in that session's workspace, on the left of the
 * Agent pane. A new session opens on the Agent pane alone
 * (session-layout.js), and `switchTo` takes no layout, so the pane is opened
 * once the switch has happened. The pane opens at its start page: its current
 * page and zoom stay with the session it was in. An artifact opens as an
 * artifact tile; the new session does not own that artifact, so the tile shows
 * its origin notice (artifact-origin.js).
 *
 * The switch is the one every `+` makes (`switchTo({new: true})`). A live turn
 * asks first, and a declined question leaves everything as it was. A switch
 * that throws leaves the workspace unchanged and is logged the way the Agent
 * bar's `+` logs it.
 *
 * @module tile-new-session
 */

import { switchTo } from './session-model.js';
import { hasDockedTile, openPanelLeftOfAgent } from './panel-placement.js';
import { artifactIdOf, artifactTileKey, holdArtifactTile } from './artifact-tiles.js';
import { showUndoNotice } from './undo-notice.js';

/** @typedef {import('./panel-context-menu.js').MenuItem} MenuItem */
/** @typedef {import('./artifacts-pane-header.js').ArtifactEntry} ArtifactEntry */

/** What the notice says when the new session has no room for the pane. */
export const OPEN_FAILED_MESSAGE = 'The new session could not open the pane.';

/**
 * Start a new session, then run `place`, which opens the pane and returns its
 * key. A pane the dock does not hold afterwards, or a `place` that throws,
 * leaves the new session on the Agent pane with a notice.
 * @param {() => string | null} place
 * @param {Document} [noticeIn]  where a failed open says so
 * @returns {Promise<boolean>} whether the new session opened the pane
 */
async function inNewSession(place, noticeIn) {
  try {
    if (!(await switchTo({ new: true }))) return false;
  } catch (err) {
    console.error('Failed to start new session:', err);
    return false;
  }
  try {
    const key = place();
    if (key !== null && hasDockedTile(key)) return true;
  } catch (err) {
    console.error('Failed to open the pane in the new session:', err);
  }
  showUndoNotice({ message: OPEN_FAILED_MESSAGE, doc: noticeIn });
  return false;
}

/**
 * Open a pane in the workspace of a new session.
 * @param {string} id  a service panel id, or an artifact tile's key
 * @param {Document} [noticeIn]
 * @returns {Promise<boolean>}
 */
export function openInNewSession(id, noticeIn) {
  return inNewSession(() => {
    const artifactId = artifactIdOf(id);
    if (artifactId !== null) holdArtifactTile({ id: artifactId, entry: null, title: '' });
    openPanelLeftOfAgent(id);
    return id;
  }, noticeIn);
}

/**
 * Open an artifact, as listed in the session tree, in a new session's
 * workspace as an artifact tile, with the entry and title the row holds.
 * @param {{id: string, entry: ArtifactEntry | null, title: string}} artifact
 * @param {Document} [noticeIn]
 * @returns {Promise<boolean>}
 */
export function openArtifactInNewSession({ id, entry, title }, noticeIn) {
  return inNewSession(() => {
    const key = holdArtifactTile({ id, entry, title });
    if (key === null) return null;
    openPanelLeftOfAgent(artifactTileKey(id));
    return key;
  }, noticeIn);
}

/**
 * The menu row for a docked service tile, a service panel's sidebar row, or an
 * artifact tile.
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
