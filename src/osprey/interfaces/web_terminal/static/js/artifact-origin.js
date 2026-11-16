// @ts-check
/* OSPREY Web Terminal — the session an artifact comes from
 *
 * An artifact can show in a session it was not made in: an artifact tile
 * (artifact-tiles.js) dragged in from another session's rows, or the
 * Artifacts tile showing a row picked in another, expanded session
 * (sidebar-artifacts.js). Such an artifact is FOREIGN: its entry names an
 * owning session (`owner`, a transcript id) that is not the active session's
 * transcript (session-model.js `activeRow`). An artifact with no owner never
 * is; on a new session with no transcript yet, every owned artifact is.
 *
 * The Info card of the tile (artifacts-pane-header.js) names the owning
 * session in a `Session` row, a link that switches to it unless it is the
 * active session. While a tile shows a foreign artifact it has not
 * acknowledged, its Info icon takes the warning tone and the card opens with
 * a notice whose `Got it` acknowledges it. Nothing else does.
 *
 * THE ACKNOWLEDGEMENT is the tile's own: its dockview params (dock-apply.js
 * `setPanelParams`), under {@link ACK_PARAM}. An artifact tile stores its
 * artifact's id; the Artifacts tile, which shows one artifact after another,
 * a list of the last {@link ACK_LIMIT} ids. The params are the tile's record
 * in the dock form, so each session's layout keeps its own
 * (session-layout.js), a tile in its own window keeps them (dock-detached.js),
 * and a closed tile forgets them.
 *
 * {@link onOriginChange} tells the bars and an open card to look again: when
 * the active session or the session rows change (a title arrives late), on a
 * settled dock change (a session's layout brings its tiles' params), and on
 * an acknowledgement.
 *
 * @module artifact-origin
 */

import { getLayoutHolder } from './dock-apply.js';
import { DOCK_VIEW_EVENT } from './dock-view.js';
import { activeRow, sessionOfArtifact, subscribe, switchTo } from './session-model.js';

/** @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry */

/** The tile param that holds what the tile acknowledged. */
export const ACK_PARAM = 'originAck';

/** How many acknowledged ids the Artifacts tile keeps, the most recent. */
export const ACK_LIMIT = 50;

/** The Artifacts tile's key: the one tile that keeps a list. */
const GALLERY_KEY = 'artifacts';

/** The owning session's name when the session rows do not list it. */
export const UNKNOWN_SESSION = 'Unknown session';

/** A listed session with no title, as the sidebar names it. */
const UNTITLED_SESSION = 'Untitled session';

/** The model changes that can change an artifact's origin. */
const SESSION_REASONS = new Set(['pointer', 'switched', 'new', 'rows', 'named']);

/**
 * Whether an artifact belongs to a session other than the active one.
 * @param {Pick<ArtifactEntry, 'owner'> | null | undefined} entry
 * @returns {boolean}
 */
export function isForeign(entry) {
  const owner = entry?.owner;
  if (!owner) return false;
  return owner !== (activeRow()?.transcript_id ?? null);
}

/**
 * The artifact ids a tile acknowledged.
 * @param {string} panelKey  `artifacts`, or an artifact tile's key
 * @returns {string[]}
 */
function acknowledged(panelKey) {
  const value = getLayoutHolder()?.panelParams(panelKey)?.[ACK_PARAM];
  if (Array.isArray(value)) return value.filter((id) => typeof id === 'string');
  return typeof value === 'string' ? [value] : [];
}

/**
 * Whether the tile showing an artifact should warn about its origin: the
 * artifact is foreign and the tile has not acknowledged it.
 * @param {ArtifactEntry | null | undefined} entry
 * @param {string} panelKey
 * @returns {boolean}
 */
export function needsOriginNotice(entry, panelKey) {
  return !!entry && isForeign(entry) && !acknowledged(panelKey).includes(entry.id);
}

/**
 * Acknowledge an artifact's origin on the tile showing it, and announce it.
 * False, and nothing changes, when the dock has no such tile.
 * @param {string} panelKey
 * @param {string} artifactId
 * @returns {boolean}
 */
export function acknowledgeOrigin(panelKey, artifactId) {
  const holder = getLayoutHolder();
  if (!holder) return false;
  const value =
    panelKey === GALLERY_KEY
      ? [...acknowledged(panelKey).filter((id) => id !== artifactId), artifactId].slice(-ACK_LIMIT)
      : artifactId;
  if (!holder.setPanelParams(panelKey, { [ACK_PARAM]: value })) return false;
  announce();
  return true;
}

/**
 * What the Info card says about an artifact's origin.
 * @typedef {object} ArtifactOrigin
 * @property {string} title  the owning session's title, or {@link UNKNOWN_SESSION}
 * @property {boolean} listed  whether the session rows list the owning session
 * @property {(() => void) | null} open  switches to the owning session; null
 *   for the active session and for one not listed
 * @property {boolean} notice  whether the card leads with the notice
 * @property {() => void} acknowledge  the notice's `Got it`
 */

/**
 * The origin of an artifact shown in a tile, or null for one with no owner.
 * @param {ArtifactEntry} entry
 * @param {string} panelKey
 * @returns {ArtifactOrigin | null}
 */
export function artifactOrigin(entry, panelKey) {
  if (!entry.owner) return null;
  const row = sessionOfArtifact(entry);
  return {
    title: row ? row.title || UNTITLED_SESSION : UNKNOWN_SESSION,
    listed: row !== null,
    open: row && isForeign(entry) ? () => void switchTo(row) : null,
    notice: needsOriginNotice(entry, panelKey),
    acknowledge: () => void acknowledgeOrigin(panelKey, entry.id),
  };
}

// ---- Looking again ----

/** @type {Set<() => void>} */
const listeners = new Set();
/** @type {(() => void) | null} */
let unwire = null;

/** Tell every listener to look again. */
function announce() {
  for (const listener of [...listeners]) listener();
}

/** @param {{reason: string}} change */
function onSessionChange({ reason }) {
  if (SESSION_REASONS.has(reason)) announce();
}

/**
 * Hear every change that can change an artifact's origin notice.
 * @param {() => void} listener
 * @returns {() => void} unsubscribe
 */
export function onOriginChange(listener) {
  if (!unwire) {
    const offSessions = subscribe(onSessionChange);
    document.addEventListener(DOCK_VIEW_EVENT, announce);
    unwire = () => {
      offSessions();
      document.removeEventListener(DOCK_VIEW_EVENT, announce);
    };
  }
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Forget every listener (tests). */
export function resetArtifactOrigin() {
  listeners.clear();
  unwire?.();
  unwire = null;
}
