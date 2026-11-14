// @ts-check
/* OSPREY Web Terminal — artifact tiles: one tile pinned to one artifact
 *
 * An artifact row dragged from the session tree (the sidebar's tree, the
 * collapsed sidebar's Sessions flyout, a session's hover peek) into the dock
 * opens that artifact in a tile of its own. The single Artifacts tile
 * (`artifacts`) is not one of these: it shows whatever the tree picks, and
 * stays the target of a row click, the agent's focus and the palette.
 *
 * THE KEY. An artifact tile's layout key is `artifact:<artifactId>`
 * ({@link artifactTileKey}); its dockview id is that key under the service
 * prefix (`iframe:artifact:<id>`, dock-apply.js `panelIdOf`). One tile per
 * artifact: a row whose tile is already docked is a `{tab}` drag source and
 * moves that tile (rail-drag.js `railSourceOf`). The key is not a panel id:
 * the modules that keep only known panels (the live prune, the restore repair,
 * the hydrate, the pop-out gate, the server's layout report) admit it while
 * the Artifacts panel is served, and the per-session layout keeps the tile but
 * no page for it (session-layout.js).
 *
 * THE FRAME. The tile frames the Artifacts panel's own gallery: the same
 * address, embed parameters and sandbox, built by panel-iframe-factory.js
 * under the tile's key (so dock-iframe.js adopts it, and the tile bar and the
 * header contribution find it, by that key). The viewer opens empty and says
 * `osprey-artifact-ready`; the tile answers with `osprey-artifact-select` for
 * its fixed id. A frame in its own window reaches the hub through
 * attached-window.js's relay, so the answer is the same there. The hub's
 * artifact event stream (sidebar-artifact-frames.js) hands every update and
 * delete to {@link onArtifactChanged}, which forwards the tile's own as
 * `osprey-artifact-changed`.
 *
 * THE BAR. Each tile's bar is the Artifacts tile's for a shown artifact
 * (artifacts-pane-header.js `headerItems`: the crumb "Artifact › <title>",
 * Info, Download), contributed under the tile's key, and its ⋯ "This panel"
 * rows are the viewer's verbs (artifact-verbs.js) for that artifact. The
 * dockview title, which the pop-out window and the menus read, is the
 * artifact's `titleOf`. A deleted artifact leaves the tile open: the viewer
 * shows its empty state and the bar its disabled icons. An artifact from
 * another session gives Info the warning tone until the tile acknowledges it
 * (artifact-origin.js, kept in the tile's params); every tile's bar is drawn
 * again on each change of that (`onOriginChange`).
 *
 * A closed tile's frame leaves the page on the dock's next settled change
 * ({@link releaseClosedArtifactTiles}); a tile that comes back (Undo, a
 * session's layout) frames the gallery again. The dock adapter keeps its
 * entry for the key and takes the new frame in its place (dock-iframe.js
 * `adoptIframe`).
 *
 * What lives in panel-manager's private state (the panel records, their
 * state, the host the factory observes) arrives through
 * {@link initArtifactTiles}; panel-manager routes an artifact tile's
 * activation, hydrate and pop-out frame here.
 *
 * @module artifact-tiles
 */

import { keysOf } from '/static/vendor/dock-engine.mjs';
import { getLayoutHolder, isDetached } from './dock-apply.js';
import { focusPanel, managedIframe } from './dock-iframe.js';
import { withEchoSuppressed } from './dock-sync.js';
import { DOCK_VIEW_EVENT } from './dock-view.js';
import { createPanelIframe } from './panel-iframe-factory.js';
import { setServiceMenuExtras } from './panel-menu-policy.js';
import {
  sendArtifactActionToIframe,
  sendArtifactChangedToIframe,
  sendArtifactSelectToIframe,
} from './panel-iframe-sync.js';
import { contributeLocal } from './tile-header-contrib.js';
import { downloadArtifact, headerItems, showArtifactInfo } from './artifacts-pane-header.js';
import { artifactMenuItems } from './artifact-verbs.js';
import { fetchArtifact } from './sidebar-artifact-listing.js';
import { titleOf } from './sidebar-artifact-rows.js';
import { needsOriginNotice, onOriginChange } from './artifact-origin.js';

/**
 * @typedef {import('./artifacts-pane-header.js').ArtifactEntry} ArtifactEntry
 * @typedef {import('./panel-iframe-factory.js').PanelSpec} PanelSpec
 * @typedef {import('./tile-menu.js').PanelRow} PanelRow
 */

/** The panel whose gallery every artifact tile frames. */
export const GALLERY_PANEL_ID = 'artifacts';

/** The layout-key prefix of an artifact tile. */
export const ARTIFACT_TILE_PREFIX = 'artifact:';

/** The artifact ids a key may carry (the store's are 12 hex digits). */
const ARTIFACT_ID_SHAPE = /^[A-Za-z0-9_-]{1,64}$/;

/** A tile's name before its artifact is known. */
const UNKNOWN_TITLE = 'Artifact';

// ---- The key namespace ----

/**
 * The layout key of an artifact's tile.
 * @param {string} artifactId
 * @returns {string}
 */
export function artifactTileKey(artifactId) {
  return ARTIFACT_TILE_PREFIX + artifactId;
}

/**
 * The artifact id an artifact tile's key carries, or null for any other key.
 * @param {unknown} key
 * @returns {string | null}
 */
export function artifactIdOf(key) {
  if (typeof key !== 'string' || !key.startsWith(ARTIFACT_TILE_PREFIX)) return null;
  const id = key.slice(ARTIFACT_TILE_PREFIX.length);
  return ARTIFACT_ID_SHAPE.test(id) ? id : null;
}

/**
 * Whether a layout key is an artifact tile's.
 * @param {unknown} key
 * @returns {boolean}
 */
export function isArtifactTileKey(key) {
  return artifactIdOf(key) !== null;
}

/**
 * Whether the dock keeps an artifact tile now: the Artifacts panel it frames
 * is among the known panels, and the tile is not in a window of its own. False
 * for any other key, which the caller judges as a panel.
 * @param {string} key
 * @param {Set<string>} knownIds
 * @returns {boolean}
 */
export function artifactTileStays(key, knownIds) {
  return isArtifactTileKey(key) && knownIds.has(GALLERY_PANEL_ID) && !isDetached(key);
}

// ---- The tiles ----

/**
 * What panel-manager hands this module.
 * @typedef {object} ArtifactTileDeps
 * @property {readonly PanelSpec[]} panels - the panel records (PANELS)
 * @property {Record<string, {healthy: boolean, url: string | null}>} panelState
 * @property {HTMLElement} contentEl - the host the factory's resize observer watches
 */

/** @type {ArtifactTileDeps | null} */
let deps = null;

/**
 * One artifact tile: its artifact as last read (null until read, or once
 * unreadable), whether it was deleted, the title its dockview tab carries, and
 * its frame record for the factory.
 * @typedef {object} ArtifactTile
 * @property {string} id
 * @property {ArtifactEntry | null} entry
 * @property {boolean} deleted
 * @property {boolean} reading
 * @property {string} title
 * @property {{url: string | null, pendingUrl: string | null, iframe: HTMLIFrameElement | null}} frame
 */

/** @type {Map<string, ArtifactTile>} by tile key */
const tiles = new Map();

let listening = false;
/** @type {(() => void) | null} */
let offOrigin = null;

/**
 * Hand this module what panel-manager owns, and start answering the tiles'
 * viewers and following the dock. Safe to call again; the last call wins.
 * @param {ArtifactTileDeps} part
 */
export function initArtifactTiles(part) {
  deps = part;
  if (!listening) {
    listening = true;
    window.addEventListener('message', onMessage);
    document.addEventListener(DOCK_VIEW_EVENT, releaseClosedArtifactTiles);
    offOrigin = onOriginChange(paintAll);
  }
}

/**
 * The Artifacts panel's record and address while it is served and answering.
 * @returns {{panel: PanelSpec, url: string} | null}
 */
function gallery() {
  const panel = deps?.panels.find((p) => p.id === GALLERY_PANEL_ID);
  const state = deps?.panelState[GALLERY_PANEL_ID];
  return panel && state?.healthy && state.url ? { panel, url: state.url } : null;
}

/**
 * The tile record of a key, made on first use.
 * @param {string} key  an artifact tile key
 * @returns {ArtifactTile}
 */
function tileOf(key) {
  let tile = tiles.get(key);
  if (!tile) {
    tile = {
      id: /** @type {string} */ (artifactIdOf(key)),
      entry: null,
      deleted: false,
      reading: false,
      title: UNKNOWN_TITLE,
      frame: { url: null, pendingUrl: null, iframe: null },
    };
    tiles.set(key, tile);
  }
  return tile;
}

/**
 * The name an artifact tile's tab is built with: its artifact's title, once
 * known; null for a key that is no artifact tile's.
 * @param {string} key
 * @returns {string | null}
 */
export function artifactTileTitle(key) {
  if (!isArtifactTileKey(key)) return null;
  return tiles.get(key)?.title ?? UNKNOWN_TITLE;
}

/**
 * Record what a tile shows and paint its bar: the crumb and both icons for a
 * readable artifact, the disabled icons for one deleted or not read.
 * @param {string} key
 * @param {ArtifactTile} tile
 */
function paint(key, tile) {
  const entry = tile.deleted ? null : tile.entry;
  if (entry) tile.title = titleOf(entry);
  const notice = needsOriginNotice(entry, key);
  contributeLocal(key, headerItems(entry, { notice }), (action) => onHeaderAction(key, action));
  setServiceMenuExtras(key, () => menuRows(key));
}

/** Paint every tile's bar again: its artifact's origin notice may have changed. */
function paintAll() {
  for (const [key, tile] of tiles) paint(key, tile);
}

/**
 * Hold an artifact for its tile (a row about to be dragged, rail-drag.js
 * `railDragStart` with the key returned): the tab's name and the bar come
 * from its entry at once. A row the tree holds no entry for (a run's part)
 * names the tab with its own title, and the entry is read.
 * @param {{id: string, entry: ArtifactEntry | null, title: string}} row
 *   the row's artifact id, its entry as the tree holds it, and its title
 * @returns {string | null} the tile key; null for an id no key can carry
 */
export function holdArtifactTile({ id, entry, title }) {
  if (artifactIdOf(artifactTileKey(id)) !== id) return null;
  const key = artifactTileKey(id);
  const tile = tileOf(key);
  tile.deleted = false;
  if (entry) tile.entry = entry;
  else if (title) tile.title = title;
  paint(key, tile);
  void readEntry(key, tile);
  return key;
}

/**
 * Read a tile's artifact from the store when it has none (a tile restored by
 * a reload or a session's layout). An unreadable one leaves the bar disabled.
 * @param {string} key
 * @param {ArtifactTile} tile
 */
async function readEntry(key, tile) {
  if (tile.entry || tile.deleted || tile.reading) return;
  tile.reading = true;
  const entry = await fetchArtifact(tile.id);
  tile.reading = false;
  if (entry?.id !== tile.id || tile.entry || tile.deleted) return;
  tile.entry = entry;
  paint(key, tile);
}

// ---- The frame ----

/**
 * Whether a tile can get a frame now: the Artifacts panel is served and answering.
 * @param {string} key
 * @returns {boolean}
 */
export function canLoadArtifactFrame(key) {
  return isArtifactTileKey(key) && gallery() !== null;
}

/**
 * Build a tile's frame, unless it has one: the Artifacts panel's gallery,
 * under the tile's key. `dock: false` builds it with no tile, on its way to a
 * window of its own (pane-window.js).
 * @param {string} key
 * @param {{dock?: boolean}} [options]
 * @returns {boolean} whether the tile has a frame afterwards
 */
export function loadArtifactFrame(key, { dock = true } = {}) {
  if (!isArtifactTileKey(key)) return false;
  if (frameOf(key)) return true;
  const served = gallery();
  if (!deps || !served) return false;
  const tile = tileOf(key);
  paint(key, tile);
  void readEntry(key, tile);
  tile.frame.url = served.url;
  tile.frame.iframe = null;
  const spec = { ...served.panel, label: tile.title };
  createPanelIframe(spec, key, tile.frame, deps.contentEl, { dock });
  return frameOf(key) !== null;
}

/**
 * pane-window.js's frame gate and loader, for artifact tiles as well as
 * panels: an artifact tile's key goes to this module, any other to `panel`.
 * @template {{canLoadFrame: (id: string) => boolean, loadFrame: (id: string) => void}} T
 * @param {T} panel
 * @returns {T}
 */
export function withArtifactFrames(panel) {
  return {
    ...panel,
    canLoadFrame: (id) =>
      isArtifactTileKey(id) ? canLoadArtifactFrame(id) : panel.canLoadFrame(id),
    loadFrame: (id) =>
      isArtifactTileKey(id) ? loadArtifactFrame(id, { dock: false }) : panel.loadFrame(id),
  };
}

/**
 * Bring a tile forward (its drop's landing, a focus of its tab): frame it if
 * it has no frame, then focus it. An applied change, never a person's focus
 * gesture (dock-sync.js's echo guard).
 * @param {string} key
 */
export function showArtifactTile(key) {
  withEchoSuppressed(() => {
    if (loadArtifactFrame(key)) focusPanel(key);
  });
}

/**
 * Frame every docked artifact tile that has no frame: after a layout apply,
 * on a health settle (a restored tile waits for the Artifacts panel).
 */
export function hydrateArtifactTiles() {
  withEchoSuppressed(() => {
    for (const key of keysOf(getLayoutHolder()?.layout ?? null)) {
      if (isArtifactTileKey(key) && !frameOf(key)) loadArtifactFrame(key);
    }
  });
}

/**
 * Take out of the page the frame of every artifact tile the dock no longer
 * holds and that is not in a window of its own: an artifact tile is cheap to
 * frame again and there is one per artifact ever dragged, so a closed one
 * keeps no document. Runs on every settled dock change (dock-view.js).
 */
export function releaseClosedArtifactTiles() {
  const docked = new Set(keysOf(getLayoutHolder()?.layout ?? null));
  for (const [key, tile] of tiles) {
    if (docked.has(key) || isDetached(key)) continue;
    frameOf(key)?.remove();
    tile.frame.iframe = null;
  }
}

/**
 * A tile's frame while it is in a page (the hub's, or its own window's).
 * @param {string} key
 * @returns {HTMLIFrameElement | null}
 */
function frameOf(key) {
  const iframe = managedIframe(key)?.iframe;
  return iframe?.isConnected ? iframe : null;
}

// ---- The viewer protocol ----

/**
 * A tile's viewer is ready: tell it which artifact to show (none, once deleted).
 * @param {MessageEvent} event
 */
function onMessage(event) {
  if (event.origin !== window.location.origin) return;
  if (event.data?.type !== 'osprey-artifact-ready' || !event.source) return;
  for (const [key, tile] of tiles) {
    const iframe = frameOf(key);
    if (iframe && iframe.contentWindow === event.source) {
      sendArtifactSelectToIframe(iframe, tile.deleted ? null : tile.id);
      return;
    }
  }
}

/**
 * An artifact changed on the server (the hub's event stream): its tile, if
 * any, takes the new copy, or its deleted state, and its viewer is told.
 * @param {string} artifactId
 * @param {'updated' | 'deleted'} change
 * @param {ArtifactEntry} [artifact]  updated: the server's copy
 */
export function onArtifactChanged(artifactId, change, artifact) {
  const key = artifactTileKey(artifactId);
  const tile = tiles.get(key);
  if (!tile) return;
  if (change === 'deleted') {
    tile.deleted = true;
  } else if (artifact) {
    tile.entry = { ...tile.entry, ...artifact };
  }
  paint(key, tile);
  sendArtifactChangedToIframe(frameOf(key), artifactId, change, artifact);
}

// ---- The bar and the ⋯ rows ----

/**
 * @param {string} key
 * @param {string} action  the bar item's id
 */
function onHeaderAction(key, action) {
  const tile = tiles.get(key);
  const entry = tile && !tile.deleted ? tile.entry : null;
  if (!entry) return;
  if (action === 'info') showArtifactInfo(entry, key);
  else if (action === 'download') downloadArtifact(entry);
}

/**
 * A tile's ⋯ "This panel" rows: artifact-verbs.js's rows for its artifact,
 * with the viewer's verbs sent to the tile's frame, as the Artifacts tile's
 * (sidebar-artifact-viewer.js `viewerMenuItems`). None while its artifact is
 * deleted, unread, or has no frame to run the verbs. The tile menu draws its
 * own divider before Delete….
 * @param {string} key
 * @returns {PanelRow[]}
 */
function menuRows(key) {
  const tile = tiles.get(key);
  const entry = tile && !tile.deleted ? tile.entry : null;
  if (!entry || !frameOf(key)) return [];
  const viewer = (/** @type {'print' | 'logbook'} */ verb) =>
    sendArtifactActionToIframe(frameOf(key), entry.id, verb);
  return /** @type {PanelRow[]} */ (
    artifactMenuItems(entry, { viewer }).filter((item) => !item.divider)
  );
}

/** Forget every tile (tests). */
export function resetArtifactTiles() {
  tiles.clear();
  deps = null;
  window.removeEventListener('message', onMessage);
  document.removeEventListener(DOCK_VIEW_EVENT, releaseClosedArtifactTiles);
  offOrigin?.();
  offOrigin = null;
  listening = false;
}
