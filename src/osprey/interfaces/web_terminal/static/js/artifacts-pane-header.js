// @ts-check
/* OSPREY Web Terminal — the Artifacts tile's bar items and ⋯ rows
 *
 * The hub fills the Artifacts tile's bar itself (tile-header-contrib.js
 * `contributeLocal`), because the artifacts iframe is a viewer of one
 * artifact and the list it is chosen from is the sidebar's. With an artifact
 * shown, the bar names it in place of the tile's own name: the hub's `crumb`,
 * "Artifact › <title>", the title as the sidebar row spells it (`titleOf`).
 * That replaces the name rather than adding a subtitle beside it, the same as
 * the Agent tile naming its session. The bar then holds two panel icons: Info
 * opens the artifact's details (info-popover.js: title, description, created
 * time, size, and the file path with a Copy button) and Download saves its
 * file. With no artifact the tile keeps its name and both icons are disabled.
 *
 * Info and Download act here, on the hub's side of the proxy, in the click's
 * own task: the hub holds the entry, and the file URL is the gallery's own
 * (artifacts state.js `fileUrl`) under the proxy's mount. Only the directory
 * the store writes to is the gallery's to know: the viewer page carries it
 * as its `osprey-artifact-dir` meta, read from the same-origin frame when
 * Info opens.
 *
 * An artifact tile (artifact-tiles.js), pinned to one artifact, draws the
 * same bar under its own key: Info hangs from that tile's bar and reads that
 * tile's frame.
 *
 * An artifact made in a session names it in the card's `Session` row, and an
 * artifact from another session warns until the tile acknowledges it: the
 * Info icon takes the warning tone and the card leads with a notice
 * (artifact-origin.js). An open card is shown again in place whenever its
 * origin may have changed (`onOriginChange`): a switch, a late title, `Got it`.
 *
 * The tile's ⋯ "This panel" rows are artifact-verbs.js's for the shown
 * artifact: the viewer runs logbook and print, and Delete… asks in the hub. Print
 * opens its window from the viewer's message task, which a popup blocker
 * lets through: the viewer is a first-party frame on the hub's origin with
 * `allow-popups`, and the HTML standard gives a click's activation to the
 * same-origin frames below it (Chromium does; Firefox is not verified).
 *
 * @module artifacts-pane-header
 */

import { formatFacilityTime, viewerSharesFacilityClock } from '/design-system/js/facility-time.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { withPrefix } from './api.js';
import { queryAllDocuments } from './attached-window.js';
import { panelIdOf } from './dock-apply.js';
import { managedIframe } from './dock-iframe.js';
import { formatBytes } from './activity-log-format.js';
import { ARTIFACTS_BASE } from './sidebar-artifact-listing.js';
import { titleOf } from './sidebar-artifact-rows.js';
import { artifactOrigin, onOriginChange } from './artifact-origin.js';

/**
 * An artifact as the bar reads it: the listed fields, plus the description
 * and size every entry the panel sends carries.
 * @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry
 *   & {description?: string, size_bytes?: number}} ArtifactEntry
 */

/**
 * The bar of the tile showing `panelKey`: the docked tile's (its dockview id),
 * a pane window's (the window bar carries the bare key, pane-window.js), else
 * the fallback bar.
 * @param {string} panelKey  `artifacts`, or an artifact tile's key
 */
const barSelector = (panelKey) =>
  `.tile-tab[data-panel-id="${CSS.escape(panelIdOf(panelKey))}"], ` +
  `.tile-tab[data-panel-id="${CSS.escape(panelKey)}"], .fallback-tile-bar`;

/**
 * The bar items, in their order on the bar. `notice`: the shown artifact's
 * origin is not acknowledged, so Info takes the warning tone.
 * @param {ArtifactEntry | null} shown
 * @param {{notice?: boolean}} [options]
 * @returns {object[]}
 */
export function headerItems(shown, { notice = false } = {}) {
  const info = {
    kind: 'icon',
    id: 'info',
    icon: 'info',
    title: 'Artifact details',
    ...(shown && notice ? { tone: 'warning' } : {}),
  };
  const download = { kind: 'icon', id: 'download', icon: 'download', title: 'Download' };
  if (!shown)
    return [
      { ...info, disabled: true },
      { ...download, disabled: true },
    ];
  const size = shown.size_bytes;
  return [
    { kind: 'crumb', id: 'name', root: 'Artifact', leaf: titleOf(shown) },
    info,
    typeof size === 'number' ? { ...download, title: `Download (${formatBytes(size)})` } : download,
  ];
}

/**
 * The gallery's file route for an artifact, through the hub's proxy.
 * @param {ArtifactEntry} entry
 * @returns {string}
 */
export function fileUrl(entry) {
  const id = encodeURIComponent(entry.id);
  const name = encodeURIComponent(entry.filename ?? '');
  return withPrefix(`${ARTIFACTS_BASE}/files/${id}/${name}`);
}

/**
 * Download: save the artifact's file under its own name.
 * @param {ArtifactEntry} entry
 */
export function downloadArtifact(entry) {
  const link = document.createElement('a');
  link.href = fileUrl(entry);
  link.download = entry.filename ?? '';
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
}

/**
 * The directory the gallery's store writes to, as the viewer page in the
 * tile's frame states it; null while the viewer has not loaded.
 * @param {string} panelKey
 * @returns {string | null}
 */
function artifactDir(panelKey) {
  // Looked up by its panel: in its own window the frame is not in this document.
  const frame = managedIframe(panelKey)?.iframe;
  if (!frame) return null;
  try {
    const meta = frame.contentDocument?.querySelector('meta[name="osprey-artifact-dir"]');
    return meta?.getAttribute('content') || null;
  } catch {
    // Not readable (not loaded, or not same-origin): no directory to name.
    return null;
  }
}

/**
 * The created time as the gallery shows it: on the facility clock, naming
 * the zone when the viewer's own clock differs.
 * @param {string | undefined} timestamp
 * @returns {string}
 */
function createdOn(timestamp) {
  if (!timestamp) return '';
  return formatFacilityTime(timestamp, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    ...(viewerSharesFacilityClock(timestamp) ? {} : { timeZoneName: 'short' }),
  });
}

/**
 * The notice of an artifact from another session: its owning session, as a
 * link to switch to it when it is listed, and `Got it`.
 * @param {import('./artifact-origin.js').ArtifactOrigin} origin
 * @returns {import('/design-system/js/info-popover.js').InfoNotice}
 */
function originNotice(origin) {
  // The quotes are part of the link's text, so a wrap never leaves one behind.
  const parts = origin.open
    ? ['From another session: ', { text: `"${origin.title}"`, onClick: origin.open }]
    : [`From another session: ${origin.listed ? `"${origin.title}"` : origin.title}`];
  return { parts, actionLabel: 'Got it', onAction: origin.acknowledge };
}

/**
 * What the Info popover shows for an artifact. With an origin (an artifact
 * made in a session), the `Session` row names that session, a link to it
 * unless it is the active one, and an unacknowledged artifact from another
 * session leads with the notice. The file row is the store's path when the
 * directory is known, else the bare file name.
 * @param {ArtifactEntry} entry
 * @param {string | null} dir
 * @param {import('./artifact-origin.js').ArtifactOrigin | null} [origin]
 * @returns {import('/design-system/js/info-popover.js').InfoContent}
 */
export function infoContent(entry, dir, origin = null) {
  /** @type {import('/design-system/js/info-popover.js').InfoRow[]} */
  const rows = [];
  if (origin) {
    rows.push({
      label: 'Session',
      value: origin.title,
      ...(origin.open ? { onClick: origin.open } : {}),
    });
  }
  const created = createdOn(entry.timestamp);
  if (created) rows.push({ label: 'Created', value: created });
  if (typeof entry.size_bytes === 'number') {
    rows.push({ label: 'Size', value: formatBytes(entry.size_bytes) });
  }
  if (entry.filename) {
    rows.push({
      label: 'File',
      value: dir ? `${dir}/${entry.filename}` : entry.filename,
      copy: true,
    });
  }
  return {
    ...(origin?.notice ? { notice: originNotice(origin) } : {}),
    title: titleOf(entry),
    text: entry.description || undefined,
    rows,
  };
}

/**
 * The element the Info popover hangs from: the bar's Info icon, or the bar
 * itself while a narrow tile has folded the icon into ⋯. The bar is looked up
 * in the hub and in every pane window (attached-window.js
 * `queryAllDocuments`): a popped-out tile's bar is in its window, and the
 * popover opens in the anchor's document.
 * @param {string} panelKey
 * @returns {HTMLElement | null}
 */
function infoAnchor(panelKey) {
  const [bar] = /** @type {HTMLElement[]} */ (queryAllDocuments(barSelector(panelKey)));
  if (!bar) return null;
  const icon = /** @type {HTMLElement | null} */ (
    bar.querySelector('[data-contrib-key="icon:info"]')
  );
  if (icon && !icon.classList.contains('contrib-hidden')) return icon;
  return bar;
}

/**
 * The card on screen: what it shows, and the handle that shows it again.
 * @type {{content: () => import('/design-system/js/info-popover.js').InfoContent,
 *   handle: import('/design-system/js/info-popover.js').InfoPopoverHandle} | null}
 */
let card = null;
/** Whether {@link refreshCard} follows the origin changes. */
let followingOrigin = false;

/** Show the open card again for its artifact's origin now; forget a closed one. */
function refreshCard() {
  if (card && !card.handle.update(card.content())) card = null;
}

/**
 * Info: open the artifact's details under the Info icon of the bar of the
 * tile showing it, or close them when they already show there.
 * @param {ArtifactEntry} entry
 * @param {string} panelKey  the tile's key: `artifacts`, or an artifact tile's
 */
export function showArtifactInfo(entry, panelKey) {
  const anchor = infoAnchor(panelKey);
  if (!anchor) return;
  const content = () => infoContent(entry, artifactDir(panelKey), artifactOrigin(entry, panelKey));
  card = { content, handle: openInfoPopover(anchor, content()) };
  if (!followingOrigin) {
    followingOrigin = true;
    onOriginChange(refreshCard);
  }
}
