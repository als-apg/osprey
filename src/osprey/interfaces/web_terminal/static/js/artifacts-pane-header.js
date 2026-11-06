// @ts-check
/* OSPREY Web Terminal — the Artifacts tile's bar items and ⋯ rows
 *
 * The hub fills the Artifacts tile's bar itself (tile-header-contrib.js
 * `contributeLocal`), because the artifacts iframe is a viewer of one
 * artifact and the list it is chosen from is the sidebar's. The bar names
 * the shown artifact (the subtitle beside the title) and holds two panel
 * icons: Info opens the artifact's details (info-popover.js: title,
 * description, created time, size, and the file path with a Copy button) and
 * Download saves its file. With no artifact there is no subtitle and both
 * icons are disabled.
 *
 * Info and Download act here, on the hub's side of the proxy, in the click's
 * own task: the hub holds the entry, and the file URL is the gallery's own
 * (artifacts state.js `fileUrl`) under the proxy's mount. Only the directory
 * the store writes to is the gallery's to know: the viewer page carries it
 * as its `osprey-artifact-dir` meta, read from the same-origin frame when
 * Info opens.
 *
 * The tile's ⋯ "This panel" section gets the artifact verbs the viewer runs
 * (logbook, print, pin, delete); Delete keeps the viewer's confirm. Print
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
import { fmtBytes } from './session-helpers.js';
import { ARTIFACTS_BASE } from './sidebar-artifact-listing.js';
import { titleOf } from './sidebar-artifact-rows.js';

/**
 * An artifact as the bar reads it: the listed fields, plus the description
 * and size every entry the panel sends carries.
 * @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry
 *   & {description?: string, size_bytes?: number}} ArtifactEntry
 * @typedef {'pin' | 'unpin' | 'print' | 'logbook' | 'delete'} ArtifactAction
 */

/** The Artifacts tile's bar: the docked tile's, else the fallback bar. */
const BAR_SELECTOR = '.tile-tab[data-panel-id="iframe:artifacts"], .fallback-tile-bar';

/**
 * The bar items, in their order on the bar.
 * @param {ArtifactEntry | null} shown
 * @returns {object[]}
 */
export function headerItems(shown) {
  const info = { kind: 'icon', id: 'info', icon: 'info', title: 'Artifact details' };
  const download = { kind: 'icon', id: 'download', icon: 'download', title: 'Download' };
  if (!shown)
    return [
      { ...info, disabled: true },
      { ...download, disabled: true },
    ];
  const size = shown.size_bytes;
  return [
    { kind: 'text', id: 'name', text: titleOf(shown) },
    info,
    typeof size === 'number' ? { ...download, title: `Download (${fmtBytes(size)})` } : download,
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
 * The directory the gallery's store writes to, as the viewer page states it;
 * null while the viewer has not loaded.
 * @returns {string | null}
 */
function artifactDir() {
  const frame = document.querySelector('iframe.panel-iframe[data-panel-id="artifacts"]');
  if (!(frame instanceof HTMLIFrameElement)) return null;
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
 * What the Info popover shows for an artifact. The file row is the store's
 * path when the directory is known, else the bare file name.
 * @param {ArtifactEntry} entry
 * @param {string | null} dir
 * @returns {import('/design-system/js/info-popover.js').InfoContent}
 */
export function infoContent(entry, dir) {
  /** @type {{label: string, value: string, copy?: boolean}[]} */
  const rows = [];
  const created = createdOn(entry.timestamp);
  if (created) rows.push({ label: 'Created', value: created });
  if (typeof entry.size_bytes === 'number') {
    rows.push({ label: 'Size', value: fmtBytes(entry.size_bytes) });
  }
  if (entry.filename) {
    rows.push({
      label: 'File',
      value: dir ? `${dir}/${entry.filename}` : entry.filename,
      copy: true,
    });
  }
  return { title: titleOf(entry), text: entry.description || undefined, rows };
}

/**
 * The element the Info popover hangs from: the bar's Info icon, or the bar
 * itself while a narrow tile has folded the icon into ⋯.
 * @returns {HTMLElement | null}
 */
function infoAnchor() {
  const bar = document.querySelector(BAR_SELECTOR);
  if (!(bar instanceof HTMLElement)) return null;
  const icon = bar.querySelector('[data-contrib-key="icon:info"]');
  if (icon instanceof HTMLElement && !icon.classList.contains('contrib-hidden')) return icon;
  return bar;
}

/**
 * Info: open the artifact's details under the bar's Info icon, or close them
 * when they already show there.
 * @param {ArtifactEntry} entry
 */
export function showArtifactInfo(entry) {
  const anchor = infoAnchor();
  if (anchor) openInfoPopover(anchor, infoContent(entry, artifactDir()));
}

/**
 * The artifact verbs of the tile's ⋯ "This panel" section, each sending its
 * action for the artifact to the viewer. Delete comes last, set apart, in the
 * danger colour; the viewer asks before it deletes.
 * @param {ArtifactEntry} shown
 * @param {(action: ArtifactAction) => void} send
 * @returns {import('./tile-menu.js').PanelRow[]}
 */
export function menuItems(shown, send) {
  const pin = shown.pinned
    ? { label: 'Unpin artifact', run: () => send('unpin') }
    : { label: 'Pin artifact', run: () => send('pin') };
  return [
    { label: 'Compose logbook entry', run: () => send('logbook') },
    { label: 'Print artifact', run: () => send('print') },
    pin,
    { label: 'Delete artifact…', danger: true, separator: true, run: () => send('delete') },
  ];
}
