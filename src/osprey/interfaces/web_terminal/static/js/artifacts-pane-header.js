// @ts-check
/* OSPREY Web Terminal — the artifacts pane header and tile menu (layout F)
 *
 * The artifacts pane's one header row: `Artifacts › <title>`, ‹ › with
 * `N of M` to step through the list the shown artifact is in, then Open full
 * size and Save. The hub draws it (tile-header-contrib.js `contributeLocal`;
 * artifacts-header.css places the items), because the artifacts iframe is a
 * viewer of one artifact and the list it steps through is the sidebar's.
 * With no artifact the header is the crumb's root alone and the two buttons,
 * disabled.
 *
 * Open full size and Save act here, on the hub's side of the proxy: a new tab
 * opened from inside the iframe in reply to a message would lack the click's
 * user activation, and the browser would block it. The URLs are the gallery's
 * own (artifacts types.js `openUrl`, state.js `fileUrl`) under the proxy's
 * mount. The tile's ⋯ menu gets the artifact verbs the viewer runs (pin,
 * print, logbook, delete).
 *
 * @module artifacts-pane-header
 */

import { getTheme } from '/design-system/js/theme-manager.js';
import { withPrefix } from './api.js';
import { ARTIFACTS_BASE } from './sidebar-artifact-listing.js';
import { titleOf } from './sidebar-artifact-rows.js';

/**
 * @typedef {import('./sidebar-artifact-listing.js').ArtifactEntry} ArtifactEntry
 * @typedef {'pin' | 'unpin' | 'print' | 'logbook' | 'delete'} ArtifactAction
 */

/**
 * Where the shown artifact sits in the list ‹ › steps through.
 * @typedef {object} PagerPosition
 * @property {number} index  its index in the drawn order
 * @property {number} count  how many the list holds on the server
 * @property {boolean} last  no artifact follows it, held or not
 */

/**
 * The header items, in their order on the bar.
 * @param {ArtifactEntry | null} shown
 * @param {PagerPosition | null} position
 * @returns {object[]}
 */
export function headerItems(shown, position) {
  const crumb = {
    kind: 'crumb',
    id: 'crumb',
    root: 'Artifacts',
    leaf: shown ? titleOf(shown) : '',
  };
  const actions = [
    {
      kind: 'button',
      id: 'open',
      icon: 'maximize',
      label: 'Open full size',
      title: 'Open full size',
      tone: 'accent',
      priority: 5,
      disabled: !shown,
    },
    {
      kind: 'button',
      id: 'save',
      icon: 'save',
      label: 'Save',
      title: 'Save',
      priority: 5,
      disabled: !shown,
    },
  ];
  if (!shown || !position) return [crumb, ...actions];
  const { index, count, last } = position;
  return [
    crumb,
    {
      kind: 'button',
      id: 'prev',
      icon: 'chevron-left',
      title: 'Previous artifact',
      priority: 9,
      disabled: index <= 0,
    },
    { kind: 'text', id: 'pager', text: `${index + 1} of ${count}`, priority: 9 },
    {
      kind: 'button',
      id: 'next',
      icon: 'chevron-right',
      title: 'Next artifact',
      priority: 9,
      disabled: last,
    },
    { kind: 'sep', id: 'sep' },
    ...actions,
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
 * The page an artifact opens on in its own tab: the rendered page for
 * markdown and notebooks, the file otherwise. It carries the theme on screen,
 * since the tab has no hub to follow.
 * @param {ArtifactEntry} entry
 * @returns {string}
 */
export function pageUrl(entry) {
  const id = encodeURIComponent(entry.id);
  let url = fileUrl(entry);
  if (entry.artifact_type === 'markdown') {
    url = withPrefix(`${ARTIFACTS_BASE}/api/markdown/${id}/rendered`);
  } else if (entry.artifact_type === 'notebook') {
    url = withPrefix(`${ARTIFACTS_BASE}/api/notebooks/${id}/rendered`);
  }
  const theme = getTheme();
  return theme ? `${url}?theme=${encodeURIComponent(theme)}` : url;
}

/**
 * Open full size: the artifact's page in a new tab. Call it from the click.
 * @param {ArtifactEntry} entry
 */
export function openArtifact(entry) {
  window.open(pageUrl(entry), '_blank', 'noopener');
}

/**
 * Save: download the artifact's file under its own name.
 * @param {ArtifactEntry} entry
 */
export function saveArtifact(entry) {
  const link = document.createElement('a');
  link.href = fileUrl(entry);
  link.download = entry.filename ?? '';
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
}

/**
 * The artifact verbs of the tile's ⋯ menu, each sending its action for the
 * artifact to the viewer.
 * @param {ArtifactEntry} shown
 * @param {(action: ArtifactAction) => void} send
 * @returns {{label: string, run: () => void, danger?: boolean}[]}
 */
export function menuItems(shown, send) {
  const pin = shown.pinned
    ? { label: 'Unpin artifact', run: () => send('unpin') }
    : { label: 'Pin artifact', run: () => send('pin') };
  return [
    pin,
    { label: 'Print artifact', run: () => send('print') },
    { label: 'Compose logbook entry', run: () => send('logbook') },
    { label: 'Delete artifact', danger: true, run: () => send('delete') },
  ];
}
