// @ts-check
/* OSPREY Web Terminal — panel links and reload
 *
 * Three verbs that every service tile shares, whether they come from the tile
 * bar's shared controls or from the tile ⋯ menu: copy a panel's link, open it
 * in a browser tab, and reload it. All three need the panel's live iframe, so
 * they find it through tile-header-contrib's iframeForPanel and hold no state
 * of their own.
 *
 * The deep link is the panel's standalone URL carrying its current state. The
 * iframe's own location is the best source (a panel keeps its filters and
 * selection in its URL), so it is read first. That read throws for a
 * cross-origin frame and answers `about:blank` for a frame that has not
 * loaded, and in both cases the `src` attribute stands in. The `embedded`,
 * `popout` and `theme` query params describe the frame, not the panel, so
 * they are stripped: a pasted link opens the full page in the reader's own
 * theme.
 */

import { copyText } from '/design-system/js/clipboard.js';

import { iframeForPanel } from './tile-header-contrib.js';

/** Query params that describe the hosting frame, not the panel's state. */
const FRAME_PARAMS = ['embedded', 'popout', 'theme'];

/**
 * The URL the iframe shows right now, or its `src` when that is unreadable.
 *
 * @param {HTMLIFrameElement} iframe
 * @returns {string | null}
 */
function currentUrl(iframe) {
  try {
    const href = iframe.contentWindow?.location.href;
    if (typeof href === 'string' && href.startsWith('http')) return href;
  } catch {
    // Cross-origin frame: the location is not readable. Use `src` below.
  }
  return iframe.getAttribute('src');
}

/**
 * The panel's deep link: its standalone URL with its current state.
 *
 * @param {string} panelId
 * @returns {string | null} An absolute URL without the `embedded`, `popout`
 *   and `theme` params, or null when the panel has no iframe or no `src`.
 */
export function panelLink(panelId) {
  const iframe = iframeForPanel(panelId);
  if (iframe == null) return null;
  const raw = currentUrl(iframe);
  if (raw == null || raw === '') return null;

  const url = new URL(raw, window.location.origin);
  for (const param of FRAME_PARAMS) {
    if (url.searchParams.has(param)) url.searchParams.delete(param);
  }
  return url.toString();
}

/**
 * Reload the panel's iframe.
 *
 * A same-origin frame reloads in place and keeps its history entry. A frame
 * that refuses (cross-origin) is reloaded by setting its `src` again.
 *
 * @param {string} panelId
 * @returns {boolean} False when the panel has no iframe.
 */
export function reloadPanel(panelId) {
  const iframe = iframeForPanel(panelId);
  if (iframe == null) return false;
  try {
    /** @type {Window} */ (iframe.contentWindow).location.reload();
  } catch {
    const src = iframe.getAttribute('src');
    if (src == null) return false;
    iframe.src = src;
  }
  return true;
}

/**
 * Copy the panel's deep link to the clipboard.
 *
 * @param {string} panelId
 * @returns {Promise<boolean>} False when the panel has no link or the copy
 *   failed.
 */
export async function copyPanelLink(panelId) {
  const link = panelLink(panelId);
  if (link == null) return false;
  return copyText(link);
}

/**
 * Open the panel's deep link in a new browser tab.
 *
 * @param {string} panelId
 * @returns {boolean} False when the panel has no link.
 */
export function openPanelInTab(panelId) {
  const link = panelLink(panelId);
  if (link == null) return false;
  window.open(link, '_blank', 'noopener');
  return true;
}
