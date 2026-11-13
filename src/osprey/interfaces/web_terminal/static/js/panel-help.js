// @ts-check
/* OSPREY Web Terminal — panel help pages
 *
 * Maps a panel to the page of the deployment's documentation that describes it,
 * for the tile bar's Help control and the tile ⋯ menu's Help row.
 *
 * The map is static and lists only panels whose page exists in this repo's
 * docs (`docs/source/how-to/`, checked by panel-help.test.mjs). An application
 * panel names its own page in its manifest (`panel.help`), carried on its
 * catalog entry, and that page wins over the map. A panel with neither has no
 * Help: the control stays hidden rather than pointing at a page that may not
 * exist. Paths are relative to the docs root and name the built `.html` page;
 * an application's page may instead be an absolute https:// URL, which needs
 * no docs root.
 *
 * The docs root is per deployment. It reaches the page as
 * `<html data-docs-url="…">` and is absent when `web.docs_url` is blank, an
 * air-gapped control room with nothing to point at. Then no panel has a help
 * URL, for the same reason the sidebar's Docs row stays hidden
 * (rail-utility.js).
 */

import { panelEntry } from './panel-catalog.js';

/**
 * Builtin and sidecar panel id → docs page, relative to the docs root.
 *
 * @type {Readonly<Record<string, string>>}
 */
export const HELP_PAGES = Object.freeze({
  'channel-finder': 'how-to/use-channel-finder.html',
  ariel: 'how-to/ariel/web-interface.html',
  okf: 'how-to/facility-knowledge/okf-bundle.html',
  'system-health': 'how-to/health-and-monitoring/index.html',
  jupyter: 'how-to/web-terminal/notebooks.html',
  canopy: 'how-to/web-terminal/canopy.html',
  bluesky: 'how-to/bluesky/index.html',
  events: 'how-to/agent-interfaces/event-dispatch.html#the-events-panel',
});

/**
 * The deployment's docs root, from `<html data-docs-url>`.
 *
 * @returns {string} The trimmed value, or '' when the attribute is absent or
 *   blank.
 */
export function docsRoot() {
  return (document.documentElement.dataset.docsUrl ?? '').trim();
}

/**
 * The docs URL of a panel's help page.
 *
 * @param {string} panelId
 * @param {string} [docsUrl] - the docs root; defaults to {@link docsRoot}.
 * @returns {string | null} An application panel's https:// page as declared;
 *   otherwise the root joined to the page with exactly one slash between them,
 *   or null when the root is blank or the panel has no page.
 */
export function helpUrl(panelId, docsUrl = docsRoot()) {
  const declared = panelEntry(panelId)?.help;
  if (declared?.startsWith('https://')) return declared;
  const root = docsUrl.trim();
  if (root === '') return null;
  const page = declared ?? (Object.hasOwn(HELP_PAGES, panelId) ? HELP_PAGES[panelId] : null);
  if (page == null) return null;
  return `${root.replace(/\/+$/, '')}/${page}`;
}

/**
 * Open a panel's help page in a new browser tab.
 *
 * @param {string} panelId
 * @returns {boolean} False when the panel has no help page or no docs root.
 */
export function openHelp(panelId) {
  const url = helpUrl(panelId);
  if (url == null) return false;
  window.open(url, '_blank', 'noopener');
  return true;
}
