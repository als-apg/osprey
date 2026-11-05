// @ts-check
/* OSPREY Web Terminal — panel help pages
 *
 * Maps a panel to the page of the deployment's documentation that describes it,
 * for the tile bar's Help control and the tile ⋯ menu's Help row.
 *
 * The map is static and lists only panels whose page exists in this repo's
 * docs (`docs/source/how-to/`, checked by panel-help.test.mjs). A panel with
 * no entry has no Help: the control stays hidden rather than pointing at a
 * page that may not exist. Paths are relative to the docs root and name the
 * built `.html` page.
 *
 * The docs root is per deployment. It reaches the page as
 * `<html data-docs-url="…">` and is absent when `web.docs_url` is blank, an
 * air-gapped control room with nothing to point at. Then no panel has a help
 * URL, for the same reason the sidebar's Docs row stays hidden
 * (rail-utility.js).
 */

/**
 * Builtin and sidecar panel id → docs page, relative to the docs root.
 *
 * @type {Readonly<Record<string, string>>}
 */
export const HELP_PAGES = Object.freeze({
  'channel-finder': 'how-to/use-channel-finder.html',
  ariel: 'how-to/ariel/web-interface.html',
  lattice: 'how-to/web-terminal/lattice-dashboard.html',
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
 * @returns {string | null} The root joined to the page with exactly one slash
 *   between them, or null when the root is blank or the panel has no page.
 */
export function helpUrl(panelId, docsUrl = docsRoot()) {
  const root = docsUrl.trim();
  if (root === '') return null;
  if (!Object.hasOwn(HELP_PAGES, panelId)) return null;
  return `${root.replace(/\/+$/, '')}/${HELP_PAGES[panelId]}`;
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
