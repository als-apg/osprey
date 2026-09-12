// @ts-check
/**
 * The built-in panel labels, as the server stamps them on `<html>`.
 *
 * `panel-catalog.js` reads `data-panel-labels` at MODULE LOAD, so a suite that
 * asserts a panel's display label has to stamp the roster before it imports
 * anything that reaches the catalog — otherwise every panel wears its id.
 * These pairs are pinned to `osprey.profiles.web_panels.BUILTIN_PANEL_LABELS`
 * by `test_panel_catalog_roster.py`, so a renamed label cannot leave this
 * fixture behind.
 */

/** @type {Readonly<Record<string, string>>} */
export const BUILTIN_PANEL_LABELS = Object.freeze({
  'artifacts': 'WORKSPACE',
  'ariel': 'ARIEL',
  'channel-finder': 'CHANNELS',
  'lattice': 'LATTICE',
  'jupyter': 'JUPYTER',
  'okf': 'KNOWLEDGE',
  'system-health': 'SYSTEM',
});

/**
 * Stamp the roster on a document's `<html>`, exactly as `root()` renders it.
 * @param {Document} [doc]
 */
export function stampPanelLabels(doc = document) {
  doc.documentElement.setAttribute('data-panel-labels', JSON.stringify(BUILTIN_PANEL_LABELS));
}
