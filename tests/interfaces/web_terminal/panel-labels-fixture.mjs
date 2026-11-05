// @ts-check
/**
 * The built-in panel labels, as the server stamps them on `<html>`.
 *
 * `panel-catalog.js` reads `data-panel-labels` at MODULE LOAD, so a suite that
 * asserts a panel's display label has to stamp the roster before it imports
 * anything that reaches the catalog — otherwise every panel wears its id, and
 * the sidecar panels are missing.
 * These pairs are pinned to `osprey.profiles.web_panels.BUILTIN_PANEL_LABELS`
 * by `test_panel_catalog_roster.py`, so a renamed label cannot leave this
 * fixture behind.
 */

/** @type {Readonly<Record<string, string>>} */
export const BUILTIN_PANEL_LABELS = Object.freeze({
  artifacts: 'Artifacts',
  ariel: 'LOGBOOK',
  'channel-finder': 'CHANNELS',
  jupyter: 'JUPYTER',
  canopy: 'CANOPY',
  okf: 'KNOWLEDGE',
  'system-health': 'SYSTEM',
});

/**
 * The sidecar panel descriptors, as the server stamps them on `<html>`. The
 * catalog builds its sidecar rows from `data-sidecar-panels` at module load,
 * so without this stamp it has none. Pinned to
 * `osprey.profiles.web_panels.sidecar_panel_descriptors()` by the same test.
 */
export const SIDECAR_PANELS = Object.freeze([
  Object.freeze({
    id: 'jupyter',
    label: 'JUPYTER',
    icon: 'notebook',
    description: 'JupyterLab with kernels that follow the terminal session',
    configEndpoint: '/api/jupyter-server',
    healthEndpoint: '/api/status',
    startEndpoint: '/api/panels/jupyter/start',
  }),
  Object.freeze({
    id: 'canopy',
    label: 'CANOPY',
    icon: 'layers',
    description: 'Machine screens and Trace, which the agent reads and authors',
    configEndpoint: '/api/panels/canopy/server',
    healthEndpoint: '/health',
    startEndpoint: '/api/panels/canopy/start',
  }),
]);

/**
 * Stamp the roster on a document's `<html>`, exactly as `root()` renders it:
 * the labels, and the sidecar descriptors beside them.
 * @param {Document} [doc]
 */
export function stampPanelLabels(doc = document) {
  doc.documentElement.setAttribute('data-panel-labels', JSON.stringify(BUILTIN_PANEL_LABELS));
  doc.documentElement.setAttribute('data-sidecar-panels', JSON.stringify(SIDECAR_PANELS));
}
