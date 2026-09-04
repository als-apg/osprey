// @ts-check
/* OSPREY Web Terminal — Shipped Service Panel Catalog
 *
 * The static registry of built-in service panels plus the terminal rail
 * constants — pure data, extracted from panel-manager.js to keep that module
 * under the max-lines cap. panel-manager.js filters the array in place against
 * /api/panels at init; panel-lifecycle.js appends runtime-registered panels;
 * panel-sse.js reads it for the close fallback.
 */

/**
 * @typedef {object} Panel
 * @property {string} id
 * @property {string} label
 * @property {string | null} configEndpoint
 * @property {string | null} [healthEndpoint] - null/undefined means skip health polling
 * @property {string} [path] - iframe subpath for custom panels (e.g. "/panel/")
 */

/** Rail id of the terminal/chat tile. NOT a service panel: it has no iframe,
 *  no health poll, and no membership — its open/closed state is dock-layout
 *  state (dock-workspace.js owns the tile; panel-manager only renders its rail
 *  entry and routes activate/close to the dock). Kept out of PANELS so every
 *  service-panel iteration stays terminal-free. */
export const TERMINAL_RAIL_ID = 'terminal';
export const TERMINAL_RAIL_LABEL = 'SESSION';

/** Fallback default panel when /api/panels doesn't pin one (kept in sync with
 *  osprey.profiles.web_panels.DEFAULT_PANEL_FALLBACK on the backend). */
export const DEFAULT_PANEL_FALLBACK = 'artifacts';

/** @type {Panel[]} */
export const PANELS = [
  {
    id: 'artifacts',
    label: 'WORKSPACE',
    configEndpoint: '/api/artifact-server',
    healthEndpoint: null,    // embedded same-origin — skip health polling
  },
  {
    id: 'ariel',
    label: 'ARIEL',
    configEndpoint: '/api/ariel-server',
  },
  {
    id: 'channel-finder',
    label: 'CHANNELS',
    configEndpoint: '/api/channel-finder-server',
  },
  {
    id: 'lattice',
    label: 'LATTICE',
    configEndpoint: '/api/lattice-server',
  },
  {
    id: 'jupyter',
    label: 'NOTEBOOKS',
    configEndpoint: '/api/jupyter-server',
    healthEndpoint: '/api/status', // the sidecar's own status route, reached through the panel proxy
  },
  {
    id: 'okf',
    label: 'KNOWLEDGE',
    configEndpoint: '/api/okf-server',
  },
  {
    id: 'system-health',
    label: 'SYSTEM',
    configEndpoint: '/api/system-health-server', // data string; fetchJSON prefixes it in initPanel()
    healthEndpoint: '/health', // EXPLICIT — omitting/null skips polling and pins the panel healthy, which would leave the rail entry enabled with the sidecar down
  },
];
