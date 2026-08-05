// @ts-check
/* OSPREY Web Terminal — Shipped Service Panel Catalog
 *
 * The static registry of built-in service panels plus the terminal rail
 * constants — pure data, extracted from panel-manager.js (its only consumer,
 * which filters/extends the array in place against /api/panels at init) to
 * keep that module under the max-lines cap.
 */

/**
 * @typedef {object} Panel
 * @property {string} id
 * @property {string} label
 * @property {string | null} configEndpoint
 * @property {string | null} [healthEndpoint] - null/undefined means skip health polling
 * @property {string | null} statusBarId
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
    statusBarId: null,       // no dedicated status-bar item
  },
  {
    id: 'ariel',
    label: 'ARIEL',
    configEndpoint: '/api/ariel-server',
    statusBarId: 'ariel-status',
  },
  {
    id: 'channel-finder',
    label: 'CHANNELS',
    configEndpoint: '/api/channel-finder-server',
    statusBarId: 'channel-finder-status',
  },
  {
    id: 'lattice',
    label: 'LATTICE',
    configEndpoint: '/api/lattice-server',
    statusBarId: null,
  },
  {
    id: 'okf',
    label: 'KNOWLEDGE',
    configEndpoint: '/api/okf-server',
    statusBarId: null,
  },
  {
    id: 'system-health',
    label: 'SYSTEM',
    configEndpoint: '/api/system-health-server', // data string; fetchJSON prefixes it in initPanel()
    healthEndpoint: '/health', // EXPLICIT — omitting/null skips polling and pins the panel healthy, which would leave the rail entry enabled with the sidecar down
    statusBarId: null,
  },
];
