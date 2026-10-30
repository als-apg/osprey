// @ts-check
/* OSPREY Web Terminal — Shipped Service Panel Catalog
 *
 * The static registry of built-in service panels plus the terminal row
 * constants, extracted from panel-manager.js to keep that module under the
 * max-lines cap. panel-manager.js filters the array in place against
 * /api/panels at init; panel-lifecycle.js appends runtime-registered panels;
 * panel-sse.js reads it for the close fallback.
 *
 * The ids, endpoints and health posture below are the browser's own — they are
 * what it builds proxy paths and polls from. The DISPLAY LABELS are not: the
 * backend registry owns them, and the page carries them here as a stamp on
 * <html>. /api/panels cannot seed them, for two reasons: PANELS is module
 * state read before that request resolves, and the route answers for the
 * ENABLED panels only, so it could never name the full roster the "+"
 * menu's catalog list needs.
 *
 * The sidebar shows each label in title case ({@link titleCase}) and each
 * panel with an icons.js glyph ({@link panelIconName}).
 */

/**
 * @typedef {object} Panel
 * @property {string} id
 * @property {string} label
 * @property {string | null} configEndpoint
 * @property {string | null} [healthEndpoint] - null/undefined means skip health polling
 * @property {string} [path] - iframe subpath for custom panels (e.g. "/panel/")
 * @property {string} [startEndpoint] - POST target that starts a failed sidecar again
 */

/** Sidebar id of the Agent pane (the terminal/chat tile). NOT a service panel:
 *  it has no iframe, no health poll, and no membership — its open/closed state
 *  is dock-layout state (dock-workspace.js owns the tile; panel-manager only
 *  renders its sidebar row and routes activate/close to the dock). Kept out of
 *  PANELS so every service-panel iteration stays terminal-free. */
export const TERMINAL_RAIL_ID = 'terminal';
export const TERMINAL_RAIL_LABEL = 'Agent';

/**
 * The icons.js glyph of each known panel's sidebar row. A panel not named here
 * (a custom or runtime-registered one) wears {@link FALLBACK_PANEL_ICON}.
 * @type {Readonly<Record<string, string>>}
 */
export const PANEL_ICONS = Object.freeze({
  terminal: 'agent',
  artifacts: 'layers',
  events: 'activity',
  jupyter: 'notebook',
  ariel: 'book-open',
  lattice: 'hexagon',
  okf: 'library',
  'channel-finder': 'radio',
  bluesky: 'flask',
  'system-health': 'activity',
});

/** The glyph of a panel {@link PANEL_ICONS} does not name. */
export const FALLBACK_PANEL_ICON = 'workspace';

/**
 * The icons.js name for a panel's sidebar row.
 * @param {string} id
 * @returns {string}
 */
export function panelIconName(id) {
  return Object.hasOwn(PANEL_ICONS, id) ? PANEL_ICONS[id] : FALLBACK_PANEL_ICON;
}

/**
 * A panel label as the sidebar shows it. An all-capitals label (`ARIEL`,
 * `CHANNELS`, a custom panel named after its id) reads in title case:
 * every word, split at spaces and hyphens, keeps only its first letter
 * capital. A label with any lower-case letter is the deployment's own spelling
 * (`EPICS Archiver`, `pyAT`) and is left as it is.
 * @param {string} label
 * @returns {string}
 */
export function titleCase(label) {
  if (label !== label.toUpperCase()) return label;
  return label.toLowerCase().replace(/(^|[\s-])(\p{L})/gu, (_, sep, ch) => sep + ch.toUpperCase());
}

/** Fallback default panel when /api/panels doesn't pin one (kept in sync with
 *  osprey.profiles.web_panels.DEFAULT_PANEL_FALLBACK on the backend). */
export const DEFAULT_PANEL_FALLBACK = 'artifacts';

/** Attribute carrying the registry's built-in panel labels, JSON. */
const PANEL_LABELS_ATTR = 'data-panel-labels';

/**
 * The built-in panel labels this page was served with. An absent or unreadable
 * stamp answers with an empty roster — every panel then wears its own id,
 * which is wrong but readable, and says so once in the console.
 * @returns {Record<string, string>}
 */
function panelLabels() {
  const raw =
    typeof document === 'undefined'
      ? null
      : document.documentElement?.getAttribute(PANEL_LABELS_ATTR);
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') return parsed;
    } catch {
      // Falls through to the warning below.
    }
  }
  console.warn(
    `[panel-catalog] no readable ${PANEL_LABELS_ATTR} on this page; ` + 'panels are labelled by id'
  );
  return {};
}

const PANEL_LABELS = panelLabels();

/**
 * The label the server gave this panel, or its id when the stamp does not
 * name it.
 * @param {string} id
 * @returns {string}
 */
function labelOf(id) {
  const label = PANEL_LABELS[id];
  return typeof label === 'string' && label ? label : id.toUpperCase();
}

/** @type {Panel[]} */
export const PANELS = [
  {
    id: 'artifacts',
    label: labelOf('artifacts'),
    configEndpoint: '/api/artifact-server',
    healthEndpoint: null, // embedded same-origin — skip health polling
  },
  {
    id: 'ariel',
    label: labelOf('ariel'),
    configEndpoint: '/api/ariel-server',
  },
  {
    id: 'channel-finder',
    label: labelOf('channel-finder'),
    configEndpoint: '/api/channel-finder-server',
  },
  {
    id: 'lattice',
    label: labelOf('lattice'),
    configEndpoint: '/api/lattice-server',
  },
  {
    id: 'jupyter',
    label: labelOf('jupyter'),
    configEndpoint: '/api/jupyter-server',
    healthEndpoint: '/api/status', // the sidecar's own status route, reached through the panel proxy
    startEndpoint: '/api/panels/jupyter/start', // data string; apiRequest prefixes it in retryPanelStart()
  },
  {
    id: 'okf',
    label: labelOf('okf'),
    configEndpoint: '/api/okf-server',
  },
  {
    id: 'system-health',
    label: labelOf('system-health'),
    configEndpoint: '/api/system-health-server', // data string; fetchJSON prefixes it in initPanel()
    healthEndpoint: '/health', // EXPLICIT — omitting/null skips polling and pins the panel healthy, so its rail entry would never go disabled (down at boot) or unreachable (down later)
  },
];
