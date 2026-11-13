// @ts-check
/* OSPREY Web Terminal — Shipped Service Panel Catalog
 *
 * The static registry of built-in service panels plus the terminal row
 * constants, extracted from panel-manager.js to keep that module under the
 * max-lines cap. panel-manager.js filters the array in place against
 * /api/panels at init; panel-lifecycle.js appends runtime-registered panels;
 * panel-sse.js reads it for the close fallback.
 *
 * For the companion panels, the ids, endpoints and health posture below are
 * the browser's own — they are what it builds proxy paths and polls from. The
 * DISPLAY LABELS are not: the backend registry owns them, and the page carries
 * them here as a stamp on <html>. /api/panels cannot seed them, for two
 * reasons: PANELS is module state read before that request resolves, and the
 * route answers for the ENABLED panels only, so it could never name the full
 * roster the "+" menu's catalog list needs.
 *
 * The SIDECAR panels — the ones the terminal starts itself — are not listed
 * here at all. The backend registry describes each one whole (id, label,
 * icon, description, endpoints), the page stamps those descriptors on <html>,
 * and this module builds one descriptor per entry.
 *
 * An application panel reaches the browser as a custom def of /api/panels,
 * which panel-manager.js turns into a catalog entry with {@link customPanelEntry}.
 * Its tooltip and help page are the def's own, declared in the application's
 * manifest, under the same names a sidecar descriptor uses; its glyph travels
 * with every other configured glyph, in the page's icon stamp.
 *
 * The sidebar shows each label in title case ({@link titleCase}) and each
 * panel with an icons.js glyph ({@link panelIconName}). A deployment names a
 * built-in panel its own way, and picks any panel's glyph, in `web.panels.<id>`
 * (`label`, `icon`); the labels arrive already resolved in the roster stamp,
 * the glyphs in a stamp of their own.
 */

import { ICON_NAMES } from '/design-system/js/icons.js';

/**
 * @typedef {object} Panel
 * @property {string} id
 * @property {string} label
 * @property {string | null} configEndpoint
 * @property {string | null} [healthEndpoint] - null/undefined means skip health polling
 * @property {string} [path] - iframe subpath for custom panels (e.g. "/panel/")
 * @property {string} [startEndpoint] - POST target that starts a failed sidecar again
 * @property {string} [icon] - icons.js glyph a sidecar's registry entry names
 * @property {string} [description] - tooltip sentence of a sidecar or application panel
 * @property {string} [help] - an application panel's help page: docs-root-relative, or an https:// URL
 */

/** Sidebar id of the Agent pane (the terminal/chat tile). NOT a service panel:
 *  it has no iframe, no health poll, and no membership — its open/closed state
 *  is dock-layout state (dock-workspace.js owns the tile; panel-manager only
 *  renders its sidebar row and routes activate/close to the dock). Kept out of
 *  PANELS so every service-panel iteration stays terminal-free. */
export const TERMINAL_RAIL_ID = 'terminal';
/** The sidebar row's name: it lists the agent's sessions. The pane it opens keeps its own name. */
export const TERMINAL_RAIL_LABEL = 'Sessions';
/** The Agent pane's name: its tile, its header menu, its pop-out window. */
export const TERMINAL_PANE_LABEL = 'Agent';

/** Attribute carrying the registry's sidecar panel descriptors, a JSON array. */
const SIDECAR_PANELS_ATTR = 'data-sidecar-panels';

/**
 * A string field of a stamped descriptor, or `fallback` when it is absent or
 * not a non-empty string.
 * @template T
 * @param {unknown} value
 * @param {T} fallback
 * @returns {string | T}
 */
function stringOr(value, fallback) {
  return typeof value === 'string' && value ? value : fallback;
}

/**
 * The sidecar panels this page was served with, one descriptor each. An
 * absent or unreadable stamp answers with none — the sidecar panels are then
 * missing, and the console says so once. An entry without an id is skipped.
 *
 * `healthEndpoint` is carried as the registry states it: a path means poll it,
 * null means the panel is not polled.
 * @returns {Panel[]}
 */
function sidecarPanels() {
  const raw =
    typeof document === 'undefined'
      ? null
      : document.documentElement?.getAttribute(SIDECAR_PANELS_ATTR);
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed
          .filter((entry) => entry && typeof entry.id === 'string' && entry.id)
          .map((entry) => {
            /** @type {Panel} */
            const panel = {
              id: entry.id,
              label: titleCase(stringOr(entry.label, entry.id.toUpperCase())),
              configEndpoint: stringOr(entry.configEndpoint, `/api/panels/${entry.id}/server`),
              healthEndpoint: stringOr(entry.healthEndpoint, null),
              // Data strings; apiRequest prefixes the start one in retryPanelStart().
              startEndpoint: stringOr(entry.startEndpoint, `/api/panels/${entry.id}/start`),
            };
            const icon = stringOr(entry.icon, null);
            if (icon) panel.icon = icon;
            const description = stringOr(entry.description, null);
            if (description) panel.description = description;
            return panel;
          });
      }
    } catch {
      // Falls through to the warning below.
    }
  }
  console.warn(
    `[panel-catalog] no readable ${SIDECAR_PANELS_ATTR} on this page; ` +
      'sidecar panels are missing'
  );
  return [];
}

/**
 * The catalog entry of a custom panel /api/panels answers with: a
 * config-defined one (an application's panel among them) or a runtime-registered
 * one. The def's `description` and `help` are carried when they are non-empty
 * strings.
 * @param {{ id: string, label?: string, healthEndpoint?: string | null, path?: string,
 *   description?: unknown, help?: unknown }} def
 * @returns {Panel}
 */
export function customPanelEntry(def) {
  /** @type {Panel} */
  const panel = {
    id: def.id,
    label: titleCase(def.label || def.id.toUpperCase()),
    configEndpoint: null,
    healthEndpoint: def.healthEndpoint, // null = skip health polling
    path: def.path || '/', // subpath for iframe (e.g. "/panel/")
  };
  const description = stringOr(def.description, null);
  if (description) panel.description = description;
  const help = stringOr(def.help, null);
  if (help) panel.help = help;
  return panel;
}

/**
 * The catalog entry for a panel id: the live roster's first, then the
 * page's sidecar descriptors (which the roster may have filtered out).
 * @param {string} id
 * @returns {Panel | undefined}
 */
export function panelEntry(id) {
  return PANELS.find((p) => p.id === id) ?? SIDECAR_PANELS.find((p) => p.id === id);
}

/**
 * The icons.js glyph of each known companion panel's sidebar row. A sidecar
 * panel's glyph is its registry entry's; an application panel's is the one its
 * manifest names, which reaches the page as that panel's `web.panels.<id>.icon`;
 * a panel named in none of these places (another custom or runtime-registered
 * one) wears {@link FALLBACK_PANEL_ICON} unless its config block names one.
 * @type {Readonly<Record<string, string>>}
 */
export const PANEL_ICONS = Object.freeze({
  terminal: 'agent',
  artifacts: 'layers',
  events: 'activity',
  ariel: 'book-open',
  okf: 'library',
  'channel-finder': 'radio',
  bluesky: 'bluesky',
  optimizer: 'crosshair',
  'system-health': 'stethoscope',
});

/** The glyph of a panel {@link PANEL_ICONS} does not name. */
export const FALLBACK_PANEL_ICON = 'workspace';

/** Attribute carrying the glyph each `web.panels.<id>.icon` names, JSON. */
const PANEL_ICONS_ATTR = 'data-panel-icons';

/**
 * The glyphs this deployment's config gives its panels, by panel id. A name
 * icons.js does not draw is dropped (the build refuses one, so this is a
 * hand-edited config) and the panel keeps the glyph it would have had. An
 * absent stamp is a deployment that set none.
 * @returns {Record<string, string>}
 */
function configuredIcons() {
  const raw =
    typeof document === 'undefined'
      ? null
      : document.documentElement?.getAttribute(PANEL_ICONS_ATTR);
  /** @type {Record<string, string>} */
  const icons = {};
  if (!raw) return icons;
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return icons;
    for (const [id, name] of Object.entries(parsed)) {
      if (typeof name === 'string' && ICON_NAMES.includes(name)) icons[id] = name;
      else console.warn(`[panel-catalog] web.panels.${id}.icon names no icon: ${String(name)}`);
    }
  } catch {
    console.warn(`[panel-catalog] unreadable ${PANEL_ICONS_ATTR} on this page`);
  }
  return icons;
}

const CONFIGURED_ICONS = configuredIcons();

/**
 * The icons.js name for a panel's sidebar row: the glyph `web.panels.<id>.icon`
 * names for any panel (a deployment's own choice, or an application manifest's),
 * else the panel's own glyph. A sidecar glyph icons.js does not draw wears
 * {@link FALLBACK_PANEL_ICON}.
 * @param {string} id
 * @returns {string}
 */
export function panelIconName(id) {
  if (Object.hasOwn(CONFIGURED_ICONS, id)) return CONFIGURED_ICONS[id];
  const sidecar = SIDECAR_PANELS.find((p) => p.id === id);
  if (sidecar) {
    return sidecar.icon && ICON_NAMES.includes(sidecar.icon) ? sidecar.icon : FALLBACK_PANEL_ICON;
  }
  return Object.hasOwn(PANEL_ICONS, id) ? PANEL_ICONS[id] : FALLBACK_PANEL_ICON;
}

/**
 * One plain sentence per known companion panel, shown as the explanatory line
 * of its sidebar tooltip. A sidecar or application panel's sentence is its own
 * entry's; a panel named in none of these places (another custom or
 * runtime-registered one) has no description, and its tooltip stays a bare label.
 * @type {Readonly<Record<string, string>>}
 */
export const PANEL_DESCRIPTIONS = Object.freeze({
  terminal: 'Chat with the agent, approve its actions and run commands',
  artifacts: 'Files, plots and notebooks the agent produced',
  events: 'Event dispatcher dashboard',
  ariel: 'Search the electronic logbook',
  'channel-finder': 'Find control-system channels by name or property',
  okf: 'Browse the facility knowledge bundle',
  'system-health': 'Health checks for the framework and its services',
  bluesky: 'Write plans, run the queue and watch live results',
  optimizer: 'Review and start an optimizer run, then watch it converge',
});

/**
 * The tooltip sentence for a panel's sidebar row.
 * @param {string} id
 * @returns {string | null} `null` when the panel has no description
 */
export function describePanel(id) {
  const entry = panelEntry(id);
  if (entry?.description) return entry.description;
  if (SIDECAR_PANELS.some((p) => p.id === id)) return null;
  return Object.hasOwn(PANEL_DESCRIPTIONS, id) ? PANEL_DESCRIPTIONS[id] : null;
}

/** Attribute carrying the labels the deployment wrote, a JSON list. */
const AUTHORED_LABELS_ATTR = 'data-panel-authored-labels';

/**
 * The panel labels the deployment wrote itself. An absent or unreadable stamp
 * is an empty set: every all-capitals label then reads in title case.
 * @returns {Set<string>}
 */
function authoredLabels() {
  const raw =
    typeof document === 'undefined'
      ? null
      : document.documentElement?.getAttribute(AUTHORED_LABELS_ATTR);
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return new Set(parsed.filter((l) => typeof l === 'string'));
    } catch {
      // An unreadable stamp names no label.
    }
  }
  return new Set();
}

const AUTHORED_LABELS = authoredLabels();

/**
 * A panel label as the sidebar shows it. An all-capitals label the framework
 * derived (`ARIEL`, `CHANNELS`, a custom panel named after its id) reads in
 * title case: every word, split at spaces and hyphens, keeps only its first
 * letter capital. A label the deployment wrote is its own spelling and is left
 * as it is: one the page stamps as authored (`ASCC`), and any label with a
 * lower-case letter (`EPICS Archiver`, `pyAT`).
 * @param {string} label
 * @returns {string}
 */
export function titleCase(label) {
  if (AUTHORED_LABELS.has(label) || label !== label.toUpperCase()) return label;
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
  return titleCase(typeof label === 'string' && label ? label : id.toUpperCase());
}

/** The sidecar panels of this page, built once from the stamp. */
const SIDECAR_PANELS = sidecarPanels();

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
  // The sidecar panels, from the page's stamp, where the notebook panel has
  // always sat in the roster.
  ...SIDECAR_PANELS,
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
