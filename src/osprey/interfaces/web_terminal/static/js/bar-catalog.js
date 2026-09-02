/* OSPREY Web Terminal — Bar item catalog.
 *
 * The single declaration of what may live in the global header and the status
 * bar. Both bars are item hosts rendering one ordered list each; every entry
 * here answers the questions a host, the layout model and the customize UI all
 * ask about a type, so none of them carries a private table of its own:
 *
 *   hosts          which bar may hold it — a HARD capability, not a default.
 *                  `logo/identity/control-target/search/display` are header-only
 *                  because their bodies cannot render in a 24–26 px bar.
 *   densities      which render densities the body supports. One density per
 *                  host (see DENSITY_BY_HOST) — the axes are kept apart because
 *                  `hosts` is placement policy and `densities` is a rendering
 *                  fact about the body.
 *   options        per-type option spec: key, scalar kind, bounds, default.
 *                  bar-layout.js validates against this; the options popover
 *                  renders from it. Bounds live here so the client, the route
 *                  validator and the UI cannot disagree.
 *   priority       fold order for the overflow ladder — LOWEST folds first.
 *   locked         movable, never removable. Exactly four: logo, identity,
 *                  control-target, display. Deployments may lock more via
 *                  `web.bar_items.locked`; they may never unlock these.
 *   align          `baseline` items are wrapped in a shared baseline run by the
 *                  host so the wordmark and the identity text sit on one line.
 *   flex           hint stamped on the item SHELL. Spacing back-pressure is
 *                  CSS-owned: gaps and spaces shrink continuously through their
 *                  declared flex-shrink ahead of every JS ladder rung, so
 *                  locked chrome can never be clipped by a spacing item.
 *   overflowLabel  non-null ⇒ the item folds into the overflow popover under
 *                  this label. null ⇒ it NEVER folds. Exactly six fold.
 *   available      false ⇒ the item is absent from this deployment regardless
 *                  of lock, because the deployment does not render its body.
 *
 * Pure data plus pure functions: no DOM, no storage, no network. The bodies
 * themselves are built elsewhere (bar-host.js), keyed by `type`.
 */

import { PANELS } from './panel-catalog.js';

/** @typedef {'header' | 'status'} BarHost */
/** @typedef {'comfortable' | 'compact'} BarDensity */
/** @typedef {'center' | 'baseline'} BarAlign */

/**
 * A placed item's own options, already merged over the type's defaults.
 * @typedef {Readonly<Record<string, string | number | boolean>>} BarItemOptions
 */

/**
 * One option's spec. `kind` is the scalar type; number options carry inclusive
 * bounds and enum options carry a closed value list.
 * @typedef {{kind: 'number', min: number, max: number, step: number,
 *            unit: string | null, default: number}
 *          | {kind: 'boolean', default: boolean}
 *          | {kind: 'enum', values: readonly string[], default: string}} BarOptionSpec
 */

/**
 * What the shell's inline style declares for an item that participates in the
 * bar's spare-space arithmetic. `null` from `flex()` means "stamp nothing".
 * @typedef {{flex: string, minWidth?: string}} BarFlexHint
 */

/**
 * Deployment and placement facts the catalog's predicates read. Every field is
 * optional and an ABSENT fact reads as "not present", so a caller that knows
 * nothing yet gets the conservative answer rather than a phantom item.
 * @typedef {object} BarContext
 * @property {BarHost} [host] - which bar the item is being considered for
 * @property {BarItemOptions} [options] - the placed item's own options
 * @property {boolean} [identityAvailable] - deployment renders the header identity block
 * @property {boolean} [blueskyAvailable] - deployment runs a bluesky bridge
 * @property {readonly string[]} [statusBarIds] - ids of the status dots this
 *   deployment actually renders, one per ENABLED built-in panel. A subset of
 *   {@link PANEL_HEALTH_STATUS_BAR_IDS}, which is every id that could appear.
 */

/**
 * One catalog entry.
 * @typedef {object} BarItemType
 * @property {string} type
 * @property {string} label - human name used by tiles and context menus
 * @property {readonly BarHost[]} hosts
 * @property {readonly BarDensity[]} densities
 * @property {Readonly<Record<string, BarOptionSpec>>} options
 * @property {number} priority
 * @property {boolean} locked
 * @property {BarAlign} align
 * @property {(options: BarItemOptions) => BarFlexHint | null} flex
 * @property {(ctx: BarContext) => string | null} overflowLabel
 * @property {(ctx: BarContext) => boolean} available
 */

/** The two hosts, in render order. */
export const BAR_HOSTS = /** @type {readonly BarHost[]} */ (Object.freeze(['header', 'status']));

/** Each host renders at exactly one density (`--bar-item-size`: 28 px / 20 px). */
export const DENSITY_BY_HOST = Object.freeze(
  /** @type {Readonly<Record<BarHost, BarDensity>>} */ ({
    header: 'comfortable',
    status: 'compact',
  })
);

const HEADER_ONLY = /** @type {readonly BarHost[]} */ (Object.freeze(['header']));
const BOTH_HOSTS = /** @type {readonly BarHost[]} */ (Object.freeze(['header', 'status']));
const COMFORTABLE = /** @type {readonly BarDensity[]} */ (Object.freeze(['comfortable']));
const BOTH_DENSITIES = /** @type {readonly BarDensity[]} */ (
  Object.freeze(['comfortable', 'compact'])
);

/** @type {Readonly<Record<string, BarOptionSpec>>} */
const NO_OPTIONS = Object.freeze({});

/** Shared predicates — the overwhelmingly common answers, named once. */
const ALWAYS = () => true;
/** @type {() => string | null} */
const NEVER_FOLDS = () => null;
/** @type {() => BarFlexHint | null} */
const NO_FLEX = () => null;

/**
 * The status-bar ids the panel-health item may own: a CLOSED set, derived from
 * the shipped panel catalog rather than retyped, so a panel that gains or loses
 * a `statusBarId` cannot leave a stale id behind here. Facility panels
 * registered at runtime are deliberately excluded — panel-health renders one
 * dot per enabled BUILT-IN panel.
 * @type {readonly string[]}
 */
export const PANEL_HEALTH_STATUS_BAR_IDS = Object.freeze(
  PANELS.reduce((ids, panel) => {
    if (panel.statusBarId) ids.push(panel.statusBarId);
    return ids;
  }, /** @type {string[]} */ ([]))
);

/**
 * Build a bounded numeric option spec. A builder rather than a bare literal so
 * the discriminant stays the literal type `'number'` under `Object.freeze`.
 * @param {number} min
 * @param {number} max
 * @param {number} fallback - the default value, which must sit inside the bounds
 * @param {string | null} [unit]
 * @param {number} [step]
 * @returns {BarOptionSpec}
 */
function numberSpec(min, max, fallback, unit = null, step = 1) {
  return { kind: 'number', min, max, step, unit, default: fallback };
}

/**
 * Build a boolean option spec.
 * @param {boolean} fallback
 * @returns {BarOptionSpec}
 */
function booleanSpec(fallback) {
  return { kind: 'boolean', default: fallback };
}

/**
 * Build a closed-value option spec.
 * @param {readonly string[]} values
 * @param {string} fallback
 * @returns {BarOptionSpec}
 */
function enumSpec(values, fallback) {
  return { kind: 'enum', values, default: fallback };
}

/**
 * Read a numeric option, falling back to the spec's default when the stored
 * value is missing or not a finite number.
 * @param {BarItemOptions} options
 * @param {string} key
 * @param {number} fallback
 * @returns {number}
 */
function numberOption(options, key, fallback) {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** @type {Readonly<Record<string, BarItemType>>} */
export const BAR_CATALOG = Object.freeze({
  logo: {
    type: 'logo',
    label: 'Logo',
    hosts: HEADER_ONLY,
    densities: COMFORTABLE,
    options: NO_OPTIONS,
    priority: 100,
    locked: true,
    align: 'baseline',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  identity: {
    type: 'identity',
    label: 'Identity',
    hosts: HEADER_ONLY,
    densities: COMFORTABLE,
    options: NO_OPTIONS,
    priority: 100,
    locked: true,
    align: 'baseline',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    // Single-user deployments render no identity block at all; the item is then
    // absent rather than locked-but-empty.
    available: (ctx) => ctx.identityAvailable === true,
  },

  'control-target': {
    type: 'control-target',
    label: 'Control target',
    hosts: HEADER_ONLY,
    densities: COMFORTABLE,
    options: NO_OPTIONS,
    priority: 100,
    locked: true,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  display: {
    type: 'display',
    label: 'Display',
    hosts: HEADER_ONLY,
    densities: COMFORTABLE,
    options: NO_OPTIONS,
    priority: 100,
    locked: true,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  search: {
    type: 'search',
    label: 'Command palette',
    hosts: HEADER_ONLY,
    densities: COMFORTABLE,
    options: NO_OPTIONS,
    priority: 90,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    // Search collapses to its magnifier under pressure and stops there; it is
    // never parked in the overflow popover.
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  connection: {
    type: 'connection',
    label: 'Connection',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 80,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  'panel-health': {
    type: 'panel-health',
    label: 'Panel health',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 70,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: (ctx) =>
      (ctx.statusBarIds ?? []).some((id) => PANEL_HEALTH_STATUS_BAR_IDS.includes(id)),
  },

  activity: {
    type: 'activity',
    label: 'Activity',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 60,
    locked: false,
    align: 'center',
    // The one item that absorbs the bar's spare space; `min-width: 0` lets its
    // aria-live text ellipsize instead of pushing the locked chrome out.
    flex: () => ({ flex: '1 1 0', minWidth: '0' }),
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  docs: {
    type: 'docs',
    label: 'Documentation',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 50,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Documentation',
    available: ALWAYS,
  },

  feedback: {
    type: 'feedback',
    label: 'Feedback',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 45,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Feedback',
    available: ALWAYS,
  },

  clock: {
    type: 'clock',
    label: 'Clock',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: Object.freeze({
      zone: enumSpec(Object.freeze(['local', 'utc', 'both']), 'local'),
      seconds: booleanSpec(false),
    }),
    priority: 40,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Clock',
    available: ALWAYS,
  },

  'terminal-size': {
    type: 'terminal-size',
    label: 'Terminal size',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 30,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Terminal size',
    available: ALWAYS,
  },

  'bluesky-queue': {
    type: 'bluesky-queue',
    label: 'Plan queue',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 20,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Plan queue',
    available: (ctx) => ctx.blueskyAvailable === true,
  },

  stopwatch: {
    type: 'stopwatch',
    label: 'Stopwatch',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 10,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Stopwatch',
    available: ALWAYS,
  },

  // --- Spacing. Never folds: these YIELD instead, shrinking continuously via
  // the flex-shrink they declare below, which is why they carry a top priority
  // the JS ladder never consults.

  space: {
    type: 'space',
    label: 'Flexible space',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: Object.freeze({ share: numberSpec(1, 3, 1) }),
    priority: 100,
    locked: false,
    align: 'center',
    flex: (options) => ({ flex: `${numberOption(options, 'share', 1)} 1 0` }),
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  gap: {
    type: 'gap',
    label: 'Fixed gap',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: Object.freeze({ size: numberSpec(4, 400, 12, 'px') }),
    priority: 100,
    locked: false,
    align: 'center',
    // Basis is the requested width, shrink is 1: a gap holds its size until the
    // bar runs out of room, then gives it up before any real item is touched.
    flex: (options) => ({ flex: `0 1 ${numberOption(options, 'size', 12)}px`, minWidth: '0' }),
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  separator: {
    type: 'separator',
    label: 'Separator',
    hosts: BOTH_HOSTS,
    densities: BOTH_DENSITIES,
    options: NO_OPTIONS,
    priority: 100,
    locked: false,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },
});

/** Every declared type, in catalog order. @type {readonly string[]} */
export const BAR_ITEM_TYPES = Object.freeze(Object.keys(BAR_CATALOG));

/** Types that may never be removed from a layout. @type {readonly string[]} */
export const LOCKED_BAR_ITEM_TYPES = Object.freeze(
  BAR_ITEM_TYPES.filter((type) => BAR_CATALOG[type].locked)
);

/**
 * Look a type up. Returns null for an unknown type rather than throwing:
 * normalization drops unknown types silently, and a stale saved layout from a
 * newer deployment is a normal thing to receive, not an error.
 * @param {string} type
 * @returns {BarItemType | null}
 */
export function barItemType(type) {
  return Object.prototype.hasOwnProperty.call(BAR_CATALOG, type) ? BAR_CATALOG[type] : null;
}

/**
 * Whether a type may be placed in a host. Unknown types are refused.
 * @param {string} type
 * @param {BarHost} host
 * @returns {boolean}
 */
export function supportsHost(type, host) {
  const entry = barItemType(type);
  return entry ? entry.hosts.includes(host) : false;
}

/**
 * The density a host renders at.
 * @param {BarHost} host
 * @returns {BarDensity}
 */
export function densityForHost(host) {
  return DENSITY_BY_HOST[host];
}

/**
 * The type's option defaults as a fresh object. Callers merge stored options
 * over this, so an option added in a later version arrives with its default
 * rather than as `undefined`.
 * @param {string} type
 * @returns {Record<string, string | number | boolean>}
 */
export function defaultOptions(type) {
  const entry = barItemType(type);
  /** @type {Record<string, string | number | boolean>} */
  const defaults = {};
  if (!entry) return defaults;
  for (const [key, spec] of Object.entries(entry.options)) defaults[key] = spec.default;
  return defaults;
}
