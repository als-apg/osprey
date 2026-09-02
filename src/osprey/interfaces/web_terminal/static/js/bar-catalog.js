/* OSPREY Web Terminal — Bar item catalog.
 *
 * The single declaration of what may live in the global header and the status
 * bar. Both bars are item hosts rendering one ordered list each; every entry
 * here answers the questions a host, the layout model and the customize UI all
 * ask about a type, so none of them carries a private table of its own:
 *
 *   group          the heading the customize sheet files the tile under; the
 *                  headings render in BAR_GROUPS order.
 *   multi          true ⇒ a layout may place the type more than once. False
 *                  for every item whose body is one server-rendered node or
 *                  one id-owning dot: a second copy could only ever be empty.
 *                  Every type may live in either bar: there is no placement
 *                  axis. A host renders every body at its own density (see
 *                  DENSITY_BY_HOST), and a body that looks wrong at one of the
 *                  two densities is a CSS bug, never a reason to refuse a bar.
 *   options        per-type option spec: key, scalar kind, bounds, default.
 *                  bar-layout.js validates against this; the options popover
 *                  renders from it. Bounds live here so the client, the route
 *                  validator and the UI cannot disagree.
 *   priority       fold order for the overflow ladder — LOWEST folds first.
 *   align          `baseline` items are wrapped in a shared baseline run by the
 *                  host so the wordmark and the identity text sit on one line.
 *   flex           hint stamped on the item SHELL. Spacing back-pressure is
 *                  CSS-owned: a space shrinks continuously through its
 *                  declared flex-shrink ahead of every JS ladder rung, so
 *                  the chrome beside it can never be clipped by a spacing item.
 *   overflowLabel  non-null ⇒ the item folds into the overflow popover under
 *                  this label. null ⇒ it NEVER folds. Exactly six fold.
 *   available      false ⇒ the item is absent from this deployment, because
 *                  the deployment does not render its body.
 *
 * Pure data plus pure functions: no DOM, no storage, no network. The bodies
 * themselves are built elsewhere (bar-host.js), keyed by `type`.
 */

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
 * @property {boolean} [blueskyAvailable] - deployment declares the Bluesky panel,
 *   whose proxy is where the plan queue is read
 * @property {boolean} [systemHealthAvailable] - deployment enables the SYSTEM
 *   panel, whose proxy is where the health report is read
 */

/**
 * One catalog entry.
 * @typedef {object} BarItemType
 * @property {string} type
 * @property {string} label - human name used by tiles and context menus
 * @property {string} group - the sheet heading the tile sits under
 * @property {boolean} multi - may be placed more than once per layout
 * @property {Readonly<Record<string, BarOptionSpec>>} options
 * @property {number} priority
 * @property {BarAlign} align
 * @property {(options: BarItemOptions) => BarFlexHint | null} flex
 * @property {(ctx: BarContext) => string | null} overflowLabel
 * @property {(ctx: BarContext) => boolean} available
 */

/** The two hosts, in render order. */
export const BAR_HOSTS = /** @type {readonly BarHost[]} */ (Object.freeze(['header', 'status']));

/** The sheet's headings, in the order it renders them. */
export const BAR_GROUPS = Object.freeze([
  'Identity',
  'Machine',
  'Panels',
  'System',
  'Tools',
  'Layout',
]);

/** Each host renders at exactly one density (`--bar-item-size`: 28 px / 20 px). */
export const DENSITY_BY_HOST = Object.freeze(
  /** @type {Readonly<Record<BarHost, BarDensity>>} */ ({
    header: 'comfortable',
    status: 'compact',
  })
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
    group: 'Identity',
    multi: false,
    options: NO_OPTIONS,
    priority: 100,
    align: 'baseline',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  identity: {
    type: 'identity',
    label: 'Identity',
    group: 'Identity',
    multi: false,
    options: NO_OPTIONS,
    priority: 100,
    align: 'baseline',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    // Single-user deployments render no identity block at all; the item is then
    // absent rather than empty.
    available: (ctx) => ctx.identityAvailable === true,
  },

  'control-target': {
    type: 'control-target',
    label: 'Control target',
    group: 'Machine',
    multi: false,
    options: NO_OPTIONS,
    priority: 100,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  display: {
    type: 'display',
    label: 'Display',
    group: 'Tools',
    multi: false,
    options: NO_OPTIONS,
    priority: 100,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  search: {
    type: 'search',
    label: 'Command palette',
    group: 'Tools',
    multi: false,
    options: NO_OPTIONS,
    priority: 90,
    align: 'center',
    flex: NO_FLEX,
    // Search collapses to its magnifier under pressure and stops there; it is
    // never parked in the overflow popover.
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },

  'system-health': {
    type: 'system-health',
    label: 'System health',
    group: 'System',
    multi: false,
    // `text` is what the chip says beside its dot: nothing, or the suite's
    // worst status in a word. `detail` is what the card lists: one row per
    // category, or every check.
    options: Object.freeze({
      text: enumSpec(Object.freeze(['none', 'status']), 'none'),
      detail: enumSpec(Object.freeze(['categories', 'checks']), 'categories'),
    }),
    priority: 55,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'System health',
    // The report is read through the SYSTEM panel's proxy, so the item exists
    // exactly where that panel is enabled.
    available: (ctx) => ctx.systemHealthAvailable === true,
  },

  'bluesky-queue': {
    type: 'bluesky-queue',
    label: 'Bluesky queue',
    group: 'Panels',
    multi: false,
    // `controls` is what the popover may DO to the queue: nothing (it only
    // opens the panel), the plain stop, or the panel's full set with the
    // emergency abort. The chip itself never acts on a click — it opens.
    // `progress` and `count` are what the chip SAYS beside its dot.
    options: Object.freeze({
      controls: enumSpec(['none', 'stop', 'full'], 'none'),
      progress: booleanSpec(true),
      count: booleanSpec(true),
    }),
    priority: 20,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Bluesky queue',
    // The queue is read through the Bluesky panel's proxy, so the item exists
    // exactly where that panel is declared.
    available: (ctx) => ctx.blueskyAvailable === true,
  },

  docs: {
    type: 'docs',
    label: 'Documentation',
    group: 'Tools',
    multi: false,
    options: NO_OPTIONS,
    priority: 50,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Documentation',
    available: ALWAYS,
  },

  feedback: {
    type: 'feedback',
    label: 'Feedback',
    group: 'Tools',
    multi: false,
    options: NO_OPTIONS,
    priority: 45,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Feedback',
    available: ALWAYS,
  },

  clock: {
    type: 'clock',
    label: 'Clock',
    group: 'System',
    multi: true,
    // `none` is the plain clock: local time, no zone suffix anywhere. `local`
    // is the same time with the zone's name beside it; `utc` and `both` say
    // what they are. `format` is the hour cycle, 24h or 12h with AM/PM.
    options: Object.freeze({
      zone: enumSpec(Object.freeze(['none', 'local', 'utc', 'both']), 'none'),
      format: enumSpec(Object.freeze(['24h', '12h']), '24h'),
      seconds: booleanSpec(false),
    }),
    priority: 40,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Clock',
    available: ALWAYS,
  },

  stopwatch: {
    type: 'stopwatch',
    label: 'Stopwatch',
    group: 'System',
    multi: true,
    options: NO_OPTIONS,
    priority: 10,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: () => 'Stopwatch',
    available: ALWAYS,
  },

  // --- Spacing. Never folds: a space YIELDS instead, shrinking continuously
  // via the flex-shrink it declares below, which is why it carries a top
  // priority the JS ladder never consults.

  space: {
    type: 'space',
    label: 'Space',
    group: 'Layout',
    multi: true,
    // `width` 0 is the flexible space: it takes whatever room is left over.
    // Any other value is a fixed width the operator set by dragging the
    // space's edge, held until the bar runs out of room and then given up
    // before any real item is touched. The ceiling is wider than any bar.
    options: Object.freeze({ width: numberSpec(0, 2000, 0, 'px') }),
    priority: 100,
    align: 'center',
    flex: (options) => {
      const width = numberOption(options, 'width', 0);
      return width > 0 ? { flex: `0 1 ${width}px`, minWidth: '0' } : { flex: '1 1 0' };
    },
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },


  separator: {
    type: 'separator',
    label: 'Separator',
    group: 'Layout',
    multi: true,
    options: NO_OPTIONS,
    priority: 100,
    align: 'center',
    flex: NO_FLEX,
    overflowLabel: NEVER_FOLDS,
    available: ALWAYS,
  },
});

/** Every declared type, in catalog order. @type {readonly string[]} */
export const BAR_ITEM_TYPES = Object.freeze(Object.keys(BAR_CATALOG));

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
