/* OSPREY Web Terminal — bar layout model.
 *
 * The schema-v1 layout document and the one function that turns anything a
 * server, a config file or an older build hands us into a document this build
 * can render:
 *
 *   {version, rev, header[], status[], status_visible}
 *
 * `normalize()` is the whole contract. It never throws and always returns a
 * valid document, because every input it sees is untrusted: a saved layout
 * written by a NEWER deployment, a hand-edited config block, a half-written
 * file. Items it cannot honour are dropped silently — unknown types, types the
 * host cannot render, items this deployment has no body for, and anything past
 * `MAX_ITEMS_PER_HOST`.
 *
 * Silence is safe only because the result says what happened. Dropping is
 * loss, and a client that wrote the dropped-down document back would DESTROY
 * the user's layout the moment a rollback, a disabled bridge or a stale build
 * made an item unrenderable. So the result carries `readonly`: content was
 * dropped, a stored value was overwritten, or the version was unknown, and the
 * client must render what it got, offer no Customize entry, and issue ZERO
 * PUTs until the user explicitly resets. `changed` answers the weaker question
 * — is this document byte-identical to the input — and is true for lossless
 * completions too, such as an option arriving with its declared default.
 *
 * `merge()` is the second half of the contract: a user document says what the
 * user arranged, and a LOCKED item is one they were never allowed to remove.
 * Merge puts back the locked items a document is missing — at the index the
 * deployment default gives them — and touches nothing else.
 *
 * `PRESETS` is the stock arrangements the customize sheet offers. They are
 * plain documents, so applying one is the same operation as any other edit.
 *
 * Pure: no DOM, no storage, no network, and no runtime dependency on the
 * catalog either. The catalog is a PARAMETER, so this module can be exercised
 * against a fixture catalog and cannot drift into holding item knowledge of
 * its own — including which types are locked, which `merge()` is TOLD.
 */

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */
/** @typedef {import('./bar-catalog.js').BarItemType} BarItemType */
/** @typedef {import('./bar-catalog.js').BarItemOptions} BarItemOptions */
/** @typedef {import('./bar-catalog.js').BarOptionSpec} BarOptionSpec */
/** @typedef {import('./bar-catalog.js').BarContext} BarCatalogContext */

/** The catalog as this module consumes it: types keyed by name. */
/** @typedef {Readonly<Record<string, BarItemType>>} BarCatalog */

/**
 * Deployment facts, plus the deployment's own default document. The default is
 * what an unknown `version` falls back to; when it is absent the fallback is an
 * empty document, which renders nothing rather than guessing an order.
 * @typedef {BarCatalogContext & {defaultLayout?: unknown}} BarLayoutContext
 */

/**
 * One placed item. `type` keys the catalog; `options` is always complete —
 * every option the type declares is present, merged over its default.
 * @typedef {object} BarLayoutItem
 * @property {string} type
 * @property {BarItemOptions} options
 */

/**
 * The schema-v1 document. `rev` is the server's monotonic revision (0 for a
 * document that has never been saved); `status_visible` hides the status bar
 * without emptying it.
 * @typedef {object} BarLayout
 * @property {number} version
 * @property {number} rev
 * @property {readonly BarLayoutItem[]} header
 * @property {readonly BarLayoutItem[]} status
 * @property {boolean} status_visible
 */

/**
 * Why one entry did not survive. `index` is its position in the RAW list, so a
 * log line points at the stored document rather than at the result.
 * @typedef {object} BarLayoutDrop
 * @property {BarHost} host
 * @property {number} index
 * @property {string} type - '' when the entry carried no readable type
 * @property {'malformed' | 'unknown-type' | 'host-mismatch' | 'unavailable' | 'overflow'} reason
 */

/**
 * One stock arrangement offered by the customize sheet. `layout` is an ordinary
 * document at `rev: 0` — applying a preset is an edit like any other, so it
 * goes through the same normalize/merge/PUT path a drag does.
 * @typedef {object} BarPreset
 * @property {string} id
 * @property {string} label - the sheet's button text
 * @property {string} description - one line, what the arrangement is for
 * @property {BarLayout} layout
 */

/**
 * @typedef {object} BarLayoutResult
 * @property {BarLayout} layout - always valid, always frozen
 * @property {boolean} changed - the layout is not identical to the input
 * @property {boolean} readonly - content was lost: render it, but never PUT it
 * @property {readonly BarLayoutDrop[]} dropped
 */

/** The only schema version this build can read. */
export const BAR_LAYOUT_VERSION = 1;

/**
 * Per host, not per document: two capped lists already bound the total, and a
 * separate total cap would make a legal header edit fail because of the status
 * bar.
 */
export const MAX_ITEMS_PER_HOST = 20;

/** Internal per-host outcome, before the two hosts are folded into a result. */
/**
 * @typedef {object} HostResult
 * @property {BarLayoutItem[]} items
 * @property {boolean} changed
 * @property {boolean} lossy
 * @property {BarLayoutDrop[]} dropped
 */

/**
 * A valid, empty v1 document. The last-resort fallback: it renders no items
 * rather than inventing an order the deployment never declared.
 * @returns {BarLayout}
 */
export function emptyLayout() {
  return freezeLayout({
    version: BAR_LAYOUT_VERSION,
    rev: 0,
    header: [],
    status: [],
    status_visible: true,
  });
}

/**
 * Turn an untrusted document into one this build can render.
 *
 * @param {unknown} raw - a stored, served or configured document; anything at all
 * @param {BarCatalog} catalog - the item catalog to validate against
 * @param {BarLayoutContext} [ctx] - deployment facts, plus `defaultLayout`
 * @returns {BarLayoutResult}
 */
export function normalize(raw, catalog, ctx = {}) {
  const doc = asRecord(raw);
  // An unreadable document and an unknown version take the same exit: this
  // build cannot claim to understand the content, so it must not overwrite it.
  if (!doc || doc.version !== BAR_LAYOUT_VERSION) return fallback(catalog, ctx);
  return normalizeDocument(doc, catalog, ctx, false);
}

/**
 * Put back the locked items a user document is missing.
 *
 * Locked means movable, never removable — so a document that lacks one is not
 * a user preference to be honoured but a document that predates the lock, or
 * came from a deployment that did not have it. The missing item returns at the
 * index the DEFAULT gives it, which is the only opinion about placement anyone
 * has; everything the user did arrange keeps its order and its options.
 *
 * Presence is asked of the WHOLE document, not one host: a locked item the user
 * moved to the other bar is still present, and re-inserting it in its default
 * host would silently duplicate it. Locked types the default never places are
 * not invented — there is no index to place them at.
 *
 * Which types are locked is the CALLER's to state: it unions the catalog's own
 * locked types with the deployment's `web.bar_items.locked`, so a deployment
 * can lock more without this module learning any item names.
 *
 * Pure and idempotent: `merge(d, merge(d, u, l), l)` is `merge(d, u, l)`.
 *
 * @param {BarLayout} defaultLayout - the deployment default; the source of both
 *   the restored items and their indices
 * @param {BarLayout} userLayout - a NORMALIZED user document
 * @param {readonly string[]} [locked] - every type this deployment locks
 * @returns {BarLayout}
 */
export function merge(defaultLayout, userLayout, locked = []) {
  const lockedTypes = new Set(locked);
  const present = new Set(
    [...userLayout.header, ...userLayout.status].map((entry) => entry.type)
  );
  return freezeLayout({
    version: BAR_LAYOUT_VERSION,
    rev: userLayout.rev,
    header: restoreLocked(defaultLayout.header, userLayout.header, lockedTypes, present),
    status: restoreLocked(defaultLayout.status, userLayout.status, lockedTypes, present),
    status_visible: userLayout.status_visible,
  });
}

/**
 * One host's list with its missing locked items spliced back in.
 *
 * Insertion walks the default in order and places each missing item at its own
 * default index, clamped to the current length, so a run of restored items
 * lands in default-relative order around whatever the user kept. `present` is
 * shared across both hosts and grows as items are restored — a type restored
 * into the header is not restored into the status bar too.
 *
 * The result is capped exactly as `normalizeHost` caps: the tail goes. Without
 * that, restoring an item into a full bar would hand back a document whose next
 * normalization drops something and turns the layout read-only.
 *
 * @param {readonly BarLayoutItem[]} defaults
 * @param {readonly BarLayoutItem[]} userItems
 * @param {ReadonlySet<string>} lockedTypes
 * @param {Set<string>} present - types already placed anywhere in the document
 * @returns {BarLayoutItem[]}
 */
function restoreLocked(defaults, userItems, lockedTypes, present) {
  const items = [...userItems];
  defaults.forEach((entry, index) => {
    if (!lockedTypes.has(entry.type) || present.has(entry.type)) return;
    items.splice(Math.min(index, items.length), 0, entry);
    present.add(entry.type);
  });
  return items.slice(0, MAX_ITEMS_PER_HOST);
}

/**
 * The deployment default, itself normalized, flagged read-only. Normalizing the
 * default too means a deployment that ships an item this build cannot render
 * degrades the same way a user document does, instead of rendering a hole.
 * @param {BarCatalog} catalog
 * @param {BarLayoutContext} ctx
 * @returns {BarLayoutResult}
 */
function fallback(catalog, ctx) {
  const source = asRecord(ctx.defaultLayout);
  if (!source) return { layout: emptyLayout(), changed: true, readonly: true, dropped: [] };
  const result = normalizeDocument(source, catalog, ctx, true);
  return { ...result, changed: true, readonly: true };
}

/**
 * Normalize a document's CONTENT, version already decided.
 * @param {Record<string, unknown>} doc
 * @param {BarCatalog} catalog
 * @param {BarLayoutContext} ctx
 * @param {boolean} isDefault - a deployment default has no revision of its own
 * @returns {BarLayoutResult}
 */
function normalizeDocument(doc, catalog, ctx, isDefault) {
  const header = normalizeHost(doc.header, 'header', catalog, ctx);
  const status = normalizeHost(doc.status, 'status', catalog, ctx);

  const rev = isDefault ? 0 : asRev(doc.rev);
  const revChanged = !isDefault && rev !== doc.rev;
  const statusVisible = typeof doc.status_visible === 'boolean' ? doc.status_visible : true;
  const visibleChanged = typeof doc.status_visible !== 'boolean';

  return {
    layout: freezeLayout({
      version: BAR_LAYOUT_VERSION,
      rev,
      header: header.items,
      status: status.items,
      status_visible: statusVisible,
    }),
    changed: header.changed || status.changed || revChanged || visibleChanged,
    readonly: header.lossy || status.lossy,
    dropped: [...header.dropped, ...status.dropped],
  };
}

/**
 * Filter one host's list. Order is the layout, so surviving items keep their
 * relative order and the cap takes the TAIL — the items a user would have to
 * scroll a popover to reach, not the chrome at the start of the bar.
 * @param {unknown} rawList
 * @param {BarHost} host
 * @param {BarCatalog} catalog
 * @param {BarLayoutContext} ctx
 * @returns {HostResult}
 */
function normalizeHost(rawList, host, catalog, ctx) {
  /** @type {BarLayoutItem[]} */
  const items = [];
  /** @type {BarLayoutDrop[]} */
  const dropped = [];
  let changed = false;
  let lossy = false;

  // A missing or non-array list is a malformed document, not an empty bar: the
  // document never stated a list we could honour, so refuse to write it back.
  if (!Array.isArray(rawList)) return { items, changed: true, lossy: true, dropped };

  const hostCtx = { ...ctx, host };
  rawList.forEach((entry, index) => {
    const raw = asRecord(entry);
    const type = raw && typeof raw.type === 'string' ? raw.type : '';
    const reason = refuse(raw, type, host, catalog, hostCtx, items.length);
    if (reason) {
      dropped.push({ host, index, type, reason });
      return;
    }
    const options = normalizeOptions(raw?.options, catalog[type]);
    if (options.changed) changed = true;
    if (options.lossy) lossy = true;
    items.push({ type, options: options.value });
  });

  return {
    items,
    changed: changed || dropped.length > 0,
    lossy: lossy || dropped.length > 0,
    dropped,
  };
}

/**
 * Why this entry cannot be placed, or null when it can. The order matters: a
 * type must be known before its hosts can be asked, and the cap is checked LAST
 * so a rejected entry never consumes a slot a good one could have used.
 * @param {Record<string, unknown> | null} raw
 * @param {string} type
 * @param {BarHost} host
 * @param {BarCatalog} catalog
 * @param {BarCatalogContext} hostCtx
 * @param {number} kept - how many items already survived in this host
 * @returns {BarLayoutDrop['reason'] | null}
 */
function refuse(raw, type, host, catalog, hostCtx, kept) {
  if (!raw || !type) return 'malformed';
  if (!hasOwn(catalog, type)) return 'unknown-type';
  const entry = catalog[type];
  // `hosts` is a hard capability: a header-only body cannot render in a 26 px
  // status bar, so a status placement is refused rather than re-homed.
  if (!entry.hosts.includes(host)) return 'host-mismatch';
  if (!entry.available(hostCtx)) return 'unavailable';
  if (kept >= MAX_ITEMS_PER_HOST) return 'overflow';
  return null;
}

/**
 * Complete an item's options against the type's spec: every declared option
 * present, unknown keys discarded, out-of-spec values repaired.
 *
 * The two outcomes are deliberately different in weight. Supplying a MISSING
 * default loses nothing (`changed`, still writable) — that is how an option
 * added in a later build reaches an older document. Discarding an unknown key
 * or repairing a stored value DOES lose what was written (`lossy`), so the
 * document becomes read-only rather than being silently rewritten.
 *
 * @param {unknown} rawOptions
 * @param {BarItemType} entry
 * @returns {{value: BarItemOptions, changed: boolean, lossy: boolean}}
 */
function normalizeOptions(rawOptions, entry) {
  const stored = asRecord(rawOptions) ?? {};
  /** @type {Record<string, string | number | boolean>} */
  const value = {};
  let changed = false;
  let lossy = false;

  for (const [key, spec] of Object.entries(entry.options)) {
    const raw = stored[key];
    if (raw === undefined) {
      value[key] = spec.default;
      changed = true;
      continue;
    }
    const coerced = coerceOption(raw, spec);
    value[key] = coerced ?? spec.default;
    if (coerced !== raw) lossy = true;
  }

  for (const key of Object.keys(stored)) {
    if (!hasOwn(entry.options, key)) lossy = true;
  }

  return { value: Object.freeze(value), changed: changed || lossy, lossy };
}

/**
 * A stored option value as its spec allows it, or null when nothing usable can
 * be made of it. Numbers are CLAMPED rather than reset: a 900 px gap from a
 * build with wider bounds keeps the user's intent at this build's maximum.
 * @param {unknown} raw
 * @param {BarOptionSpec} spec
 * @returns {string | number | boolean | null}
 */
function coerceOption(raw, spec) {
  if (spec.kind === 'number') {
    if (typeof raw !== 'number' || !Number.isFinite(raw)) return null;
    return Math.min(spec.max, Math.max(spec.min, raw));
  }
  if (spec.kind === 'boolean') return typeof raw === 'boolean' ? raw : null;
  return typeof raw === 'string' && spec.values.includes(raw) ? raw : null;
}

/**
 * A revision is a non-negative integer; anything else reads as "never saved",
 * which the server answers with a 409 carrying the real document.
 * @param {unknown} raw
 * @returns {number}
 */
function asRev(raw) {
  return typeof raw === 'number' && Number.isInteger(raw) && raw >= 0 ? raw : 0;
}

/**
 * @param {unknown} value
 * @returns {Record<string, unknown> | null}
 */
function asRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? /** @type {Record<string, unknown>} */ (value)
    : null;
}

/**
 * Own-property test that cannot be answered by an inherited member — `toString`
 * is not an item type and `constructor` is not an option key.
 * @param {object} target
 * @param {string} key
 * @returns {boolean}
 */
function hasOwn(target, key) {
  return Object.prototype.hasOwnProperty.call(target, key);
}

/**
 * Freeze a document through its items, so a consumer that mutates what it was
 * handed fails loudly instead of corrupting the next comparison.
 * @param {{version: number, rev: number, header: BarLayoutItem[],
 *          status: BarLayoutItem[], status_visible: boolean}} layout
 * @returns {BarLayout}
 */
function freezeLayout(layout) {
  return Object.freeze({
    ...layout,
    header: Object.freeze(layout.header.map(freezeItem)),
    status: Object.freeze(layout.status.map(freezeItem)),
  });
}

/**
 * @param {BarLayoutItem} item
 * @returns {BarLayoutItem}
 */
function freezeItem(item) {
  return Object.freeze({ type: item.type, options: Object.freeze(item.options) });
}

// --- Presets -----------------------------------------------------------------
//
// Stock arrangements for the customize sheet. Each is a complete document, with
// every option written out at the value the catalog declares as its default, so
// a preset normalizes CLEAN — a preset that arrived `changed` would make a fresh
// user document look edited, and one that arrived `readonly` could not be
// applied at all. That is the property the suite pins.
//
// The item names live here as data, not as knowledge: nothing in this module
// reads them, and a deployment whose catalog lacks one of these types simply
// normalizes it away, exactly as it does for a saved layout.

/**
 * One placed item at its declared options.
 * @param {string} type
 * @param {Record<string, string | number | boolean>} [options]
 * @returns {BarLayoutItem}
 */
function presetItem(type, options = {}) {
  return { type, options };
}

/**
 * @param {string} id
 * @param {string} label
 * @param {string} description
 * @param {BarLayoutItem[]} header
 * @param {BarLayoutItem[]} status
 * @returns {BarPreset}
 */
function preset(id, label, description, header, status) {
  return Object.freeze({
    id,
    label,
    description,
    layout: freezeLayout({
      version: BAR_LAYOUT_VERSION,
      rev: 0,
      header,
      status,
      status_visible: true,
    }),
  });
}

/**
 * The header every preset ships: the four locked items, the command palette and
 * the display menu, with a flexible space holding the wordmark and the identity
 * text apart from the action cluster. Every preset carries all four locked ids,
 * so applying one never depends on `merge()` to repair it.
 * @returns {BarLayoutItem[]}
 */
function standardHeader() {
  return [
    presetItem('logo'),
    presetItem('identity'),
    presetItem('space', { share: 1 }),
    presetItem('control-target'),
    presetItem('search'),
    presetItem('display'),
  ];
}

/**
 * The arrangements the sheet offers, in the order it renders them.
 *
 * `full` is the status bar as it stood before the bars became item hosts;
 * `clean` is the shipped deployment default — activity stretching across to a
 * right-hand cluster of documentation, the panel dots and the clock, with the
 * connection dot and the terminal size available but not placed. `minimal`
 * strips back to the locked chrome plus what an unattended screen needs.
 *
 * There is no "deployment default" preset: resetting to the deployment's own
 * layout is a DELETE, not a document the client can hold, since a deployment
 * may have configured something none of these describe.
 *
 * @type {readonly BarPreset[]}
 */
export const PRESETS = Object.freeze([
  preset(
    'clean',
    'Clean',
    'Activity across the bar, with documentation, panel health and the clock at the right.',
    standardHeader(),
    [
      presetItem('activity'),
      presetItem('panel-health'),
      presetItem('docs'),
      presetItem('clock', { zone: 'local', seconds: false }),
    ]
  ),
  preset(
    'full',
    'Everything',
    'Adds the connection dot and the terminal size to the clean status bar.',
    standardHeader(),
    [
      presetItem('connection'),
      presetItem('panel-health'),
      presetItem('terminal-size'),
      presetItem('activity'),
      presetItem('docs'),
      presetItem('clock', { zone: 'local', seconds: false }),
    ]
  ),
  preset(
    'minimal',
    'Minimal',
    'The chrome that cannot be removed, plus activity and the clock.',
    [
      presetItem('logo'),
      presetItem('identity'),
      presetItem('space', { share: 1 }),
      presetItem('control-target'),
      presetItem('display'),
    ],
    [presetItem('activity'), presetItem('clock', { zone: 'local', seconds: false })]
  ),
]);
