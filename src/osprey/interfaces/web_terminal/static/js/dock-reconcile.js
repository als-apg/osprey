// @ts-check
/* OSPREY Web Terminal — Dock Layout Reconcile
 *
 * The pure layout-repair algorithm run on EVERY dockview layout apply (storage
 * load, mode restore, reset): a persisted SerializedDockview is reconciled
 * against the panels the dock can currently build. Unknown ids are dropped,
 * newly-registered panels are appended, and only genuinely corrupt/unusable
 * input falls back (returns null).
 *
 * PURE by contract — nothing in this module reads or mutates live dock/DOM
 * state, so it is unit-testable without a browser (dock-reconcile.test.mjs).
 * The side-effectful plumbing (fromJSON apply, localStorage, project key)
 * lives in dock-workspace.js, this module's only production consumer.
 */

/** dockview panel-id namespace for the iframe adapter's placeholder panels,
 *  kept distinct from the shell's own 'terminal' id and from panel-manager's
 *  rail ids. Defined here because reconcile() treats the namespace as
 *  always-recreatable; dock-iframe.js re-exports it as the adapter's public
 *  constant. */
export const PLACEHOLDER_PREFIX = 'iframe:';

/**
 * @typedef {object} PanelDescriptor
 * @property {string} id
 * @property {string} [contentComponent]  dockview component name (defaults to id).
 * @property {string} [tabComponent]
 * @property {string} [title]
 * @property {any}    [params]
 */

/**
 * Reconcile a persisted dockview layout against the panels the dock can build
 * right now. PURE — it never reads or mutates live dock/DOM state and never
 * touches the argument object (a defensive deep copy is reconciled and
 * returned). Runs on EVERY layout apply (storage load, mode restore, reset) so a
 * stored layout can neither reference a panel that no longer exists nor silently
 * omit one that now does.
 *
 * @param {any} layout  A SerializedDockview object, or its raw JSON string
 *   (localStorage holds the string). A string is parsed here; a parse failure is
 *   treated as corrupt input.
 * @param {Array<PanelDescriptor|string>} registeredPanels  The panels the dock
 *   can (re)create. A bare id string is accepted as `{id, contentComponent: id}`.
 * @returns {any|null}  The reconciled layout, or `null` when the input is
 *   unparseable / structurally unusable — the ONLY case that warrants a full
 *   fallback to the synthesized default. A merely-stale (but well-formed) layout
 *   is repaired in place, never discarded.
 */
export function reconcile(layout, registeredPanels) {
  const descriptors = normalizeDescriptors(registeredPanels);
  const registered = new Set(descriptors.map((d) => d.id));

  let data;
  try {
    data = typeof layout === 'string' ? JSON.parse(layout) : JSON.parse(JSON.stringify(layout));
  } catch {
    return null;
  }
  if (!isSerializedDockview(data)) return null;

  // 1. Panels map: keep registered ids — and EVERY `iframe:` placeholder id.
  //    Placeholders are always recreatable (their component is the adapter's
  //    neutral factory), and at apply time the adapter may not have created
  //    this session's placeholders yet (panel registration is async), so
  //    pruning "unregistered" ones would silently discard the stored
  //    arrangement of every service panel. A placeholder whose service truly
  //    no longer exists is removed by the adapter once panel-manager's
  //    registry is known (dock-iframe.js setKnownServicePanels).
  const kept = new Set();
  for (const id of Object.keys(data.panels)) {
    if (registered.has(id) || id.startsWith(PLACEHOLDER_PREFIX)) kept.add(id);
    else delete data.panels[id];
  }

  // 2. Grid tree: drop unknown ids from every group's view list, prune groups
  //    left empty, and collapse single-child branches. A grid with nothing left
  //    is unusable — signal a full fallback.
  const pruned = pruneGridNode(data.grid.root, kept);
  if (!pruned) return null;
  // dockview 7.0.2's fromJSON hard-throws unless the grid ROOT is a branch. An
  // inner single-child branch legitimately collapses to a leaf, but if the
  // collapse reaches the root (e.g. a single-group layout, or every group but
  // one pruned away) we must re-wrap it as a branch or the whole layout gets
  // rejected on reload / simple→expert restore.
  const root = pruned.type === 'branch' ? pruned : { type: 'branch', data: [pruned] };
  data.grid.root = root;

  // 3. Append registered-but-unreferenced panels to a sensible default group —
  //    the largest surviving tab-stack — materializing a panels-map entry from
  //    each descriptor so fromJSON can recreate it.
  const leaves = collectLeaves(root);
  const referenced = new Set();
  for (const leaf of leaves) for (const view of leaf.data.views) referenced.add(view);

  const target = leaves.reduce(
    (best, leaf) => (leaf.data.views.length > best.data.views.length ? leaf : best),
    leaves[0],
  );
  for (const descriptor of descriptors) {
    if (referenced.has(descriptor.id)) continue;
    data.panels[descriptor.id] = panelStateFromDescriptor(descriptor);
    target.data.views.push(descriptor.id);
    if (!target.data.activeView) target.data.activeView = descriptor.id;
  }

  // Repoint activeGroup if the group it referenced didn't survive the prune —
  // otherwise fromJSON restores with a dangling active-group id. The group that
  // received the appended panels (else the largest surviving one) is the sensible
  // new focus.
  const survivingGroupIds = new Set(leaves.map((leaf) => leaf.data.id));
  if (data.activeGroup != null && !survivingGroupIds.has(data.activeGroup)) {
    data.activeGroup = target.data.id;
  }

  return data;
}

/**
 * Coerce a registered-panels list into clean descriptors: drop malformed
 * entries, and expand a bare id string into `{id, contentComponent: id}`.
 * @param {Array<PanelDescriptor|string>} list
 * @returns {PanelDescriptor[]}
 */
function normalizeDescriptors(list) {
  if (!Array.isArray(list)) return [];
  /** @type {PanelDescriptor[]} */
  const out = [];
  for (const item of list) {
    if (typeof item === 'string') {
      if (item) out.push({ id: item, contentComponent: item });
    } else if (item && typeof item === 'object' && typeof item.id === 'string' && item.id) {
      out.push(item);
    }
  }
  return out;
}

/**
 * Build a SerializedDockview panel-map entry from a descriptor (contentComponent
 * defaults to the id; optional fields are omitted when absent).
 * @param {PanelDescriptor} descriptor
 * @returns {any}
 */
function panelStateFromDescriptor(descriptor) {
  /** @type {any} */
  const state = { id: descriptor.id, contentComponent: descriptor.contentComponent ?? descriptor.id };
  if (descriptor.tabComponent) state.tabComponent = descriptor.tabComponent;
  if (descriptor.title != null) state.title = descriptor.title;
  if (descriptor.params != null) state.params = descriptor.params;
  return state;
}

/**
 * Shallow structural guard for a SerializedDockview before reconciling it.
 * @param {any} data
 * @returns {boolean}
 */
function isSerializedDockview(data) {
  return !!data && typeof data === 'object'
    && !!data.grid && typeof data.grid === 'object'
    && !!data.grid.root && typeof data.grid.root === 'object'
    && !!data.panels && typeof data.panels === 'object';
}

/**
 * Recursively prune a SerializedGridObject to only `kept` panel ids, returning a
 * fresh node (never mutating the input) or `null` when the node has no surviving
 * content. Leaves whose view list empties are dropped; branches left with one
 * child collapse to that child; branches left empty are dropped.
 * @param {any} node
 * @param {Set<string>} kept
 * @returns {any|null}
 */
function pruneGridNode(node, kept) {
  if (!node || typeof node !== 'object') return null;

  if (node.type === 'leaf') {
    const data = node.data;
    if (!data || !Array.isArray(data.views)) return null;
    const views = data.views.filter((/** @type {string} */ v) => kept.has(v));
    if (views.length === 0) return null;
    const activeView = views.includes(data.activeView) ? data.activeView : views[0];
    return { ...node, data: { ...data, views, activeView } };
  }

  if (node.type === 'branch' && Array.isArray(node.data)) {
    const children = node.data
      .map((/** @type {any} */ child) => pruneGridNode(child, kept))
      .filter((/** @type {any} */ child) => child != null);
    if (children.length === 0) return null;
    if (children.length === 1) return children[0];
    return { ...node, data: children };
  }

  return null;
}

/**
 * Depth-first collect every leaf node of a pruned grid tree.
 * @param {any} node
 * @param {any[]} [out]
 * @returns {any[]}
 */
function collectLeaves(node, out = []) {
  if (!node || typeof node !== 'object') return out;
  if (node.type === 'leaf') out.push(node);
  else if (node.type === 'branch' && Array.isArray(node.data)) {
    for (const child of node.data) collectLeaves(child, out);
  }
  return out;
}
