// @ts-check
/**
 * OSPREY Channel Finder — class-tree layout for the graph explorer (pure logic).
 *
 * Turns a flat list of ontology classes (each naming its parent classes by uri)
 * into placed nodes and orthogonal "elbow" edge paths that an SVG renderer can
 * draw directly. There is no DOM, no fetch and no module state here, so the
 * layout is unit-testable under Vitest
 * (see tests/interfaces/channel_finder/graph-tree-layout.test.mjs).
 *
 * Layout rules, all deterministic:
 *   - The class graph is reduced to a forest: a class with several known parents
 *     is attached under the parent whose **uri** sorts first lexicographically;
 *     its remaining known parents are reported on `extraParents` so the renderer
 *     can annotate the multi-parent case. Parents that name a uri absent from
 *     the input are ignored; a class with no known parent is a root.
 *   - Roots are laid out consecutively in one shared vertical run (stacked
 *     forests), ordered by name then uri, with no gap row between them.
 *   - Depth is the column index: `x = depth * colWidth`. Leaves take successive
 *     vertical slots (`y = slot * rowHeight`) and every parent is centred on the
 *     span of its own children. Siblings are ordered by name then uri.
 *   - `x` / `y` are the node box's top-left corner.
 *   - Node widths come from the injected `measure(name, rollup)` callback, so a
 *     renderer can pass a canvas text measurer while tests pass a stub.
 *
 * A `parents` cycle (A -> B -> A) cannot be drawn as a tree, so it is broken
 * deterministically: the cycle member whose uri sorts first loses its parent
 * link (the dropped parent moves to `extraParents`) and becomes a root.
 */

/** Default vertical distance between leaf rows, in px. */
export const DEFAULT_ROW_HEIGHT = 28;

/** Default horizontal distance between depth columns, in px. */
export const DEFAULT_COL_WIDTH = 200;

/** Default node box height, in px. */
export const DEFAULT_NODE_HEIGHT = 22;

/** Fallback width used when `measure` returns a non-finite or non-positive value. */
const FALLBACK_NODE_WIDTH = 120;

/**
 * @typedef {object} ClassInput
 * @property {string} uri - Stable class identifier.
 * @property {string} name - Display name.
 * @property {string[]} [altLabel] - Alternative labels, if any.
 * @property {string[]} [parents] - Parent class uris (any that are unknown here are ignored).
 * @property {number} [rollup] - Instance count rolled up over the subtree.
 */

/**
 * @typedef {object} LayoutNode
 * @property {string} uri - Stable class identifier.
 * @property {string} name - Display name.
 * @property {string[]} altLabel - Alternative labels (never null; empty when none).
 * @property {number} rollup - Instance count rolled up over the subtree (0 when absent).
 * @property {string[]} extraParents - Known parent uris this node is NOT drawn under.
 * @property {number} x - Left edge, px.
 * @property {number} y - Top edge, px.
 * @property {number} width - Box width from `measure`, px.
 * @property {number} height - Box height, px.
 * @property {number} depth - Column index (0 for roots).
 * @property {boolean} isRoot - True when the node has no drawn parent.
 */

/**
 * @typedef {object} LayoutEdge
 * @property {string} from - Parent uri.
 * @property {string} to - Child uri.
 * @property {string} d - Orthogonal SVG path ("M x y H x V y H x").
 */

/**
 * @typedef {object} ForestLayout
 * @property {LayoutNode[]} nodes - Placed nodes, in draw order (pre-order per root).
 * @property {LayoutEdge[]} edges - One elbow edge per drawn parent/child link.
 * @property {number} width - Overall width of the placed content, px.
 * @property {number} height - Overall height of the placed content, px.
 */

/**
 * @typedef {object} LayoutOptions
 * @property {number} [rowHeight] - Vertical distance between leaf rows, px.
 * @property {number} [colWidth] - Horizontal distance between depth columns, px.
 * @property {number} [nodeHeight] - Node box height, px.
 * @property {(name: string, rollup: number) => number} [measure] - Node width callback.
 */

/**
 * @typedef {object} TreeNode
 * @property {string} uri
 * @property {string} name
 * @property {string[]} altLabel
 * @property {number} rollup
 * @property {string[]} rawParents - Parent uris exactly as declared by the input.
 * @property {string|null} parent - Parent this node is drawn under.
 * @property {string[]} extraParents
 * @property {TreeNode[]} children
 * @property {number} depth
 * @property {number} x
 * @property {number} y
 * @property {number} width
 */

/** Round to 2 decimals so emitted path strings stay short and stable. */
const round2 = (/** @type {number} */ value) => Math.round(value * 100) / 100;

/**
 * Deterministic sibling / root ordering: by display name, uri as tie-breaker.
 *
 * @param {TreeNode} a - First node.
 * @param {TreeNode} b - Second node.
 * @returns {number} Negative, zero or positive per `Array.prototype.sort`.
 */
function byNameThenUri(a, b) {
  if (a.name !== b.name) return a.name < b.name ? -1 : 1;
  if (a.uri === b.uri) return 0;
  return a.uri < b.uri ? -1 : 1;
}

/**
 * Build the internal records, dropping entries with no uri and keeping the first
 * record for any repeated uri.
 *
 * @param {ClassInput[]} classes - Raw input classes.
 * @returns {Map<string, TreeNode>} Records keyed by uri, in input order.
 */
function buildRecords(classes) {
  /** @type {Map<string, TreeNode>} */
  const records = new Map();
  for (const cls of classes || []) {
    if (!cls || typeof cls.uri !== 'string' || cls.uri === '') continue;
    if (records.has(cls.uri)) continue;
    records.set(cls.uri, {
      uri: cls.uri,
      name: typeof cls.name === 'string' && cls.name !== '' ? cls.name : cls.uri,
      altLabel: Array.isArray(cls.altLabel) ? cls.altLabel.slice() : [],
      rollup: typeof cls.rollup === 'number' && Number.isFinite(cls.rollup) ? cls.rollup : 0,
      rawParents: Array.isArray(cls.parents) ? cls.parents.slice() : [],
      parent: null,
      extraParents: [],
      children: [],
      depth: 0,
      x: 0,
      y: 0,
      width: FALLBACK_NODE_WIDTH,
    });
  }
  return records;
}

/**
 * Resolve each record's known parents and pick the drawn one (uri-lexicographic
 * first); the rest become `extraParents`.
 *
 * @param {Map<string, TreeNode>} records - Records to annotate in place.
 * @returns {void}
 */
function resolveParents(records) {
  for (const record of records.values()) {
    /** @type {Set<string>} */
    const known = new Set();
    for (const parentUri of record.rawParents) {
      if (typeof parentUri !== 'string') continue;
      if (parentUri === record.uri) continue;
      if (!records.has(parentUri)) continue;
      known.add(parentUri);
    }
    const sorted = Array.from(known).sort();
    record.parent = sorted.length > 0 ? sorted[0] : null;
    record.extraParents = sorted.slice(1);
  }
}

/**
 * Break every `parents` cycle so the parent chain always terminates. The cycle
 * member whose uri sorts first is demoted to a root and its dropped parent link
 * is preserved on `extraParents`.
 *
 * @param {Map<string, TreeNode>} records - Records to fix up in place.
 * @returns {void}
 */
function breakCycles(records) {
  /** @type {Set<string>} */
  const settled = new Set();
  for (const start of records.values()) {
    /** @type {TreeNode[]} */
    const path = [];
    /** @type {Set<string>} */
    const onPath = new Set();
    /** @type {TreeNode|undefined} */
    let current = start;
    while (current && !settled.has(current.uri)) {
      if (onPath.has(current.uri)) {
        const cycleStart = current.uri;
        const from = path.findIndex((node) => node.uri === cycleStart);
        const cycle = path.slice(from);
        let victim = cycle[0];
        for (const node of cycle) {
          if (node.uri < victim.uri) victim = node;
        }
        if (victim.parent !== null) {
          victim.extraParents = victim.extraParents.concat([victim.parent]).sort();
          victim.parent = null;
        }
        break;
      }
      onPath.add(current.uri);
      path.push(current);
      current = current.parent === null ? undefined : records.get(current.parent);
    }
    for (const node of path) settled.add(node.uri);
  }
}

/**
 * Attach children to their drawn parent and collect the ordered roots.
 *
 * @param {Map<string, TreeNode>} records - Records with resolved, cycle-free parents.
 * @returns {TreeNode[]} Roots ordered by name then uri.
 */
function attachChildren(records) {
  /** @type {TreeNode[]} */
  const roots = [];
  for (const record of records.values()) {
    const parent = record.parent === null ? undefined : records.get(record.parent);
    if (parent) parent.children.push(record);
    else roots.push(record);
  }
  for (const record of records.values()) record.children.sort(byNameThenUri);
  roots.sort(byNameThenUri);
  return roots;
}

/**
 * Place every node: depth columns left to right, leaves on successive rows, each
 * parent centred on its children. Iterative post-order — no recursion depth
 * limit on deep chains.
 *
 * @param {TreeNode[]} roots - Ordered roots.
 * @param {number} rowHeight - Vertical distance between leaf rows.
 * @param {number} colWidth - Horizontal distance between depth columns.
 * @param {(name: string, rollup: number) => number} measure - Width callback.
 * @returns {TreeNode[]} All nodes in draw order (pre-order per root).
 */
function placeNodes(roots, rowHeight, colWidth, measure) {
  /** @type {TreeNode[]} */
  const ordered = [];
  /** @type {{node: TreeNode, depth: number, expanded: boolean}[]} */
  const stack = [];
  for (let i = roots.length - 1; i >= 0; i -= 1) {
    stack.push({ node: roots[i], depth: 0, expanded: false });
  }
  let leafSlot = 0;
  while (stack.length > 0) {
    const frame = stack[stack.length - 1];
    const node = frame.node;
    if (!frame.expanded) {
      frame.expanded = true;
      node.depth = frame.depth;
      node.x = frame.depth * colWidth;
      const measured = measure(node.name, node.rollup);
      node.width = Number.isFinite(measured) && measured > 0 ? measured : FALLBACK_NODE_WIDTH;
      ordered.push(node);
      for (let i = node.children.length - 1; i >= 0; i -= 1) {
        stack.push({ node: node.children[i], depth: frame.depth + 1, expanded: false });
      }
      continue;
    }
    stack.pop();
    if (node.children.length === 0) {
      node.y = leafSlot * rowHeight;
      leafSlot += 1;
    } else {
      const first = node.children[0];
      const last = node.children[node.children.length - 1];
      node.y = (first.y + last.y) / 2;
    }
  }
  return ordered;
}

/**
 * Build one orthogonal elbow path from a parent's right edge to a child's left
 * edge, turning at the horizontal midpoint between the two columns.
 *
 * @param {TreeNode} parent - Parent node (already placed).
 * @param {TreeNode} child - Child node (already placed).
 * @param {number} nodeHeight - Node box height.
 * @returns {LayoutEdge} Edge with its SVG path string.
 */
function elbow(parent, child, nodeHeight) {
  const startX = round2(parent.x + parent.width);
  const startY = round2(parent.y + nodeHeight / 2);
  const endX = round2(child.x);
  const endY = round2(child.y + nodeHeight / 2);
  const midX = round2(startX + (endX - startX) / 2);
  return {
    from: parent.uri,
    to: child.uri,
    d: `M ${startX} ${startY} H ${midX} V ${endY} H ${endX}`,
  };
}

/**
 * Lay out a forest of ontology classes for the graph explorer's class tree.
 *
 * Pure and deterministic: the same input always yields the same coordinates and
 * the same node/edge ordering. Terminates on any input, including one whose
 * `parents` links form a cycle.
 *
 * @param {ClassInput[]} classes - Flat class list; `parents` names parent uris.
 * @param {LayoutOptions} [options] - Metrics and the node-width callback.
 * @returns {ForestLayout} Placed nodes, elbow edges and the overall extent.
 */
export function layoutForest(classes, options = {}) {
  const rowHeight = options.rowHeight ?? DEFAULT_ROW_HEIGHT;
  const colWidth = options.colWidth ?? DEFAULT_COL_WIDTH;
  const nodeHeight = options.nodeHeight ?? DEFAULT_NODE_HEIGHT;
  const measure = options.measure ?? ((name) => FALLBACK_NODE_WIDTH + name.length * 7);

  const records = buildRecords(classes);
  resolveParents(records);
  breakCycles(records);
  const roots = attachChildren(records);
  const ordered = placeNodes(roots, rowHeight, colWidth, measure);

  /** @type {LayoutNode[]} */
  const nodes = [];
  /** @type {LayoutEdge[]} */
  const edges = [];
  let width = 0;
  let height = 0;
  for (const record of ordered) {
    nodes.push({
      uri: record.uri,
      name: record.name,
      altLabel: record.altLabel,
      rollup: record.rollup,
      extraParents: record.extraParents,
      x: record.x,
      y: record.y,
      width: record.width,
      height: nodeHeight,
      depth: record.depth,
      isRoot: record.parent === null,
    });
    for (const child of record.children) edges.push(elbow(record, child, nodeHeight));
    width = Math.max(width, record.x + record.width);
    height = Math.max(height, record.y + nodeHeight);
  }
  return { nodes, edges, width: round2(width), height: round2(height) };
}
