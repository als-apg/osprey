// @ts-check
/*
 * OKF Knowledge Panel — the Graph view's model: what is drawn, and where.
 *
 * Pure functions over the `/api/graph` payload (okf_panel/graph.py); nothing
 * here touches the DOM. The facility graph is thousands of nodes, so it is
 * never drawn whole. Two views cut it down:
 *
 *   overview        one node per device class and one per other kind, with
 *                   the relations between their members counted onto one edge
 *   neighbourhood   the nodes within a few hops of one focus node
 *
 * A neighbourhood stops at a hub (a node with more than HUB_DEGREE shown
 * relations) unless that hub is the focus, and at MAX_NODES nodes; what it
 * leaves out is counted, per node and in total, so the view can say so.
 *
 * A drawn node's `key` is the payload index of the node it stands for (a
 * number), or a group's key (a string: `c:<class index>` or `g:<kind>`).
 */

/** @typedef {'place' | 'device' | 'channel' | 'signal' | 'class'} NodeKind */
/**
 * @typedef {{id: string, kind: NodeKind, label: string, type: string | null,
 *   props: Record<string, string>, record?: string, pages?: string[]}} GraphNode
 */
/** @typedef {[number, number, string]} GraphEdge  source index, target index, relation */
/** @typedef {{nodes: GraphNode[], edges: GraphEdge[], derived?: string[]}} GraphPayload */
/** @typedef {number | string} NodeKey */
/**
 * @typedef {{key: NodeKey, label: string, kind: NodeKind, r: number, badge?: number,
 *   members?: number[], classNode?: number}} DrawnNode
 */
/**
 * @typedef {{a: NodeKey, b: NodeKey, relation: string, label: string, width?: number}} DrawnEdge
 */
/** @typedef {{nodes: DrawnNode[], edges: DrawnEdge[], hidden: number}} DrawnGraph */
/** @typedef {{x: number, y: number}} Point */
/** @typedef {{x: number, y: number, k: number}} Transform */
/**
 * @typedef {object} GraphIndex
 * @property {GraphNode[]} nodes
 * @property {GraphEdge[]} edges
 * @property {Set<string>} derived relations the graph does not hold as an edge
 * @property {[number, string, 1 | -1][][]} adjacency per node: neighbour, relation,
 *   and 1 for an edge leaving the node, -1 for one arriving
 * @property {Map<string, number>} relationCounts in display order
 * @property {Map<NodeKind, number>} kindCounts
 * @property {Map<string, number>} byId node index by node id
 * @property {Map<string, number>} classIndex class node index by class id
 */

/** The node kinds, in the order the view lists them. */
export const NODE_KINDS = /** @type {const} */ (['place', 'device', 'channel', 'signal', 'class']);

/** @type {Record<NodeKind, string>} */
export const KIND_NAMES = {
  place: 'Place',
  device: 'Device',
  channel: 'Channel',
  signal: 'Signal',
  class: 'Class',
};

/** The relations the stylesheet has a line style for, in display order. */
export const KNOWN_RELATIONS = [
  'hasBinding',
  'readsSignal',
  'writesSignal',
  'type',
  'subClassOf',
  'inPlace',
];

/** A neighbourhood does not walk through a node with more shown relations than this. */
export const HUB_DEGREE = 12;
/** A neighbourhood holds at most this many nodes. */
export const MAX_NODES = 120;

const CHANNEL_RADIUS = 5;
const NODE_RADIUS = 8;

/**
 * The style key of a relation: itself when the stylesheet knows it, else `other`.
 * @param {string} relation
 * @returns {string}
 */
export function relationStyle(relation) {
  return KNOWN_RELATIONS.includes(relation) ? relation : 'other';
}

/**
 * Index a payload for the lookups the view makes.
 * @param {GraphPayload} payload
 * @returns {GraphIndex}
 */
export function indexGraph(payload) {
  const nodes = payload.nodes || [];
  const edges = payload.edges || [];
  /** @type {[number, string, 1 | -1][][]} */
  const adjacency = nodes.map(() => []);
  /** @type {Map<string, number>} */
  const counted = new Map();
  for (const [a, b, relation] of edges) {
    adjacency[a].push([b, relation, 1]);
    adjacency[b].push([a, relation, -1]);
    counted.set(relation, (counted.get(relation) || 0) + 1);
  }
  const order = [
    ...KNOWN_RELATIONS.filter((relation) => counted.has(relation)),
    ...[...counted.keys()].filter((relation) => !KNOWN_RELATIONS.includes(relation)).sort(),
  ];
  /** @type {Map<NodeKind, number>} */
  const kindCounts = new Map();
  /** @type {Map<string, number>} */
  const byId = new Map();
  /** @type {Map<string, number>} */
  const classIndex = new Map();
  nodes.forEach((node, i) => {
    kindCounts.set(node.kind, (kindCounts.get(node.kind) || 0) + 1);
    byId.set(node.id, i);
    if (node.kind === 'class') classIndex.set(node.id, i);
  });
  return {
    nodes,
    edges,
    derived: new Set(payload.derived || []),
    adjacency,
    relationCounts: new Map(order.map((relation) => [relation, counted.get(relation) || 0])),
    kindCounts,
    byId,
    classIndex,
  };
}

/**
 * The nodes within `hops` of `focus`, over the relations in `relations`.
 * @param {GraphIndex} graph
 * @param {{focus: number, hops: number, relations: Set<string>, expanded?: Set<number>}} opts
 *   `expanded`: hubs to walk through all the same
 * @returns {DrawnGraph}
 */
export function neighbourhood(graph, { focus, hops, relations, expanded = new Set() }) {
  /** @type {Map<number, number>} */
  const distance = new Map([[focus, 0]]);
  /** @type {Map<number, number>} */
  const hiddenBy = new Map();
  const queue = [focus];
  let hidden = 0;
  for (let head = 0; head < queue.length; head++) {
    const i = queue[head];
    const d = /** @type {number} */ (distance.get(i));
    if (d >= hops) continue;
    const shown = graph.adjacency[i].filter(([, relation]) => relations.has(relation));
    const walk = i === focus || expanded.has(i) || shown.length <= HUB_DEGREE;
    for (const [j] of shown) {
      if (distance.has(j)) continue;
      if (!walk || distance.size >= MAX_NODES) {
        hidden++;
        hiddenBy.set(i, (hiddenBy.get(i) || 0) + 1);
        continue;
      }
      distance.set(j, d + 1);
      queue.push(j);
    }
  }
  /** @type {DrawnNode[]} */
  const nodes = [...distance.keys()].map((i) => {
    const node = graph.nodes[i];
    /** @type {DrawnNode} */
    const drawn = {
      key: i,
      label: node.label,
      kind: node.kind,
      r: node.kind === 'channel' ? CHANNEL_RADIUS : NODE_RADIUS,
    };
    const badge = hiddenBy.get(i);
    if (badge) drawn.badge = badge;
    return drawn;
  });
  const edges = graph.edges
    .filter(([a, b, relation]) => relations.has(relation) && distance.has(a) && distance.has(b))
    .map(([a, b, relation]) => ({ a, b, relation, label: relation }));
  return { nodes, edges, hidden };
}

/**
 * The whole graph, grouped: a device stands in its class's node, every other
 * node in its kind's. `type` edges are what the grouping already says, so
 * they are not drawn.
 * @param {GraphIndex} graph
 * @param {Set<string>} relations
 * @returns {DrawnGraph}
 */
export function overview(graph, relations) {
  /** @param {number} i */
  const groupOf = (i) => {
    const node = graph.nodes[i];
    if (node.kind === 'class') return 'c:' + i;
    const cls = node.kind === 'device' && node.type ? graph.classIndex.get(node.type) : undefined;
    return cls === undefined ? 'g:' + node.kind : 'c:' + cls;
  };
  /** @type {Map<string, number[]>} */
  const members = new Map();
  graph.nodes.forEach((node, i) => {
    if (node.kind === 'class') return;
    const key = groupOf(i);
    const list = members.get(key);
    if (list) list.push(i);
    else members.set(key, [i]);
  });

  /** @type {DrawnNode[]} */
  const nodes = [];
  for (const kind of NODE_KINDS) {
    if (kind === 'class') continue;
    const list = members.get('g:' + kind) || [];
    nodes.push({
      key: 'g:' + kind,
      label: `${KIND_NAMES[kind]}s · ${count(list.length)}`,
      kind,
      r: NODE_RADIUS + Math.sqrt(list.length) / 2.2,
      members: list,
    });
  }
  graph.nodes.forEach((node, i) => {
    if (node.kind !== 'class') return;
    const list = members.get('c:' + i) || [];
    nodes.push({
      key: 'c:' + i,
      label: list.length ? `${node.label} · ${count(list.length)}` : node.label,
      kind: list.length ? 'device' : 'class',
      r: list.length ? 7 + Math.sqrt(list.length) : CHANNEL_RADIUS,
      members: list,
      classNode: i,
    });
  });

  /** @type {Map<string, {a: string, b: string, relation: string, n: number}>} */
  const joined = new Map();
  for (const [a, b, relation] of graph.edges) {
    if (relation === 'type' || !relations.has(relation)) continue;
    const ga = groupOf(a);
    const gb = groupOf(b);
    if (ga === gb) continue;
    const key = ga + '|' + gb + '|' + relation;
    const edge = joined.get(key);
    if (edge) edge.n++;
    else joined.set(key, { a: ga, b: gb, relation, n: 1 });
  }
  const edges = [...joined.values()].map(({ a, b, relation, n }) => ({
    a,
    b,
    relation,
    label: n > 1 ? `${relation} ×${count(n)}` : relation,
    width: 1 + Math.log10(n),
  }));
  const linked = new Set(edges.flatMap((edge) => [edge.a, edge.b]));
  return {
    nodes: nodes.filter(
      (node) => (node.members && node.members.length > 0) || linked.has(String(node.key))
    ),
    edges,
    hidden: 0,
  };
}

/**
 * A count as the view writes it: grouped digits, the same in every locale.
 * @param {number} n
 * @returns {string}
 */
export function count(n) {
  return n.toLocaleString('en-US');
}

/**
 * Place a drawn graph: Fruchterman-Reingold, started on a spiral so one graph
 * always gets one picture. A node already in `positions` starts from there,
 * so a redraw after a filter change moves little.
 * @param {DrawnGraph} drawn
 * @param {Map<NodeKey, Point>} positions read for the start, written with the result
 * @param {{spacing: number, gravity: number, pin?: NodeKey | null, fresh?: boolean,
 *   iterations?: number}} opts `pin` is held at the origin; `fresh` ignores `positions`
 */
export function layout(
  drawn,
  positions,
  { spacing, gravity, pin = null, fresh = false, iterations = 300 }
) {
  const n = drawn.nodes.length;
  if (!n) return;
  const index = new Map(drawn.nodes.map((node, i) => [node.key, i]));
  const points = drawn.nodes.map((node, i) => {
    const held = fresh ? undefined : positions.get(node.key);
    if (held) return { x: held.x, y: held.y };
    const angle = (i / n) * 2 * Math.PI * 3.3;
    const radius = spacing * Math.sqrt(i + 1);
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  });
  /** @type {[number, number][]} */
  const links = [];
  for (const edge of drawn.edges) {
    const a = index.get(edge.a);
    const b = index.get(edge.b);
    if (a !== undefined && b !== undefined) links.push([a, b]);
  }
  for (let step = 0, heat = spacing * 2; step < iterations; step++, heat *= 0.985) {
    const force = points.map(() => ({ x: 0, y: 0 }));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = points[i].x - points[j].x;
        let dy = points[i].y - points[j].y;
        // Marks repel from their rims, so a large one keeps its neighbours off.
        const apart = Math.hypot(dx, dy) || 0.1;
        const gap = Math.max(apart - drawn.nodes[i].r - drawn.nodes[j].r, apart * 0.2);
        const f = (spacing * spacing) / (gap * apart);
        dx *= f;
        dy *= f;
        force[i].x += dx;
        force[i].y += dy;
        force[j].x -= dx;
        force[j].y -= dy;
      }
    }
    for (const [a, b] of links) {
      let dx = points[a].x - points[b].x;
      let dy = points[a].y - points[b].y;
      const f = (Math.hypot(dx, dy) || 0.01) / spacing;
      dx *= f;
      dy *= f;
      force[a].x -= dx;
      force[a].y -= dy;
      force[b].x += dx;
      force[b].y += dy;
    }
    for (let i = 0; i < n; i++) {
      if (drawn.nodes[i].key === pin) {
        points[i].x = 0;
        points[i].y = 0;
        continue;
      }
      force[i].x -= points[i].x * gravity;
      force[i].y -= points[i].y * gravity;
      const size = Math.hypot(force[i].x, force[i].y) || 1;
      const scale = Math.min(size, heat) / size;
      points[i].x += force[i].x * scale;
      points[i].y += force[i].y * scale;
    }
  }
  drawn.nodes.forEach((node, i) => positions.set(node.key, points[i]));
}

/**
 * The transform that shows every point inside a `width` × `height` box.
 * @param {Point[]} points
 * @param {number} width
 * @param {number} height
 * @param {{pad?: number, maxScale?: number}} [opts]
 * @returns {Transform | null} null with no points or no box
 */
export function fitTransform(points, width, height, { pad = 90, maxScale = 1.6 } = {}) {
  if (!points.length || !(width > 0) || !(height > 0)) return null;
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const p of points) {
    x0 = Math.min(x0, p.x);
    x1 = Math.max(x1, p.x);
    y0 = Math.min(y0, p.y);
    y1 = Math.max(y1, p.y);
  }
  // A small box keeps a readable margin by giving up the padding first.
  const margin = Math.min(pad, width / 4, height / 4);
  const k = Math.min(
    maxScale,
    (width - 2 * margin) / (x1 - x0 || 1),
    (height - 2 * margin) / (y1 - y0 || 1)
  );
  return { k, x: width / 2 - (k * (x0 + x1)) / 2, y: height / 2 - (k * (y0 + y1)) / 2 };
}

/**
 * Zoom by `factor`, keeping the point under (`cx`, `cy`) where it is.
 * @param {Transform} tf
 * @param {number} factor
 * @param {number} cx
 * @param {number} cy
 * @returns {Transform}
 */
export function zoomAt(tf, factor, cx, cy) {
  const k = Math.max(0.15, Math.min(4, tf.k * factor));
  const ratio = k / tf.k;
  return { k, x: cx - (cx - tf.x) * ratio, y: cy - (cy - tf.y) * ratio };
}

/**
 * The nodes a search names: a label that starts with the query first, then
 * one that holds it, then an id that holds it. Case is ignored.
 * @param {GraphIndex} graph
 * @param {string} query
 * @param {number} [limit]
 * @returns {number[]} node indexes, at most `limit`
 */
export function searchNodes(graph, query, limit = 12) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  /** @type {number[][]} */
  const ranked = [[], [], []];
  graph.nodes.forEach((node, i) => {
    const label = node.label.toLowerCase();
    if (label.startsWith(q)) ranked[0].push(i);
    else if (label.includes(q)) ranked[1].push(i);
    else if (node.id.toLowerCase().includes(q)) ranked[2].push(i);
  });
  return ranked.flat().slice(0, limit);
}

/**
 * A node's relations, grouped by direction and relation in first-seen order.
 * @param {GraphIndex} graph
 * @param {number} i
 * @returns {{relation: string, outgoing: boolean, nodes: number[]}[]}
 */
export function relationsOf(graph, i) {
  /** @type {Map<string, {relation: string, outgoing: boolean, nodes: number[]}>} */
  const groups = new Map();
  for (const [j, relation, direction] of graph.adjacency[i]) {
    const key = direction + relation;
    const group = groups.get(key);
    if (group) group.nodes.push(j);
    else groups.set(key, { relation, outgoing: direction > 0, nodes: [j] });
  }
  return [...groups.values()];
}

/**
 * The node that stands for a facility record.
 * @param {GraphIndex} graph
 * @param {string} kind the record's kind
 * @param {string} id the record's id in the facility file
 * @returns {number} the node index, or -1
 */
export function nodeOfRecord(graph, kind, id) {
  return graph.nodes.findIndex((node) => node.kind === kind && node.record === id);
}
