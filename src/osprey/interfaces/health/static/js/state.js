// @ts-check
/*
 * System Health dashboard — shared view state and the model derived from it.
 *
 * The /checks envelope, its index (tiles by id, drill nodes by key), the open
 * drawer, per-tile drill paths, in-flight probes, provider pings and the MCP
 * introspection cache, plus the pure-ish lookups every view module reads.
 * Reassigned values live on ``store`` so other modules can update them.
 */

import { mcpFact, mcpSummary, sevOf } from './helpers.js';

/**
 * @typedef {Object} Row
 * @property {string} name
 * @property {string} category
 * @property {string} status
 * @property {string} message
 * @property {string} [value]
 * @property {string} [display] The short shown value (value or message, with latency).
 * @property {number} [latency_ms]
 * @property {string} [details]
 * @property {string} [fix]
 * @property {string[]} [used_by]
 * @property {string} [layer]
 * @property {string} [component]
 * @property {string} [label]
 * @property {string} [method]
 * @property {string | null} [group]
 * @property {number} [probed_at]
 *
 * @typedef {Object} Group
 * @property {string} id
 * @property {string} name
 * @property {string} status
 * @property {string} [summary]
 * @property {number[]} rows
 * @property {string | null} [fix]
 *
 * @typedef {Object} Comp
 * @property {string} id
 * @property {string} name
 * @property {string} status
 * @property {string} [fact]
 * @property {string | null} [issue]
 * @property {string | null} [header]
 * @property {string[]} [used_by]
 * @property {number[]} [rows]
 * @property {Group[]} [groups]
 * @property {number[]} [unused]
 *
 * @typedef {{id: string, label: string, status: string, components: Comp[]}} Layer
 *
 * @typedef {Object} DrillNode
 * @property {string} key
 * @property {string} name
 * @property {string} kind      'group' | 'server' | 'tool'
 * @property {string} sev
 * @property {string} parent
 * @property {number[]} [rows]
 * @property {string | null} [fix]
 * @property {string} [summary]
 * @property {any} [srv]
 * @property {any} [tool]
 * @property {DrillNode[]} children
 */

/** Values the modules reassign, kept on one object so every module sees the update. */
export const store = {
  /** @type {any} */
  env: null,
  R: /** @type {Row[]} */ ([]),
  LAYERS: /** @type {Layer[]} */ ([]),
  TILES: /** @type {Record<string, {comp: Comp, band: string}>} */ ({}),
  NODES: /** @type {Record<string, DrillNode>} */ ({}),
  /** Signature of the open drawer's data, to skip no-op re-renders on poll. */
  drawerSig: '',
  polling: false,
  /** The rows come from the `?demo` fixture, not a deployment. */
  demo: false,
  /** An MCP deep-link path waiting for the introspection. */
  pendingMcpPath: /** @type {string[] | null} */ (null),
  pollTimer: /** @type {ReturnType<typeof setTimeout> | undefined} */ (undefined),
  api: /** @type {import('./api.js').Api} */ (/** @type {unknown} */ (null)),
};

export const MCP_ID = 'mcp-servers';

export const PING_COMPONENTS = new Set(['model-provider', 'embed-vision']);

export const UNUSED_ID = '__unused';

export const UNUSED_NAME = 'not used by this deployment';

export const DEFAULT_INTERVAL_S = 60;

export const WARMING_REPOLL_S = 3;

export const RANK = { MAJOR: 3, MINOR: 2, OK: 1, DISABLED: 0 };

export const GAUGE_R = 80;

export const GAUGE_C = 2 * Math.PI * GAUGE_R;

export const SVG_NS = 'http://www.w3.org/2000/svg';

export const NARROW_MILL_PX = 900;

/** Open drawer: band id and either a tile id or 'band:<id>'. */
export const S = {
  band: /** @type {string | null} */ (null),
  key: /** @type {string | null} */ (null),
};

/** Drill state per tile. @type {Record<string, {path: string[], col: number, q: string}>} */
export const DS = {};

/** Row names with a re-probe in flight. */
export const busyRows = new Set();

/** Provider ping results by provider name. @type {Map<string, {value: string, method: string, status: string, t: number}>} */
export const pings = new Map();

/** MCP introspection state. */
export const mcp = { data: /** @type {any} */ (null), loading: false, error: '', busy: new Set() };

/** @param {Comp} c */
export function compRows(c) {
  return [...(c.rows || []), ...(c.groups || []).flatMap((g) => g.rows), ...(c.unused || [])];
}

/** The MCP tile's live severity and fact, from the cached introspection when present. */
export function mcpTileState() {
  const t = store.TILES[MCP_ID];
  const base = t ? sevOf(t.comp.status) : 'OK';
  if (!mcp.data) return { sev: base, fact: mcpFact(store.env && store.env.mcp), issue: null };
  const s = mcpSummary(mcp.data);
  const failed = s.stdio - s.answered;
  const fact = mcpFact(s);
  if (failed > 0) return { sev: 'MAJOR', fact, issue: s.answered + ' / ' + s.stdio + ' answered' };
  return { sev: base, fact, issue: null };
}

/** @param {Comp} c */
export function tileSev(c) {
  return c.id === MCP_ID ? mcpTileState().sev : sevOf(c.status);
}

/** @param {Comp} c */
export function tileFact(c) {
  if (c.id === MCP_ID) {
    const st = mcpTileState();
    return st.issue || st.fact;
  }
  const sev = sevOf(c.status);
  if ((sev === 'MINOR' || sev === 'MAJOR') && c.issue) return c.issue;
  return c.fact || '';
}

/** Rebuild store.TILES and the drill store.NODES of every grouped tile. */
export function indexModel() {
  store.TILES = {};
  store.NODES = {};
  for (const layer of store.LAYERS) {
    for (const c of layer.components) {
      store.TILES[c.id] = { comp: c, band: layer.id };
      if (c.id === MCP_ID) continue;
      const groups = c.groups || [];
      if (!groups.length) continue;
      /** @type {DrillNode[]} */
      const kids = groups.map((g) => ({
        key: c.id + '/' + g.id,
        name: g.name,
        kind: 'group',
        sev: sevOf(g.status),
        parent: c.id,
        rows: g.rows,
        fix: g.fix || null,
        summary: g.summary || '',
        children: [],
      }));
      if ((c.unused || []).length) {
        kids.push({
          key: c.id + '/' + UNUSED_ID,
          name: UNUSED_NAME,
          kind: 'group',
          sev: 'DISABLED',
          parent: c.id,
          rows: c.unused || [],
          fix: null,
          summary: (c.unused || []).length + ' providers',
          children: [],
        });
      }
      for (const k of kids) store.NODES[k.key] = k;
      store.NODES[c.id] = /** @type {DrillNode} */ ({
        key: c.id,
        name: c.name,
        kind: 'tile',
        sev: sevOf(c.status),
        parent: '',
        children: kids,
      });
    }
  }
  indexMcp();
}

/** Server and tool nodes from the cached introspection. */
export function indexMcp() {
  if (!store.TILES[MCP_ID]) return;
  for (const k of Object.keys(store.NODES))
    if (k === MCP_ID || k.startsWith(MCP_ID + '/')) delete store.NODES[k];
  const servers = mcp.data ? mcp.data.servers || [] : [];
  /** @type {DrillNode[]} */
  const kids = servers.map((/** @type {any} */ s) => {
    const key = MCP_ID + '/' + s.name;
    const http = s.transport === 'http';
    /** @type {DrillNode} */
    const node = {
      key,
      name: s.name,
      kind: 'server',
      sev: http ? 'DISABLED' : Array.isArray(s.tools) ? 'OK' : 'MAJOR',
      parent: MCP_ID,
      srv: s,
      children: [],
    };
    node.children = (s.tools || []).map((/** @type {any} */ t) => ({
      key: key + '/' + t.name,
      name: t.name,
      kind: 'tool',
      sev: 'OK',
      parent: key,
      tool: t,
      children: [],
    }));
    for (const t of node.children) store.NODES[t.key] = t;
    store.NODES[key] = node;
    return node;
  });
  store.NODES[MCP_ID] = {
    key: MCP_ID,
    name: 'MCP servers',
    kind: 'tile',
    sev: 'OK',
    parent: '',
    children: kids,
  };
  // A deep link into the servers resolves once they have been introspected.
  if (store.pendingMcpPath && mcp.data) {
    const path = store.pendingMcpPath.filter((k) => store.NODES[k]);
    store.pendingMcpPath = null;
    if (path.length) {
      const st = ds(MCP_ID);
      st.path = path;
      st.col = Math.max(0, path.length - 1);
    }
  }
}

/** @param {string} id */
export function ds(id) {
  if (!DS[id]) {
    const node = store.NODES[id];
    const kids = node ? node.children : [];
    const bad = kids.find((k) => k.sev === 'MINOR' || k.sev === 'MAJOR');
    const first = id === MCP_ID ? null : bad || kids[0];
    DS[id] = { path: first ? [first.key] : [], col: 0, q: '' };
  }
  return DS[id];
}

/** @param {any} t @param {string} q */
export function toolMatch(t, q) {
  return (t.name + ' ' + (t.tool.description || '')).toLowerCase().includes(q.toLowerCase());
}

/** @param {DrillNode} n @param {string} q */
export function kidsOf(n, q) {
  let k = n.children || [];
  if (q && n.key === MCP_ID) k = k.filter((s) => s.children.some((t) => toolMatch(t, q)));
  if (q && n.kind === 'server') k = k.filter((t) => toolMatch(t, q));
  return k;
}

/** @param {string} key */
export function chainOf(key) {
  /** @type {string[]} */
  const out = [];
  let n = store.NODES[key];
  while (n && n.parent) {
    out.unshift(n.key);
    n = store.NODES[n.parent];
  }
  return out;
}

/** Rows referenced by the layers (the panel's scope). */
export function layerRows() {
  return store.LAYERS.flatMap((l) => l.components.flatMap(compRows))
    .map((i) => store.R[i])
    .filter(Boolean);
}

/** @param {number} i */
export function rowView(i) {
  const r = store.R[i];
  const ping = r.category === 'providers' ? pings.get(r.name) : undefined;
  return {
    sev: ping ? sevOf(ping.status) : sevOf(r.status),
    label: r.label || r.name,
    value: ping ? ping.value : r.display || r.value || r.message || '',
    full: ping
      ? ping.value
      : [r.display || r.value, r.message].filter((x, k, a) => x && a.indexOf(x) === k).join(' — '),
    method: ping ? ping.method : r.method || '',
    t: ping ? ping.t : r.probed_at || (store.env && store.env.scanned_at) || 0,
  };
}
