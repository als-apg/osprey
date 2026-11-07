// @ts-check
/*
 * System Health dashboard — pure, framework-free helpers.
 *
 * Every function defined here is self-contained and side-effect-free apart
 * from DOM node construction. Keep it that way so the module stays trivially
 * testable in isolation. Ported from an earlier integration-status dashboard,
 * with the DOM builder aligned to the okf panel's el(tag, attrs, children)
 * signature and every status mapped to a CSS class name — never an inline
 * color literal.
 *
 * The rules this dashboard shares with the web terminal's health bar item —
 * how a check name is shown, how severe a status is, which status wins — are
 * not defined here. That item shows the same checks and this bundle is served
 * through the panel proxy rather than to that page, so every shared rule lives
 * in the design system and is re-exported below for this dashboard's own
 * importers.
 */

export { fmtName } from '/design-system/js/check-name.js';
export { STATUS_RANK, worstStatus } from '/design-system/js/check-status.js';

/** @type {HTMLDivElement | null} */
let escDiv = null;

/**
 * HTML-escape an arbitrary server value by round-tripping it through a detached
 * element's `textContent`. `null`/`undefined` become the empty string. The
 * scratch element is created lazily and reused across calls.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function esc(value) {
  if (value == null) return '';
  if (!escDiv) escDiv = document.createElement('div');
  escDiv.textContent = String(value);
  return escDiv.innerHTML;
}

/**
 * Create a DOM element with a flat string-valued attribute map and a list of
 * child nodes. Signature-compatible with the okf panel's `el`.
 *
 * @param {string} tag
 * @param {Record<string, string>|null} [attrs] String-valued attribute map.
 *   The key `"class"` sets `className`; `"text"` sets `textContent`; every other
 *   key is set via `setAttribute`.
 * @param {(Node|null|undefined)[]} [children] Falsy entries are skipped.
 * @returns {HTMLElement}
 */
export function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const k in attrs) {
      if (k === 'class') node.className = attrs[k];
      else if (k === 'text') node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
  }
  if (children) {
    for (const child of children) {
      if (child) node.appendChild(child);
    }
  }
  return node;
}

/**
 * Format a latency in milliseconds for display. Zero, negative, or missing
 * values yield the empty string — matching `CheckResult.to_dict()`, which omits
 * `latency_ms` unless it is greater than zero.
 *
 * @param {number} [ms]
 * @returns {string}
 */
export function fmtMs(ms) {
  if (!ms || ms <= 0) return '';
  return ms < 1000 ? Math.round(ms) + 'ms' : (ms / 1000).toFixed(1) + 's';
}

/**
 * Map a latency to a bucket CSS class — fast (`ms-f`), medium (`ms-m`), or slow
 * (`ms-s`). Zero, negative, or missing latency yields the empty string.
 *
 * @param {number} [ms]
 * @returns {string}
 */
export function msCls(ms) {
  if (!ms || ms <= 0) return '';
  if (ms < 100) return 'ms-f';
  if (ms < 500) return 'ms-m';
  return 'ms-s';
}

/**
 * Group check results by their `category`, preserving first-seen order.
 *
 * @template {{category: string}} T
 * @param {T[]} results
 * @returns {Map<string, T[]>}
 */
export function byCategory(results) {
  /** @type {Map<string, T[]>} */
  const m = new Map();
  for (const r of results) {
    const bucket = m.get(r.category);
    if (bucket) bucket.push(r);
    else m.set(r.category, [r]);
  }
  return m;
}

// -- layer-stack view helpers --------------------------------------------------

/**
 * Wire status → the severity label the panel shows (EPICS-style).
 *
 * @type {Readonly<Record<string, string>>}
 */
export const SEVERITY = Object.freeze({
  ok: 'OK',
  warning: 'MINOR',
  error: 'MAJOR',
  skip: 'DISABLED',
});

/** Severity label → rank, highest wins. */
const SEV_RANK = Object.freeze({ MAJOR: 3, MINOR: 2, OK: 1, DISABLED: 0 });

/**
 * The panel severity label for a wire status; unknown statuses read DISABLED.
 *
 * @param {string | undefined | null} status
 * @returns {string}
 */
export function sevOf(status) {
  return (status && SEVERITY[status]) || 'DISABLED';
}

/**
 * The worst of several severity labels. DISABLED only when every one is
 * DISABLED (or the list is empty).
 *
 * @param {string[]} sevs
 * @returns {string}
 */
export function worstSev(sevs) {
  /** @type {keyof typeof SEV_RANK} */
  let w = 'DISABLED';
  for (const s of sevs) {
    const k = /** @type {keyof typeof SEV_RANK} */ (s);
    if ((SEV_RANK[k] ?? 0) > SEV_RANK[w]) w = k;
  }
  return w;
}

/**
 * The probe method several rows share, or ``null``. Rows share a method when
 * their method strings agree word by word up to a common prefix and differ at
 * most in one trailing word (a path or argument): the group row then carries
 * the prefix once and the rows drop theirs.
 *
 * @param {(string | undefined | null)[]} methods
 * @returns {string | null}
 */
export function commonMethod(methods) {
  if (!methods || methods.length < 2) return null;
  const words = methods.map((m) => String(m || '').split(' '));
  const n = Math.min(...words.map((w) => w.length));
  let k = 0;
  while (k < n && words.every((w) => w[k] === words[0][k])) k++;
  if (!k || !words[0][0]) return null;
  return words.every((w) => w.length - k <= 1) ? words[0].slice(0, k).join(' ') : null;
}

const WRITES =
  /(^|_)(create|delete|register|add|remove|drop|close|open|arrange|manage|submit|draft|clear|patch|export|pin|focus|publish|capture|write|set|to_artifact|execute|measure|start|stop)(_|$)/;

/**
 * The exception tag of an MCP tool: "" for a read tool (the default carries
 * no tag), "write", or "write · gated". The render's hook contract decides
 * when the server sends it (``tool.hooks``, read from hook_config.json): only
 * it can say a tool is gated. Without it, MCP annotations or the tool name
 * (``derived`` true) can say a tool writes, never that it is gated.
 *
 * @param {{name: string, annotations?: Record<string, unknown> | null,
 *   hooks?: {write: boolean, gated: boolean}}} tool
 * @returns {{tag: string, derived: boolean, from: string}}
 */
export function tagOf(tool) {
  const g = tool.hooks;
  if (g && typeof g === 'object') {
    return {
      tag: g.gated ? 'write · gated' : g.write ? 'write' : '',
      derived: false,
      from: 'hooks',
    };
  }
  const a = tool.annotations;
  if (a && typeof a === 'object' && ('readOnlyHint' in a || 'destructiveHint' in a)) {
    return { tag: a.readOnlyHint ? '' : 'write', derived: false, from: 'annotations' };
  }
  return { tag: WRITES.test(tool.name) ? 'write' : '', derived: true, from: 'name' };
}

/**
 * The page title: the configured title, marked when the rows are the `?demo`
 * fixture so fabricated rows can never be read as the deployment's own.
 *
 * @param {string} title
 * @param {boolean} demo
 * @returns {string}
 */
export function pageTitle(title, demo) {
  const t = title || 'System Health';
  return demo ? t + ' — demo data' : t;
}

/**
 * Age of a probe as one short phrase.
 *
 * @param {number} seconds Elapsed seconds (negative clamps to zero).
 * @returns {string}
 */
export function fmtAge(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 3) return 'just now';
  if (s < 60) return s + ' s ago';
  if (s < 3600) return Math.floor(s / 60) + ' m ago';
  if (s < 86400) return Math.floor(s / 3600) + ' h ago';
  return Math.floor(s / 86400) + ' d ago';
}

/**
 * UTC wall-clock time of an epoch-seconds stamp, e.g. ``06:31:07Z``.
 *
 * @param {number | null | undefined} epochS
 * @returns {string}
 */
export function fmtClock(epochS) {
  if (!epochS) return '--:--:--Z';
  return new Date(epochS * 1000).toISOString().slice(11, 19) + 'Z';
}

/**
 * Format a duration in milliseconds the way the MCP table shows it: under a
 * second in ms, otherwise seconds with one decimal.
 *
 * @param {number | null | undefined} ms
 * @returns {string}
 */
export function fmtDur(ms) {
  if (ms == null || !isFinite(ms)) return '—';
  return ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(1) + ' s';
}

/**
 * Split a location hash into its decoded path parts.
 *
 * @param {string} hash e.g. ``#mcp-servers/controls/channel_write``
 * @returns {string[]}
 */
export function parseHash(hash) {
  const h = String(hash || '').replace(/^#/, '');
  if (!h) return [];
  return h.split('/').map((p) => {
    try {
      return decodeURIComponent(p);
    } catch {
      return p;
    }
  });
}

/**
 * Inverse of {@link parseHash}.
 *
 * @param {string[]} parts
 * @returns {string}
 */
export function formatHash(parts) {
  return parts.length ? '#' + parts.map(encodeURIComponent).join('/') : '';
}

/**
 * @typedef {Object} LayoutComponent
 * @property {string} id
 * @property {string} name
 * @property {string} status
 * @property {string | null} [issue]
 */

/**
 * Counts and faulty components for the headline line.
 *
 * @param {{components: LayoutComponent[]}[]} layers
 * @param {{status: string}[]} rows The result rows the layers reference.
 * @returns {{minor: number, major: number, faults: {id: string, name: string, issue: string, sev: string}[]}}
 */
export function headline(layers, rows) {
  let minor = 0;
  let major = 0;
  for (const r of rows) {
    if (r.status === 'warning') minor++;
    else if (r.status === 'error') major++;
  }
  /** @type {{id: string, name: string, issue: string, sev: string}[]} */
  const faults = [];
  for (const layer of layers) {
    for (const c of layer.components) {
      const sev = sevOf(c.status);
      if (sev === 'MINOR' || sev === 'MAJOR') {
        faults.push({ id: c.id, name: c.name, issue: c.issue || SEVERITY[c.status], sev });
      }
    }
  }
  faults.sort(
    (a, b) => SEV_RANK[/** @type {'MAJOR'} */ (b.sev)] - SEV_RANK[/** @type {'MAJOR'} */ (a.sev)]
  );
  return { minor, major, faults };
}

/**
 * Resolve a local ``$ref`` against the schema root's ``$defs``/``definitions``.
 *
 * @param {any} s
 * @param {any} root
 * @returns {any}
 */
export function resolveRef(s, root) {
  if (s && s.$ref) {
    const k = String(s.$ref).split('/').pop() || '';
    return ((root && (root.$defs || root.definitions)) || {})[k] || s;
  }
  return s || {};
}

/**
 * One-line type of a JSON-schema node: ``string``, ``array<string>``,
 * ``"a" | "b"``, ``integer | null``, ``object{3}``.
 *
 * @param {any} schema
 * @param {any} [root]
 * @returns {string}
 */
export function ty(schema, root) {
  const s = resolveRef(schema, root || schema);
  if (s.enum) return s.enum.map((/** @type {unknown} */ v) => JSON.stringify(v)).join(' | ');
  if ('const' in s) return JSON.stringify(s.const);
  const alt = s.anyOf || s.oneOf;
  if (alt) return alt.map((/** @type {any} */ a) => ty(a, root || schema)).join(' | ');
  if (Array.isArray(s.type)) return s.type.join(' | ');
  if (s.type === 'array') return 'array<' + (s.items ? ty(s.items, root || schema) : 'any') + '>';
  if (s.type === 'object')
    return s.properties ? 'object{' + Object.keys(s.properties).length + '}' : 'object';
  return s.type || 'any';
}

/**
 * The object schema an argument nests (its own properties, its array items',
 * or the first ``anyOf``/``oneOf`` branch that has some), or ``null``.
 *
 * @param {any} s
 * @param {any} root
 * @returns {any}
 */
function subObject(s, root) {
  const r = resolveRef(s, root);
  if (r.properties) return r;
  if (r.items) {
    const i = resolveRef(r.items, root);
    if (i.properties) return i;
  }
  for (const a of r.anyOf || r.oneOf || []) {
    const o = subObject(a, root);
    if (o) return o;
  }
  return null;
}

/**
 * @typedef {Object} ArgNode
 * @property {string} name
 * @property {boolean} required
 * @property {string} desc   Description folded to one line.
 * @property {string} type
 * @property {string} def    JSON of the default, or "".
 * @property {ArgNode[]} children
 */

/**
 * The argument tree of a tool's ``inputSchema``.
 *
 * @param {any} schema
 * @param {any} [root]
 * @returns {ArgNode[]}
 */
export function argTree(schema, root) {
  const rt = root || schema || {};
  const s = schema || {};
  const req = s.required || [];
  return Object.entries(s.properties || {}).map(([name, p]) => {
    const pr = resolveRef(p, rt);
    const sub = subObject(pr, rt);
    return {
      name,
      required: req.includes(name),
      desc: pr.description
        ? String(pr.description)
            .replace(/\s*\n\s*/g, ' ')
            .trim()
        : '',
      type: ty(pr, rt),
      def: 'default' in pr ? JSON.stringify(pr.default) : '',
      children: sub && sub !== s ? argTree(sub, rt) : [],
    };
  });
}

/**
 * @typedef {{code: boolean, text: string}} Span
 * @typedef {{kind: 'p', spans: Span[]} | {kind: 'ul', lead: Span[], items: Span[][]}} Block
 */

/**
 * Split inline text into plain and ``code`` spans (single or double
 * backticks).
 *
 * @param {string} text
 * @returns {Span[]}
 */
export function inlineSpans(text) {
  /** @type {Span[]} */
  const out = [];
  const re = /``(.+?)``|`([^`]+?)`/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ code: false, text: text.slice(last, m.index) });
    out.push({ code: true, text: m[1] || m[2] });
    last = re.lastIndex;
  }
  if (last < text.length) out.push({ code: false, text: text.slice(last) });
  return out;
}

/**
 * A tool description as paragraphs and bullet lists — markdown-lite, no HTML.
 *
 * @param {string | null | undefined} s
 * @returns {Block[]}
 */
export function mdBlocks(s) {
  const src = String(s || '').trim();
  if (!src) return [];
  return src.split(/\n\s*\n/).map((para) => {
    const lines = para.split('\n');
    if (lines.some((l) => /^\s*[-*] /.test(l))) {
      /** @type {string[]} */
      const items = [];
      /** @type {string[]} */
      const lead = [];
      for (const l of lines) {
        if (/^\s*[-*] /.test(l)) items.push(l.replace(/^\s*[-*] /, ''));
        else if (items.length) items[items.length - 1] += ' ' + l.trim();
        else lead.push(l.trim());
      }
      return { kind: 'ul', lead: inlineSpans(lead.join(' ')), items: items.map(inlineSpans) };
    }
    return { kind: 'p', spans: inlineSpans(lines.map((l) => l.trim()).join(' ')) };
  });
}

/**
 * First non-empty line of a text.
 *
 * @param {string | null | undefined} s
 * @returns {string}
 */
export function firstLine(s) {
  return (
    String(s || '')
      .trim()
      .split(/\n/)[0] || ''
  );
}

/**
 * The MCP tile's one fact. The answered count is out of the servers that can
 * answer the stdio probe (``stdio``, when given; else every configured one).
 *
 * @param {{configured?: string[], stdio?: number, answered?: number | null, tools?: number | null} | null | undefined} mcp
 * @returns {string}
 */
export function mcpFact(mcp) {
  if (!mcp) return '';
  const n = (mcp.configured || []).length;
  if (mcp.answered == null) return n + (n === 1 ? ' server' : ' servers');
  return mcp.answered + ' / ' + (mcp.stdio ?? n) + ' answered · ' + (mcp.tools ?? 0) + ' tools';
}

/**
 * Summary of an introspection payload, in the envelope's ``mcp`` shape.
 *
 * @param {{probed_at?: number | null, servers: {name: string, transport?: string, tools?: unknown[] | null}[]}} data
 * @returns {{configured: string[], answered: number, tools: number, probed_at: number | null, stdio: number}}
 */
export function mcpSummary(data) {
  const servers = data.servers || [];
  const stdio = servers.filter((s) => s.transport !== 'http');
  const answered = stdio.filter((s) => Array.isArray(s.tools));
  return {
    configured: servers.map((s) => s.name),
    answered: answered.length,
    stdio: stdio.length,
    tools: answered.reduce((a, s) => a + (s.tools ? s.tools.length : 0), 0),
    probed_at: data.probed_at ?? null,
  };
}

/**
 * The row text of a provider ping result.
 *
 * @param {{kind?: string, model?: string, latency_ms?: number, input_tokens?: number | null,
 *          output_tokens?: number | null, message?: string, status?: string}} p
 * @returns {{value: string, method: string}}
 */
export function pingText(p) {
  const method = p.kind === 'embedding' ? '1 embedding' : '1 completion';
  if (p.status && p.status !== 'ok')
    return { value: p.message || SEVERITY[p.status] || '', method };
  const parts = [];
  if (p.model) parts.push(p.model);
  if (p.latency_ms != null) parts.push(Math.round(p.latency_ms) + ' ms');
  if (p.input_tokens != null) {
    parts.push(
      p.input_tokens +
        ' in' +
        (p.kind !== 'embedding' && p.output_tokens != null ? ' / ' + p.output_tokens + ' out' : '')
    );
  }
  return { value: parts.join(' · ') || p.message || '', method };
}
