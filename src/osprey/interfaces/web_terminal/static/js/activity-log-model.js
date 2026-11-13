// @ts-check
/**
 * What the Activity page derives from one `/api/session-activity` answer: the
 * counts in its header, the turns of its timeline, the calls a filter or a
 * search keeps, and the way to a call inside a subagent. Pure functions of the
 * answer; nothing here touches the DOM.
 *
 * @module activity-log-model
 */

/**
 * @typedef {Object} TrailEntry  One audit entry filed for a call.
 * @property {string} ts
 * @property {string | null} surface  The gate that wrote it (a server name, `hook_approval`).
 * @property {string | null} decision
 * @property {string | null} reason
 * @property {unknown} [detail]
 */

/**
 * @typedef {Object} Approval  The operator's part in a call.
 * @property {string | null} asked_ts
 * @property {string | null} decided_ts
 * @property {number | null} wait_ms
 * @property {string} [outcome]
 * @property {string} [approver]
 * @property {string} [permission_mode]
 * @property {unknown} [position]
 */

/**
 * @typedef {Object} ResultView  The shown part of a call's result.
 * @property {string} text
 * @property {number} size  Bytes of the whole result.
 * @property {boolean} cut  True when `text` is not all of it.
 * @property {string} key  Record key the whole result is read by.
 */

/**
 * @typedef {Object} SubagentRun
 * @property {string} id
 * @property {string} type
 * @property {string} description
 * @property {ActivityNode[]} children
 * @property {number | null} ms
 */

/**
 * @typedef {Object} ToolNode  One tool call, with what the record joined to it.
 * @property {'tool'} t
 * @property {string} ts
 * @property {string} id  The call id; empty on an orphan group.
 * @property {string} key
 * @property {string} name  The full tool name (`mcp__controls__channel_read`).
 * @property {string} server  The MCP server, `''` for a built-in tool.
 * @property {string} tool  The name without its server.
 * @property {string} agent  `main`, or the subagent type that made the call.
 * @property {Record<string, unknown>} input
 * @property {'allowed' | 'approved' | 'asked' | 'denied' | 'none'} verdict
 * @property {string | null} source
 * @property {number | null} duration_ms  Measured by the tool server.
 * @property {number | null} elapsed_ms  Call to result, by the record's clock.
 * @property {string | null} target
 * @property {string | null} posture
 * @property {Approval | null} approval
 * @property {ResultView | null} result
 * @property {boolean} error
 * @property {string} errmsg
 * @property {TrailEntry[]} trail
 * @property {boolean} incomplete  The record has not received all of the call.
 * @property {boolean} running  No result yet.
 * @property {boolean} orphan  A subagent whose parent call the record does not hold.
 * @property {SubagentRun} [sub]
 */

/** @typedef {{ t: 'prompt', ts: string, text: string, command?: string }} PromptNode */
/** @typedef {{ t: 'brief' | 'thinking' | 'answer', ts: string, text: string }} TextNode */
/** @typedef {{ t: 'note', ts: string, icon: string, text: string, size?: number, body?: string | null }} NoteNode */
/** @typedef {{ t: 'usage', ts: string, cost: number, ms: number }} UsageNode */
/** @typedef {PromptNode | TextNode | NoteNode | UsageNode | ToolNode} ActivityNode */

/**
 * @typedef {Object} ActivitySession
 * @property {string} id  The transcript id.
 * @property {string | null} session_key
 * @property {string} title
 * @property {string} identity
 * @property {string | null} start
 * @property {string | null} end
 * @property {string} state  `live`, `idle` or `ended`.
 * @property {number} entries  Record entries the reading was made from.
 * @property {Record<string, number>} kinds
 * @property {string | null} target
 * @property {string | null} posture
 * @property {number | null} retention_days  `null` when the deployment names none
 */

/**
 * @typedef {Object} ArtifactRow
 * @property {string} id
 * @property {string} title
 * @property {string} call  The call that saved it.
 * @property {string} tool
 * @property {string} agent
 * @property {string} ts
 * @property {string} form
 * @property {string} form_label
 * @property {string} type
 * @property {number} size
 */

/**
 * @typedef {Object} ActivityPayload  One `/api/session-activity` answer.
 * @property {ActivitySession | null} session
 * @property {ActivityNode[]} nodes
 * @property {ArtifactRow[]} artifacts
 * @property {'complete' | 'incomplete' | 'unavailable'} record
 * @property {string} [record_reason]
 */

/**
 * @typedef {Object} ActivitySummary
 * @property {ToolNode[]} tools  Every call, at any depth, in order.
 * @property {SubagentRun[]} subs
 * @property {ToolNode[]} approvals  Calls the operator was asked about.
 * @property {ToolNode[]} errors  Calls that failed.
 * @property {ToolNode[]} gated  Calls a server gate filed an entry for.
 * @property {number} decided  Calls with a recorded decision.
 * @property {number} prompts  Operator prompts.
 * @property {number} cost  Model cost in USD.
 * @property {number} modelMs  Model time in ms.
 * @property {number | null} ms  First entry to last, in ms.
 */

/** @typedef {{ prompt: PromptNode | null, nodes: ActivityNode[] }} Turn */

/**
 * The name a call goes by in page state and links: its call id, or, for an
 * orphan group that has none, its subagent.
 * @param {ToolNode} node
 * @returns {string}
 */
export function callKey(node) {
  return node.id || (node.sub ? `agent:${node.sub.id}` : node.key);
}

/**
 * Visit every node at any depth, in timeline order.
 * @param {ActivityNode[]} nodes
 * @param {(node: ActivityNode) => void} visit
 */
export function walk(nodes, visit) {
  for (const node of nodes) {
    visit(node);
    if (node.t === 'tool' && node.sub) walk(node.sub.children, visit);
  }
}

/**
 * The counts the page's header and views read.
 * @param {ActivityPayload} payload
 * @returns {ActivitySummary}
 */
export function summarize(payload) {
  /** @type {ToolNode[]} */
  const tools = [];
  /** @type {SubagentRun[]} */
  const subs = [];
  let cost = 0;
  let modelMs = 0;
  walk(payload.nodes, (node) => {
    if (node.t === 'usage') {
      cost += node.cost || 0;
      modelMs += node.ms || 0;
    } else if (node.t === 'tool') {
      // An orphan group stands in for a call the record does not hold.
      if (!node.orphan) tools.push(node);
      if (node.sub) subs.push(node.sub);
    }
  });
  const start = Date.parse(payload.session?.start ?? '');
  const end = Date.parse(payload.session?.end ?? '');
  return {
    tools,
    subs,
    approvals: tools.filter((n) => n.approval),
    errors: tools.filter((n) => n.error),
    gated: tools.filter((n) =>
      n.trail.some((entry) => entry.surface && entry.surface !== 'hook_approval')
    ),
    decided: tools.filter((n) => n.verdict !== 'none').length,
    prompts: payload.nodes.filter((n) => n.t === 'prompt').length,
    cost,
    modelMs,
    ms: Number.isNaN(start) || Number.isNaN(end) ? null : end - start,
  };
}

/**
 * The timeline in turns: a prompt and everything up to the next one. Entries
 * before the first prompt form a turn without one.
 * @param {ActivityNode[]} nodes  top-level nodes
 * @returns {Turn[]}
 */
export function turnsOf(nodes) {
  /** @type {Turn[]} */
  const turns = [];
  for (const node of nodes) {
    if (node.t === 'prompt') turns.push({ prompt: node, nodes: [] });
    else if (turns.length === 0) turns.push({ prompt: null, nodes: [node] });
    else turns[turns.length - 1].nodes.push(node);
  }
  return turns;
}

/**
 * The text of *node* a search reads.
 * @param {ActivityNode} node
 * @returns {string}
 */
function haystack(node) {
  switch (node.t) {
    case 'tool':
      return [
        node.name,
        JSON.stringify(node.input),
        node.result?.text,
        node.errmsg,
        node.sub?.type,
        node.sub?.description,
      ].join(' ');
    case 'prompt':
      return `${node.text} ${node.command ?? ''}`;
    case 'note':
      return `${node.text} ${node.body ?? ''}`;
    case 'usage':
      return '';
    default:
      return node.text;
  }
}

/**
 * Whether *node* holds the search text *q*, in any case. An empty search
 * matches everything.
 * @param {ActivityNode} node
 * @param {string | null | undefined} q
 * @returns {boolean}
 */
export function matches(node, q) {
  const needle = (q ?? '').trim().toLowerCase();
  if (!needle) return true;
  return haystack(node).toLowerCase().includes(needle);
}

/**
 * The calls at any depth, in order, that a filter and a search keep.
 * @param {ActivityNode[]} nodes
 * @param {{ filter?: string, q?: string }} [options] `filter` is `approvals`
 *   or `errors`; anything else keeps every call
 * @returns {ToolNode[]}
 */
export function flatCalls(nodes, { filter, q } = {}) {
  /** @type {ToolNode[]} */
  const calls = [];
  walk(nodes, (node) => {
    if (node.t !== 'tool' || node.orphan) return;
    if (filter === 'approvals' && !node.approval) return;
    if (filter === 'errors' && !node.error) return;
    if (matches(node, q)) calls.push(node);
  });
  return calls;
}

/**
 * The way to the call named *key*: the keys of the calls whose subagent holds
 * it, outermost first.
 * @param {ActivityNode[]} nodes
 * @param {string} key  a {@link callKey}
 * @returns {string[] | null} `[]` for a top-level call, null when no call has that key
 */
export function findCallPath(nodes, key) {
  for (const node of nodes) {
    if (node.t !== 'tool') continue;
    if (callKey(node) === key) return [];
    const inner = node.sub ? findCallPath(node.sub.children, key) : null;
    if (inner) return [callKey(node), ...inner];
  }
  return null;
}

/**
 * A string that is equal for two answers exactly when they would render the
 * same, so a poll that brought nothing new leaves the page alone.
 * @param {ActivityPayload} payload
 * @returns {string}
 */
export function payloadSignature(payload) {
  return JSON.stringify(payload);
}
