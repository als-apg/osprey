// @ts-check
/**
 * The Timeline view of the Activity page: a session as its turns, each a
 * prompt and what the agent did about it, with subagents as groups; or, under
 * a search or a call filter, the entries that are kept, flat.
 *
 * @module activity-log-timeline
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { facilityZoneLabel } from '/design-system/js/facility-time.js';

import { clock, formatBytes, formatDuration } from './activity-log-format.js';
import { callKey, matches, turnsOf, walk } from './activity-log-model.js';
import { chip, foldRow, iconHtml, stateBlock, toolRow } from './activity-log-rows.js';

/** @typedef {import('./activity-log-model.js').ActivityNode} ActivityNode */
/** @typedef {import('./activity-log-model.js').ActivityPayload} ActivityPayload */
/** @typedef {import('./activity-log-model.js').ActivitySummary} ActivitySummary */
/** @typedef {import('./activity-log-model.js').NoteNode} NoteNode */
/** @typedef {import('./activity-log-model.js').PromptNode} PromptNode */
/** @typedef {import('./activity-log-model.js').TextNode} TextNode */
/** @typedef {import('./activity-log-model.js').ToolNode} ToolNode */
/** @typedef {import('./activity-log-rows.js').UiState} UiState */

/** The timeline's filters, in the order of the switcher. */
const FILTERS = [
  ['all', 'All'],
  ['conversation', 'Conversation'],
  ['tools', 'Tool calls'],
  ['approvals', 'Approvals'],
  ['errors', 'Errors'],
];

/** The filters that keep calls only, and so flatten the timeline. */
const CALL_FILTERS = new Set(['tools', 'approvals', 'errors']);

/** What a call filter says when it keeps nothing: icon, message. @type {Record<string, [string, string]>} */
const NOTHING_KEPT = {
  errors: ['check', 'No failed calls in this session'],
  approvals: ['shield-check', 'No call in this session asked the operator'],
  tools: ['wrench', 'No tool calls in this session'],
};

/** The agent's own task-list tools: a run of them folds into one row. */
const TASK_TOOLS = new Set(['TaskCreate', 'TaskUpdate', 'TaskList', 'TaskGet', 'TodoWrite']);

/** The design-system icon of each note icon the server names. @type {Record<string, string>} */
const NOTE_ICONS = {
  book: 'book-open',
  reply: 'corner-down-right',
  info: 'info',
  alert: 'triangle-alert',
  stop: 'circle-x',
};

/**
 * @param {number} n
 * @param {string} noun
 * @returns {string} `1 call`, `2 calls`
 */
function count(n, noun) {
  return `${n} ${noun}${n === 1 ? '' : 's'}`;
}

/**
 * @param {string} text
 * @returns {string} the first line of *text* that holds anything
 */
function firstLine(text) {
  return (
    text
      .split('\n')
      .find((line) => line.trim())
      ?.trim() ?? ''
  );
}

/**
 * @param {PromptNode} node
 * @param {string} identity  who the session belongs to
 * @returns {string}
 */
function promptBlock(node, identity) {
  const command = node.command ? `<code>${escapeHtml(node.command)}</code> ` : '';
  const who = identity ? `${escapeHtml(identity)} asked` : 'Operator asked';
  return (
    `<div class="act-prompt"><span class="act-t">${clock(node.ts)}</span>` +
    `<span class="act-i">${iconHtml('user')}</span><div><div class="act-who">${who}</div>` +
    `<div class="act-prompt-text">${command}${escapeHtml(node.text)}</div></div></div>`
  );
}

/**
 * The agent's answer, as text; the page upgrades `[data-md]` to markdown once
 * it is in the document.
 * @param {TextNode} node
 * @returns {string}
 */
function answerRow(node) {
  return (
    `<div class="act-row" data-kind="answer"><span class="act-t">${clock(node.ts)}</span>` +
    `<span class="act-i">${iconHtml('agent')}</span>` +
    `<div class="act-md" data-md>${escapeHtml(node.text)}</div><span></span></div>`
  );
}

/**
 * A row that is not a call: thinking, a subagent's brief, an answer or a note.
 * @param {TextNode | NoteNode} node
 * @param {string} id  the row's name in `ui.open`, when it folds
 * @param {UiState} ui
 * @returns {string}
 */
function textRow(node, id, ui) {
  if (node.t === 'answer') return answerRow(node);
  const folded = node.t === 'note' ? node.body : node.text;
  const body = () => `<div class="act-fold">${escapeHtml(folded)}</div>`;
  if (node.t === 'thinking') {
    const label = `<span class="act-think">${escapeHtml(firstLine(node.text))}</span>`;
    return foldRow({ id, kind: 'thinking', icon: 'bulb', ts: node.ts, label, body }, ui);
  }
  if (node.t !== 'note') {
    const label = '<span class="act-muted">Task given to the subagent</span>';
    return foldRow({ id, kind: 'note', icon: 'file-text', ts: node.ts, label, body }, ui);
  }
  const name = NOTE_ICONS[node.icon] ?? 'info';
  const label = `<span class="act-muted">${escapeHtml(node.text)}</span>`;
  const size = node.size ? `<span class="act-dur soft">${formatBytes(node.size)}</span>` : '';
  if (node.body) {
    return foldRow({ id, kind: 'note', icon: name, ts: node.ts, label, body, right: size }, ui);
  }
  return (
    `<div class="act-row" data-kind="note"><span class="act-t">${clock(node.ts)}</span>` +
    `<span class="act-i">${iconHtml(name)}</span><span class="act-line">${label}</span>` +
    `<span class="act-r">${size}</span></div>`
  );
}

/**
 * A subagent: a head that collapses it, and its own steps.
 * @param {ToolNode & { sub: NonNullable<ToolNode['sub']> }} n
 * @param {UiState} ui
 * @returns {string}
 */
function subGroup(n, ui) {
  const sub = n.sub;
  const key = callKey(n);
  const open = !ui.closed.has(key);
  let calls = 0;
  let failed = 0;
  walk(sub.children, (child) => {
    if (child.t !== 'tool' || child.orphan) return;
    calls += 1;
    if (child.error) failed += 1;
  });
  const description = sub.description
    ? `<span class="act-muted">${escapeHtml(sub.description)}</span>`
    : '';
  const lost = n.orphan
    ? ' title="The record does not hold the call that started this subagent"'
    : '';
  const took =
    n.running && ui.live
      ? '<span class="act-running">running</span>'
      : `<span class="act-dur">${formatDuration(sub.ms)}</span>`;
  return (
    `<div class="act-group" id="row-${escapeHtml(key)}">` +
    `<button class="act-row act-group-head" type="button" data-kind="tool"${lost}` +
    ` data-collapse="${escapeHtml(key)}" aria-expanded="${open}">` +
    `<span class="act-t">${clock(n.ts)}</span><span class="act-i">${iconHtml('agent')}</span>` +
    `<span class="act-line"><span class="act-srv">subagent</span>` +
    `<span class="act-tool">${escapeHtml(sub.type)}</span>${description}</span>` +
    `<span class="act-r">${failed ? chip('err', `${failed} failed`) : ''}` +
    `<span class="act-dur soft">${count(calls, 'call')}</span>` +
    `${took}<span class="act-chev">${iconHtml('chevron-right')}</span></span></button>` +
    `${open ? `<div class="act-steps">${steps(sub.children, key, ui)}</div>` : ''}</div>`
  );
}

/**
 * @param {ActivityNode | undefined} node
 * @returns {node is ToolNode}
 */
function isTaskCall(node) {
  return node?.t === 'tool' && !node.sub && !node.server && TASK_TOOLS.has(node.tool);
}

/**
 * The rows of one level of the timeline. Fold ids are `<key>:<index>`, so they
 * are stable while the session only grows.
 * @param {ActivityNode[]} nodes
 * @param {string} key  the level's name: a turn, or the call of a subagent
 * @param {UiState} ui
 * @returns {string}
 */
function steps(nodes, key, ui) {
  let html = '';
  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    const id = `${key}:${i}`;
    if (node.t === 'usage' || node.t === 'prompt') continue;
    if (node.t !== 'tool') {
      html += textRow(node, id, ui);
    } else if (node.sub) {
      html += subGroup({ ...node, sub: node.sub }, ui);
    } else if (!isTaskCall(node)) {
      html += toolRow(node, ui);
    } else {
      // A run of task-list calls, with the model requests between them.
      const run = [node];
      let next = i + 1;
      while (nodes[next]?.t === 'usage' || isTaskCall(nodes[next])) {
        const following = nodes[next];
        if (isTaskCall(following)) run.push(following);
        next += 1;
      }
      i = next - 1;
      if (run.length === 1) {
        html += toolRow(node, ui);
      } else {
        const label = `<span class="act-muted">Task list: ${run.length} updates</span>`;
        const body = () =>
          `<div class="act-steps">${run.map((item) => toolRow(item, ui)).join('')}</div>`;
        html += foldRow({ id, kind: 'note', icon: 'list-tree', ts: node.ts, label, body }, ui);
      }
    }
  }
  return html;
}

/**
 * What a turn cost: its model requests, at any depth.
 * @param {ActivityNode[]} nodes
 * @param {string} label  what the requests belong to
 * @returns {string} `''` for a turn with no model request on record
 */
function turnFooter(nodes, label) {
  let requests = 0;
  let cost = 0;
  let ms = 0;
  walk(nodes, (node) => {
    if (node.t !== 'usage') return;
    requests += 1;
    cost += node.cost || 0;
    ms += node.ms || 0;
  });
  if (!requests) return '';
  const text =
    `${label}: ${count(requests, 'model request')}, ${formatDuration(ms)} model time, ` +
    `$${cost.toFixed(2)}`;
  return (
    `<div class="act-row act-turn-foot" data-kind="note"><span class="act-t"></span>` +
    `<span class="act-i">${iconHtml('clock')}</span>` +
    `<span class="act-line"><span class="act-muted">${text}</span></span><span></span></div>`
  );
}

/**
 * The session in turns.
 * @param {ActivityPayload} payload
 * @param {UiState} ui
 * @returns {string}
 */
function turns(payload, ui) {
  const identity = payload.session?.identity ?? '';
  const talk = ui.filter === 'conversation';
  return turnsOf(payload.nodes)
    .map((turn, index) => {
      const shown = talk ? turn.nodes.filter((node) => node.t === 'answer') : turn.nodes;
      // What the record holds ahead of the first prompt is not a turn of the session.
      const label = turn.prompt ? 'Turn' : 'Before the first prompt';
      const rows = steps(shown, `t${index}`, ui) + (talk ? '' : turnFooter(turn.nodes, label));
      return (
        `<section class="act-turn">${turn.prompt ? promptBlock(turn.prompt, identity) : ''}` +
        `${rows ? `<div class="act-steps">${rows}</div>` : ''}</section>`
      );
    })
    .join('');
}

/**
 * What a search or a call filter keeps, at any depth, without the turns.
 * @param {ActivityPayload} payload
 * @param {UiState} ui
 * @returns {string} `''` when nothing is kept
 */
function flat(payload, ui) {
  const identity = payload.session?.identity ?? '';
  const callsOnly = CALL_FILTERS.has(ui.filter);
  let html = '';
  let index = 0;
  walk(payload.nodes, (node) => {
    const id = `f:${index}`;
    index += 1;
    if (node.t === 'usage' || !matches(node, ui.q)) return;
    if (node.t === 'tool') {
      // A subagent is a group, not a call; a search still finds it by its task.
      if (node.sub && (callsOnly || node.orphan)) return;
      if (ui.filter === 'conversation') return;
      if (ui.filter === 'approvals' && !node.approval) return;
      if (ui.filter === 'errors' && !node.error) return;
      html += toolRow(node, ui);
    } else if (callsOnly) {
      // Calls only.
    } else if (node.t === 'prompt') {
      html += promptBlock(node, identity);
    } else if (node.t === 'answer' || ui.filter === 'all') {
      html += textRow(node, id, ui);
    }
  });
  return html;
}

/**
 * The filter switcher, the clock the times are on, and expand and collapse.
 * @param {ActivityPayload} payload
 * @param {UiState} ui
 * @returns {string}
 */
function bar(payload, ui) {
  const buttons = FILTERS.map(
    ([value, label]) =>
      `<button type="button" data-filter="${value}" aria-pressed="${ui.filter === value}">` +
      `${label}</button>`
  );
  const zone = escapeHtml(facilityZoneLabel(payload.session?.start ?? undefined));
  return (
    `<div class="act-sub"><div class="osprey-seg" role="group" aria-label="Show">` +
    `${buttons.join('')}</div><span class="spacer"></span>` +
    `<span class="hint">Times in ${zone}</span>` +
    `<button class="osprey-btn" type="button" data-act="expand">Expand all</button>` +
    `<button class="osprey-btn" type="button" data-act="collapse">Collapse all</button></div>`
  );
}

/**
 * The Timeline view.
 * @param {ActivityPayload} payload
 * @param {ActivitySummary} _model
 * @param {UiState} state
 * @returns {string}
 */
export function renderTimeline(payload, _model, state) {
  if (payload.nodes.length === 0) {
    return stateBlock(
      'activity',
      'No activity in this session yet',
      'Entries appear here as the agent record receives them.'
    );
  }
  const ui = { ...state, live: payload.session?.state === 'live' };
  const q = ui.q.trim();
  if (!q && !CALL_FILTERS.has(ui.filter)) return bar(payload, ui) + turns(payload, ui);
  const rows = flat(payload, ui);
  if (rows) return `${bar(payload, ui)}<div class="act-steps">${rows}</div>`;
  const [name, message] = q
    ? ['search', `Nothing in this session matches “${q}”`]
    : NOTHING_KEPT[ui.filter];
  return bar(payload, ui) + stateBlock(name, message);
}
