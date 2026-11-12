// @ts-check
/**
 * The Tool calls, Safety and Artifacts views of the Activity page: the same
 * session as a sortable table of its calls, as the decisions made about them,
 * and as what the agent saved.
 *
 * @module activity-log-views
 */

import { escapeHtml } from '/design-system/js/dom.js';

import { argSummary, clock, formatBytes, formatDuration } from './activity-log-format.js';
import { callKey, findCallPath, matches } from './activity-log-model.js';
import {
  durationCell,
  iconHtml,
  stateBlock,
  statusCell,
  toolDetail,
  trailList,
  verdictChip,
} from './activity-log-rows.js';

/** @typedef {import('./activity-log-model.js').ActivityPayload} ActivityPayload */
/** @typedef {import('./activity-log-model.js').ActivitySummary} ActivitySummary */
/** @typedef {import('./activity-log-model.js').ArtifactRow} ArtifactRow */
/** @typedef {import('./activity-log-model.js').ToolNode} ToolNode */
/** @typedef {import('./activity-log-rows.js').UiState} UiState */

/** What each sortable column of the call table orders by. @type {Record<string, (n: ToolNode) => number | string>} */
const SORT_KEYS = {
  ts: (n) => Date.parse(n.ts) || 0,
  tool: (n) => `${n.server} ${n.tool}`.toLowerCase(),
  dur: (n) => n.duration_ms ?? n.elapsed_ms ?? -1,
  size: (n) => n.result?.size ?? -1,
};

/** The icon of an artifact type; anything else is a file. @type {Record<string, string>} */
const ARTIFACT_ICONS = {
  plot: 'chart-line',
  plot_html: 'chart-line',
  json: 'database',
  table: 'table',
  csv: 'table',
};

/**
 * A call's server and tool name.
 * @param {ToolNode} n
 * @returns {string}
 */
function callName(n) {
  const server = n.server ? `<span class="act-srv">${escapeHtml(n.server)}</span>` : '';
  return `${server}<span class="act-tool">${escapeHtml(n.tool)}</span>`;
}

/**
 * A table's cell for a call's name: cut to its column, whole in the tooltip.
 * @param {ToolNode} n
 * @returns {string}
 */
function nameCell(n) {
  return `<td class="act-name" title="${escapeHtml(n.name)}">${callName(n)}</td>`;
}

/**
 * A section heading with its count.
 * @param {string} iconName
 * @param {string} label
 * @param {number} count
 * @returns {string}
 */
function heading(iconName, label, count) {
  return (
    `<h2 class="act-h2">${iconHtml(iconName)}${label}` +
    `<span class="osprey-count">${count}</span></h2>`
  );
}

/**
 * The Tool calls view: every call of the session as one sortable table.
 * @param {ActivityPayload} payload
 * @param {ActivitySummary} model
 * @param {UiState} state
 * @returns {string}
 */
export function renderTools(payload, model, state) {
  const ui = { ...state, live: payload.session?.state === 'live' };
  const mcp = model.tools.filter((n) => n.server).length;
  const scopes = [
    ['all', 'All', model.tools.length],
    ['mcp', 'Control system and services', mcp],
    ['builtin', 'Built in', model.tools.length - mcp],
  ];
  const key = SORT_KEYS[ui.sort.k] ?? SORT_KEYS.ts;
  const rows = model.tools
    .filter((n) => ui.scope === 'all' || (ui.scope === 'mcp') === Boolean(n.server))
    .filter((n) => matches(n, ui.q))
    .sort((a, b) => {
      const [left, right] = [key(a), key(b)];
      return (left < right ? -1 : left > right ? 1 : 0) * ui.sort.dir;
    });
  const buttons = scopes.map(
    ([value, label, count]) =>
      `<button type="button" data-scope="${value}" aria-pressed="${ui.scope === value}">` +
      `${label}<span class="osprey-count">${count}</span></button>`
  );
  const bar =
    `<div class="act-sub"><div class="osprey-seg" role="group" aria-label="Tools">` +
    `${buttons.join('')}</div><span class="spacer"></span>` +
    `<span class="hint">${rows.length} of ${model.tools.length} calls</span></div>`;
  if (rows.length === 0) return bar + stateBlock('search', 'No tool call matches');

  /** @param {string} k @param {string} label @param {string} [cls] */
  const th = (k, label, cls = '') => {
    const sorted = ui.sort.k === k;
    const order = sorted ? (ui.sort.dir > 0 ? 'ascending' : 'descending') : 'none';
    const arrow = sorted ? (ui.sort.dir > 0 ? ' ↑' : ' ↓') : '';
    return (
      `<th data-sort="${k}" tabindex="0" aria-sort="${order}"` +
      `${cls ? ` class="${cls}"` : ''}>${label}${arrow}</th>`
    );
  };
  const head =
    `<tr>${th('ts', 'Time')}${th('tool', 'Tool')}<th>Arguments</th><th>Agent</th>` +
    `<th>Decision</th>${th('dur', 'Duration', 'num')}${th('size', 'Result', 'num')}` +
    `<th aria-label="Outcome"></th></tr>`;
  const body = rows.map((n) => {
    const id = escapeHtml(callKey(n));
    const open = ui.open.has(callKey(n));
    const args = n.sub ? n.sub.description : argSummary(n.input);
    const agent = n.agent === 'main' ? '<span class="act-muted">main</span>' : escapeHtml(n.agent);
    return (
      `<tr class="row" tabindex="0" data-toggle="${id}" aria-expanded="${open}">` +
      `<td class="act-t">${clock(n.ts)}</td>${nameCell(n)}` +
      `<td class="wide act-args">${escapeHtml(args)}</td><td>${agent}</td>` +
      `<td>${verdictChip(n, true)}</td><td class="num">${durationCell(n)}</td>` +
      `<td class="num">${n.result ? formatBytes(n.result.size) : ''}</td>` +
      `<td>${statusCell(n, ui.live)}</td></tr>` +
      `${open ? `<tr class="detail"><td colspan="8">${toolDetail(n, ui)}</td></tr>` : ''}`
    );
  });
  return (
    `${bar}<table class="act-table"><thead>${head}</thead>` +
    `<tbody>${body.join('')}</tbody></table>`
  );
}

/**
 * One count of the Safety view.
 * @param {string} title
 * @param {number} count
 * @param {string} text
 * @param {string} [tone]  applied only when the count is not zero
 * @returns {string}
 */
function card(title, count, text, tone = '') {
  return (
    `<div class="osprey-card act-card" data-tone="${count ? tone : ''}">` +
    `<div class="act-card-n">${count}</div><h3>${title}</h3><p>${text}</p></div>`
  );
}

/**
 * A call the operator was asked about: what was asked, how long the answer
 * took, and what happened to the call.
 * @param {ToolNode} n
 * @param {string} identity
 * @returns {string}
 */
function approvalCard(n, identity) {
  const wait = formatDuration(n.approval?.wait_ms);
  const waited = wait
    ? `${wait}${identity ? ` for ${escapeHtml(identity)}` : ''}`
    : n.verdict === 'asked'
      ? 'Waiting for the operator'
      : 'Not recorded';
  const rows = [
    `<dt>Arguments</dt><dd class="mono">${escapeHtml(argSummary(n.input))}</dd>`,
    n.target ? `<dt>Target</dt><dd>${escapeHtml(n.target)}</dd>` : '',
    `<dt>Agent</dt><dd>${escapeHtml(n.agent)}</dd>`,
    `<dt>Waited</dt><dd>${waited}</dd>`,
  ];
  return (
    `<div class="osprey-card act-approval"><div><h3>${callName(n)}${verdictChip(n, false)}</h3>` +
    `<dl class="act-kv">${rows.join('')}</dl><div class="act-more">` +
    `<button class="osprey-btn" type="button" data-jump="${escapeHtml(callKey(n))}">` +
    `Show in timeline</button></div></div>${trailList(n)}</div>`
  );
}

/**
 * One call a server gate filed an entry for.
 * @param {ToolNode} n
 * @returns {string}
 */
function gateRow(n) {
  const gates = new Set();
  for (const entry of n.trail) {
    if (entry.surface && entry.surface !== 'hook_approval') gates.add(entry.surface);
  }
  const writes = n.posture === 'writes' ? 'enabled' : (n.posture ?? '');
  return (
    `<tr class="row" tabindex="0" data-jump="${escapeHtml(callKey(n))}">` +
    `<td class="act-t">${clock(n.ts)}</td>${nameCell(n)}` +
    `<td class="wide act-args">${escapeHtml(argSummary(n.input))}</td>` +
    `<td>${escapeHtml([...gates].join(', '))}</td><td>${escapeHtml(writes)}</td>` +
    `<td>${escapeHtml(n.target)}</td><td>${verdictChip(n, true)}</td></tr>`
  );
}

/**
 * The Safety view: how many calls the rules let through, how many the operator
 * was asked about, and what each server gate filed.
 * @param {ActivityPayload} payload
 * @param {ActivitySummary} model
 * @returns {string}
 */
export function renderSafety(payload, model) {
  if (!model.decided && model.gated.length === 0) {
    return stateBlock(
      'shield-check',
      'No safety decisions in the record for this session',
      'The record holds the conversation and the tool calls of this session, ' +
        'but no decision or audit entries.'
    );
  }
  const identity = payload.session?.identity ?? '';
  const allowed = model.tools.filter((n) => n.verdict === 'allowed').length;
  const denied = model.tools.filter((n) => n.verdict === 'denied').length;
  const approved = model.approvals.filter((n) => n.verdict === 'approved').length;
  const declined = model.approvals.filter((n) => n.verdict === 'denied').length;
  const asked = model.approvals.length
    ? `${approved} approved, ${declined} declined.`
    : 'No call needed an approval.';
  const cards =
    `<div class="act-cards">` +
    card(
      'Allowed by configuration',
      allowed,
      "Calls the deployment's rules permit without asking."
    ) +
    card('Asked the operator', model.approvals.length, asked, 'warn') +
    card('Denied', denied, denied ? 'Calls the rules refused.' : 'No call was refused.', 'err') +
    `</div>`;
  const approvals = model.approvals.length
    ? heading('user', 'Approvals', model.approvals.length) +
      model.approvals.map((n) => approvalCard(n, identity)).join('')
    : '';
  const gated = model.gated.length
    ? `${heading('shield-check', 'Calls through a server gate', model.gated.length)}` +
      `<table class="act-table"><thead><tr><th>Time</th><th>Call</th><th>Arguments</th>` +
      `<th>Gate</th><th>Writes</th><th>Target</th><th>Decision</th></tr></thead>` +
      `<tbody>${model.gated.map(gateRow).join('')}</tbody></table>`
    : '';
  return `<div class="act-view">${cards}${approvals}${gated}</div>`;
}

/**
 * One saved artifact.
 * @param {ArtifactRow} a
 * @param {boolean} inRecord  whether the record holds the call that saved it
 * @returns {string}
 */
function artifactCard(a, inRecord) {
  const by = a.tool ? ` by ${a.tool}` : '';
  const within = a.agent && a.agent !== 'main' ? ` in subagent ${a.agent}` : '';
  const facts = [
    a.form_label || a.form,
    a.type,
    formatBytes(a.size),
    `Saved ${clock(a.ts)}${by}${within}`,
  ];
  const jump = inRecord
    ? `<button class="osprey-btn" type="button" data-jump="${escapeHtml(a.call)}">` +
      `Show in timeline</button>`
    : '';
  return (
    `<div class="osprey-card act-art">` +
    `<span class="act-art-ic">${iconHtml(ARTIFACT_ICONS[a.type] ?? 'file-text')}</span>` +
    `<div class="act-art-text"><h3>${escapeHtml(a.title)}</h3><p>` +
    `${facts
      .filter(Boolean)
      .map((fact) => `<span>${escapeHtml(fact)}</span>`)
      .join('')}</p></div>` +
    `<div class="act-art-actions">${jump}<button class="osprey-btn primary" type="button"` +
    ` data-open-artifact="${escapeHtml(a.id)}"><span class="osprey-btn-label">Open</span>` +
    `</button></div></div>`
  );
}

/**
 * The Artifacts view: what the agent saved, each with the way to its call.
 * @param {ActivityPayload} payload
 * @returns {string}
 */
export function renderArtifacts(payload) {
  if (payload.artifacts.length === 0) {
    return stateBlock(
      'layers',
      'No artifacts in this session',
      'Plots, tables and files the agent saves appear here.'
    );
  }
  const cards = payload.artifacts.map((a) =>
    artifactCard(a, Boolean(a.call) && findCallPath(payload.nodes, a.call) !== null)
  );
  return `<div class="act-view">${cards.join('')}</div>`;
}
