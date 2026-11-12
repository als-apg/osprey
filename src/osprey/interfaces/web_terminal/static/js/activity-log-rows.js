// @ts-check
/**
 * The rows of the Activity page: a tool call and its detail, a folded row, the
 * decision chip, the audit trail of a call, and the empty state. Every
 * function returns an HTML string, and every string that came from the agent,
 * the operator or the record is escaped on its way in.
 *
 * @module activity-log-rows
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { icon } from '/design-system/js/icons.js';

import { argSummary, clock, formatBytes, formatDuration } from './activity-log-format.js';
import { callKey } from './activity-log-model.js';

/** @typedef {import('./activity-log-model.js').ToolNode} ToolNode */

/**
 * @typedef {Object} UiState  What the operator has chosen on the page.
 * @property {string} view  `timeline`, `tools`, `safety` or `artifacts`.
 * @property {string} filter  Timeline filter: `all`, `conversation`, `tools`, `approvals`, `errors`.
 * @property {string} scope  Tool-calls scope: `all`, `mcp` or `builtin`.
 * @property {string} q  Search text.
 * @property {{ k: string, dir: number }} sort  Tool-calls sort: a column key, `1` up or `-1` down.
 * @property {Set<string>} open  Rows that are expanded, by call key or fold id.
 * @property {Set<string>} closed  Subagent groups that are collapsed, by call key.
 * @property {Map<string, string>} full  Whole results loaded so far, by call key.
 * @property {boolean} [live]  The session shown is live; set by the views.
 */

/** The icon of each server the framework ships; any other gets the wrench. */
const SERVER_ICONS = new Map([
  ['bluesky', 'bluesky'],
  ['controls', 'crosshair'],
  ['osprey_workspace', 'workspace'],
  ['python', 'terminal'],
  ['ariel', 'book-open'],
  ['osprey_facility_knowledge', 'book-open'],
  ['channel-finder', 'search'],
]);

/** A call's decision as the detail spells it out. @type {Record<string, string>} */
const DECISIONS = {
  approved: 'Asked the operator, approved',
  asked: 'Waiting for the operator',
  denied: 'Denied',
  none: 'Not recorded',
};

/** Icon markup by name: the same string every time it is asked for. @type {Map<string, string>} */
const iconCache = new Map();

/**
 * A design-system icon as markup.
 * @param {string} name  one of the design system's icon names
 * @returns {string}
 */
export function iconHtml(name) {
  let html = iconCache.get(name);
  if (html === undefined) {
    html = icon(name).outerHTML;
    iconCache.set(name, html);
  }
  return html;
}

/**
 * The icon of an MCP server.
 * @param {string} server  `''` for a built-in tool
 * @returns {string}
 */
export function serverIcon(server) {
  return iconHtml(SERVER_ICONS.get(server) ?? 'wrench');
}

/**
 * A design-system chip.
 * @param {'ok' | 'warn' | 'err' | 'info' | ''} tone
 * @param {string} label  plain text; escaped here
 * @param {string} [title]
 * @returns {string}
 */
export function chip(tone, label, title = '') {
  const cls = tone ? `osprey-chip osprey-chip--${tone}` : 'osprey-chip';
  const tip = title ? ` title="${escapeHtml(title)}"` : '';
  return `<span class="${cls}"${tip}><span class="osprey-chip-label">${escapeHtml(label)}</span></span>`;
}

/**
 * The decision on a call. The short form shows only the decisions the operator
 * was part of; the long form names every one, "not recorded" included.
 * @param {ToolNode} n
 * @param {boolean} long
 * @returns {string}
 */
export function verdictChip(n, long) {
  if (n.verdict === 'approved') {
    const wait = formatDuration(n.approval?.wait_ms);
    const label = `Approved by operator${wait ? ` · ${wait}` : ''}`;
    return chip('warn', label, 'Asked the operator; approved');
  }
  if (n.verdict === 'asked') return chip('warn', 'Waiting for approval');
  if (n.verdict === 'denied') return chip('err', 'Denied');
  if (!long) return '';
  if (n.verdict === 'allowed') {
    return `<span class="act-verdict" data-v="allowed">${iconHtml('shield-check')}Allowed</span>`;
  }
  return '<span class="act-verdict" data-v="none">not recorded</span>';
}

/**
 * How a call ended: failed, completed, still running, or no result on record.
 * A call without a result is running only while its session is live.
 * @param {ToolNode} n
 * @param {boolean} [live]
 * @returns {string}
 */
export function statusCell(n, live = false) {
  if (n.error) return `<span class="act-bad" title="Failed">${iconHtml('circle-x')}</span>`;
  if (n.result || n.sub) {
    return `<span class="act-ok" title="Completed">${iconHtml('check')}</span>`;
  }
  if (n.running && live) return '<span class="act-running">running</span>';
  return '<span class="act-dur soft" title="No result in the record">–</span>';
}

/**
 * A call's duration: the tool server's measurement, else call to result.
 * @param {ToolNode} n
 * @returns {string}
 */
export function durationCell(n) {
  if (typeof n.duration_ms === 'number') {
    const text = formatDuration(n.duration_ms);
    return `<span class="act-dur" title="Measured by the tool server">${text}</span>`;
  }
  if (typeof n.elapsed_ms === 'number' && !n.sub) {
    const text = formatDuration(n.elapsed_ms);
    return `<span class="act-dur soft" title="Call to result">${text}</span>`;
  }
  return '<span class="act-dur"></span>';
}

/**
 * What happened to a call, in order: issued, asked, decided, passed or refused
 * by each server gate, executed.
 * @param {ToolNode} n
 * @returns {string}
 */
export function trailList(n) {
  /** @type {{ ts: string | null, text: string, tone: string }[]} */
  const steps = [{ ts: n.ts, text: 'Call issued by the agent', tone: '' }];
  const ask = 'Approval requested from the operator';
  let asked = false;
  for (const entry of n.trail) {
    const gate = entry.surface ?? '';
    if (entry.decision === 'ask') {
      asked = true;
      steps.push({ ts: entry.ts, text: ask, tone: 'warn' });
    } else if (gate === 'hook_approval') {
      // The approval hook's own entry; the approval below says the same.
    } else if (entry.decision === 'allowed') {
      const target = n.target ? `, target ${n.target}` : '';
      steps.push({ ts: entry.ts, text: `Passed the ${gate} server gate${target}`, tone: 'ok' });
    } else if (entry.reason === 'tool_error') {
      steps.push({ ts: entry.ts, text: `The ${gate} server returned an error`, tone: 'err' });
    } else {
      const reason = entry.reason ? ` (${entry.reason})` : '';
      steps.push({
        ts: entry.ts,
        text: `Refused by the ${gate} server gate${reason}`,
        tone: 'err',
      });
    }
  }
  const approval = n.approval;
  if (approval?.asked_ts && !asked) steps.push({ ts: approval.asked_ts, text: ask, tone: 'warn' });
  if (approval?.decided_ts) {
    const who =
      !approval.approver || approval.approver === 'human' ? 'the operator' : approval.approver;
    const approved = approval.outcome === 'approved';
    const once = approved && n.source === 'user_temporary' ? ', this call only,' : '';
    const wait = formatDuration(approval.wait_ms);
    steps.push({
      ts: approval.decided_ts,
      text: `${approved ? 'Approved' : 'Denied'} by ${who}${once}${wait ? ` after ${wait}` : ''}`,
      tone: approved ? 'ok' : 'err',
    });
  }
  steps.sort((a, b) => (Date.parse(a.ts ?? '') || 0) - (Date.parse(b.ts ?? '') || 0));
  if (typeof n.duration_ms === 'number') {
    const failed = n.error ? ', failed' : '';
    steps.push({
      ts: null,
      text: `Executed in ${formatDuration(n.duration_ms)}${failed}`,
      tone: n.error ? 'err' : '',
    });
  }
  const items = steps.map(
    (step) =>
      `<li data-tone="${step.tone}"><time>${step.ts ? clock(step.ts, { ms: true }) : ''}</time>` +
      `<span>${escapeHtml(step.text)}</span></li>`
  );
  return `<ol class="act-trail">${items.join('')}</ol>`;
}

/**
 * A call's arguments: the single-line ones as one JSON block, each multi-line
 * string as a block of its own.
 * @param {Record<string, unknown>} input
 * @returns {string}
 */
function argsBlock(input) {
  /** @type {Record<string, unknown>} */
  const plain = {};
  let blocks = '';
  for (const [key, value] of Object.entries(input ?? {})) {
    if (typeof value === 'string' && value.includes('\n')) {
      const lines = value.split('\n').length;
      blocks +=
        `<h4>${escapeHtml(key)}<span class="grow"></span><span>${lines} lines</span></h4>` +
        `<pre class="act-code">${escapeHtml(value)}</pre>`;
    } else {
      plain[key] = value;
    }
  }
  if (Object.keys(plain).length === 0) {
    return blocks || '<h4>Arguments</h4><div class="act-more">None</div>';
  }
  const json = escapeHtml(JSON.stringify(plain, null, 2));
  return `<h4>Arguments</h4><pre class="act-code">${json}</pre>${blocks}`;
}

/**
 * A call's result: the shown part, and the way to the whole one when it was cut.
 * @param {ToolNode} n
 * @param {UiState} ui
 * @returns {string}
 */
function resultBlock(n, ui) {
  if (n.sub) return '';
  const result = n.result;
  if (!result) {
    const why =
      n.running && ui.live
        ? 'The call has not returned yet.'
        : 'No result in the record for this call.';
    return `<h4>Result</h4><div class="act-more">${why}</div>`;
  }
  const key = callKey(n);
  const whole = ui.full.get(key);
  const cut = result.cut && whole === undefined;
  const size = formatBytes(result.size);
  const more = cut
    ? `<div class="act-more"><button class="osprey-btn" type="button" data-full="${escapeHtml(key)}"` +
      ` data-key="${escapeHtml(result.key)}">Load full result (${size})</button>` +
      `<span>Showing the first ${result.text.length} characters</span></div>`
    : '';
  return (
    `<h4>Result<span class="grow"></span><span>${size}</span></h4>` +
    `<pre class="act-code act-result">${escapeHtml(whole ?? result.text)}${cut ? '\n…' : ''}</pre>` +
    more
  );
}

/**
 * @param {ToolNode} n
 * @returns {string}
 */
function decisionText(n) {
  if (n.verdict !== 'allowed') return DECISIONS[n.verdict] ?? DECISIONS.none;
  if (n.source === 'config') return 'Allowed by configuration';
  return n.source === 'hook' ? 'Allowed by a hook' : 'Allowed';
}

/**
 * The expanded part of a call: arguments and result, the safety facts, and the
 * trail when a gate or the operator had a part in it.
 * @param {ToolNode} n
 * @param {UiState} ui
 * @returns {string}
 */
export function toolDetail(n, ui) {
  /** @type {[string, string, string?][]} */
  const facts = [['Decision', decisionText(n)]];
  if (n.posture)
    facts.push(['Writes', n.posture === 'writes' ? 'Enabled for this session' : n.posture]);
  if (n.target) facts.push(['Target', n.target]);
  facts.push(['Tool', n.server ? `MCP server ${n.server}` : 'Built into the agent']);
  facts.push(['Agent', n.agent]);
  if (n.id) facts.push(['Call id', n.id, 'mono']);
  const rows = facts.map(
    ([label, value, cls]) =>
      `<dt>${label}</dt><dd${cls ? ` class="${cls}"` : ''}>${escapeHtml(value)}</dd>`
  );
  const partial = n.incomplete
    ? '<div class="act-more">The record has not received all of this call.</div>'
    : '';
  const trail = n.trail.length > 0 || n.approval ? trailList(n) : '';
  return (
    `<div class="act-detail"><div>${partial}${argsBlock(n.input)}${resultBlock(n, ui)}</div>` +
    `<div><h4>Safety</h4><dl class="act-kv">${rows.join('')}</dl>${trail}</div></div>`
  );
}

/**
 * One tool call as a row that opens onto its detail.
 * @param {ToolNode} n
 * @param {UiState} ui
 * @returns {string}
 */
export function toolRow(n, ui) {
  const key = escapeHtml(callKey(n));
  const open = ui.open.has(callKey(n));
  const tone = n.error ? 'err' : n.approval ? 'warn' : '';
  const summary = n.error && n.errmsg ? n.errmsg : argSummary(n.input);
  const server = n.server ? `<span class="act-srv">${escapeHtml(n.server)}</span>` : '';
  const partial = n.incomplete
    ? chip('warn', 'Incomplete', 'The record has not received all of this call')
    : '';
  return (
    `<button class="act-row" type="button" data-kind="tool" data-tone="${tone}"` +
    ` data-toggle="${key}" id="row-${key}" aria-expanded="${open}">` +
    `<span class="act-t">${clock(n.ts)}</span>` +
    `<span class="act-i">${serverIcon(n.server)}</span>` +
    `<span class="act-line">${server}<span class="act-tool">${escapeHtml(n.tool)}</span>` +
    `<span class="act-args">${escapeHtml(summary)}</span></span>` +
    `<span class="act-r">${partial}${verdictChip(n, false)}${durationCell(n)}` +
    `${statusCell(n, ui.live)}<span class="act-chev">${iconHtml('chevron-right')}</span></span>` +
    `</button>${open ? `<div class="act-body-block">${toolDetail(n, ui)}</div>` : ''}`
  );
}

/**
 * A row that opens onto a body built only when it is open.
 * @param {Object} row
 * @param {string} row.id  the fold's name in `ui.open`
 * @param {string} row.kind  `thinking` or `note`
 * @param {string} row.icon
 * @param {string | null | undefined} row.ts
 * @param {string} row.label  markup, already escaped
 * @param {() => string} row.body  markup, already escaped
 * @param {string} [row.right]  markup for the right-hand cluster
 * @param {UiState} ui
 * @returns {string}
 */
export function foldRow({ id, kind, icon: name, ts, label, body, right = '' }, ui) {
  const open = ui.open.has(id);
  return (
    `<button class="act-row" type="button" data-kind="${kind}" data-toggle="${escapeHtml(id)}"` +
    ` aria-expanded="${open}"><span class="act-t">${clock(ts)}</span>` +
    `<span class="act-i">${iconHtml(name)}</span><span class="act-line">${label}</span>` +
    `<span class="act-r">${right}<span class="act-chev">${iconHtml('chevron-right')}</span></span>` +
    `</button>${open ? `<div class="act-body-block">${body()}</div>` : ''}`
  );
}

/**
 * The design-system empty state as markup.
 * @param {string} iconName
 * @param {string} message  plain text; escaped here
 * @param {string} [detail]  plain text; escaped here
 * @param {{ label: string, act: string }} [action]  a button the page's click
 *   handler reads by `data-act`
 * @returns {string}
 */
export function stateBlock(iconName, message, detail = '', action) {
  const more = detail ? `<p class="osprey-state-detail">${escapeHtml(detail)}</p>` : '';
  const button = action
    ? `<button class="osprey-btn" type="button" data-act="${escapeHtml(action.act)}">` +
      `${escapeHtml(action.label)}</button>`
    : '';
  return (
    `<div class="osprey-state" role="status">${iconHtml(iconName)}` +
    `<p class="osprey-state-message">${escapeHtml(message)}</p>${more}${button}</div>`
  );
}
