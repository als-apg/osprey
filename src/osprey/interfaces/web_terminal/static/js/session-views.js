// @ts-check
/* OSPREY Web Terminal — Session Activity Log: View Renderers
 *
 * The four Activity-log views (Agents, Tool Log, Artifacts, Conversation)
 * for session.html. Each renderer takes a small dependency-injection
 * context --
 * { apiFetch, showToast, cache } -- instead of importing session.js
 * directly, so every renderer can be unit tested with a stubbed apiFetch
 * and a throwaway cache object (see session-views.test.mjs) without
 * booting the whole page (nav wiring, refresh loop, message listener --
 * session.js's job).
 *
 * `logFilters`/`logData` are kept as module-level state (the same
 * singleton pattern theme-manager.js uses for role/preference) since they
 * only matter to the Tool Log view's own filter bar and must survive
 * across refreshes of that one view.
 *
 * The API response bodies each view fetches (session-agents,
 * session-agent-timeline, session-log, session-summary, session-chat) have
 * no formal shared type anywhere in the codebase yet -- they're rendered
 * defensively (every field access already guards against `undefined`, the
 * same way the original inline script did), so this module leans on
 * `apiFetch`'s `Promise<any>` return rather than inventing response
 * typedefs that could drift from the real API.
 *
 * @module session-views
 */

import { escapeHtml as esc } from '/design-system/js/dom.js';
import { serverClass, ts, fmtBytes, emptyState, typeIcon } from './session-helpers.js';

/**
 * @typedef {{
 *   apiFetch: (path: string) => Promise<any>,
 *   showToast: (msg: string) => void,
 *   cache: Record<string, unknown>,
 * }} RenderCtx
 */

// Shared formatting/lookup helpers (serverClass, ts, fmtBytes, emptyState,
// typeIcon) live in session-helpers.js -- re-exported so existing importers
// of this module keep working unchanged.
export { serverClass, ts, fmtBytes, emptyState, typeIcon };

// ---- Agents View ----

/**
 * @param {RenderCtx} ctx
 * @returns {Promise<void>}
 */
export async function renderAgents({ apiFetch, showToast, cache }) {
  const el = document.getElementById('view-agents');
  if (!el) return;
  let data;
  try {
    data = await apiFetch('/api/session-agents');
  } catch {
    showToast('Failed to load agents');
    if (!cache.agents) el.innerHTML = emptyState('ERROR', 'Could not load agent data');
    return;
  }
  if (!data || (!data.agents.length && data.total_events === 0)) {
    el.innerHTML = emptyState('NO AGENTS', 'No subagent activity recorded');
    cache.agents = null;
    return;
  }
  cache.agents = data;

  const agents = data.agents || [];
  const toolsByAgent = data.tool_calls_by_agent || {};
  const agentCount = agents.filter((/** @type {any} */ a) => a.agent_id && a.agent_id !== 'main').length;

  let html = `<div class="summary-bar">
    <span>Events: <span class="val">${data.total_events}</span></span>
    <span>Agents: <span class="val">${agentCount}</span></span>
  </div>`;

  for (const agent of agents) {
    const aid = agent.agent_id || 'main';
    const isRoot = !agent.agent_id || agent.agent_id === 'main';
    const name = isRoot ? 'ROOT SESSION' : (agent.agent_type || aid);
    const srvCls = isRoot ? '' : serverClass(agent.server || agent.agent_type);
    const tools = toolsByAgent[aid] || [];
    const errCount = tools.filter((/** @type {any} */ t) => t.is_error).length;
    const dur = agent.duration ? agent.duration : '';

    html += `<div class="agent-card ${srvCls}" style="border-left-color:${isRoot ? 'var(--color-accent-light)' : `var(--srv, var(--color-accent-light))`}" data-agent="${esc(aid)}">
      <div class="agent-card-header">
        <span class="agent-name">${esc(name)}</span>
        ${agent.agent_type ? `<span class="badge">${esc(agent.agent_type)}</span>` : ''}
        ${dur ? `<span class="badge">${esc(dur)}</span>` : ''}
        <span class="badge">${tools.length} tools</span>
        ${errCount ? `<span class="badge badge-error">${errCount} errors</span>` : ''}
      </div>
      <div class="agent-tools">`;

    for (let ti = 0; ti < tools.length; ti++) {
      const tc = tools[ti];
      const isErr = tc.is_error;
      const tcSrv = serverClass(tc.server_name || tc.server);
      const toolKey = `${aid}-${ti}`;
      html += `<div class="tool-row ${isErr ? 'is-error' : ''} ${tcSrv}" data-tool-key="${toolKey}">
        <span class="tool-time">${ts(tc.timestamp)}</span>
        <span class="tool-name">${esc(tc.tool_name || tc.tool || '')}</span>
        <span class="server-badge ${tcSrv}">${esc(tc.server_name || tc.server || '')}</span>
      </div>
      <div class="log-detail" data-tool-detail="${toolKey}">
        ${tc.arguments ? `<div class="detail-label">Arguments</div><pre>${esc(typeof tc.arguments === 'string' ? tc.arguments : JSON.stringify(tc.arguments, null, 2))}</pre>` : ''}
        ${tc.result_summary ? `<div class="detail-label">Result</div><pre>${esc(tc.result_summary)}</pre>` : ''}
      </div>`;
    }

    if (!isRoot) {
      html += `<div class="agent-timeline" data-tl-agent="${esc(aid)}">
        <div class="tl-loading">LOADING TIMELINE…</div>
      </div>`;
    }

    html += `</div>`;
  }

  el.innerHTML = html;

  const cards = /** @type {NodeListOf<HTMLElement>} */ (el.querySelectorAll('.agent-card'));
  cards.forEach((card) => {
    const header = card.querySelector('.agent-card-header');
    if (!header) return;
    header.addEventListener('click', () => {
      const wasExpanded = card.classList.contains('expanded');
      card.classList.toggle('expanded');
      if (!wasExpanded) {
        const tlEl = /** @type {HTMLElement|null} */ (card.querySelector('.agent-timeline'));
        if (tlEl && !tlEl.dataset.loaded) {
          loadAgentTimeline(tlEl, card.dataset.agent || '', apiFetch);
        }
      }
    });
  });

  const toolRows = /** @type {NodeListOf<HTMLElement>} */ (el.querySelectorAll('.agent-tools .tool-row'));
  toolRows.forEach((row) => {
    row.addEventListener('click', (e) => {
      e.stopPropagation();
      const key = row.dataset.toolKey;
      const detail = el.querySelector(`.log-detail[data-tool-detail="${key}"]`);
      if (detail) detail.classList.toggle('expanded');
    });
  });
}

// ---- Agent Timeline ----

/**
 * @param {HTMLElement} tlEl
 * @param {string} agentId
 * @param {(path: string) => Promise<any>} apiFetch
 * @returns {Promise<void>}
 */
async function loadAgentTimeline(tlEl, agentId, apiFetch) {
  tlEl.dataset.loaded = '1';
  try {
    const data = await apiFetch(`/api/session-agent-timeline?agent_id=${encodeURIComponent(agentId)}`);
    if (!data || !data.timeline || !data.timeline.length) {
      tlEl.innerHTML = '<div class="tl-loading">NO INTERNAL STATE AVAILABLE</div>';
      return;
    }
    renderTimeline(tlEl, data.timeline);
  } catch {
    tlEl.innerHTML = '<div class="tl-loading">FAILED TO LOAD TIMELINE</div>';
  }
}

/**
 * @param {HTMLElement} container
 * @param {any[]} timeline
 * @returns {void}
 */
function renderTimeline(container, timeline) {
  let html = '';
  for (const entry of timeline) {
    const kind = entry.kind || 'unknown';
    const errCls = entry.is_error ? ' tl-error' : '';
    const kindLabel = kind === 'tool_call' ? entry.tool || 'tool call'
                     : kind === 'tool_result' ? 'result'
                     : kind;

    html += `<div class="tl-entry tl-${esc(kind)}${errCls}">`;
    html += `<div class="tl-kind">${esc(kindLabel)}</div>`;

    if (kind === 'tool_call') {
      const args = entry.arguments;
      const argStr = args ? (typeof args === 'string' ? args : JSON.stringify(args, null, 2)) : '';
      html += `<div class="tl-tool-name">${esc(entry.tool || '')}</div>`;
      if (argStr) {
        html += `<div class="tl-args">${esc(argStr)}</div>`;
      }
    } else {
      const text = entry.text || '';
      const needsCollapse = text.length > 200;
      html += `<div class="tl-text${needsCollapse ? ' tl-collapsed' : ''}">${esc(text)}</div>`;
    }

    html += `</div>`;
  }
  container.innerHTML = html;

  const collapsedTexts = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.tl-text.tl-collapsed'));
  collapsedTexts.forEach((el) => {
    el.addEventListener('click', () => el.classList.toggle('tl-collapsed'));
  });
}

// ---- Tool Log View ----

const logFilters = { agent: '', tool: '', errorsOnly: false };
/** @type {any[]} */
let logData = [];

/**
 * @param {RenderCtx} ctx
 * @returns {Promise<void>}
 */
export async function renderToolLog({ apiFetch, showToast, cache }) {
  const el = document.getElementById('view-toollog');
  if (!el) return;
  const ctx = { apiFetch, showToast, cache };
  let data;
  try {
    const qp = new URLSearchParams();
    if (logFilters.agent) qp.set('agent_id', logFilters.agent);
    if (logFilters.tool) qp.set('tool', logFilters.tool);
    if (logFilters.errorsOnly) qp.set('errors_only', 'true');
    qp.set('last_n', '500');
    data = await apiFetch('/api/session-log?' + qp.toString());
  } catch {
    showToast('Failed to load tool log');
    if (!cache.toollog) el.innerHTML = emptyState('ERROR', 'Could not load tool log');
    return;
  }
  if (!data || !data.events.length) {
    el.innerHTML = buildFilterBar() + emptyState('NO TOOL CALLS', 'No OSPREY tool calls in this session');
    attachFilterListeners(el, ctx);
    cache.toollog = null;
    return;
  }
  cache.toollog = data;
  logData = data.events;

  const agentIds = [...new Set(logData.map((e) => e.agent_id || 'main'))];

  let html = buildFilterBar(agentIds);

  html += `<div class="log-headers">
    <span class="col-time">Time</span>
    <span class="col-tool">Tool</span>
    <span class="col-server">Server</span>
    <span class="col-agent">Agent</span>
  </div>`;

  for (let i = 0; i < logData.length; i++) {
    const ev = logData[i];
    const isErr = ev.is_error;
    const srvCls = serverClass(ev.server_name);
    html += `<div class="log-row ${isErr ? 'is-error' : ''}" data-idx="${i}">
      <span class="col-time">${ts(ev.timestamp)}</span>
      <span class="col-tool">${esc(ev.tool_name || '')}</span>
      <span class="col-server"><span class="server-badge ${srvCls}">${esc(ev.server_name || '')}</span></span>
      <span class="col-agent">${esc((ev.agent_id || 'main').slice(0, 8))}</span>
    </div>
    <div class="log-detail" data-detail="${i}">
      ${ev.arguments ? `<div class="detail-label">Arguments</div><pre>${esc(typeof ev.arguments === 'string' ? ev.arguments : JSON.stringify(ev.arguments, null, 2))}</pre>` : ''}
      ${ev.result_summary ? `<div class="detail-label">Result</div><pre>${esc(ev.result_summary)}</pre>` : ''}
    </div>`;
  }

  el.innerHTML = html;
  attachFilterListeners(el, ctx);

  const logRows = /** @type {NodeListOf<HTMLElement>} */ (el.querySelectorAll('.log-row'));
  logRows.forEach((row) => {
    row.addEventListener('click', () => {
      const idx = row.dataset.idx;
      const detail = el.querySelector(`.log-detail[data-detail="${idx}"]`);
      if (detail) detail.classList.toggle('expanded');
    });
  });
}

/**
 * @param {string[]} [agentIds]
 * @returns {string}
 */
function buildFilterBar(agentIds) {
  let opts = '<option value="">All agents</option>';
  if (agentIds) {
    for (const id of agentIds) {
      const sel = logFilters.agent === id ? ' selected' : '';
      opts += `<option value="${esc(id)}"${sel}>${esc(id.slice(0, 12))}</option>`;
    }
  }
  return `<div class="filter-bar">
    <select id="filter-agent">${opts}</select>
    <input type="text" id="filter-tool" placeholder="Tool name..." value="${esc(logFilters.tool)}">
    <label><input type="checkbox" id="filter-errors" ${logFilters.errorsOnly ? 'checked' : ''}> Errors only</label>
  </div>`;
}

/**
 * @param {HTMLElement} el
 * @param {RenderCtx} ctx
 * @returns {void}
 */
function attachFilterListeners(el, ctx) {
  const agentSel = /** @type {HTMLSelectElement|null} */ (el.querySelector('#filter-agent'));
  const toolInput = /** @type {HTMLInputElement|null} */ (el.querySelector('#filter-tool'));
  const errCb = /** @type {HTMLInputElement|null} */ (el.querySelector('#filter-errors'));
  if (agentSel) agentSel.addEventListener('change', () => { logFilters.agent = agentSel.value; renderToolLog(ctx); });
  if (toolInput) {
    /** @type {ReturnType<typeof setTimeout>|undefined} */
    let debounce;
    toolInput.addEventListener('input', () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => { logFilters.tool = toolInput.value; renderToolLog(ctx); }, 300);
    });
  }
  if (errCb) errCb.addEventListener('change', () => { logFilters.errorsOnly = errCb.checked; renderToolLog(ctx); });
}

// ---- Artifacts View ----

/**
 * @param {RenderCtx} ctx
 * @returns {Promise<void>}
 */
export async function renderArtifacts({ apiFetch, showToast, cache }) {
  const el = document.getElementById('view-artifacts');
  if (!el) return;
  let data;
  try {
    data = await apiFetch('/api/session-summary');
  } catch {
    showToast('Failed to load artifacts');
    if (!cache.artifacts) el.innerHTML = emptyState('ERROR', 'Could not load artifact data');
    return;
  }
  if (!data || !data.entries.length) {
    el.innerHTML = emptyState('NO ARTIFACTS', 'No data or artifacts saved yet');
    cache.artifacts = null;
    return;
  }
  cache.artifacts = data;

  const totals = data.totals || {};
  const cats = totals.categories || {};

  let html = `<div class="totals-bar">
    <span><span class="val">${totals.entry_count || 0}</span> entries</span>
    <span><span class="val">${fmtBytes(totals.total_bytes)}</span></span>`;
  for (const [cat, count] of Object.entries(cats)) {
    html += `<span class="cat-pill">${esc(cat)}: ${count}</span>`;
  }
  html += `</div>`;

  /** @type {Record<string, any[]>} */
  const grouped = {};
  for (const entry of data.entries) {
    const cat = entry.category || 'uncategorized';
    (grouped[cat] = grouped[cat] || []).push(entry);
  }

  for (const [cat, entries] of Object.entries(grouped)) {
    html += `<details class="artifact-group" open>
      <summary>${esc(cat)} (${entries.length})</summary>`;
    for (const entry of entries) {
      const channels = entry.channels || [];
      html += `<div class="artifact-entry">
        <span class="artifact-icon">${typeIcon(entry.artifact_type)}</span>
        <div class="artifact-body">
          <div class="artifact-title">${esc(entry.title || entry.name || 'Untitled')}</div>
          ${entry.description ? `<div class="artifact-desc">${esc(entry.description)}</div>` : ''}
          <div class="artifact-meta">
            <span>${fmtBytes(entry.size_bytes)}</span>
            ${entry.timestamp ? `<span>${ts(entry.timestamp)}</span>` : ''}
            ${entry.artifact_type ? `<span>${esc(entry.artifact_type)}</span>` : ''}
          </div>
          ${channels.length ? `<div class="artifact-channels">${channels.map((/** @type {string} */ c) => `<span class="channel-tag">${esc(c)}</span>`).join('')}</div>` : ''}
        </div>
      </div>`;
    }
    html += `</details>`;
  }

  el.innerHTML = html;
}

// ---- Conversation View ----

/**
 * @param {RenderCtx} ctx
 * @returns {Promise<void>}
 */
export async function renderConversation({ apiFetch, showToast, cache }) {
  const el = document.getElementById('view-conversation');
  if (!el) return;
  let data;
  try {
    data = await apiFetch('/api/session-chat');
  } catch {
    showToast('Failed to load conversation');
    if (!cache.conversation) el.innerHTML = emptyState('ERROR', 'Could not load conversation');
    return;
  }
  if (!data || !data.turns || !data.turns.length) {
    el.innerHTML = emptyState('NO CONVERSATION', 'Session transcript is empty');
    cache.conversation = null;
    return;
  }
  cache.conversation = data;

  let html = '<div class="chat-container">';
  for (const turn of data.turns) {
    const role = (turn.role || 'unknown').toLowerCase();
    const cls = role === 'user' ? 'user' : 'assistant';
    const content = turn.content || turn.text || '';
    html += `<div class="chat-msg ${cls}">
      <div class="chat-role">${esc(role.toUpperCase())} ${turn.timestamp ? `<span class="chat-ts">${ts(turn.timestamp)}</span>` : ''}</div>
      <div class="chat-content">${esc(content)}</div>
    </div>`;
  }
  html += '</div>';
  el.innerHTML = html;

  const container = /** @type {HTMLElement|null} */ (el.querySelector('.chat-container'));
  if (container) container.scrollTop = container.scrollHeight;
}
