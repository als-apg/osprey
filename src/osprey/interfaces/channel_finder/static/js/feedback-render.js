// @ts-check
/**
 * OSPREY Channel Finder — Feedback pure render/parse helpers.
 *
 * Stateless, DOM-free string builders and parsers extracted from feedback.js so
 * they can be unit-tested in isolation (see feedback-render.test.mjs). Every
 * server-derived value routed into markup here is escaped through `esc`.
 */

import { esc } from './utils.js';

/**
 * Short badge label for a captured tool name.
 * @param {string} [toolName]
 * @returns {string}
 */
export function _toolLabel(toolName) {
  if (!toolName) return '';
  if (toolName.includes('build_channels')) return 'build';
  if (toolName.includes('get_channels')) return 'get';
  return '';
}

/**
 * Build a compact "system / family / device / ..." path from a selections map.
 * Preserves the canonical field order, merges a trailing `subfield` into the
 * `field` part, truncates long arrays, and appends any non-standard keys.
 * @param {Record<string, unknown>} [selections]
 * @returns {string}
 */
export function _buildSelectionPath(selections) {
  if (!selections || Object.keys(selections).length === 0) return '';
  const order = ['system', 'family', 'device', 'field', 'subfield'];
  /** @type {string[]} */
  const parts = [];
  /** @type {Set<string>} */
  const used = new Set();

  for (const key of order) {
    if (!(key in selections)) continue;
    used.add(key);
    const val = selections[key];
    if (key === 'subfield') {
      // Merge with previous field part if field was present
      if (parts.length > 0 && 'field' in selections) {
        parts[parts.length - 1] += ':' + (Array.isArray(val) ? val.join(', ') : String(val));
        continue;
      }
    }
    if (Array.isArray(val)) {
      if (val.length <= 2) {
        parts.push(val.join(', '));
      } else {
        parts.push(`${val[0]}, ${val[1]} +${val.length - 2}`);
      }
    } else {
      parts.push(String(val));
    }
  }

  // Append non-standard keys
  for (const [k, v] of Object.entries(selections)) {
    if (used.has(k)) continue;
    const vs = Array.isArray(v) ? v.join(', ') : String(v);
    parts.push(`${k}:${vs}`);
  }

  return parts.join(' / ');
}

/**
 * Summarize a pending-review item for its card header.
 * @param {Record<string, any>} item
 * @returns {string}
 */
export function _buildCardSummary(item) {
  if (item.query && item.query.trim()) return item.query.trim();
  const path = _buildSelectionPath(item.selections);
  if (path) return path;
  const label = _toolLabel(item.tool_name);
  if (label) return label;
  return 'Agent-captured search';
}

/**
 * Parse a channel-name list out of a (possibly double-encoded) tool response.
 * @param {unknown} toolResponse - raw string or already-parsed object.
 * @returns {string[]}
 */
export function _parseChannelsFromResponse(toolResponse) {
  if (!toolResponse) return [];
  try {
    /** @type {any} */
    let parsed = typeof toolResponse === 'string' ? JSON.parse(toolResponse) : toolResponse;
    // Double-encoded: {result: "..."} envelope
    if (parsed.result && typeof parsed.result === 'string') {
      try { parsed = JSON.parse(parsed.result); } catch { /* use as-is */ }
    }
    const channels = parsed.channels;
    if (!Array.isArray(channels)) return [];
    return channels.map(ch => {
      if (typeof ch === 'string') return ch;
      if (ch && typeof ch === 'object') return ch.name || ch.pv || JSON.stringify(ch);
      return String(ch);
    });
  } catch {
    return [];
  }
}

/**
 * Render a channel-name list with a collapsible overflow section.
 * @param {string[]} channels
 * @param {number} [visibleCount]
 * @returns {string}
 */
export function _renderChannelList(channels, visibleCount = 3) {
  if (!channels || channels.length === 0) return '';
  const visible = channels.slice(0, visibleCount);
  const overflow = channels.slice(visibleCount);
  let html = '<div class="fb-pending-channels">';
  html += '<div class="fb-pending-channels-label">Channels</div>';
  html += '<div class="fb-pending-channels-list">';
  html += visible.map(ch => `<span class="fb-pending-pv">${esc(ch)}</span>`).join('');
  if (overflow.length > 0) {
    html += '<div class="fb-pending-channels-overflow" style="display:none">';
    html += overflow.map(ch => `<span class="fb-pending-pv">${esc(ch)}</span>`).join('');
    html += '</div>';
    html += `<button type="button" class="fb-pending-channels-toggle" data-full="${overflow.length}">+${overflow.length} more</button>`;
  }
  html += '</div></div>';
  return html;
}

/**
 * Render selection key/value pairs as escaped chips.
 * @param {Record<string, unknown>} [selections]
 * @returns {string}
 */
export function _renderSelections(selections) {
  if (!selections || Object.keys(selections).length === 0) {
    return '<span style="color: var(--text-muted); font-size: var(--cf-text-xs);">(empty)</span>';
  }
  return Object.entries(selections).map(([k, v]) => {
    const val = Array.isArray(v) ? v.join(', ') : String(v);
    return `<span class="fb-sel-pair"><span class="fb-sel-key">${esc(k)}</span><span class="fb-sel-value">${esc(val)}</span></span>`;
  }).join('');
}

/**
 * Parse a "key: value" textarea (one pair per line) into a selections map.
 * @param {string} [text]
 * @returns {Record<string, string>}
 */
export function _parseSelections(text) {
  /** @type {Record<string, string>} */
  const selections = {};
  if (!text) return selections;
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const colonIdx = trimmed.indexOf(':');
    if (colonIdx > 0) {
      const key = trimmed.slice(0, colonIdx).trim();
      const value = trimmed.slice(colonIdx + 1).trim();
      if (key) selections[key] = value;
    }
  }
  return selections;
}

/**
 * Format an ISO timestamp as a compact local "Mon D, HH:MM" string.
 * @param {string} [isoStr]
 * @returns {string}
 */
export function _formatTime(isoStr) {
  if (!isoStr) return '';
  try {
    const d = new Date(isoStr);
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return isoStr;
  }
}

/**
 * Render the "Pending Reviews" section: the searches the agent ran, captured
 * by the PostToolUse hook and waiting for an operator to look at them.
 *
 * Shared by both feedback views, and `canPromote` is the entire difference
 * between them. Approving promotes a captured search into the navigation
 * hints a pipeline reads back on its next search; a deployment with no
 * feedback store has nowhere to put one, and the server refuses the call
 * (`pending_review_api._get_feedback_store`). So the button is not drawn
 * there rather than drawn and answered with an error. Dismiss stays in both:
 * it only deletes from the capture queue and needs no feedback store.
 *
 * @param {any[]} items - pending review items, newest first.
 * @param {{canPromote?: boolean}} [options]
 * @returns {string}
 */
export function _renderPendingSection(items, options = {}) {
  const canPromote = options.canPromote !== false;
  const pendingItems = items || [];

  if (pendingItems.length === 0) {
    return `
      <div class="fb-pending-section">
        <div class="fb-pending-header">
          <span>Pending Reviews</span>
          <span class="fb-pending-subtitle">Agent-captured searches awaiting review</span>
        </div>
        <div class="fb-pending-empty">No pending reviews</div>
      </div>
    `;
  }

  return `
    <div class="fb-pending-section">
      <div class="fb-pending-header">
        <span>Pending Reviews <span class="fb-pending-badge">${pendingItems.length}</span></span>
        <span class="fb-pending-subtitle">Agent-captured searches awaiting review</span>
      </div>
      ${pendingItems.map((/** @type {any} */ item) => {
        const summary = _buildCardSummary(item);
        const label = _toolLabel(item.tool_name);
        const channels = _parseChannelsFromResponse(item.tool_response);
        const selHtml = item.selections && Object.keys(item.selections).length > 0
          ? `<div class="fb-selections">${_renderSelections(item.selections)}</div>` : '';
        const chHtml = _renderChannelList(channels);
        const taskHtml = item.agent_task && item.agent_task !== summary
          ? `<div class="fb-pending-agent-task">${esc(item.agent_task)}</div>` : '';
        const artifactHtml = item.artifact && item.artifact.filename
          ? `<a class="fb-pending-artifact-link" href="/api/artifacts/${esc(item.artifact.filename)}" target="_blank">View Result</a>` : '';
        const approveHtml = canPromote
          ? `<button class="btn btn-sm btn-primary fb-pending-approve" data-id="${esc(item.id)}">Approve &amp; Update Prompt</button>` : '';
        return `
        <div class="fb-pending-card" data-id="${esc(item.id)}">
          ${taskHtml}
          <div class="fb-pending-header">
            <div class="fb-pending-summary">${esc(summary)}</div>
            ${label ? `<span class="fb-pending-tool-badge">${esc(label)}</span>` : ''}
          </div>
          ${selHtml}
          ${chHtml}
          <div class="fb-pending-footer">
            <div class="fb-pending-meta">
              <span>${item.channel_count || 0} channels</span>
              <span>${_formatTime(item.captured_at)}</span>
            </div>
            ${artifactHtml}
            <div class="fb-pending-actions">
              ${approveHtml}
              <button class="btn btn-sm btn-danger fb-pending-dismiss" data-id="${esc(item.id)}">Dismiss</button>
            </div>
          </div>
        </div>`;
      }).join('')}
    </div>
  `;
}
