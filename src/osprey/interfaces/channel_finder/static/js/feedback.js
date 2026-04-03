/**
 * OSPREY Channel Finder — Feedback Management View
 *
 * Browse and manage feedback entries (successful/failed navigation paths)
 * used by the hierarchical pipeline for search hints.
 */

import { fetchJSON, postJSON, putJSON, deleteJSON } from './api.js';
import { esc } from './utils.js';
import { formModal, confirmModal } from './modal.js';
import { showToast } from './app.js';

let _container = null;
let _currentKey = null; // When set, we're in detail view

// ---- Public API ----

export function mountFeedback(container) {
  _container = container;
  _currentKey = null;
  _render();
}

export function unmountFeedback() {
  _container = null;
  _currentKey = null;
}

// ---- Rendering ----

async function _render() {
  if (!_container) return;
  _container.innerHTML = '<div class="loading-center"><div class="loading-spinner"></div> Loading feedback...</div>';

  try {
    const status = await fetchJSON('/api/feedback/status');
    if (!status.available) {
      _renderDisabled();
      return;
    }
    if (_currentKey) {
      await _renderDetail();
    } else {
      await _renderList();
    }
  } catch (e) {
    _container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">!</div>Failed to load feedback: ${esc(e.message)}</div>`;
  }
}

function _renderDisabled() {
  _container.innerHTML = `
    <div class="fb-disabled">
      <div class="fb-disabled-icon">&#128274;</div>
      <div class="fb-disabled-title">Feedback Store Not Available</div>
      <div class="fb-disabled-hint">
        Enable feedback in your config to start recording navigation hints:<br>
        <code>channel_finder.pipelines.hierarchical.feedback.enabled: true</code>
      </div>
    </div>
  `;
}

// ---- List View ----

async function _renderList() {
  const [data, pendingData] = await Promise.all([
    fetchJSON('/api/feedback'),
    fetchJSON('/api/pending-reviews/status')
      .then(s => s.available ? fetchJSON('/api/pending-reviews') : { items: [] })
      .catch(() => ({ items: [] })),
  ]);
  const entries = data.entries || [];
  const pendingItems = pendingData.items || [];

  const totalSuccesses = entries.reduce((s, e) => s + e.success_count, 0);
  const totalFailures = entries.reduce((s, e) => s + e.failure_count, 0);

  let html = `
    <div class="fb-toolbar">
      <div class="fb-toolbar-left">
        <div class="fb-stat">
          Entries: <span class="fb-stat-value">${entries.length}</span>
        </div>
        <div class="fb-stat">
          Successes: <span class="fb-stat-value fb-stat-success">${totalSuccesses}</span>
        </div>
        <div class="fb-stat">
          Failures: <span class="fb-stat-value fb-stat-failure">${totalFailures}</span>
        </div>
      </div>
      <div class="fb-toolbar-right">
        <button class="btn btn-secondary btn-sm" id="fb-add-btn">+ Add Entry</button>
        <button class="btn btn-secondary btn-sm" id="fb-export-btn">Export</button>
        ${entries.length > 0 ? '<button class="btn btn-danger btn-sm" id="fb-clear-btn">Clear All</button>' : ''}
      </div>
    </div>
  `;

  if (entries.length === 0) {
    html += '<div class="empty-state"><div class="empty-state-icon">&#128203;</div>No feedback entries yet.<br>Use the Search tab to record feedback, or review agent-captured searches below.</div>';
  } else {
    html += `
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>Query</th>
              <th>Facility</th>
              <th>Successes</th>
              <th>Failures</th>
              <th>Last Activity</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            ${entries.map(e => `
              <tr class="fb-table-row" data-key="${esc(e.key)}">
                <td>${esc(e.query)}</td>
                <td><span class="font-mono" style="font-size: var(--text-xs);">${esc(e.facility)}</span></td>
                <td><span class="fb-stat-value fb-stat-success">${e.success_count}</span></td>
                <td><span class="fb-stat-value fb-stat-failure">${e.failure_count}</span></td>
                <td style="color: var(--text-muted); font-size: var(--text-xs);">${_formatTime(e.last_activity)}</td>
                <td>
                  <button class="item-action-btn action-delete fb-row-delete" data-key="${esc(e.key)}" title="Delete entry">&times;</button>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  // ---- Pending Reviews Section ----
  if (pendingItems.length > 0) {
    html += `
      <div class="fb-pending-section">
        <div class="fb-pending-header">
          <span>Pending Reviews <span class="fb-pending-badge">${pendingItems.length}</span></span>
          <span class="fb-pending-subtitle">Agent-captured searches awaiting review</span>
        </div>
        ${pendingItems.map(item => {
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
                <button class="btn btn-sm btn-primary fb-pending-approve" data-id="${esc(item.id)}">Approve &amp; Update Prompt</button>
                <button class="btn btn-sm btn-danger fb-pending-dismiss" data-id="${esc(item.id)}">Dismiss</button>
              </div>
            </div>
          </div>`;
        }).join('')}
      </div>
    `;
  } else {
    html += `
      <div class="fb-pending-section">
        <div class="fb-pending-header">
          <span>Pending Reviews</span>
          <span class="fb-pending-subtitle">Agent-captured searches awaiting review</span>
        </div>
        <div class="fb-pending-empty">No pending reviews</div>
      </div>
    `;
  }

  _container.innerHTML = html;
  _bindListEvents();
  _bindPendingEvents();
}

function _bindListEvents() {
  // Row click → detail view
  _container.querySelectorAll('.fb-table-row').forEach(row => {
    row.addEventListener('click', (e) => {
      if (e.target.closest('.fb-row-delete')) return;
      _currentKey = row.dataset.key;
      _render();
    });
  });

  // Delete entry buttons
  _container.querySelectorAll('.fb-row-delete').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const key = btn.dataset.key;
      const ok = await confirmModal({
        title: 'Delete Feedback Entry',
        message: 'This will remove the entire feedback entry and all its records.',
        confirmLabel: 'Delete',
        danger: true,
      });
      if (ok) {
        try {
          await deleteJSON(`/api/feedback/${key}`);
          showToast('Entry deleted', 'success');
          _render();
        } catch (err) {
          showToast(`Delete failed: ${err.message}`, 'error');
        }
      }
    });
  });

  // Add entry
  const addBtn = _container.querySelector('#fb-add-btn');
  if (addBtn) addBtn.addEventListener('click', _handleAddEntry);

  // Export
  const exportBtn = _container.querySelector('#fb-export-btn');
  if (exportBtn) exportBtn.addEventListener('click', _handleExport);

  // Clear all
  const clearBtn = _container.querySelector('#fb-clear-btn');
  if (clearBtn) clearBtn.addEventListener('click', _handleClearAll);
}

function _bindPendingEvents() {
  // Approve buttons — directly promote to feedback store (no modal)
  _container.querySelectorAll('.fb-pending-approve').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      try {
        await postJSON(`/api/pending-reviews/${id}/approve`);
        showToast('Review approved and recorded', 'success');
        _render();
      } catch (err) {
        showToast(`Approve failed: ${err.message}`, 'error');
      }
    });
  });

  // Dismiss buttons
  _container.querySelectorAll('.fb-pending-dismiss').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.id;
      const ok = await confirmModal({
        title: 'Dismiss Pending Review',
        message: 'This will discard the captured search result without recording feedback.',
        confirmLabel: 'Dismiss',
        danger: true,
      });
      if (!ok) return;

      try {
        await deleteJSON(`/api/pending-reviews/${id}`);
        showToast('Review dismissed', 'success');
        _render();
      } catch (err) {
        showToast(`Dismiss failed: ${err.message}`, 'error');
      }
    });
  });

  // Expand/collapse channel list toggles
  _container.querySelectorAll('.fb-pending-channels-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const overflow = btn.previousElementSibling;
      if (!overflow) return;
      const isHidden = overflow.style.display === 'none';
      overflow.style.display = isHidden ? 'contents' : 'none';
      const count = btn.dataset.full;
      btn.textContent = isHidden ? 'show less' : `+${count} more`;
    });
  });
}

// ---- Detail View ----

async function _renderDetail() {
  let entry;
  try {
    entry = await fetchJSON(`/api/feedback/${_currentKey}`);
  } catch (e) {
    showToast('Entry not found', 'error');
    _currentKey = null;
    await _renderList();
    return;
  }

  const meta = entry._meta || {};
  const successes = entry.successes || [];
  const failures = entry.failures || [];

  let html = `
    <div class="fb-breadcrumb">
      <a id="fb-back-link">Feedback</a>
      <span class="fb-breadcrumb-sep">/</span>
      <span class="fb-breadcrumb-current">"${esc(meta.query || '(unknown)')}" (${esc(meta.facility || '?')})</span>
    </div>

    <div class="fb-detail-grid">
      <div class="fb-detail-section">
        <div class="fb-detail-section-header fb-success-header">
          <span>Successes (${successes.length})</span>
          <button class="btn btn-sm btn-secondary" id="fb-add-success">+ Add</button>
        </div>
        ${successes.length === 0 ? '<div class="empty-state" style="padding: var(--space-4);">No successes</div>' : ''}
        ${successes.map((s, i) => _renderSuccessCard(s, i)).join('')}
      </div>

      <div class="fb-detail-section">
        <div class="fb-detail-section-header fb-failure-header">
          <span>Failures (${failures.length})</span>
          <button class="btn btn-sm btn-secondary" id="fb-add-failure">+ Add</button>
        </div>
        ${failures.length === 0 ? '<div class="empty-state" style="padding: var(--space-4);">No failures</div>' : ''}
        ${failures.map((f, i) => _renderFailureCard(f, i)).join('')}
      </div>
    </div>
  `;

  _container.innerHTML = html;
  _bindDetailEvents(meta);
}

function _renderSuccessCard(record, index) {
  return `
    <div class="fb-record-card" data-type="successes" data-index="${index}">
      <div class="fb-selections">${_renderSelections(record.selections)}</div>
      <div class="fb-record-meta">
        <span class="fb-channel-count">${record.channel_count} channels</span>
        <span class="fb-timestamp">${_formatTime(record.timestamp)}</span>
        <span class="fb-record-actions">
          <button class="item-action-btn action-edit fb-edit-record" title="Edit">&#9998;</button>
          <button class="item-action-btn action-delete fb-delete-record" title="Delete">&times;</button>
        </span>
      </div>
    </div>
  `;
}

function _renderFailureCard(record, index) {
  return `
    <div class="fb-record-card" data-type="failures" data-index="${index}">
      <div class="fb-selections">${_renderSelections(record.partial_selections)}</div>
      ${record.reason ? `<div class="fb-record-reason">${esc(record.reason)}</div>` : ''}
      <div class="fb-record-meta">
        <span class="fb-timestamp">${_formatTime(record.timestamp)}</span>
        <span class="fb-record-actions">
          <button class="item-action-btn action-edit fb-edit-record" title="Edit">&#9998;</button>
          <button class="item-action-btn action-delete fb-delete-record" title="Delete">&times;</button>
        </span>
      </div>
    </div>
  `;
}

function _toolLabel(toolName) {
  if (!toolName) return '';
  if (toolName.includes('build_channels')) return 'build';
  if (toolName.includes('get_channels')) return 'get';
  return '';
}

function _buildSelectionPath(selections) {
  if (!selections || Object.keys(selections).length === 0) return '';
  const order = ['system', 'family', 'device', 'field', 'subfield'];
  const parts = [];
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

function _buildCardSummary(item) {
  if (item.query && item.query.trim()) return item.query.trim();
  const path = _buildSelectionPath(item.selections);
  if (path) return path;
  const label = _toolLabel(item.tool_name);
  if (label) return label;
  return 'Agent-captured search';
}

function _parseChannelsFromResponse(toolResponse) {
  if (!toolResponse) return [];
  try {
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

function _renderChannelList(channels, visibleCount = 3) {
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

function _renderSelections(selections) {
  if (!selections || Object.keys(selections).length === 0) {
    return '<span style="color: var(--text-muted); font-size: var(--text-xs);">(empty)</span>';
  }
  return Object.entries(selections).map(([k, v]) => {
    const val = Array.isArray(v) ? v.join(', ') : String(v);
    return `<span class="fb-sel-pair"><span class="fb-sel-key">${esc(k)}</span><span class="fb-sel-value">${esc(val)}</span></span>`;
  }).join('');
}

function _bindDetailEvents(meta) {
  // Back link
  _container.querySelector('#fb-back-link')?.addEventListener('click', () => {
    _currentKey = null;
    _render();
  });

  // Add success to existing entry
  _container.querySelector('#fb-add-success')?.addEventListener('click', async () => {
    await _handleAddRecordToEntry(meta, 'success');
  });

  // Add failure to existing entry
  _container.querySelector('#fb-add-failure')?.addEventListener('click', async () => {
    await _handleAddRecordToEntry(meta, 'failure');
  });

  // Edit record buttons
  _container.querySelectorAll('.fb-edit-record').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const card = btn.closest('.fb-record-card');
      const type = card.dataset.type;
      const index = parseInt(card.dataset.index, 10);
      await _handleEditRecord(type, index);
    });
  });

  // Delete record buttons
  _container.querySelectorAll('.fb-delete-record').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const card = btn.closest('.fb-record-card');
      const type = card.dataset.type;
      const index = parseInt(card.dataset.index, 10);
      await _handleDeleteRecord(type, index);
    });
  });
}

// ---- Actions ----

async function _handleAddEntry() {
  const result = await formModal({
    title: 'Add Feedback Entry',
    fields: [
      { name: 'query', label: 'Query', type: 'text', required: true, placeholder: 'e.g. show me magnets' },
      { name: 'facility', label: 'Facility', type: 'text', required: true, placeholder: 'e.g. ALS' },
      { name: 'entry_type', label: 'Type', type: 'select', options: [
        { value: 'success', label: 'Success' },
        { value: 'failure', label: 'Failure' },
      ], value: 'success' },
      { name: 'selections', label: 'Selections (key: value, one per line)', type: 'textarea', placeholder: 'system: MAG\ndevice: QF1' },
    ],
    submitLabel: 'Add Entry',
  });

  if (!result) return;

  const selections = _parseSelections(result.selections);
  try {
    await postJSON('/api/feedback', {
      query: result.query,
      facility: result.facility,
      entry_type: result.entry_type,
      selections,
      channel_count: 0,
      reason: '',
    });
    showToast('Entry added', 'success');
    _render();
  } catch (err) {
    showToast(`Add failed: ${err.message}`, 'error');
  }
}

async function _handleAddRecordToEntry(meta, entryType) {
  const fields = [
    { name: 'selections', label: 'Selections (key: value, one per line)', type: 'textarea', placeholder: 'system: MAG\ndevice: QF1' },
  ];
  if (entryType === 'success') {
    fields.push({ name: 'channel_count', label: 'Channel Count', type: 'number', value: '0' });
  } else {
    fields.push({ name: 'reason', label: 'Failure Reason', type: 'text', placeholder: 'e.g. no options at family level' });
  }

  const result = await formModal({
    title: `Add ${entryType === 'success' ? 'Success' : 'Failure'} Record`,
    fields,
    submitLabel: 'Add',
  });

  if (!result) return;

  const selections = _parseSelections(result.selections);
  try {
    await postJSON('/api/feedback', {
      query: meta.query || '',
      facility: meta.facility || '',
      entry_type: entryType,
      selections,
      channel_count: parseInt(result.channel_count || '0', 10),
      reason: result.reason || '',
    });
    showToast('Record added', 'success');
    _render();
  } catch (err) {
    showToast(`Add failed: ${err.message}`, 'error');
  }
}

async function _handleEditRecord(recordType, index) {
  // Fetch fresh entry data
  let entry;
  try {
    entry = await fetchJSON(`/api/feedback/${_currentKey}`);
  } catch (e) {
    showToast('Entry not found', 'error');
    return;
  }

  const records = entry[recordType] || [];
  const record = records[index];
  if (!record) {
    showToast('Record not found', 'error');
    return;
  }

  const isSuccess = recordType === 'successes';
  const currentSelections = isSuccess ? record.selections : record.partial_selections;
  const selStr = Object.entries(currentSelections || {}).map(([k, v]) => `${k}: ${v}`).join('\n');

  const fields = [
    { name: 'selections', label: 'Selections (key: value, one per line)', type: 'textarea', value: selStr },
  ];
  if (isSuccess) {
    fields.push({ name: 'channel_count', label: 'Channel Count', type: 'number', value: String(record.channel_count || 0) });
  } else {
    fields.push({ name: 'reason', label: 'Failure Reason', type: 'text', value: record.reason || '' });
  }

  const result = await formModal({
    title: `Edit ${isSuccess ? 'Success' : 'Failure'} Record`,
    fields,
    submitLabel: 'Save',
  });

  if (!result) return;

  const selections = _parseSelections(result.selections);
  const body = { expected_timestamp: record.timestamp };
  if (isSuccess) {
    body.selections = selections;
    body.channel_count = parseInt(result.channel_count || '0', 10);
  } else {
    body.partial_selections = selections;
    body.reason = result.reason || '';
  }

  try {
    await putJSON(`/api/feedback/${_currentKey}/${recordType}/${index}`, body);
    showToast('Record updated', 'success');
    _render();
  } catch (err) {
    if (err.status === 409) {
      showToast('Record changed by another user, refreshing...', 'info');
      _render();
    } else {
      showToast(`Edit failed: ${err.message}`, 'error');
    }
  }
}

async function _handleDeleteRecord(recordType, index) {
  // Fetch fresh timestamp
  let entry;
  try {
    entry = await fetchJSON(`/api/feedback/${_currentKey}`);
  } catch (e) {
    showToast('Entry not found', 'error');
    return;
  }

  const records = entry[recordType] || [];
  const record = records[index];
  if (!record) {
    showToast('Record not found', 'error');
    return;
  }

  const ok = await confirmModal({
    title: 'Delete Record',
    message: `Delete this ${recordType === 'successes' ? 'success' : 'failure'} record?`,
    confirmLabel: 'Delete',
    danger: true,
  });
  if (!ok) return;

  try {
    await deleteJSON(`/api/feedback/${_currentKey}/${recordType}/${index}`, {
      expected_timestamp: record.timestamp,
    });
    showToast('Record deleted', 'success');
    _render();
  } catch (err) {
    if (err.status === 409) {
      showToast('Record changed, refreshing...', 'info');
      _render();
    } else {
      showToast(`Delete failed: ${err.message}`, 'error');
    }
  }
}

async function _handleExport() {
  try {
    const resp = await fetch('/api/feedback/export');
    if (!resp.ok) throw new Error('Export failed');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'feedback_export.json';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Export downloaded', 'success');
  } catch (err) {
    showToast(`Export failed: ${err.message}`, 'error');
  }
}

async function _handleClearAll() {
  const ok = await confirmModal({
    title: 'Clear All Feedback',
    message: 'This will permanently delete all feedback entries.',
    impact: 'This action cannot be undone.',
    confirmLabel: 'Clear All',
    danger: true,
  });
  if (!ok) return;

  try {
    await deleteJSON('/api/feedback?confirm=true');
    showToast('All feedback cleared', 'success');
    _render();
  } catch (err) {
    showToast(`Clear failed: ${err.message}`, 'error');
  }
}

// ---- Helpers ----

function _parseSelections(text) {
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

function _formatTime(isoStr) {
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
