// @ts-check
/**
 * OSPREY Channel Finder — Feedback Management View (lifecycle + list).
 *
 * Browse and manage feedback entries (successful/failed navigation paths)
 * used by the hierarchical pipeline for search hints. Detail-view rendering and
 * per-record mutations live in feedback-detail.js; pure string/parse helpers in
 * feedback-render.js; shared view-state in feedback-state.js.
 */

import { fetchJSON, postJSON, deleteJSON } from './api.js';
import { esc, messageOf } from './utils.js';
import { formModal, confirmModal } from './modal.js';
import { showToast } from './app.js';
import { getContainer, setContainer, getCurrentKey, setCurrentKey, setRerender, setRenderList } from './feedback-state.js';
import { _renderDetail } from './feedback-detail.js';
import {
  _toolLabel,
  _buildCardSummary,
  _parseChannelsFromResponse,
  _renderChannelList,
  _renderSelections,
  _parseSelections,
  _formatTime,
} from './feedback-render.js';

// ---- Public API ----

/**
 * @param {HTMLElement} container
 */
export function mountFeedback(container) {
  setContainer(container);
  setCurrentKey(null);
  _render();
}

export function unmountFeedback() {
  setContainer(null);
  setCurrentKey(null);
}

// ---- Rendering ----

async function _render() {
  const container = getContainer();
  if (!container) return;
  container.innerHTML = '<div class="loading-center"><div class="loading-spinner"></div> Loading feedback...</div>';

  try {
    const status = await fetchJSON('/api/feedback/status');
    if (!status.available) {
      _renderDisabled();
      return;
    }
    if (getCurrentKey()) {
      await _renderDetail();
    } else {
      await _renderList();
    }
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">!</div>Failed to load feedback: ${esc(messageOf(e))}</div>`;
  }
}

function _renderDisabled() {
  const container = getContainer();
  if (!container) return;
  container.innerHTML = `
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
  const container = getContainer();
  if (!container) return;

  const [data, pendingData] = await Promise.all([
    fetchJSON('/api/feedback'),
    fetchJSON('/api/pending-reviews/status')
      .then((/** @type {any} */ s) => s.available ? fetchJSON('/api/pending-reviews') : { items: [] })
      .catch(() => ({ items: [] })),
  ]);
  const entries = data.entries || [];
  const pendingItems = pendingData.items || [];

  const totalSuccesses = entries.reduce((/** @type {number} */ s, /** @type {any} */ e) => s + e.success_count, 0);
  const totalFailures = entries.reduce((/** @type {number} */ s, /** @type {any} */ e) => s + e.failure_count, 0);

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
            ${entries.map((/** @type {any} */ e) => `
              <tr class="fb-table-row" data-key="${esc(e.key)}">
                <td>${esc(e.query)}</td>
                <td><span class="font-mono" style="font-size: var(--cf-text-xs);">${esc(e.facility)}</span></td>
                <td><span class="fb-stat-value fb-stat-success">${e.success_count}</span></td>
                <td><span class="fb-stat-value fb-stat-failure">${e.failure_count}</span></td>
                <td style="color: var(--text-muted); font-size: var(--cf-text-xs);">${_formatTime(e.last_activity)}</td>
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

  container.innerHTML = html;
  _bindListEvents();
  _bindPendingEvents();
}

function _bindListEvents() {
  const container = getContainer();
  if (!container) return;

  // Row click → detail view
  container.querySelectorAll('.fb-table-row').forEach(row => {
    row.addEventListener('click', (e) => {
      if (/** @type {HTMLElement} */ (e.target).closest('.fb-row-delete')) return;
      setCurrentKey(/** @type {HTMLElement} */ (row).dataset.key ?? null);
      _render();
    });
  });

  // Delete entry buttons
  container.querySelectorAll('.fb-row-delete').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const key = /** @type {HTMLElement} */ (btn).dataset.key;
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
          showToast(`Delete failed: ${messageOf(err)}`, 'error');
        }
      }
    });
  });

  // Add entry
  const addBtn = container.querySelector('#fb-add-btn');
  if (addBtn) addBtn.addEventListener('click', _handleAddEntry);

  // Export
  const exportBtn = container.querySelector('#fb-export-btn');
  if (exportBtn) exportBtn.addEventListener('click', _handleExport);

  // Clear all
  const clearBtn = container.querySelector('#fb-clear-btn');
  if (clearBtn) clearBtn.addEventListener('click', _handleClearAll);
}

function _bindPendingEvents() {
  const container = getContainer();
  if (!container) return;

  // Approve buttons — directly promote to feedback store (no modal)
  container.querySelectorAll('.fb-pending-approve').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = /** @type {HTMLElement} */ (btn).dataset.id;
      try {
        await postJSON(`/api/pending-reviews/${id}/approve`);
        showToast('Review approved and recorded', 'success');
        _render();
      } catch (err) {
        showToast(`Approve failed: ${messageOf(err)}`, 'error');
      }
    });
  });

  // Dismiss buttons
  container.querySelectorAll('.fb-pending-dismiss').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = /** @type {HTMLElement} */ (btn).dataset.id;
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
        showToast(`Dismiss failed: ${messageOf(err)}`, 'error');
      }
    });
  });

  // Expand/collapse channel list toggles
  container.querySelectorAll('.fb-pending-channels-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const overflow = /** @type {HTMLElement|null} */ (btn.previousElementSibling);
      if (!overflow) return;
      const isHidden = overflow.style.display === 'none';
      overflow.style.display = isHidden ? 'contents' : 'none';
      const count = /** @type {HTMLElement} */ (btn).dataset.full;
      btn.textContent = isHidden ? 'show less' : `+${count} more`;
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
    showToast(`Add failed: ${messageOf(err)}`, 'error');
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
    showToast(`Export failed: ${messageOf(err)}`, 'error');
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
    showToast(`Clear failed: ${messageOf(err)}`, 'error');
  }
}

// Register the list/detail dispatcher and the direct list renderer so
// feedback-detail.js can re-render without importing this module at eval time
// (avoids an ESM cycle). setRerender = full dispatcher; setRenderList = the
// list view directly (the detail "entry not found" path drops straight to it).
setRerender(_render);
setRenderList(_renderList);
