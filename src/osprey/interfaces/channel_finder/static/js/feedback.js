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
import { facilityZoneLabel, viewerSharesFacilityClock } from '/design-system/js/facility-time.js';
import { formModal, confirmModal } from './modal.js';
import { icon } from '/design-system/js/icons.js';
import { renderState } from '/design-system/js/panel-state.js';
import { showToast } from '/design-system/js/toast.js';
import { setResultCount } from './panel-toolbar.js';
import {
  getContainer,
  setContainer,
  getCurrentKey,
  setCurrentKey,
  setRerender,
  setRenderList,
} from './feedback-state.js';
import { _renderDetail } from './feedback-detail.js';
import { _parseSelections, _formatTime, _renderPendingSection } from './feedback-render.js';

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
  setResultCount('');
}

// ---- Rendering ----

async function _render() {
  const container = getContainer();
  if (!container) return;
  setResultCount('');
  renderState(container, { icon: 'refresh', message: 'Loading feedback' });

  try {
    const status = await fetchJSON('/api/feedback/status');
    if (getContainer() !== container) return;
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
    if (getContainer() !== container) return;
    setResultCount('');
    renderState(container, {
      icon: 'triangle-alert',
      message: 'Feedback could not be read',
      detail: messageOf(e),
      action: { label: 'Retry', onClick: () => void _render() },
    });
  }
}

/**
 * Feedback is switched off for a paradigm that supports it. Deliberately no
 * dotted config key: the state names the setting the way the Explore view's
 * panes do, because the key is hierarchical-only and this view is reached
 * from several paradigms.
 */
function _renderDisabled() {
  const container = getContainer();
  if (!container) return;
  renderState(container, {
    icon: 'settings',
    message: 'Feedback is switched off for this deployment',
    detail:
      'Check the channel finder feedback setting in the configuration, then reload this panel.',
  });
}

// ---- List View ----

async function _renderList() {
  const container = getContainer();
  if (!container) return;

  const [data, pendingData] = await Promise.all([
    fetchJSON('/api/feedback'),
    fetchJSON('/api/pending-reviews/status')
      .then((/** @type {any} */ s) =>
        s.available ? fetchJSON('/api/pending-reviews') : { items: [] }
      )
      .catch(() => ({ items: [] })),
  ]);
  if (getContainer() !== container) return;
  const entries = data.entries || [];
  const pendingItems = pendingData.items || [];

  const totalSuccesses = entries.reduce(
    (/** @type {number} */ s, /** @type {any} */ e) => s + e.success_count,
    0
  );
  const totalFailures = entries.reduce(
    (/** @type {number} */ s, /** @type {any} */ e) => s + e.failure_count,
    0
  );
  setResultCount(
    [
      counted(entries.length, 'entry', 'entries'),
      counted(totalSuccesses, 'success', 'successes'),
      counted(totalFailures, 'failure', 'failures'),
    ].join(' · ')
  );

  const activityHeader = viewerSharesFacilityClock()
    ? 'Last Activity'
    : `Last Activity (${facilityZoneLabel()})`;

  // The section's actions keep their slots: Clear all is disabled, never
  // removed, while there is nothing to clear (DESIGN.md §12.1).
  const clearTitle = entries.length > 0 ? 'Clear all' : 'No entries to clear';
  let html = `
    <section class="fb-entries">
      <div class="fb-section-head">
        <span class="osprey-section-label">Entries</span>
        <button type="button" class="osprey-icon-btn" data-action="new"
                aria-label="New entry" title="New entry"></button>
        <button type="button" class="osprey-icon-btn" data-action="export"
                aria-label="Export" title="Export"></button>
        <button type="button" class="osprey-icon-btn" data-action="clear"
                aria-label="Clear all" title="${clearTitle}"${entries.length > 0 ? '' : ' disabled'}></button>
      </div>
  `;

  if (entries.length === 0) {
    html += '<div class="fb-entries-empty"></div>';
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
              <th>${esc(activityHeader)}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            ${entries
              .map(
                (/** @type {any} */ e) => `
              <tr class="fb-table-row" data-key="${esc(e.key)}">
                <td>${esc(e.query)}</td>
                <td><span class="fb-facility">${esc(e.facility)}</span></td>
                <td><span class="fb-stat-value fb-stat-success">${e.success_count}</span></td>
                <td><span class="fb-stat-value fb-stat-failure">${e.failure_count}</span></td>
                <td class="fb-activity">${_formatTime(e.last_activity)}</td>
                <td>
                  <button type="button" class="osprey-mini fb-row-delete" data-key="${esc(e.key)}"
                          aria-label="Delete entry" title="Delete entry"></button>
                </td>
              </tr>
            `
              )
              .join('')}
          </tbody>
        </table>
      </div>
    `;
  }
  html += '</section>';

  // ---- Pending Reviews Section ----
  html += _renderPendingSection(pendingItems);

  container.innerHTML = html;
  container.querySelector('[data-action="new"]')?.append(icon('plus'));
  container.querySelector('[data-action="export"]')?.append(icon('download'));
  container.querySelector('[data-action="clear"]')?.append(icon('trash'));
  for (const remove of container.querySelectorAll('.fb-row-delete')) remove.append(icon('x'));
  const empty = /** @type {HTMLElement|null} */ (container.querySelector('.fb-entries-empty'));
  if (empty) {
    renderState(empty, {
      icon: 'file-text',
      message: 'No feedback entries yet',
      detail: 'Add one with New entry, or approve a captured search below.',
    });
  }
  _bindListEvents();
  _bindPendingEvents();
}

function _bindListEvents() {
  const container = getContainer();
  if (!container) return;

  // Row click → detail view
  container.querySelectorAll('.fb-table-row').forEach((row) => {
    row.addEventListener('click', (e) => {
      if (/** @type {HTMLElement} */ (e.target).closest('.fb-row-delete')) return;
      setCurrentKey(/** @type {HTMLElement} */ (row).dataset.key ?? null);
      _render();
    });
  });

  // Delete entry buttons
  container.querySelectorAll('.fb-row-delete').forEach((btn) => {
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
          showToast('Entry deleted', { kind: 'success' });
          _render();
        } catch (err) {
          showToast(`Delete failed: ${messageOf(err)}`, { kind: 'error' });
        }
      }
    });
  });

  container.querySelector('[data-action="new"]')?.addEventListener('click', _handleAddEntry);
  container.querySelector('[data-action="export"]')?.addEventListener('click', _handleExport);
  container.querySelector('[data-action="clear"]')?.addEventListener('click', _handleClearAll);
}

function _bindPendingEvents() {
  const container = getContainer();
  if (!container) return;

  // Approve buttons — directly promote to feedback store (no modal)
  container.querySelectorAll('.fb-pending-approve').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const id = /** @type {HTMLElement} */ (btn).dataset.id;
      try {
        await postJSON(`/api/pending-reviews/${id}/approve`);
        showToast('Review approved and recorded', { kind: 'success' });
        _render();
      } catch (err) {
        showToast(`Approve failed: ${messageOf(err)}`, { kind: 'error' });
      }
    });
  });

  // Dismiss buttons
  container.querySelectorAll('.fb-pending-dismiss').forEach((btn) => {
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
        showToast('Review dismissed', { kind: 'success' });
        _render();
      } catch (err) {
        showToast(`Dismiss failed: ${messageOf(err)}`, { kind: 'error' });
      }
    });
  });

  // Expand/collapse channel list toggles
  container.querySelectorAll('.fb-pending-channels-toggle').forEach((btn) => {
    btn.addEventListener('click', () => {
      const overflow = /** @type {HTMLElement|null} */ (btn.previousElementSibling);
      if (!overflow) return;
      const isHidden = overflow.hidden;
      overflow.hidden = !isHidden;
      const count = /** @type {HTMLElement} */ (btn).dataset.full;
      btn.textContent = isHidden ? 'show less' : `+${count} more`;
    });
  });
}

/**
 * @param {number} n
 * @param {string} one
 * @param {string} many
 * @returns {string}
 */
function counted(n, one, many) {
  return `${n} ${n === 1 ? one : many}`;
}

// ---- Actions ----

async function _handleAddEntry() {
  const result = await formModal({
    title: 'Add Feedback Entry',
    fields: [
      {
        name: 'query',
        label: 'Query',
        type: 'text',
        required: true,
        placeholder: 'e.g. show me magnets',
      },
      {
        name: 'entry_type',
        label: 'Type',
        type: 'select',
        options: [
          { value: 'success', label: 'Success' },
          { value: 'failure', label: 'Failure' },
        ],
        value: 'success',
      },
      {
        name: 'selections',
        label: 'Selections (key: value, one per line)',
        type: 'textarea',
        placeholder: 'system: MAG\ndevice: QF1',
      },
    ],
    submitLabel: 'Add Entry',
  });

  if (!result) return;

  const selections = _parseSelections(result.selections);
  try {
    // No facility field: the server fills it from the name this deployment
    // resolved, so an entry cannot be filed against a second spelling of the
    // same machine.
    await postJSON('/api/feedback', {
      query: result.query,
      entry_type: result.entry_type,
      selections,
      channel_count: 0,
      reason: '',
    });
    showToast('Entry added', { kind: 'success' });
    _render();
  } catch (err) {
    showToast(`Add failed: ${messageOf(err)}`, { kind: 'error' });
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
    showToast('Export downloaded', { kind: 'success' });
  } catch (err) {
    showToast(`Export failed: ${messageOf(err)}`, { kind: 'error' });
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
    showToast('All feedback cleared', { kind: 'success' });
    _render();
  } catch (err) {
    showToast(`Clear failed: ${messageOf(err)}`, { kind: 'error' });
  }
}

// Register the list/detail dispatcher and the direct list renderer so
// feedback-detail.js can re-render without importing this module at eval time
// (avoids an ESM cycle). setRerender = full dispatcher; setRenderList = the
// list view directly (the detail "entry not found" path drops straight to it).
setRerender(_render);
setRenderList(_renderList);
