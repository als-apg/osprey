// @ts-check
/**
 * OSPREY Channel Finder — Feedback Detail View + record actions.
 *
 * Renders a single entry's success/failure records and handles per-record
 * add/edit/delete (including the 409 optimistic-concurrency path). Re-renders
 * through the registered rerender hook from feedback-state.js rather than
 * importing feedback.js, so there is no ESM circular import.
 */

import { fetchJSON, postJSON, putJSON, deleteJSON, ApiError } from './api.js';
import { esc, messageOf } from './utils.js';
import { formModal, confirmModal } from './modal.js';
import { showToast } from './app.js';
import { getContainer, getCurrentKey, setCurrentKey, getRerender, getRenderList } from './feedback-state.js';
import { _renderSelections, _formatTime, _parseSelections } from './feedback-render.js';

/** Re-render the current view via the registered dispatcher. */
function _rerender() {
  const fn = getRerender();
  if (fn) fn();
}

// ---- Detail View ----

export async function _renderDetail() {
  const container = getContainer();
  if (!container) return;

  let entry;
  try {
    entry = await fetchJSON(`/api/feedback/${getCurrentKey()}`);
  } catch {
    showToast('Entry not found', 'error');
    setCurrentKey(null);
    // Drop straight back to the list (as the original did), not through the
    // full dispatcher — avoids a redundant status re-check + loading flash.
    const renderList = getRenderList();
    if (renderList) await renderList();
    return;
  }

  const meta = entry._meta || {};
  const successes = entry.successes || [];
  const failures = entry.failures || [];

  const html = `
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
        ${successes.length === 0 ? '<div class="empty-state" style="padding: var(--cf-space-4);">No successes</div>' : ''}
        ${successes.map((/** @type {any} */ s, /** @type {number} */ i) => _renderSuccessCard(s, i)).join('')}
      </div>

      <div class="fb-detail-section">
        <div class="fb-detail-section-header fb-failure-header">
          <span>Failures (${failures.length})</span>
          <button class="btn btn-sm btn-secondary" id="fb-add-failure">+ Add</button>
        </div>
        ${failures.length === 0 ? '<div class="empty-state" style="padding: var(--cf-space-4);">No failures</div>' : ''}
        ${failures.map((/** @type {any} */ f, /** @type {number} */ i) => _renderFailureCard(f, i)).join('')}
      </div>
    </div>
  `;

  container.innerHTML = html;
  _bindDetailEvents(container, meta);
}

/**
 * @param {any} record
 * @param {number} index
 * @returns {string}
 */
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

/**
 * @param {any} record
 * @param {number} index
 * @returns {string}
 */
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

/**
 * @param {HTMLElement} container
 * @param {any} meta
 */
function _bindDetailEvents(container, meta) {
  // Back link
  container.querySelector('#fb-back-link')?.addEventListener('click', () => {
    setCurrentKey(null);
    _rerender();
  });

  // Add success to existing entry
  container.querySelector('#fb-add-success')?.addEventListener('click', async () => {
    await _handleAddRecordToEntry(meta, 'success');
  });

  // Add failure to existing entry
  container.querySelector('#fb-add-failure')?.addEventListener('click', async () => {
    await _handleAddRecordToEntry(meta, 'failure');
  });

  // Edit record buttons
  container.querySelectorAll('.fb-edit-record').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const card = /** @type {HTMLElement|null} */ (btn.closest('.fb-record-card'));
      if (!card) return;
      const type = card.dataset.type;
      const index = parseInt(card.dataset.index ?? '', 10);
      if (!type) return;
      await _handleEditRecord(type, index);
    });
  });

  // Delete record buttons
  container.querySelectorAll('.fb-delete-record').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const card = /** @type {HTMLElement|null} */ (btn.closest('.fb-record-card'));
      if (!card) return;
      const type = card.dataset.type;
      const index = parseInt(card.dataset.index ?? '', 10);
      if (!type) return;
      await _handleDeleteRecord(type, index);
    });
  });
}

// ---- Actions ----

/**
 * @param {any} meta
 * @param {'success'|'failure'} entryType
 */
async function _handleAddRecordToEntry(meta, entryType) {
  /** @type {import('./modal.js').ModalField[]} */
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
    _rerender();
  } catch (err) {
    showToast(`Add failed: ${messageOf(err)}`, 'error');
  }
}

/**
 * @param {string} recordType - 'successes' | 'failures'
 * @param {number} index
 */
async function _handleEditRecord(recordType, index) {
  // Fetch fresh entry data
  let entry;
  try {
    entry = await fetchJSON(`/api/feedback/${getCurrentKey()}`);
  } catch {
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

  /** @type {import('./modal.js').ModalField[]} */
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
  /** @type {Record<string, any>} */
  const body = { expected_timestamp: record.timestamp };
  if (isSuccess) {
    body.selections = selections;
    body.channel_count = parseInt(result.channel_count || '0', 10);
  } else {
    body.partial_selections = selections;
    body.reason = result.reason || '';
  }

  try {
    await putJSON(`/api/feedback/${getCurrentKey()}/${recordType}/${index}`, body);
    showToast('Record updated', 'success');
    _rerender();
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      showToast('Record changed by another user, refreshing...', 'info');
      _rerender();
    } else {
      showToast(`Edit failed: ${messageOf(err)}`, 'error');
    }
  }
}

/**
 * @param {string} recordType - 'successes' | 'failures'
 * @param {number} index
 */
async function _handleDeleteRecord(recordType, index) {
  // Fetch fresh timestamp
  let entry;
  try {
    entry = await fetchJSON(`/api/feedback/${getCurrentKey()}`);
  } catch {
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
    await deleteJSON(`/api/feedback/${getCurrentKey()}/${recordType}/${index}`, {
      expected_timestamp: record.timestamp,
    });
    showToast('Record deleted', 'success');
    _rerender();
  } catch (err) {
    if (err instanceof ApiError && err.status === 409) {
      showToast('Record changed, refreshing...', 'info');
      _rerender();
    } else {
      showToast(`Delete failed: ${messageOf(err)}`, 'error');
    }
  }
}
