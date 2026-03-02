/* OSPREY Web Terminal — Hook Debug Toggle & Log Viewer */

import { fetchJSON } from './api.js';

/**
 * Initialize the hook debug toggle bar and collapsible log viewer
 * in the Safety tab of the Settings drawer.
 */
export function initHookDebug() {
  const bar = document.getElementById('hook-debug-bar');
  const logSection = document.getElementById('hook-debug-log-section');
  const safetyPanel = document.getElementById('tab-safety');
  if (!bar || !logSection || !safetyPanel) return;

  // ---- Toggle Bar ----
  bar.className = 'hook-debug-bar';
  _buildToggleBar(bar);

  const toggle = document.getElementById('hook-debug-toggle');

  toggle.addEventListener('change', async () => {
    const enabled = toggle.checked;
    try {
      const res = await fetch('/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ updates: { 'hooks.debug': enabled } }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      _showToast(bar, enabled ? 'Debug logging enabled' : 'Debug logging disabled');
    } catch (err) {
      console.error('Failed to toggle hook debug:', err);
      // Revert toggle on failure
      toggle.checked = !enabled;
      _showToast(bar, 'Failed to update setting', true);
    }
  });

  // ---- Log Viewer (Collapsible) ----
  logSection.className = 'hook-debug-log';
  _buildLogViewer(logSection);

  const logToggleHeader = document.getElementById('hook-debug-log-toggle');
  const logBody = document.getElementById('hook-debug-log-body');
  const refreshBtn = document.getElementById('hook-debug-refresh');
  let logExpanded = false;

  logToggleHeader.addEventListener('click', (e) => {
    // Don't toggle when clicking the refresh button
    if (e.target === refreshBtn || refreshBtn.contains(e.target)) return;
    logExpanded = !logExpanded;
    logSection.classList.toggle('expanded', logExpanded);
    if (logExpanded) _loadLogEntries(logBody);
  });

  refreshBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (logExpanded) _loadLogEntries(logBody);
  });

  // ---- Tab Activation ----
  safetyPanel.addEventListener('drawer:tab-activate', async () => {
    try {
      const data = await fetchJSON('/api/hooks/debug-status');
      toggle.checked = data.enabled;
    } catch {
      toggle.checked = false;
    }
    // Refresh log if already expanded
    if (logExpanded) _loadLogEntries(logBody);
  });
}

// ---- DOM Builders (safe — no raw HTML injection) ---- //

function _buildToggleBar(container) {
  const label = document.createElement('span');
  label.className = 'hook-debug-label';
  label.textContent = 'Hook Debug Logging';

  const toggleLabel = document.createElement('label');
  toggleLabel.className = 'toggle-switch';
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.id = 'hook-debug-toggle';
  const slider = document.createElement('span');
  slider.className = 'toggle-slider';
  toggleLabel.appendChild(input);
  toggleLabel.appendChild(slider);

  container.appendChild(label);
  container.appendChild(toggleLabel);
}

function _buildLogViewer(container) {
  // Header
  const header = document.createElement('div');
  header.className = 'hook-debug-log-header';
  header.id = 'hook-debug-log-toggle';

  const chevron = document.createElement('span');
  chevron.className = 'hook-debug-log-chevron';
  chevron.textContent = '\u25B6';
  header.appendChild(chevron);

  const title = document.createElement('span');
  title.className = 'hook-debug-log-title';
  title.textContent = 'Hook Activity Log';
  header.appendChild(title);

  const refresh = document.createElement('button');
  refresh.className = 'hook-debug-refresh-btn';
  refresh.id = 'hook-debug-refresh';
  refresh.title = 'Refresh log';
  refresh.textContent = '\u21BB';
  header.appendChild(refresh);

  // Body
  const body = document.createElement('div');
  body.className = 'hook-debug-log-body';
  body.id = 'hook-debug-log-body';

  const empty = document.createElement('div');
  empty.className = 'hook-debug-log-empty';
  empty.textContent = 'No entries';
  body.appendChild(empty);

  container.appendChild(header);
  container.appendChild(body);
}

async function _loadLogEntries(logBody) {
  // Clear existing content safely
  while (logBody.firstChild) logBody.removeChild(logBody.firstChild);

  try {
    const data = await fetchJSON('/api/hooks/debug-log?limit=50');
    if (!data.entries || data.entries.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'hook-debug-log-empty';
      empty.textContent = 'No entries';
      logBody.appendChild(empty);
      return;
    }

    const table = document.createElement('table');
    table.className = 'hook-debug-log-table';

    // Thead
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    for (const col of ['Time', 'Hook', 'Tool', 'Status', 'Detail']) {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Tbody
    const tbody = document.createElement('tbody');
    for (const entry of data.entries) {
      const tr = document.createElement('tr');

      const tdTs = document.createElement('td');
      tdTs.className = 'log-ts';
      tdTs.textContent = _formatTimestamp(entry.ts || entry.timestamp || '');
      tr.appendChild(tdTs);

      const tdHook = document.createElement('td');
      tdHook.textContent = entry.hook || entry.hook_event || '-';
      tr.appendChild(tdHook);

      const tdTool = document.createElement('td');
      tdTool.textContent = entry.tool || '-';
      tr.appendChild(tdTool);

      const tdStatus = document.createElement('td');
      const status = entry.status || '-';
      tdStatus.textContent = status;
      if (status === 'allowed') tdStatus.className = 'status-ok';
      else if (status === 'blocked') tdStatus.className = 'status-blocked';
      tr.appendChild(tdStatus);

      const tdDetail = document.createElement('td');
      tdDetail.className = 'log-detail';
      tdDetail.textContent = entry.detail || entry.message || '';
      tr.appendChild(tdDetail);

      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    logBody.appendChild(table);
  } catch (err) {
    console.error('Failed to load hook debug log:', err);
    const errorMsg = document.createElement('div');
    errorMsg.className = 'hook-debug-log-empty';
    errorMsg.textContent = 'Failed to load log';
    logBody.appendChild(errorMsg);
  }
}

function _formatTimestamp(ts) {
  if (!ts) return '-';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleTimeString('en-US', { hour12: false }) +
      '.' + String(d.getMilliseconds()).padStart(3, '0');
  } catch {
    return ts;
  }
}

function _showToast(container, message, isError = false) {
  // Remove any existing toast
  const existing = container.querySelector('.hook-debug-toast');
  if (existing) existing.remove();

  const toast = document.createElement('span');
  toast.className = 'hook-debug-toast' + (isError ? ' error' : '');
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 2500);
}
