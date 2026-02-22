/**
 * OSPREY Channel Finder — Search View (AI Subagent + Streaming + Validate)
 *
 * Two modes:
 * - AI Search: Creates an OperatorSession, sends natural language queries,
 *   and streams typed events (tool_use, tool_result, text, result).
 * - Validate: Validates channel names against the database.
 */

import { postJSON, deleteJSON, openWS } from './api.js';
import { showToast } from './app.js';
import { esc } from './utils.js';

let containerEl = null;
let sessionId = null;
let ws = null;
let currentMode = 'search';  // 'search' | 'validate'

export function mountSearch(container) {
  containerEl = container;

  container.innerHTML = `
    <div class="section-header">
      <div>
        <div class="section-title">Search & Validate</div>
        <div class="section-subtitle">
          Search channels with AI or validate channel names
        </div>
      </div>
    </div>
    <div class="search-mode-tabs" id="search-mode-tabs">
      <button class="search-mode-tab active" data-mode="search">AI Search</button>
      <button class="search-mode-tab" data-mode="validate">Validate</button>
    </div>
    <div id="search-mode-content"></div>
  `;

  // Wire up mode tabs
  document.querySelectorAll('#search-mode-tabs .search-mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      currentMode = tab.dataset.mode;
      document.querySelectorAll('#search-mode-tabs .search-mode-tab').forEach(t =>
        t.classList.toggle('active', t.dataset.mode === currentMode)
      );
      renderModeContent();
    });
  });

  renderModeContent();
}

export function unmountSearch() {
  closeSession();
  containerEl = null;
}

function renderModeContent() {
  const content = document.getElementById('search-mode-content');
  if (!content) return;

  if (currentMode === 'validate') {
    renderValidateMode(content);
  } else {
    renderSearchMode(content);
  }
}

// ---- AI Search Mode ----

function renderSearchMode(content) {
  content.innerHTML = `
    <div class="search-input-container">
      <input type="text" class="search-input" id="search-input"
             placeholder="Describe the channels you're looking for..."
             autocomplete="off">
      <button class="search-submit" id="search-submit">Search</button>
    </div>
    <div id="search-status" style="margin-top: var(--space-2); font-size: var(--text-xs); color: var(--text-muted);"></div>
    <div class="search-results" id="search-results"></div>
  `;

  const input = document.getElementById('search-input');
  const btn = document.getElementById('search-submit');

  btn?.addEventListener('click', () => handleSearch());
  input?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSearch();
  });
}

async function handleSearch() {
  const input = document.getElementById('search-input');
  const results = document.getElementById('search-results');
  const status = document.getElementById('search-status');
  if (!input?.value.trim()) return;

  const query = input.value.trim();

  // Clear previous results
  if (results) results.innerHTML = '';
  if (status) status.textContent = 'Connecting to AI agent...';

  try {
    // Create session if needed
    if (!sessionId) {
      const data = await postJSON('/api/search/session', {});
      sessionId = data.session_id;
    }

    // Open WebSocket for streaming
    if (ws) ws.close();
    ws = openWS(`/ws/search/${sessionId}`);

    ws.onopen = () => {
      if (status) status.textContent = 'Connected. Searching...';
      // Send the query
      ws.send(JSON.stringify({ type: 'prompt', text: query }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        renderEvent(data);

        if (data.type === 'result') {
          if (status) status.textContent = formatResult(data);
        }
      } catch {
        // Ignore non-JSON messages
      }
    };

    ws.onerror = () => {
      if (status) status.textContent = 'Connection error';
      showToast('Search connection error', 'error');
    };

    ws.onclose = () => {
      if (status && status.textContent === 'Connected. Searching...') {
        status.textContent = 'Search complete';
      }
    };
  } catch (e) {
    if (status) status.textContent = `Error: ${e.message}`;
    showToast(`Search failed: ${e.message}`, 'error');
  }
}

function renderEvent(event) {
  const results = document.getElementById('search-results');
  if (!results) return;

  const el = document.createElement('div');
  el.className = `search-event event-${event.type}`;

  switch (event.type) {
    case 'text':
      el.innerHTML = `
        <div class="search-event-content">${formatMarkdown(event.content || '')}</div>
      `;
      break;

    case 'tool_use':
      el.innerHTML = `
        <div class="search-event-label">Tool: ${esc(event.tool_name || event.tool_name_raw || 'unknown')}</div>
        <div class="search-event-content">
          <code>${esc(JSON.stringify(event.input || {}, null, 2))}</code>
        </div>
      `;
      break;

    case 'tool_result':
      el.innerHTML = `
        <div class="search-event-label">
          Result ${event.is_error ? '(error)' : ''}
        </div>
        <div class="search-event-content">
          <code>${esc(typeof event.content === 'string' ? event.content : JSON.stringify(event.content || '', null, 2))}</code>
        </div>
      `;
      break;

    case 'thinking':
      el.innerHTML = `
        <div class="search-event-label">Thinking</div>
        <div class="search-event-content" style="color: var(--text-muted); font-style: italic;">
          ${esc(event.content || '')}
        </div>
      `;
      break;

    case 'error':
      el.innerHTML = `
        <div class="search-event-label">Error</div>
        <div class="search-event-content" style="color: var(--color-error);">
          ${esc(event.message || 'Unknown error')}
        </div>
      `;
      break;

    case 'result':
      el.innerHTML = `
        <div class="search-event-label">Complete</div>
        <div class="search-event-content" style="color: var(--color-success);">
          ${formatResult(event)}
        </div>
      `;
      break;

    case 'system':
      // Skip system events (init, etc.)
      return;

    default:
      return;
  }

  results.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function formatResult(data) {
  const parts = [];
  if (data.num_turns != null) parts.push(`${data.num_turns} turns`);
  if (data.duration_ms != null) parts.push(`${(data.duration_ms / 1000).toFixed(1)}s`);
  if (data.total_cost_usd != null) parts.push(`$${data.total_cost_usd.toFixed(4)}`);
  return parts.join(' · ') || 'Done';
}

function formatMarkdown(text) {
  // Simple markdown: bold, code, newlines
  return esc(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// ---- Validate Mode ----

function renderValidateMode(content) {
  content.innerHTML = `
    <div class="card">
      <div class="card-header">
        <span class="card-title">Validate Channels</span>
      </div>
      <div style="max-width: 640px;">
        <div class="form-group">
          <label class="form-label" for="validate-input">Channel Names (one per line)</label>
          <textarea class="form-input" id="validate-input" rows="5"
                    placeholder="Enter channel names to validate..."></textarea>
        </div>
        <div style="display: flex; gap: var(--space-2); margin-top: var(--space-4);">
          <button class="btn btn-primary" id="validate-btn">Validate</button>
        </div>
      </div>
      <div id="validate-results" style="margin-top: var(--space-3);"></div>
    </div>
  `;

  document.getElementById('validate-btn')?.addEventListener('click', handleValidate);
}

async function handleValidate() {
  const input = document.getElementById('validate-input');
  const resultsEl = document.getElementById('validate-results');
  if (!input || !resultsEl) return;

  const channels = input.value.split('\n').map(s => s.trim()).filter(Boolean);
  if (channels.length === 0) {
    showToast('Enter at least one channel name', 'error');
    return;
  }

  resultsEl.innerHTML = '<div class="loading-center"><div class="loading-spinner"></div> Validating...</div>';

  try {
    const data = await postJSON('/api/validate', { channels });
    const results = data.results || [];

    resultsEl.innerHTML = `
      <div style="margin-bottom: var(--space-2); font-size: var(--text-sm);">
        <span style="color: var(--color-success);">${data.valid_count || 0} valid</span> &middot;
        <span style="color: var(--color-error);">${data.invalid_count || 0} invalid</span> &middot;
        <span style="color: var(--text-muted);">${data.total || channels.length} total</span>
      </div>
      <div class="table-wrapper" style="max-height: 300px;">
        <table class="data-table">
          <thead><tr><th>Channel</th><th>Status</th></tr></thead>
          <tbody>
            ${results.map(r => `
              <tr>
                <td class="pv-cell">${esc(r.channel || r.name || '')}</td>
                <td style="color: ${r.valid ? 'var(--color-success)' : 'var(--color-error)'};">
                  ${r.valid ? 'Valid' : 'Invalid'}
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  } catch (e) {
    resultsEl.innerHTML = `<div class="empty-state" style="color: var(--color-error);">Validation failed: ${e.message}</div>`;
  }
}

// ---- Session Management ----

async function closeSession() {
  if (ws) {
    ws.close();
    ws = null;
  }
  if (sessionId) {
    try {
      await deleteJSON(`/api/search/${sessionId}`);
    } catch { /* ignore */ }
    sessionId = null;
  }
}
