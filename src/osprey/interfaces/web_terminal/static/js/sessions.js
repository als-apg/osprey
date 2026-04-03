/* OSPREY Web Terminal — Session Picker */

import { fetchJSON } from './api.js';
import { stopTerminal, startTerminal, restartTerminal, getCurrentSessionId, notifySessionChange, switchSession } from './terminal.js';

let sessionsData = [];

/**
 * Initialize the session selector dropdown.
 */
export function initSessionSelector(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Build dropdown structure
  container.innerHTML = `
    <button class="session-picker-btn" id="session-picker-btn" title="Browse sessions">
      <span class="session-picker-icon">&#9662;</span>
    </button>
    <div class="session-dropdown" id="session-dropdown">
      <div class="session-dropdown-header">Recent Sessions</div>
      <div class="session-dropdown-list" id="session-dropdown-list"></div>
    </div>
  `;

  const btn = document.getElementById('session-picker-btn');
  const dropdown = document.getElementById('session-dropdown');

  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const isOpen = dropdown.classList.contains('open');
    if (isOpen) {
      dropdown.classList.remove('open');
    } else {
      await fetchSessions();
      renderSessionList();
      dropdown.classList.add('open');
    }
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!container.contains(e.target)) {
      dropdown.classList.remove('open');
    }
  });
}

/**
 * Fetch sessions from the backend.
 */
async function fetchSessions() {
  try {
    const data = await fetchJSON('/api/sessions');
    sessionsData = data.sessions || [];
  } catch (err) {
    console.error('Failed to fetch sessions:', err);
    sessionsData = [];
  }
}

/**
 * Render the session list in the dropdown.
 */
function renderSessionList() {
  const list = document.getElementById('session-dropdown-list');
  if (!list) return;

  if (sessionsData.length === 0) {
    list.innerHTML = '<div class="session-item empty">No previous sessions</div>';
    return;
  }

  const currentId = getCurrentSessionId();

  list.innerHTML = sessionsData.map(s => {
    const isActive = s.session_id === currentId;
    const shortId = s.session_id.slice(0, 8);
    const timeAgo = relativeTime(s.last_modified);
    const preview = escapeHtml(s.first_message || '(no message)');
    return `
      <button class="session-item${isActive ? ' active' : ''}"
              data-session-id="${s.session_id}"
              title="${s.session_id}">
        <div class="session-item-header">
          <span class="session-item-id">${shortId}</span>
          <span class="session-item-time">${timeAgo}</span>
        </div>
        <div class="session-item-preview">${preview}</div>
        <div class="session-item-meta">${s.message_count} messages</div>
      </button>
    `;
  }).join('');

  // Attach click handlers
  list.querySelectorAll('.session-item[data-session-id]').forEach(el => {
    el.addEventListener('click', () => {
      const id = el.dataset.sessionId;
      document.getElementById('session-dropdown').classList.remove('open');
      resumeSession(id);
    });
  });
}

/**
 * Resume a session by ID.
 *
 * Fast path: send switch_session over the existing WebSocket (near-instant
 * for warm sessions). Cold fallback: full stop/start cycle if no WS is open.
 */
export async function resumeSession(sessionId) {
  if (switchSession(sessionId)) {
    // Fast path — server handles everything.
    // terminal.js onMessage updates UI on session_switched.
    return;
  }

  // Cold fallback — no WS open, do full reconnect
  stopTerminal();
  const label = document.getElementById('terminal-label');
  if (label) label.textContent = `Session ${sessionId.slice(0, 8)}`;
  await new Promise(r => setTimeout(r, 100));
  startTerminal(sessionId, 'resume');
  notifySessionChange(sessionId);
}

/**
 * Start a brand new session.
 */
export async function startNewSession() {
  await restartTerminal();
  const label = document.getElementById('terminal-label');
  if (label) label.textContent = 'Session';
  startTerminal();
}

/**
 * Compute a human-readable relative time string.
 */
function relativeTime(isoString) {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now - date;
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}

/**
 * Escape HTML to prevent XSS.
 */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
