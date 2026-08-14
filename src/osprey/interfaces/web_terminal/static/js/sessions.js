/* OSPREY Web Terminal — Session Picker */

import { fetchJSON } from './api.js';
import { stopTerminal, startTerminal, restartTerminal, getCurrentSessionId, notifySessionChange, switchSession, setSessionLabel } from './terminal.js';
import { escapeHtml } from '/design-system/js/dom.js';

/**
 * A session summary as returned by the (prefix-aware) `/api/sessions` endpoint.
 * @typedef {object} SessionRecord
 * @property {string} session_id
 * @property {string} last_modified
 * @property {number} message_count
 * @property {string} [first_message]
 */

/** @type {SessionRecord[]} */
let sessionsData = [];

/**
 * The one open dropdown, or null. Tracked at module scope because the node is
 * body-level (see `openDropdown`) rather than a child of the picker container.
 * @type {{ dropdown: HTMLElement, btn: HTMLElement, off: () => void } | null}
 */
let openPicker = null;

/**
 * Initialize the session selector dropdown.
 * @param {string} containerId
 */
export function initSessionSelector(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // A re-init would orphan a body-level dropdown from the button it anchors to.
  closeDropdown();

  container.innerHTML = `
    <button class="session-picker-btn" id="session-picker-btn" title="Browse sessions"
            aria-haspopup="menu" aria-expanded="false">
      <span class="session-picker-icon">&#9662;</span>
    </button>
  `;

  const btn = /** @type {HTMLElement} */ (document.getElementById('session-picker-btn'));

  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (openPicker) {
      closeDropdown();
      return;
    }
    await fetchSessions();
    openDropdown(btn);
  });
}

/* ---- dropdown popover ----
 * Appended to document.body rather than inside the picker: dock-tab.js adopts
 * the live `.terminal-header` into dockview's tab strip, and that strip is both
 * `overflow: hidden` (one bar tall) and `transform: translate3d(...)`. The
 * transform opens a stacking context, so an in-place dropdown is clipped away
 * AND its z-index is trapped below the sibling content pane -- it rendered
 * behind the terminal. Body-level + `position: fixed` + JS placement escapes
 * both, matching `.contrib-menu-popover` in tile-header-items.js. */

/**
 * Build, mount, fill and pin the dropdown, and register it as THE open picker.
 * @param {HTMLElement} btn
 */
function openDropdown(btn) {
  closeDropdown();

  const dropdown = document.createElement('div');
  dropdown.className = 'session-dropdown';
  dropdown.id = 'session-dropdown';
  dropdown.setAttribute('role', 'menu');
  dropdown.innerHTML = `
    <div class="session-dropdown-header">Recent Sessions</div>
    <div class="session-dropdown-list" id="session-dropdown-list"></div>
  `;
  document.body.appendChild(dropdown);

  const onDown = (/** @type {Event} */ e) => {
    const t = e.target;
    if (t instanceof Node && (dropdown.contains(t) || btn.contains(t))) return;
    closeDropdown();
  };
  const onKey = (/** @type {KeyboardEvent} */ e) => {
    if (e.key === 'Escape') closeDropdown();
  };
  // Close on reflow rather than running a reposition loop: the anchor lives in
  // a scrollable tab strip, so a stale popover would drift off its button.
  const onReflow = () => closeDropdown();
  document.addEventListener('pointerdown', onDown, true);
  document.addEventListener('keydown', onKey, true);
  window.addEventListener('resize', onReflow);
  window.addEventListener('scroll', onReflow, true);

  openPicker = {
    dropdown,
    btn,
    off: () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('resize', onReflow);
      window.removeEventListener('scroll', onReflow, true);
    },
  };

  renderSessionList();
  // `.open` before placing: the node must be laid out for offsetWidth/Height.
  dropdown.classList.add('open');
  place(dropdown, btn);
  btn.setAttribute('aria-expanded', 'true');
}

/** Tear down the open dropdown, if any. */
function closeDropdown() {
  if (!openPicker) return;
  openPicker.off();
  openPicker.dropdown.remove();
  openPicker.btn.setAttribute('aria-expanded', 'false');
  openPicker = null;
}

/**
 * Pin the dropdown under the button, left edges aligned, clamped to the
 * viewport so a picker near the right or bottom edge still shows a full list.
 * @param {HTMLElement} dropdown
 * @param {HTMLElement} btn
 */
function place(dropdown, btn) {
  const r = btn.getBoundingClientRect();
  const w = dropdown.offsetWidth;
  const h = dropdown.offsetHeight;
  const margin = 6;
  const left = Math.max(margin, Math.min(r.left, window.innerWidth - w - margin));
  let top = r.bottom + margin;
  if (top + h > window.innerHeight - margin) top = Math.max(margin, r.top - h - margin);
  dropdown.style.left = `${Math.round(left)}px`;
  dropdown.style.top = `${Math.round(top)}px`;
}

/**
 * Fetch sessions from the backend.
 */
async function fetchSessions() {
  try {
    const data = await fetchJSON('/api/sessions'); // fetchJSON prefixes this internally
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
  const items = /** @type {NodeListOf<HTMLElement>} */ (
    list.querySelectorAll('.session-item[data-session-id]')
  );
  items.forEach(el => {
    el.addEventListener('click', () => {
      const id = /** @type {string} */ (el.dataset.sessionId);
      closeDropdown();
      resumeSession(id);
    });
  });
}

/**
 * Resume a session by ID.
 *
 * Fast path: send switch_session over the existing WebSocket (near-instant
 * for warm sessions). Cold fallback: full stop/start cycle if no WS is open.
 * @param {string} sessionId
 */
export async function resumeSession(sessionId) {
  if (switchSession(sessionId)) {
    // Fast path — server handles everything.
    // terminal.js onMessage updates UI on session_switched.
    return;
  }

  // Cold fallback — no WS open, do full reconnect
  stopTerminal();
  setSessionLabel(sessionId);
  await new Promise(r => setTimeout(r, 100));
  startTerminal(sessionId, 'resume');
  notifySessionChange(sessionId);
}

/**
 * Start a brand new session.
 */
export async function startNewSession() {
  await restartTerminal();
  setSessionLabel(null);
  startTerminal();
}

/**
 * Compute a human-readable relative time string.
 * @param {string} isoString
 */
function relativeTime(isoString) {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}
