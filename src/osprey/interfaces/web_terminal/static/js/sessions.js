/* OSPREY Web Terminal — Session Picker
 *
 * Two openers onto the same session list: the terminal's picker in the
 * `.terminal-header` (a dropdown under its chevron), and the chat's "Earlier
 * conversations" menu behind the Agent bar's ⋯ (an `.osprey-popover` placed by
 * popover.js). Both list what `/api/sessions` returns, switch through the same
 * resume path, and are withheld while the chat is busy.
 *
 * The ⋯ menu's last row is the Agent tile's way between its surfaces: "Open
 * terminal" in chat, "Back to chat" on the terminal, whose own ⋯ holds only that
 * row. Absent where the deployment offers no terminal. */

import { fetchJSON } from './api.js';
import {
  stopTerminal,
  startTerminal,
  restartTerminal,
  getCurrentSessionId,
  notifySessionChange,
  switchSession,
  setSessionLabel,
} from './terminal.js';
import { escapeHtml } from '/design-system/js/dom.js';
import { terminalOffered } from '/design-system/js/frame-params.js';
import { formatFacilityTime } from '/design-system/js/facility-time.js';
import { icon } from '/design-system/js/icons.js';
import { closePopover, openPopover } from '/design-system/js/popover.js';
import {
  isTerminalSurface,
  onSurfaceChange,
  showChatSurface,
  showTerminalSurface,
} from './agent-surface.js';
import { onChat } from './chat-events.js';
import { setPointer } from './session-pointer.js';

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

/** The picker button, once {@link initSessionSelector} has built it. @type {HTMLButtonElement | null} */
let pickerBtn = null;

/**
 * The Agent bars' ⋯ buttons and the one menu they open, once
 * {@link attachConversationMenu} has wired them.
 * @type {{ openers: HTMLButtonElement[], menu: HTMLElement } | null}
 */
let conversationMenu = null;

/** Whether the chat has a turn streaming or a hand-off in flight (its `busy` event). */
let chatBusy = false;

/**
 * In chat the console runs the session, so a pick while a turn streams would cut
 * the turn off and leave the agent working on a conversation nobody is watching.
 * The picker is withheld while the chat is busy, as the console's New button is,
 * and the ⋯ menu keeps only its way to the terminal, whose hand-off waits the
 * turn out (withheld too where no terminal is offered). The terminal has its own session controls, and a hidden console can stay
 * busy there after a refused hand-off, so the surface is part of the verdict and
 * a change re-evaluates it.
 */
function syncPicker() {
  const blocked = chatBusy && !isTerminalSurface();
  if (pickerBtn) {
    pickerBtn.disabled = blocked;
    if (blocked) closeDropdown();
  }
  if (conversationMenu) {
    for (const opener of conversationMenu.openers) {
      opener.disabled = blocked && !terminalOffered();
    }
    if (blocked) closePopover(conversationMenu.menu);
  }
}

onChat('busy', ({ busy }) => {
  chatBusy = busy;
  syncPicker();
});
onSurfaceChange(syncPicker);

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

  const btn = /** @type {HTMLButtonElement} */ (document.getElementById('session-picker-btn'));

  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (openPicker) {
      closeDropdown();
      return;
    }
    await fetchSessions();
    openDropdown(btn);
  });

  pickerBtn = btn;
  syncPicker();

  const more = /** @type {HTMLButtonElement | null} */ (
    document.getElementById('terminal-more-btn')
  );
  if (more) attachConversationMenu(more);
}

/**
 * Give an Agent bar's ⋯ the "Earlier conversations" menu: a body-level
 * `.osprey-popover` listing the sessions the picker lists, freshly fetched on
 * every open, the current one checked. A pick switches through the picker's
 * own resume path. Withheld while the chat is busy, exactly as the picker is
 * (see syncPicker). Called once per bar: by chat.js after it builds the
 * console's, and by {@link initSessionSelector} for the terminal header's.
 * @param {HTMLButtonElement} opener
 */
export function attachConversationMenu(opener) {
  const menu = conversationMenu?.menu ?? buildConversationMenu();

  opener.addEventListener('click', async () => {
    if (!menu.hidden) {
      closePopover(menu);
      return;
    }
    if (!isTerminalSurface() && !chatBusy) await fetchSessions();
    // The chat may have turned busy while the list loaded.
    if (opener.disabled) return;
    renderConversationMenu(menu);
    openPopover(menu, opener, { place: 'below-end' });
  });

  conversationMenu = { openers: [...(conversationMenu?.openers ?? []), opener], menu };
  syncPicker();
}

/** @returns {HTMLElement} the one ⋯ menu, hidden and in the page. */
function buildConversationMenu() {
  const menu = document.createElement('div');
  menu.className = 'osprey-popover op-conversations';
  menu.id = 'agent-conversations-menu';
  menu.setAttribute('role', 'menu');
  menu.setAttribute('aria-label', 'Agent');
  menu.hidden = true;
  document.body.append(menu);
  return menu;
}

/**
 * The menu's quiet last row, under a separator: the way to the other surface.
 * @param {HTMLElement} menu
 * @returns {HTMLElement[]}
 */
function surfaceRow(menu) {
  if (!terminalOffered()) return [];
  const sep = document.createElement('div');
  sep.className = 'osprey-popover-sep';
  sep.setAttribute('role', 'separator');
  const row = document.createElement('button');
  row.type = 'button';
  row.className = 'osprey-popover-item op-surface-row';
  row.setAttribute('role', 'menuitem');
  const onTerminal = isTerminalSurface();
  row.textContent = onTerminal ? 'Back to chat' : 'Open terminal';
  row.addEventListener('click', () => {
    closePopover(menu);
    if (onTerminal) showChatSurface();
    else showTerminalSurface();
  });
  return [sep, row];
}

/**
 * Fill the "Earlier conversations" menu from the fetched session list: one
 * `menuitemradio` per session (its first message, and how long ago), the
 * session the tab is on checked.
 * @param {HTMLElement} menu
 */
function renderConversationMenu(menu) {
  if (isTerminalSurface() || chatBusy) {
    // On the terminal the picker beside this ⋯ lists the sessions, and while a
    // chat turn streams a pick would cut it off: only the way across is left.
    menu.replaceChildren(...surfaceRow(menu).slice(1));
    return;
  }
  const title = document.createElement('div');
  title.className = 'osprey-popover-title';
  title.textContent = 'Earlier conversations';
  if (sessionsData.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'osprey-list-empty';
    empty.textContent = 'No earlier conversations';
    menu.replaceChildren(title, empty, ...surfaceRow(menu));
    return;
  }
  const currentId = getCurrentSessionId();
  const items = sessionsData.map((s) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'osprey-popover-item';
    item.setAttribute('role', 'menuitemradio');
    item.setAttribute('aria-checked', String(s.session_id === currentId));
    item.dataset.sessionId = s.session_id;
    item.title = `${s.message_count} messages · ${s.session_id}`;
    const label = document.createElement('span');
    label.className = 'op-conversation-title';
    label.textContent = s.first_message || '(no message)';
    const aside = document.createElement('span');
    aside.className = 'osprey-popover-aside';
    aside.textContent = relativeTime(s.last_modified);
    const check = icon('check');
    check.classList.add('osprey-popover-check');
    item.append(label, aside, check);
    item.addEventListener('click', () => {
      closePopover(menu);
      resumeSession(s.session_id);
    });
    return item;
  });
  menu.replaceChildren(title, ...items, ...surfaceRow(menu));
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
    if (e.key !== 'Escape') return;
    closeDropdown();
    btn.focus();
  };
  // Close on reflow rather than running a reposition loop: the anchor lives in
  // a scrollable tab strip, so a stale popover would drift off its button.
  // The scroll listener is capture-phase and sees every scroll on the page --
  // the list's own, the chat pane following new output -- so only one that
  // moves the button closes the picker.
  const onReflow = (/** @type {Event} */ e) => {
    const t = e.target;
    if (t instanceof Node && !t.contains(btn)) return;
    closeDropdown();
  };
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
  // Keyboard users arrive from the picker button, and the list sits at the end
  // of the body: focus moves into it, or Tab would skip the sessions entirely.
  const focusTarget =
    dropdown.querySelector('.session-item.active') ??
    dropdown.querySelector('.session-item[data-session-id]');
  if (focusTarget instanceof HTMLElement) focusTarget.focus();
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

  list.innerHTML = sessionsData
    .map((s) => {
      const isActive = s.session_id === currentId;
      // Every server-supplied value is escaped, the id included: it lands in two
      // attributes and the markup, and an unescaped quote would end the attribute.
      const id = escapeHtml(s.session_id);
      const shortId = escapeHtml(s.session_id.slice(0, 8));
      const timeAgo = relativeTime(s.last_modified);
      const preview = escapeHtml(s.first_message || '(no message)');
      return `
      <button class="session-item${isActive ? ' active' : ''}"
              data-session-id="${id}"
              title="${id}">
        <div class="session-item-header">
          <span class="session-item-id">${shortId}</span>
          <span class="session-item-time">${timeAgo}</span>
        </div>
        <div class="session-item-preview">${preview}</div>
        <div class="session-item-meta">${s.message_count} messages</div>
      </button>
    `;
    })
    .join('');

  // Attach click handlers
  const items = /** @type {NodeListOf<HTMLElement>} */ (
    list.querySelectorAll('.session-item[data-session-id]')
  );
  items.forEach((el) => {
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
async function resumeSession(sessionId) {
  // Chat has no terminal socket: the console is the surface running the session.
  // Moving the pointer is the whole switch, because the console is subscribed to
  // it and replays the key. Starting a terminal here would put a second surface
  // on the session's agent.
  if (!isTerminalSurface()) {
    setPointer(sessionId);
    return;
  }
  if (switchSession(sessionId)) {
    // Fast path — server handles everything.
    // terminal.js onMessage updates UI on session_switched.
    return;
  }

  // Cold fallback — no WS open, do full reconnect
  stopTerminal();
  setSessionLabel(sessionId);
  await new Promise((r) => setTimeout(r, 100));
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
  return formatFacilityTime(date, { year: 'numeric', month: 'numeric', day: 'numeric' });
}
