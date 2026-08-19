// @ts-check
/* OSPREY Web Terminal — Agent Activity History Popover
 *
 * The activity strip's live line shows only the latest action, so clicking the
 * strip opens this popover listing the recent ones. The history itself lives on
 * the server (a bounded ring, GET /api/agent-activity/recent) — this module
 * keeps no client-side ring, it just fetches on open and renders the rows.
 * Frames arriving while it is open are prepended live; because the server
 * appends to its ring BEFORE broadcasting, a refetch always re-includes them,
 * so a live row is never lost when a slower fetch resolves over it.
 *
 * The popover is a child of <body>, position:fixed, because the hub's strip
 * sits inside the footer status bar and that bar is overflow:hidden and one
 * row tall — an in-place popover would be clipped away (same reason, same
 * recipe as the tile contribution menu in tile-header-items.js).
 *
 * The strip mount is its own trigger. It is an aria-live region provided by
 * the template, so this module cannot re-role it into a button or nest a
 * persistent button inside it (the live slot is wiped on every frame).
 * Instead the mount takes aria-expanded plus keyboard activation.
 *
 * All agent-supplied strings are rendered as text nodes only — createElement +
 * textContent, never innerHTML.
 */

import { withPrefix } from './api.js';
import { formatActivity, formatRelativeTime } from './activity-format.js';

/** @typedef {import('./panel-manager.js').AgentActivityEvent} AgentActivityFrame */

/**
 * Rows requested from the server ring, and the cap on rows kept in the open
 * popover's DOM. Matches ACTIVITY_RING_MAX server-side: asking for more just
 * gets clamped, and a popover left open through a long run must not grow
 * without bound.
 */
export const HISTORY_LIMIT = 50;

/**
 * Default history reader: the server's ring, newest first. Throws on a
 * transport or HTTP failure so the popover can show its error state rather
 * than an empty list that would read as "the agent did nothing".
 * @param {number} limit
 * @returns {Promise<AgentActivityFrame[]>}
 */
async function fetchRecentActivity(limit) {
  const resp = await fetch(withPrefix(`/api/agent-activity/recent?limit=${limit}`), {
    cache: 'no-store',
  });
  if (!resp.ok) throw new Error(`recent agent activity: HTTP ${resp.status}`);
  const body = await resp.json();
  // Contract: {"events": [...]} — an object, never a bare array.
  return Array.isArray(body?.events) ? body.events : [];
}

/**
 * Build the history popover for a strip, anchored to and triggered by `mount`.
 * Wiring the trigger is part of construction: the mount is the affordance, so
 * a history that existed without it could never be opened by an operator.
 *
 * Dependencies are injected so tests drive it without a network.
 * @param {{
 *   mount: HTMLElement,
 *   fetchRecent?: (limit: number) => Promise<AgentActivityFrame[]>,
 *   labelOf?: (id: string) => string,
 * }} deps
 * @returns {{
 *   open: () => Promise<void>,
 *   close: () => void,
 *   isOpen: () => boolean,
 *   prepend: (frame: AgentActivityFrame) => void,
 * }}
 */
export function createActivityHistory({
  mount,
  fetchRecent = fetchRecentActivity,
  labelOf: panelLabel,
}) {
  /** @type {HTMLElement | null} */
  let popoverEl = null;
  let historyOpen = false;
  /** Bumped on every open and close, so a fetch from a stale open is dropped. */
  let historyGeneration = 0;

  function ensurePopover() {
    if (popoverEl) return popoverEl;
    const el = document.createElement('div');
    el.className = 'activity-history-popover';
    el.setAttribute('role', 'region');
    el.setAttribute('aria-label', 'Recent agent activity');
    el.tabIndex = -1;
    popoverEl = el;
    return el;
  }

  /**
   * Replace the popover body with a single status line (loading, empty, error).
   * @param {string} text
   */
  function showHistoryMessage(text) {
    const el = ensurePopover();
    const msg = document.createElement('div');
    msg.className = 'activity-history-message';
    msg.textContent = text;
    el.replaceChildren(msg);
  }

  /**
   * One history row: verb, subject, and a coarse age. Every agent-supplied
   * string goes in through textContent, exactly as the live entry does.
   * @param {AgentActivityFrame} frame
   * @returns {HTMLElement}
   */
  function buildHistoryRow(frame) {
    const row = document.createElement('div');
    row.className = 'activity-history-row';

    const { verb, subject } = formatActivity(frame, { labelOf: panelLabel });
    const verbEl = document.createElement('span');
    verbEl.className = 'activity-history-verb';
    verbEl.textContent = verb;
    row.appendChild(verbEl);

    if (subject) {
      const subjectEl = document.createElement('span');
      subjectEl.className = 'activity-history-subject';
      subjectEl.textContent = subject;
      row.appendChild(subjectEl);
    }

    const age = formatRelativeTime(frame.ts, Date.now());
    if (age) {
      const timeEl = document.createElement('span');
      timeEl.className = 'activity-history-time';
      timeEl.textContent = age;
      row.appendChild(timeEl);
    }
    return row;
  }

  /**
   * Render the server's history, newest first — the endpoint already orders it
   * that way, so rows go in array order, top to bottom.
   * @param {AgentActivityFrame[]} events
   */
  function renderHistory(events) {
    const el = ensurePopover();
    const usable = events.filter((e) => e && e.target);
    if (usable.length === 0) {
      showHistoryMessage('No recent agent activity');
      return;
    }
    el.replaceChildren(...usable.slice(0, HISTORY_LIMIT).map(buildHistoryRow));
  }

  /**
   * Add a frame that arrived while the popover is open at the top of the list.
   * @param {AgentActivityFrame} frame
   */
  function prependHistoryRow(frame) {
    const el = ensurePopover();
    // A message line ("No recent agent activity", an error) is not a row —
    // the first real event replaces it.
    const msg = el.querySelector('.activity-history-message');
    if (msg) el.replaceChildren();
    el.prepend(buildHistoryRow(frame));
    while (el.children.length > HISTORY_LIMIT) el.lastElementChild?.remove();
  }

  /**
   * Anchor the fixed popover to the strip. It opens upward by default: in the
   * hub the strip is the footer status bar, so above is where the room is.
   */
  function placeHistory() {
    if (!popoverEl) return;
    const anchor = mount.getBoundingClientRect();
    const w = popoverEl.offsetWidth;
    const h = popoverEl.offsetHeight;
    const margin = 4;
    let left = anchor.left + anchor.width / 2 - w / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - w - margin));
    let top = anchor.top - h - margin;
    if (top < margin) top = Math.min(anchor.bottom + margin, window.innerHeight - h - margin);
    popoverEl.style.left = `${Math.round(left)}px`;
    popoverEl.style.top = `${Math.round(Math.max(margin, top))}px`;
  }

  /** @param {MouseEvent} e */
  function onDocumentClick(e) {
    if (!(e.target instanceof Node)) return;
    if (mount.contains(e.target)) return;
    if (popoverEl && popoverEl.contains(e.target)) return;
    closeHistory();
  }

  /** @param {KeyboardEvent} e */
  function onDocumentKeydown(e) {
    if (e.key === 'Escape') {
      // Focus first, close second. Closing can drop the tab stop (the strip
      // emptied while the popover was up), and focusing an unfocusable element
      // is a silent no-op that would strand focus on <body>. While the popover
      // is open the trigger is always focusable, so this always lands.
      mount.focus();
      closeHistory();
    }
  }

  function isHistoryOpen() {
    return historyOpen;
  }

  async function openHistory() {
    if (historyOpen) return;
    historyOpen = true;
    syncFocusability();
    const generation = ++historyGeneration;

    const el = ensurePopover();
    showHistoryMessage('Loading…');
    document.body.appendChild(el);
    mount.setAttribute('aria-expanded', 'true');
    placeHistory();
    // Capture phase, so an outside click closes before it does anything else.
    document.addEventListener('click', onDocumentClick, true);
    document.addEventListener('keydown', onDocumentKeydown, true);
    window.addEventListener('resize', placeHistory);
    el.focus();

    /** @type {AgentActivityFrame[] | null} */
    let events = null;
    try {
      events = await fetchRecent(HISTORY_LIMIT);
    } catch {
      events = null;
    }
    // Closed (or closed and reopened) while the request was in flight.
    if (generation !== historyGeneration) return;
    if (events == null) showHistoryMessage('Could not load recent activity');
    else renderHistory(events);
    placeHistory();
  }

  function closeHistory() {
    if (!historyOpen) return;
    historyOpen = false;
    historyGeneration++;
    popoverEl?.remove();
    mount.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', onDocumentClick, true);
    document.removeEventListener('keydown', onDocumentKeydown, true);
    window.removeEventListener('resize', placeHistory);
    syncFocusability();
  }

  function toggleHistory() {
    if (historyOpen) closeHistory();
    else void openHistory();
  }

  mount.classList.add('activity-strip-trigger');
  mount.setAttribute('aria-expanded', 'false');
  mount.title = 'Recent agent activity';

  /**
   * Keep focusability in step with the strip's emptiness.
   *
   * An empty strip is hidden with `opacity: 0` rather than `display: none` —
   * activity-strip.css explains why: it is an aria-live region, and a
   * display-hidden region would enter the accessibility tree at the same moment
   * as its content, so screen readers could drop the announcement. The side
   * effect is that an empty strip is invisible AND zero-sized while still
   * sitting in the DOM, so an unconditional `tabIndex = 0` left a permanent tab
   * stop whose `:focus-visible` outline could not be seen. Keyboard users tabbed
   * into a hole.
   *
   * This is the keyboard half of the `pointer-events: none` already on
   * `.status-activity-floating:empty`. Three things keep the tab stop alive:
   * the strip showing something, the popover being open (the strip may empty on
   * its clear timer with the popover still up), and the trigger currently
   * holding focus — dropping `tabindex` out from under the focused element is
   * how you strand focus on `<body>`. The blur listener collects that last case
   * once focus moves on.
   */
  function syncFocusability() {
    const stillNeeded =
      mount.childNodes.length > 0 || historyOpen || document.activeElement === mount;
    if (stillNeeded) {
      if (!mount.hasAttribute('tabindex')) mount.tabIndex = 0;
    } else {
      mount.removeAttribute('tabindex');
    }
  }

  syncFocusability();
  mount.addEventListener('blur', syncFocusability);
  new MutationObserver(syncFocusability).observe(mount, { childList: true });
  mount.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleHistory();
  });
  mount.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggleHistory();
    }
  });

  return {
    open: openHistory,
    close: closeHistory,
    isOpen: isHistoryOpen,
    prepend: prependHistoryRow,
  };
}
