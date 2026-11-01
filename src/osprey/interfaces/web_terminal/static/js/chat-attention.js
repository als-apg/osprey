// @ts-check
/* OSPREY Web Terminal — Simple-mode attention signal
 *
 * Makes a wait on the operator visible from another tab and from the sidebar.
 * While an approval, plan, or question card is pending, the tab title carries
 * a count, and the chat bus carries it as `attention` `{waiting}` — emitted on
 * every change of the count, and `0` on reset and turn end — for the
 * sidebar's Agent row, which badges it while the Agent pane is closed. An
 * opt-in bell in the Agent bar sends a desktop notification when a card
 * arrives while the page is hidden, and one more when a turn ends while it is
 * hidden.
 *
 * Replayed history never badges and never notifies: a frame's `replay` flag
 * says whether it is a past turn being redrawn. Pending cards come from the
 * live stream, including the requests the server re-delivers on (re)connect.
 *
 * The title is composed, not owned. The badge keeps the title that was there
 * before it, and when another writer changes the title while the badge is up,
 * the badge re-bases on that title rather than writing over it.
 *
 * @module chat-attention
 */

import { barButton } from './chat-console.js';
import { emitChat, onChat } from './chat-events.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** Storage base for the bell's state; scoped per operator on a multi-user mount. */
const STORAGE_BASE = 'osprey-chat-notify';

/** Tag for the one "agent finished" notification; a newer one replaces it. */
const TURN_TAG = 'osprey-turn-finished';

const BELL_TITLE = 'Desktop notifications, when the agent needs you while this tab is hidden';
const BLOCKED_TITLE =
  "Notifications are blocked for this site. Allow them in the browser's site settings.";

/** The `Notification` constructor when the browser has one, else null. Read at call time. */
function notifier() {
  return typeof Notification === 'undefined' ? null : /** @type {any} */ (Notification);
}

/** @returns {boolean} whether the operator turned the bell on last time */
function readStored() {
  try {
    return localStorage.getItem(scopedStorageKey(STORAGE_BASE)) === '1';
  } catch {
    return false;
  }
}

/** @param {boolean} on */
function writeStored(on) {
  try {
    if (on) localStorage.setItem(scopedStorageKey(STORAGE_BASE), '1');
    else localStorage.removeItem(scopedStorageKey(STORAGE_BASE));
  } catch {
    /* storage blocked: the bell still works for this page view */
  }
}

/**
 * The title wording for the cards waiting now. Approvals and plans read as an
 * approval; questions read as a question; a mix reads as a plain prompt.
 * @param {string[]} kinds - the `kind` of each waiting card
 * @returns {string}
 */
function waitingLabel(kinds) {
  const questions = kinds.filter((kind) => kind === 'question').length;
  if (questions === kinds.length) return 'Question';
  if (questions === 0) return 'Approval needed';
  return 'Waiting on you';
}

/**
 * The notification text for one waiting card. The tool's name is the only
 * detail shown: a notification can appear on a lock screen, so no command or
 * input text goes in it.
 * @param {string} kind
 * @param {string} toolName
 * @returns {string}
 */
function requestText(kind, toolName) {
  if (kind === 'question') return 'Osprey: the agent has a question';
  if (kind === 'plan') return 'Osprey: the plan needs your approval';
  return `Osprey: ${toolName} needs your approval`;
}

/** @param {Record<string, any>} event */
const requestId = (event) => String(event.request_id ?? '');

/**
 * The document title, with a count prefix while cards wait. It writes only
 * what it must, and it keeps whatever title the page had as the base.
 */
function titleBadge() {
  /** @type {string | null} what this badge last wrote, or null when it is not showing */
  let written = null;
  let base = document.title;
  let shown = { count: 0, label: '' };

  const write = () => {
    const current = document.title;
    // First write, or another writer changed the title since ours: that title is the base.
    if (written === null || current !== written) base = current;
    const next = shown.count > 0 ? `(${shown.count}) ${shown.label} · ${base}` : base;
    if (next !== current) document.title = next;
    written = shown.count > 0 ? next : null;
  };

  // Another writer that replaces the title while the badge is up gets its title
  // re-based and the count re-applied; our own write matches `written` and stops there.
  const observer = new MutationObserver(() => {
    if (written !== null && document.title !== written) write();
  });
  observer.observe(document.head, { childList: true, subtree: true, characterData: true });

  return {
    /** @param {number} count @param {string} label */
    set(count, label) {
      shown = { count, label };
      write();
    },
    dispose() {
      observer.disconnect();
      shown = { count: 0, label: '' };
      write();
    },
  };
}

/**
 * The opt-in bell. It is on only when the operator turned it on and the browser
 * has granted permission. A denied or unsupported browser leaves it off: the
 * button is disabled or hidden, and nothing throws.
 * @returns {{ button: HTMLButtonElement, isOn: () => boolean }}
 */
function createBell() {
  const api = notifier();
  const button = barButton('op-attention-toggle', 'bell', 'Desktop notifications');
  button.title = BELL_TITLE;

  let on = api?.permission === 'granted' && readStored();
  button.setAttribute('aria-pressed', String(on));
  if (!api) {
    button.hidden = true;
  } else if (api.permission === 'denied') {
    button.disabled = true;
    button.title = BLOCKED_TITLE;
  }

  /** @param {boolean} next */
  const set = (next) => {
    on = next;
    writeStored(next);
    button.setAttribute('aria-pressed', String(next));
  };

  button.addEventListener('click', () => {
    if (on) {
      set(false);
      return;
    }
    const browser = notifier();
    if (!browser) return;
    if (browser.permission === 'granted') {
      set(true);
      return;
    }
    if (browser.permission !== 'default') return;
    // The request is made inside the click, so the browser treats it as the
    // operator's gesture. Dismissing or denying leaves the bell off.
    let asked;
    try {
      asked = browser.requestPermission();
    } catch {
      return;
    }
    Promise.resolve(asked).then(
      (answer) => {
        if (answer === 'granted') set(true);
      },
      () => {
        /* the browser refused to ask: the bell stays off */
      }
    );
  });

  return { button, isOn: () => on };
}

/**
 * Start the attention signal for the console. Call once, from
 * {@link ../chat-features.js initChatFeatures}.
 * @param {import('./chat-features.js').ChatFeatureContext} ctx
 * @returns {() => void} stop listening, remove the bell, and restore the title
 */
export function initChatAttention(ctx) {
  /** @type {Map<string, { kind: string, toolName: string }>} waiting cards, by request id */
  const pending = new Map();
  /** @type {Map<string, HTMLElement>} the card element of each waiting request */
  const cards = new Map();
  /** @type {WeakSet<Element>} card elements already seen, live or replayed */
  const seen = new WeakSet();
  /** @type {Set<string>} request ids a notification has been shown for */
  const notified = new Set();
  /** @type {Map<string, { close?: () => void }>} notifications still on screen, by request id */
  const open = new Map();

  const title = titleBadge();
  const bell = createBell();
  ctx.bar.append(bell.button);

  /**
   * The notification is taken away once its request is answered or cancelled.
   * @param {string} id
   */
  function closeNote(id) {
    open.get(id)?.close?.();
    open.delete(id);
  }

  /** The waiting count last put on the chat bus. */
  let emitted = 0;

  /** Show the waiting count in the title, and on the bus when it changed. */
  function render() {
    const kinds = [...pending.values()].map((waiting) => waiting.kind);
    title.set(kinds.length, waitingLabel(kinds));
    if (kinds.length !== emitted) {
      emitted = kinds.length;
      emitChat('attention', { waiting: emitted });
    }
  }

  /** The turn is over or the log was cleared: nothing waits any more. */
  function clearPending() {
    for (const id of [...pending.keys()]) closeNote(id);
    pending.clear();
    cards.clear();
    const before = emitted;
    render();
    // render() puts the zero on the bus only when the count changed. It is said
    // at every turn boundary all the same, so a listener that joined late
    // settles on zero.
    if (before === 0) emitChat('attention', { waiting: 0 });
  }

  /**
   * Show a desktop notification when the operator opted in, the page is hidden,
   * and the browser has granted permission. Returns it, or null.
   * @param {string} tag
   * @param {string} text
   * @param {() => void} [onClick]
   * @returns {{ close?: () => void } | null}
   */
  function notify(tag, text, onClick) {
    if (!bell.isOn() || document.visibilityState !== 'hidden') return null;
    const browser = notifier();
    if (!browser || browser.permission !== 'granted') return null;
    try {
      const note = new browser(text, { tag });
      note.onclick = () => {
        window.focus();
        onClick?.();
        note.close?.();
      };
      return note;
    } catch {
      return null;
    }
  }

  /**
   * The card the renderer drew for this frame: the newest pending card the
   * page has not seen yet. Every pending card is marked seen, replayed or not.
   * @returns {HTMLElement | null}
   */
  function newestNewCard() {
    const cardsNow = [...ctx.messages.querySelectorAll('.op-card[data-state="pending"]')];
    const fresh = cardsNow.filter((el) => !seen.has(el));
    for (const el of cardsNow) seen.add(el);
    const newest = fresh[fresh.length - 1];
    return newest instanceof HTMLElement ? newest : null;
  }

  /** @param {string} id */
  function focusCard(id) {
    const card = cards.get(id);
    if (card?.isConnected) card.focus();
  }

  const stopFrames = onChat('frame', ({ event, replay }) => {
    if (event.type === 'permission_request') {
      // Taken even on replay, so a replayed card is never mistaken for a new one.
      const card = newestNewCard();
      if (replay) return;
      const id = requestId(event);
      const kind = String(event.kind ?? 'tool');
      const toolName = String(event.tool_name || event.tool_name_raw || 'a tool');
      pending.set(id, { kind, toolName });
      if (card) cards.set(id, card);
      render();
      if (notified.has(id)) return;
      const note = notify(`osprey-attention-${id}`, requestText(kind, toolName), () =>
        focusCard(id)
      );
      if (note) {
        notified.add(id);
        open.set(id, note);
      }
    } else if (event.type === 'permission_resolved' && !replay) {
      const id = requestId(event);
      pending.delete(id);
      cards.delete(id);
      closeNote(id);
      render();
    }
  });

  const stopTurnEnd = onChat('turn-end', () => {
    // The renderer cancels any card still waiting when a turn ends.
    clearPending();
    notify(TURN_TAG, 'Osprey: the agent finished');
  });

  const stopReset = onChat('reset', clearPending);

  return () => {
    stopFrames();
    stopTurnEnd();
    stopReset();
    clearPending();
    title.dispose();
    bell.button.remove();
  };
}
