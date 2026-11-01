// @ts-check
/* OSPREY Web Terminal — Watchdog Alert
 *
 * The operator's answer to a watchdog flag: one overlay listing every open flag
 * of the current session, with one Stop and one Continue. Expert and simple mode
 * use this same module.
 *
 * ONE truth: `GET /api/watchdog/flags`. A `watchdog_*` frame is a change notice
 * and only that; nothing is rendered from a frame's body except the per-watchdog
 * unavailable indicator, which no stored record describes. The flags are re-read
 * on load, on every event-stream open, on any `watchdog_*` frame, when a
 * `session_held` refusal lands in the conversation, and on a low-rate poll for as
 * long as the route has not answered `enabled: false`. The poll is what recovers
 * a frame the stream dropped.
 *
 * **Only a click on Stop or Continue answers.** This is not `showConfirm`, whose
 * stale sweep and dismissal belong to other dialogs: no Escape handler, no scrim
 * click, and no button takes focus on open, so an Enter the operator was already
 * typing cannot answer a dialog they have not read. The answer names the flag ids
 * on screen when it was clicked; a flag that joins later stays open and keeps the
 * dialog up.
 *
 * Nothing here presents a session as cleared: no flags means no dialog and no
 * statement about the session at all.
 */

import { apiRequest, fetchJSON } from './api.js';
import { setSessionSource } from './control-target-chip.js';
import { fadeOutOverlay, mountOverlay } from './modal-overlay.js';
import { getCurrentSessionId } from './terminal.js';

/**
 * One open flag, as `GET /api/watchdog/flags` reports it.
 * @typedef {object} WatchdogFlag
 * @property {string} flag_id
 * @property {string} watchdog
 * @property {string} reason
 * @property {string} prompt_excerpt
 * @property {boolean} stop_unconfirmed  a Stop was answered and the turn did not
 *   stop; the flag stays open until it is answered again
 * @property {string[]} started  what that turn left running
 */

/** The flag re-read's period. A dropped frame costs at most this long. */
export const WATCHDOG_POLL_MS = 10000;

/** The status a watchdog frame carries when it could not record its own flag. */
const STATUS_FLAG_WRITE_FAILED = 'flag_write_failed';

/** What `refusalLine` in chat-tool-cards.js renders for a refused held write. */
const HELD_REFUSAL_SELECTOR = '.op-tool-refusal[data-reason="session_held"]';

let started = false;

/** `null` until the route has answered; only `false` ends the poll. */
/** @type {boolean|null} */
let enabled = null;
/** @type {ReturnType<typeof setInterval>|null} */
let pollTimer = null;
/** Ticket per read, so a slow answer cannot overwrite a newer one. */
let readSeq = 0;

/** The session and flags on screen: exactly what a click answers. */
/** @type {{key: string, flags: WatchdogFlag[]}|null} */
let shown = null;
let answering = false;

/** @type {HTMLElement|null} */
let overlay = null;
/** @type {HTMLElement|null} */
let dialog = null;
/** @type {HTMLElement|null} */
let listEl = null;
/** @type {HTMLElement|null} */
let noteEl = null;
/** @type {HTMLButtonElement|null} */
let stopBtn = null;
/** @type {HTMLButtonElement|null} */
let continueBtn = null;
/** @type {Element|null} */
let focusBefore = null;
/** The rendered list's content, so an unchanged re-read leaves the DOM alone. */
let renderedList = '';

/** Unavailable watchdogs, by session, then by watchdog name. */
/** @type {Map<string, Map<string, string>>} */
const unavailable = new Map();
/** @type {HTMLElement|null} */
let statusEl = null;

/**
 * Start the alert: first read, poll, refusal watch and the chip's session source.
 * Idempotent.
 */
export function initWatchdogAlert() {
  if (started) return;
  started = true;
  setSessionSource(getCurrentSessionId);
  watchHeldRefusals();
  syncPoll();
  void refetchFlags();
}

export { isWatchdogFrame } from './control-target-chip.js';

/**
 * Take one `watchdog_*` frame off the event stream.
 * @param {any} frame
 */
export function onWatchdogFrame(frame) {
  if (frame.type === 'watchdog_status') noteStatus(frame);
  void refetchFlags();
}

/**
 * Re-read the session's open flags and repaint.
 * @returns {Promise<void>}
 */
export async function refetchFlags() {
  const key = getCurrentSessionId();
  renderStatus();
  if (!key) return;
  const seq = ++readSeq;
  try {
    const body = await fetchJSON(`/api/watchdog/flags?session_key=${encodeURIComponent(key)}`);
    if (seq !== readSeq) return;
    enabled = body.enabled === true;
    syncPoll();
    if (!answering) setNote('');
    render(key, Array.isArray(body.flags) ? body.flags : []);
  } catch (err) {
    if (seq !== readSeq) return;
    // Not knowing is not "no flags": what is on screen stays on screen.
    console.error('osprey web_terminal: could not read the watchdog flags', err);
    setNote('Could not refresh the flags from the server.');
  }
}

function syncPoll() {
  if (enabled === false) {
    if (pollTimer !== null) clearInterval(pollTimer);
    pollTimer = null;
  } else if (pollTimer === null) {
    pollTimer = setInterval(() => void refetchFlags(), WATCHDOG_POLL_MS);
  }
}

/**
 * A refused write reaches the page as a line in the conversation, marked with the
 * refusal's machine-readable reason. Only that attribute is read, never the phrase.
 */
function watchHeldRefusals() {
  new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (
          node instanceof Element &&
          (node.matches(HELD_REFUSAL_SELECTOR) || node.querySelector(HELD_REFUSAL_SELECTOR))
        ) {
          void refetchFlags();
          return;
        }
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
}

/* ---- the dialog ---- */

/**
 * @param {string} key
 * @param {WatchdogFlag[]} flags
 */
function render(key, flags) {
  shown = flags.length ? { key, flags } : null;
  if (!flags.length) {
    closeDialog();
    return;
  }
  if (!overlay || !overlay.isConnected) openDialog();
  const content = JSON.stringify(flags);
  if (content !== renderedList && listEl) {
    renderedList = content;
    listEl.replaceChildren(...flags.map(flagItem));
  }
}

/**
 * @param {string} tag
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function elem(tag, className, text) {
  const el = document.createElement(tag);
  el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

/**
 * @param {WatchdogFlag} flag
 * @returns {HTMLElement}
 */
function flagItem(flag) {
  const item = elem('li', 'watchdog-alert-flag');
  item.dataset.flagId = flag.flag_id;
  item.append(elem('div', 'watchdog-alert-name', flag.watchdog));
  if (flag.prompt_excerpt) {
    item.append(elem('blockquote', 'watchdog-alert-prompt', flag.prompt_excerpt));
  }
  item.append(elem('p', 'watchdog-alert-reason', `This looks risky: ${flag.reason}`));
  if (flag.stop_unconfirmed) {
    const running = flag.started?.length ? ` Still running: ${flag.started.join(', ')}.` : '';
    item.append(elem('p', 'watchdog-alert-unconfirmed', `The turn did not stop.${running}`));
  }
  return item;
}

function openDialog() {
  focusBefore = document.activeElement;
  overlay = elem('div', 'watchdog-alert-overlay');
  dialog = elem('div', 'watchdog-alert-dialog');
  dialog.setAttribute('role', 'alertdialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-labelledby', 'watchdog-alert-title');
  dialog.tabIndex = -1;

  const title = elem('h2', 'watchdog-alert-title', 'A watchdog flagged this session');
  title.id = 'watchdog-alert-title';
  listEl = elem('ul', 'watchdog-alert-list');
  renderedList = '';
  noteEl = elem('p', 'watchdog-alert-note');
  noteEl.hidden = true;
  noteEl.setAttribute('role', 'status');

  stopBtn = actionButton('Stop', 'watchdog-alert-stop', 'stop');
  continueBtn = actionButton('Continue', 'watchdog-alert-continue', 'continue');
  const actions = elem('div', 'watchdog-alert-actions');
  actions.append(stopBtn, continueBtn);

  dialog.append(title, listEl, noteEl, actions);
  dialog.addEventListener('keydown', keepTabInside);
  overlay.append(dialog);
  mountOverlay(overlay);
  dialog.focus();
}

/**
 * @param {string} label
 * @param {string} className
 * @param {'stop'|'continue'} answer
 * @returns {HTMLButtonElement}
 */
function actionButton(label, className, answer) {
  const button = /** @type {HTMLButtonElement} */ (
    elem('button', `watchdog-alert-button ${className}`, label)
  );
  button.type = 'button';
  button.addEventListener('click', () => void sendAnswer(answer));
  return button;
}

function closeDialog() {
  if (!overlay) return;
  const leaving = overlay;
  const back = focusBefore;
  overlay = dialog = listEl = noteEl = stopBtn = continueBtn = focusBefore = null;
  renderedList = '';
  fadeOutOverlay(leaving);
  if (back instanceof HTMLElement && back.isConnected) back.focus();
}

/** @param {KeyboardEvent} event */
function keepTabInside(event) {
  if (event.key !== 'Tab' || !stopBtn || !continueBtn) return;
  const buttons = [stopBtn, continueBtn].filter((b) => !b.disabled);
  event.preventDefault();
  if (!buttons.length) return;
  const at = buttons.indexOf(/** @type {HTMLButtonElement} */ (document.activeElement));
  const step = event.shiftKey ? buttons.length - 1 : 1;
  buttons[at === -1 ? 0 : (at + step) % buttons.length].focus();
}

/** @param {string} text */
function setNote(text) {
  if (!noteEl) return;
  noteEl.textContent = text;
  noteEl.hidden = !text;
}

/** @param {boolean} busy */
function setBusy(busy) {
  answering = busy;
  if (stopBtn) stopBtn.disabled = busy;
  if (continueBtn) continueBtn.disabled = busy;
}

/**
 * Answer the flags on screen. A Stop waits on the turn, which can take seconds,
 * so the buttons stay disabled until the route has answered.
 * @param {'stop'|'continue'} answer
 */
async function sendAnswer(answer) {
  if (answering || !shown) return;
  const { key, flags } = shown;
  setBusy(true);
  if (answer === 'stop') setNote('Stopping the turn…');
  try {
    await apiRequest('/api/watchdog/flags/answer', {
      json: {
        session_key: key,
        flag_ids: flags.map((flag) => flag.flag_id),
        answer,
      },
      errorPrefix: 'The answer was not taken',
    });
    setBusy(false);
    await refetchFlags();
  } catch (err) {
    setBusy(false);
    setNote(err instanceof Error ? err.message : String(err));
  }
}

/* ---- the unavailable indicator ---- */

/** @param {any} frame */
function noteStatus(frame) {
  if (typeof frame.session_key !== 'string' || typeof frame.watchdog !== 'string') return;
  let names = unavailable.get(frame.session_key);
  if (!names) {
    names = new Map();
    unavailable.set(frame.session_key, names);
  }
  names.set(
    frame.watchdog,
    frame.status === STATUS_FLAG_WRITE_FAILED ? 'flag not recorded' : 'unavailable'
  );
}

/** Paint the current session's unavailable watchdogs, one line each. */
function renderStatus() {
  const key = getCurrentSessionId();
  const names = key ? unavailable.get(key) : undefined;
  if (!key || !names?.size) {
    statusEl?.remove();
    statusEl = null;
    return;
  }
  if (!statusEl || !statusEl.isConnected) {
    statusEl = elem('div', 'watchdog-status');
    statusEl.setAttribute('role', 'status');
    document.body.appendChild(statusEl);
  }
  statusEl.replaceChildren(
    ...[...names].map(([name, word]) => {
      const line = elem('div', 'watchdog-status-line');
      line.dataset.watchdog = name;
      const dismiss = elem('button', 'watchdog-status-dismiss', '×');
      dismiss.setAttribute('type', 'button');
      dismiss.setAttribute('aria-label', `Dismiss ${name}`);
      dismiss.addEventListener('click', () => {
        names.delete(name);
        renderStatus();
      });
      line.append(elem('span', 'watchdog-status-text', `Watchdog ${name}: ${word}`), dismiss);
      return line;
    })
  );
}
