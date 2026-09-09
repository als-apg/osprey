/* OSPREY Web Terminal — Session Hand-off Overlay */

import { WS_CLOSE_OUTGOING_RUNNING } from './api.js';

/**
 * The transitional state shown on the terminal card while the session is
 * being handed over from the other view, and the place a refused connection
 * says why.
 *
 * Both views are windows onto one session and only one of them may run its
 * agent at a time, so taking it over is a negotiation rather than a connect:
 * the outgoing agent finishes the turn it is on — never ended on a clock —
 * and this is what the operator watches while it does. The overlay is built
 * on demand and removed when it clears, so the card carries no extra DOM in
 * the ordinary case.
 *
 * The pending state has two faces. An outgoing agent that is idle is only
 * stopped in its view and started in this one, which takes about a second:
 * that is a restart, and it is shown as one — no clock, no way out, because
 * there is nothing to cut short. An outgoing agent that is mid-turn is
 * waited on for as long as the turn takes: that is the wait, with its
 * elapsed clock and "Stop and switch now". The server says which on the
 * frame; a restart that outlasts {@link HANDOFF_RESTART_BUDGET_MS} becomes
 * the wait on its own, and the wait never steps back to a restart.
 *
 * The module owns the overlay's DOM, copy and clock, and nothing else:
 * connecting, retrying and interrupting are the caller's, handed in as
 * callbacks.
 */
const OVERLAY_ID = 'terminal-handoff';

/** Copy shown while the outgoing view finishes the turn it is on. */
export const HANDOFF_PENDING_MESSAGE = 'Finishing in the other view · ';

/** Copy shown while an idle agent is stopped in the other view and started here. */
export const HANDOFF_RESTARTING_MESSAGE = 'Restarting the agent in this view…';

/**
 * How long a restart may take before it is shown as a wait. A restart is the
 * outgoing process exiting cleanly plus the incoming one connecting, a few
 * seconds at most; past that the operator is waiting on something, and the
 * wait's clock and way out are the honest thing to show.
 */
export const HANDOFF_RESTART_BUDGET_MS = 4000;

/** When the current hand-off began, or null when none is showing. */
/** @type {number|null} */
let startedAt = null;

/** The 1 Hz elapsed-time ticker, or null when nothing is counting. */
/** @type {number|null} */
let ticker = null;

/** The timer that turns a long restart into the wait, or null when none is armed. */
/** @type {number|null} */
let escalation = null;

/** Whether the wait is on screen; it never steps back to a restart. */
let waiting = false;

/**
 * Elapsed wait as `m:ss`.
 * @param {number} ms
 * @returns {string}
 */
function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/**
 * The overlay element, building it into the terminal body on first use.
 * Returns null where there is no terminal body to mount on.
 * @returns {HTMLElement|null}
 */
function ensureOverlay() {
  const existing = document.getElementById(OVERLAY_ID);
  if (existing) return existing;

  const body = document.querySelector('.terminal-body');
  if (!body) return null;

  const overlay = document.createElement('div');
  overlay.id = OVERLAY_ID;
  overlay.className = 'terminal-handoff';
  // A state the operator is waiting on rather than one they opened: announced
  // when it appears, never focus-trapped.
  overlay.setAttribute('role', 'status');
  overlay.setAttribute('aria-live', 'polite');

  const card = document.createElement('div');
  card.className = 'terminal-handoff-card';

  const message = document.createElement('p');
  message.className = 'terminal-handoff-message';

  const action = document.createElement('button');
  action.type = 'button';
  action.className = 'terminal-handoff-action';

  card.append(message, action);
  overlay.append(card);
  body.append(overlay);
  return overlay;
}

/**
 * Write the overlay's one line. `withElapsed` appends the live counter that
 * {@link renderElapsed} keeps current.
 * @param {HTMLElement} overlay
 * @param {string} text
 * @param {boolean} withElapsed
 */
function setMessage(overlay, text, withElapsed) {
  const message = overlay.querySelector('.terminal-handoff-message');
  if (!message) return;
  message.textContent = text;
  if (!withElapsed) return;
  const elapsed = document.createElement('span');
  elapsed.className = 'terminal-handoff-elapsed';
  // Inside the live region, but not part of what it announces: a counter
  // ticking once a second would be read out once a second, for as long as the
  // wait lasts. The sentence around it is the announcement.
  elapsed.setAttribute('aria-live', 'off');
  message.append(elapsed);
}

/** Update the elapsed counter in place. */
function renderElapsed() {
  const elapsed = document.querySelector('.terminal-handoff-elapsed');
  if (!elapsed || startedAt === null) return;
  elapsed.textContent = formatElapsed(Date.now() - startedAt);
}

/** The overlay's action button, or null when the overlay is not mounted. */
function actionButton() {
  return /** @type {HTMLButtonElement|null} */ (
    document.querySelector('.terminal-handoff-action')
  );
}

/**
 * Show the transitional state. With `busy` the other view's agent is
 * finishing its turn and this connection is waiting for it: there is no bound
 * on that wait, so the elapsed time is the honest thing to show, and the
 * button is the operator's way out of it. Without it the agent is only being
 * restarted here, and that is what is shown — until the restart outlasts its
 * budget, when the wait takes over.
 *
 * @param {() => void} onInterrupt - "Stop and switch now". The button
 *   disables itself before this runs, and a later `handoff_pending` re-enables
 *   it, so one press cannot become two.
 * @param {{ busy?: boolean }} [state] - What the server said about the other
 *   view's agent. Omitted, it is taken to be idle.
 */
export function showHandoffPending(onInterrupt, state = {}) {
  const overlay = ensureOverlay();
  if (!overlay) return;

  // A second `handoff_pending` — including the one an interrupt's reconnect
  // brings back — continues the hand-off the operator is already watching
  // rather than restarting its clock.
  if (startedAt === null) startedAt = Date.now();

  if (state.busy || waiting) {
    showWait(overlay, onInterrupt);
    return;
  }
  setMessage(overlay, HANDOFF_RESTARTING_MESSAGE, false);
  const action = actionButton();
  if (action) {
    action.hidden = true;
    action.onclick = null;
  }
  if (escalation === null) {
    escalation = window.setTimeout(() => {
      escalation = null;
      showWait(overlay, onInterrupt);
    }, HANDOFF_RESTART_BUDGET_MS);
  }
}

/**
 * The wait: the elapsed clock, counted from when the hand-off began, and the
 * way out of it.
 * @param {HTMLElement} overlay
 * @param {() => void} onInterrupt
 */
function showWait(overlay, onInterrupt) {
  waiting = true;
  disarmEscalation();
  setMessage(overlay, HANDOFF_PENDING_MESSAGE, true);
  renderElapsed();
  if (ticker === null) ticker = window.setInterval(renderElapsed, 1000);

  const action = actionButton();
  if (!action) return;
  action.textContent = 'Stop and switch now';
  action.hidden = false;
  action.disabled = false;
  action.onclick = () => {
    action.disabled = true;
    onInterrupt();
  };
  // Moves the caret out of the terminal, which takes no input while the
  // hand-off is pending.
  action.focus();
}

/** Cancel a pending restart-to-wait escalation. */
function disarmEscalation() {
  if (escalation !== null) {
    clearTimeout(escalation);
    escalation = null;
  }
}

/**
 * Show why the server refused this connection. Both refusals are final for
 * the socket that received them (see `isRefusalCloseCode` in api.js): one
 * says the session is held elsewhere, which retrying does not change, and one
 * says the outgoing agent had not finished dying, which is exactly what a
 * retry is for.
 *
 * @param {number} code - WebSocket close code.
 * @param {() => void} onRetry - Runs when the operator retries, which only
 *   the retryable refusal offers. The overlay clears first.
 */
export function showHandoffRefused(code, onRetry) {
  stopTicker();
  const overlay = ensureOverlay();
  if (!overlay) return;

  const retryable = code === WS_CLOSE_OUTGOING_RUNNING;
  setMessage(
    overlay,
    retryable
      ? 'The previous agent is still shutting down.'
      : 'This session is in use in another tab or view.',
    false,
  );

  const action = actionButton();
  if (!action) return;
  action.textContent = 'Retry';
  action.hidden = !retryable;
  action.disabled = false;
  action.onclick = retryable
    ? () => {
      hideHandoffOverlay();
      onRetry();
    }
    : null;
}

/** Stop the clock and the escalation, and forget when the hand-off began. */
function stopTicker() {
  if (ticker !== null) {
    clearInterval(ticker);
    ticker = null;
  }
  disarmEscalation();
  startedAt = null;
  waiting = false;
}

/** Clear the overlay entirely. */
export function hideHandoffOverlay() {
  stopTicker();
  const overlay = document.getElementById(OVERLAY_ID);
  if (overlay) overlay.remove();
}
