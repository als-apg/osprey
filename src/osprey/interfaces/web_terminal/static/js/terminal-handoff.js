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
 * The module owns the overlay's DOM, copy and clock, and nothing else:
 * connecting, retrying and interrupting are the caller's, handed in as
 * callbacks.
 */
const OVERLAY_ID = 'terminal-handoff';

/** When the current hand-off wait started, or null when none is showing. */
/** @type {number|null} */
let startedAt = null;

/** The 1 Hz elapsed-time ticker, or null when nothing is counting. */
/** @type {number|null} */
let ticker = null;

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
 * Show the transitional state: the other view's agent is finishing its turn
 * and this connection is waiting for it. There is no bound on that wait, so
 * the elapsed time is the honest thing to show, and the button is the
 * operator's way out of it.
 *
 * @param {() => void} onInterrupt - "Stop and switch now". The button
 *   disables itself before this runs, and a later `handoff_pending` re-enables
 *   it, so one press cannot become two.
 */
export function showHandoffPending(onInterrupt) {
  const overlay = ensureOverlay();
  if (!overlay) return;

  // A second `handoff_pending` — including the one an interrupt's reconnect
  // brings back — continues the wait the operator is already watching rather
  // than restarting its clock.
  if (startedAt === null) startedAt = Date.now();

  setMessage(overlay, 'Finishing in the other view · ', true);
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

/** Stop the elapsed counter and forget when the wait began. */
function stopTicker() {
  if (ticker !== null) {
    clearInterval(ticker);
    ticker = null;
  }
  startedAt = null;
}

/** Clear the overlay entirely. */
export function hideHandoffOverlay() {
  stopTicker();
  const overlay = document.getElementById(OVERLAY_ID);
  if (overlay) overlay.remove();
}
