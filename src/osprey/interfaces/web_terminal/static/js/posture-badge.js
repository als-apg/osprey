// @ts-check
/* OSPREY Web Terminal — Session Posture Badge
 *
 * The client half of the per-session runtime sandbox toggle. A session runs
 * either in `writes` posture (the agent may drive the control system, under
 * the deployment's usual approval/verification rules) or in `sandbox` posture
 * (the same agent, spawned with OSPREY_EXECUTION_MODE=readonly, which refuses
 * every write at the executor). The operator steps a live session between
 * them from the terminal card's header.
 *
 * ONE truth: the badge always renders what `GET /api/terminal/posture` says
 * and never what this module last did. The posture lives in a server-side
 * store that survives a container restart and is shared by every tab, so a
 * badge that trusted its own last POST would happily show `writes` for a
 * session another tab had just sandboxed. Every mutation therefore ends in a
 * re-read, and so does every refusal.
 *
 * Both directions confirm, every time, with no remembered acknowledgment —
 * unlike the settings drawer's once-per-server-session warning. Two reasons:
 * the toggle TERMINATES the session's current turn (the server has to respawn
 * the PTY for the new posture to reach the agent's environment), and the
 * sandbox direction is the one an operator most needs to not perform by
 * accident mid-incident. A dialog that can be dismissed forever is a dialog
 * that stops being read.
 *
 * The reconnect is not optional. Storing a posture terminates the session
 * server-side; without `startTerminal(sessionId, 'resume')` the card is left
 * attached to a dead PTY, and the operator's history — which the resume is
 * what preserves — looks lost. `stopTerminal()` comes first because
 * startTerminal() no-ops while a socket is still assigned.
 *
 * Both topologies use this unchanged: single-user and multi-user serve the
 * same terminal card, and every request goes through api.js's withPrefix
 * chokepoint, so the per-user mount (`/u/<name>/…`) is already handled.
 *
 * The badge carries a second line: the control target this session is on, and
 * whether THAT target is armed for writes. `rendered_writes_enabled` cannot
 * say — it is a union over every target, true as soon as one is armed — so on
 * a render that arms the virtual accelerator alone it reads "writes" to an
 * operator sitting on a live machine every write to which is refused. The
 * server answers both facts (routes/websocket.py); this module only shows
 * them, and marks a value as the deployment baseline when no session target
 * has been published yet.
 */

import { withPrefix } from './api.js';
import { mountOverlay, fadeOutOverlay } from './modal-overlay.js';
import { getCurrentSessionId, onSessionChange, startTerminal, stopTerminal } from './terminal.js';

/**
 * The payload of `GET /api/terminal/posture`.
 * @typedef {object} PostureState
 * @property {string} session_id
 * @property {'sandbox'|'writes'} posture  what this session runs as now
 * @property {boolean} rendered_writes_enabled  whether the RENDER arms writes
 *   for SOME control target — write posture is per connector type, so this is
 *   true as soon as one target is armed and says nothing about which. False
 *   means no target is armed, which makes the `writes` direction unavailable
 *   rather than merely refused (see renderBadge).
 * @property {string|null} [session_target]  the control target this session is
 *   pointed at ('live' / 'va'), or the deployment baseline when none has been
 *   published yet. Absent from an older server — the line then renders nothing.
 * @property {boolean} [target_writes_enabled]  whether writes are armed for
 *   THAT target specifically, which is the fact `rendered_writes_enabled` is
 *   unable to carry.
 * @property {'session'|'baseline'} [target_source]  'session' when the server
 *   matched a live control-target record, 'baseline' when the value is the
 *   deployment's default standing in for one.
 */

/** @type {HTMLButtonElement|null} */
let badge = null;
/** @type {HTMLElement|null} */
let badgeText = null;
/** @type {HTMLElement|null} */
let badgeTarget = null;
/** @type {PostureState|null} */
let state = null;
/** Whether a confirm dialog is up — one at a time. */
let modalOpen = false;

/**
 * How often the badge re-reads while a session is attached.
 *
 * A control-target switch happens inside the agent's turn: it fires no session
 * change and no toggle, which are the only two things that refreshed this
 * badge before. Without a poll the target line would keep naming the target
 * the operator LEFT — the one failure this whole line exists to prevent. Five
 * seconds is a cheap read of a store the server already has in memory plus one
 * directory glob, and it stops the moment the badge is detached or the card
 * has no session.
 */
const POSTURE_POLL_MS = 5000;

/** @type {ReturnType<typeof setInterval>|null} */
let pollTimer = null;

/**
 * Monotonic id for reads, so a slow one cannot overwrite a newer answer.
 *
 * The session-id guard in refreshPostureBadge does not cover this: the id is
 * unchanged. A poll tick fires a GET, the operator then confirms a toggle, the
 * POST commits and its own re-read lands — and the tick's older GET resolves
 * afterwards carrying the PRE-toggle posture, repainting the badge as `Writes`
 * on a session that is now sandboxed. Every read takes a ticket here and drops
 * its answer if a later read has since been issued.
 */
let readSeq = 0;

/** Whether a posture POST is committing right now (see applyPosture). */
let posting = false;

/**
 * Mount the badge into the terminal card header and keep it current.
 *
 * Idempotent: a second call re-renders rather than mounting a second badge.
 * Safe to call before any session exists — the badge stays hidden until the
 * terminal reports one.
 */
export function initPostureBadge() {
  const header = document.querySelector('.terminal-header');
  if (!header) return;

  if (!badge || !header.contains(badge)) {
    badge = buildBadge();
    // Left of the session selector: the posture is part of the session's
    // identity, not one of the actions ("+ New" and friends) that live at
    // the far end of the bar.
    const selector = header.querySelector('#session-selector');
    header.insertBefore(badge, selector);
    // The id is confirmed asynchronously (a `session_info` frame, a session
    // switch, or the resume-liveness timer), and every one of those paths
    // funnels through terminal.js's notifySessionChange — so this is the one
    // subscription needed to stay pointed at the right session.
    onSessionChange(() => {
      void refreshPostureBadge();
    });
  }
  void refreshPostureBadge();
}

/**
 * Re-read the posture for the current session and repaint.
 *
 * Called on mount, on every session change, and after every toggle attempt
 * (successful or refused).
 * @returns {Promise<void>}
 */
async function refreshPostureBadge() {
  if (!badge) return;
  const sessionId = getCurrentSessionId();
  if (!sessionId) {
    stopPolling();
    state = null;
    renderBadge();
    return;
  }
  startPolling();
  const seq = ++readSeq;
  try {
    const data = await postureRequest(
      `/api/terminal/posture?session_id=${encodeURIComponent(sessionId)}`
    );
    // The session can change while the read is in flight (a switch, a
    // failover resume); a stale answer must never repaint the badge.
    if (getCurrentSessionId() !== sessionId) return;
    // A newer read has been issued since this one left — most importantly the
    // re-read a toggle does — so this answer is already history.
    if (seq !== readSeq) return;
    state = data;
  } catch (err) {
    // Same stale guards as the success path: a read that failed for a session
    // this card has already left, or that a newer read has superseded, must
    // not blank the badge.
    if (getCurrentSessionId() !== sessionId || seq !== readSeq) return;
    // Unknown beats wrong: a badge that cannot read the posture says
    // nothing rather than implying a posture nobody confirmed.
    console.error('osprey web_terminal: could not read the session posture', err);
    state = null;
  }
  renderBadge();
}

/**
 * Start the refresh timer, if it is not already running.
 *
 * The tick stops itself once there is nothing left to refresh: no session on
 * the card, or a badge that is no longer in the document (the card was torn
 * down). Checking the DOM rather than trusting a teardown call means no code
 * path can leak a timer that outlives the badge it paints.
 */
function startPolling() {
  if (pollTimer !== null) return;
  pollTimer = setInterval(() => {
    if (!badge || !badge.isConnected || !getCurrentSessionId()) {
      stopPolling();
      return;
    }
    // Never read across a decision the operator is in the middle of making.
    // A confirm dialog is up, or a POST is committing: the toggle re-reads on
    // its own the moment it lands, and a tick started now would only race it.
    if (modalOpen || posting) return;
    void refreshPostureBadge();
  }, POSTURE_POLL_MS);
}

/** Stop the refresh timer. Idempotent. */
function stopPolling() {
  if (pollTimer === null) return;
  clearInterval(pollTimer);
  pollTimer = null;
}

/**
 * One request path for both verbs on `/api/terminal/posture`, owning its own
 * error contract.
 *
 * Not api.js's `apiRequest`: these routes raise `HTTPException` with a DICT
 * detail (`{error, message}` — see routes/websocket.py), and `apiRequest`
 * builds its Error from `detail.detail` directly, which stringifies an object
 * to "[object Object]". That is precisely the wording the operator most needs
 * — the 409's "send one prompt first" and the 403's explanation of a
 * writes-off render — so this module reads the body itself and unwraps the
 * message. GET and POST share the path, so both verbs fail the same way.
 * @param {string} path
 * @param {{method?: string, json?: any}} [opts]
 * @returns {Promise<any>} the parsed body
 * @throws {Error} on a non-OK response, carrying the server's own message
 *   where it sent one
 */
async function postureRequest(path, { method = 'GET', json } = {}) {
  /** @type {RequestInit} */
  const init = { method, cache: 'no-store' };
  if (json !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(json);
  }
  const resp = await fetch(withPrefix(path), init);
  const body = await resp.json().catch(() => null);
  if (!resp.ok) {
    throw new Error(refusalMessage(body, resp.status));
  }
  return body;
}

/**
 * The most specific human sentence a refusal body carries.
 *
 * Handles all three shapes rather than guessing at one: FastAPI's dict detail
 * (`{detail: {error, message}}`, what these routes raise), the plain string
 * detail every other route in this app uses, and a body with neither — where
 * the status code is all there is to say.
 * @param {any} body  parsed response body, or null when it was not JSON
 * @param {number} status
 * @returns {string}
 */
function refusalMessage(body, status) {
  const detail = body && typeof body === 'object' ? body.detail : null;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (detail && typeof detail === 'object' && typeof detail.message === 'string') {
    if (detail.message.trim()) return detail.message;
  }
  if (body && typeof body === 'object' && typeof body.message === 'string' && body.message.trim()) {
    return body.message;
  }
  return `Could not switch this session (HTTP ${status}).`;
}

/** @returns {HTMLButtonElement} */
function buildBadge() {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'posture-badge';
  el.id = 'posture-badge';
  el.hidden = true;
  const dot = document.createElement('span');
  dot.className = 'posture-badge-dot';
  dot.setAttribute('aria-hidden', 'true');
  badgeText = document.createElement('span');
  badgeText.className = 'posture-badge-text';
  badgeTarget = document.createElement('span');
  badgeTarget.className = 'posture-badge-target';
  badgeTarget.hidden = true;
  el.appendChild(dot);
  el.appendChild(badgeText);
  el.appendChild(badgeTarget);
  el.addEventListener('click', onBadgeClick);
  return el;
}

/** Paint the badge from `state` (null = nothing known yet). */
function renderBadge() {
  if (!badge || !badgeText) return;
  if (!state) {
    badge.hidden = true;
    return;
  }
  const sandbox = state.posture === 'sandbox';
  const writesAvailable = state.rendered_writes_enabled !== false;
  badge.hidden = false;
  badge.dataset.posture = sandbox ? 'sandbox' : 'writes';
  badge.dataset.writesAvailable = String(writesAvailable);
  badgeText.textContent = sandbox ? 'Sandbox' : 'Writes';

  // A sandboxed session on a render that has no writes has nowhere to go:
  // the only move available is the one the render forbids. Say so on the
  // badge rather than offering a button whose POST is guaranteed to 403.
  //
  // Deliberately NOT widened to the current target's posture: an operator may
  // step out of the sandbox and only then switch to the target they mean to
  // write on, and refusing that would be a button that lies in the other
  // direction. The target line below carries that fact instead.
  const stuck = sandbox && !writesAvailable;
  badge.disabled = stuck;
  const title = stuck
    ? 'Sandbox — this session refuses every control-system write. Writes cannot be ' +
      'enabled here: this deployment is rendered with writes off.'
    : sandbox
      ? 'Sandbox — this session refuses every control-system write. Click to allow writes again.'
      : 'Writes — this session can drive the control system. Click to sandbox it.';
  const targetTitle = renderTargetLine();
  badge.title = targetTitle ? `${title} ${targetTitle}` : title;
  badge.setAttribute(
    'aria-label',
    `Session posture: ${title}${targetTitle ? ` ${targetTitle}` : ''}`
  );
}

/**
 * Paint the control-target line and return the sentence it stands for.
 *
 * Two facts, in the fewest words that cannot be misread: which target, and
 * whether that target is armed. "not armed" is spelled out rather than left to
 * a colour or a missing word — it is the case an operator must not skim past —
 * and a baseline fallback is labelled as one so a default is never mistaken
 * for a published target.
 * @returns {string} the long form, for the tooltip and the accessible name;
 *   empty when the server reported no target (an older server, or a render it
 *   could not read).
 */
function renderTargetLine() {
  if (!badge || !badgeTarget) return '';
  const target = state?.session_target;
  if (!target) {
    badgeTarget.hidden = true;
    badgeTarget.textContent = '';
    delete badge.dataset.target;
    delete badge.dataset.targetArmed;
    delete badge.dataset.targetSource;
    return '';
  }
  const armed = state?.target_writes_enabled === true;
  const baseline = state?.target_source !== 'session';

  badgeTarget.hidden = false;
  badgeTarget.textContent = `${target}${baseline ? ' (baseline)' : ''} · ${
    armed ? 'armed' : 'not armed'
  }`;
  badge.dataset.target = target;
  badge.dataset.targetArmed = String(armed);
  badge.dataset.targetSource = baseline ? 'baseline' : 'session';

  const posture = armed
    ? `writes are armed for the ${target} target.`
    : `writes are NOT armed for the ${target} target — every write to it is refused.`;
  // Says what is KNOWN, not why. `baseline` covers "no controls server has
  // published a target yet" and "one was published but could not be resolved"
  // alike — a record another session owns, two ambiguous records, a dead
  // server, an unreadable process table. Asserting the first cause for all of
  // them would tell an operator who HAS switched that nothing was published,
  // which is the same confident-but-wrong reading this line exists to end.
  const provenance = baseline
    ? ' This is the deployment baseline; no control target has been resolved' +
      ' for this session.'
    : '';
  return `Control target: ${posture}${provenance}`;
}

function onBadgeClick() {
  if (!state || !badge || badge.disabled || modalOpen) return;
  const sessionId = getCurrentSessionId();
  if (!sessionId) return;
  /** @type {'sandbox'|'writes'} */
  const target = state.posture === 'sandbox' ? 'writes' : 'sandbox';
  if (target === 'writes' && state.rendered_writes_enabled === false) return;
  showConfirmModal(target, sessionId);
}

/**
 * The consequences an operator must agree to before either direction. Both
 * are properties of the RESPAWN, so both directions say the same thing —
 * and both sentences are load-bearing: the first is the cost, the second is
 * the reassurance that keeps the first from reading as "you lose the thread".
 * @param {'sandbox'|'writes'} target
 * @returns {{ title: string, lead: string, confirmLabel: string }}
 */
function copyFor(target) {
  if (target === 'sandbox') {
    return {
      title: 'Sandbox this session?',
      lead:
        'The agent keeps its full view of the control system and the project, and every ' +
        'write it attempts is refused for as long as this session stays sandboxed.',
      confirmLabel: 'Sandbox session',
    };
  }
  return {
    title: 'Allow writes for this session?',
    lead:
      'The agent can drive the control system from this session again, under this ' +
      "deployment's usual approval and write-verification rules.",
    confirmLabel: 'Allow writes',
  };
}

/**
 * Build and show the confirm dialog for one direction.
 *
 * Structure and lifecycle deliberately mirror the settings drawer's warning
 * dialog (settings.js): overlay appended to body, `.visible` added on the
 * next frame, Escape cancels, and one `cleanup()` shared by every dismissal
 * path so no listener outlives the dialog. What differs is the acknowledgment
 * — there is none — and that a refusal keeps the dialog up to carry the
 * server's reason.
 * @param {'sandbox'|'writes'} target
 * @param {string} sessionId  the session the dialog's copy describes; the
 *   confirm POSTs for THIS id, never for whatever is current when the button
 *   is finally clicked. A switch made while the dialog was up would otherwise
 *   silently retarget it (and the server would 409 a session that no longer
 *   matches what the operator read).
 */
function showConfirmModal(target, sessionId) {
  const { title, lead, confirmLabel } = copyFor(target);

  // A dialog dismissed a moment ago is still in the DOM, fading out. Drop it
  // now rather than stacking a second overlay on top of it.
  for (const stale of document.querySelectorAll('.posture-modal-overlay[data-closing]')) {
    stale.remove();
  }

  const overlay = document.createElement('div');
  overlay.className = 'posture-modal-overlay';

  const dialog = document.createElement('div');
  dialog.className = 'posture-modal';
  dialog.setAttribute('role', 'dialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-labelledby', 'posture-modal-title');

  const heading = document.createElement('div');
  heading.className = 'posture-modal-title';
  heading.id = 'posture-modal-title';
  heading.textContent = title;

  const body = document.createElement('div');
  body.className = 'posture-modal-body';
  const p1 = document.createElement('p');
  p1.textContent = lead;
  const p2 = document.createElement('p');
  // Assembled node by node (no innerHTML) so the emphasis can sit on the two
  // phrases that matter without any string ever being parsed as markup.
  p2.append(
    'Switching restarts the agent for this session, so ',
    strong('the current turn is terminated'),
    ' — whatever it is doing right now stops immediately. ',
    strong('Your conversation history is preserved'),
    ': the terminal reconnects to the same session and you can pick up where you left off.'
  );
  body.append(p1, p2);

  const error = document.createElement('div');
  error.className = 'posture-modal-error';
  error.setAttribute('role', 'alert');
  error.hidden = true;

  const actions = document.createElement('div');
  actions.className = 'posture-modal-actions';
  const cancelBtn = document.createElement('button');
  cancelBtn.type = 'button';
  cancelBtn.className = 'posture-modal-cancel';
  cancelBtn.textContent = 'Cancel';
  const confirmBtn = document.createElement('button');
  confirmBtn.type = 'button';
  confirmBtn.className = 'posture-modal-confirm';
  confirmBtn.textContent = confirmLabel;
  actions.append(cancelBtn, confirmBtn);

  dialog.append(heading, body, error, actions);
  overlay.appendChild(dialog);
  mountOverlay(overlay);

  /** Runs on every dismissal path — click, Escape, success. */
  const cleanup = () => {
    document.removeEventListener('keydown', onKey);
    modalOpen = false;
    // Dismissed the instant it is asked for; the node lingers only for the
    // fade, and `data-closing` is what tells the difference between "still
    // up" and "on its way out" (to a reader, to a test, and to the stale
    // sweep above).
    overlay.dataset.closing = '1';
    fadeOutOverlay(overlay);
  };

  /** @param {KeyboardEvent} e */
  const onKey = (e) => {
    if (e.key !== 'Escape') return;
    // A POST is in flight (both buttons are disabled for it): the toggle is
    // already committed server-side or about to be, so Escape must not tear
    // down the dialog that is waiting to report what happened.
    if (confirmBtn.disabled) return;
    cleanup();
  };
  document.addEventListener('keydown', onKey);
  cancelBtn.addEventListener('click', cleanup);

  confirmBtn.addEventListener('click', () => {
    void applyPosture(target, sessionId, { confirmBtn, cancelBtn, error, cleanup });
  });

  modalOpen = true;
  confirmBtn.focus();
}

/** @param {string} text @returns {HTMLElement} */
function strong(text) {
  const el = document.createElement('strong');
  el.textContent = text;
  return el;
}

/**
 * POST the toggle, then either reconnect or explain the refusal.
 *
 * A refusal (403 on a writes-off render, 409 for a session the server has
 * never seen — "send one prompt first") keeps the dialog up carrying the
 * server's own detail: it is the only place the operator is looking, and the
 * server's wording is more specific than anything this module could invent.
 * The terminal is left completely untouched on that path — nothing was
 * terminated, so nothing needs resuming.
 * @param {'sandbox'|'writes'} target
 * @param {string} sessionId  captured when the dialog opened (see
 *   showConfirmModal) — the session the operator was actually shown
 * @param {{confirmBtn: HTMLButtonElement, cancelBtn: HTMLButtonElement,
 *          error: HTMLElement, cleanup: () => void}} ui
 */
async function applyPosture(target, sessionId, { confirmBtn, cancelBtn, error, cleanup }) {
  confirmBtn.disabled = true;
  cancelBtn.disabled = true;
  posting = true;
  try {
    await postureRequest('/api/terminal/posture', {
      method: 'POST',
      json: { session_id: sessionId, posture: target },
    });
  } catch (err) {
    posting = false;
    showModalError(error, err instanceof Error ? err.message : String(err));
    confirmBtn.disabled = false;
    cancelBtn.disabled = false;
    // The refusal may itself be news about the render (a 403 means writes are
    // off), so re-read rather than keeping whatever the badge showed.
    await refreshPostureBadge();
    return;
  }
  posting = false;
  cleanup();
  // The POST terminated the session to respawn it under the new posture;
  // resume reattaches this card to the same conversation.
  stopTerminal();
  startTerminal(sessionId, 'resume');
  await refreshPostureBadge();
}

/** @param {HTMLElement} error @param {string} message */
function showModalError(error, message) {
  error.textContent = message;
  error.hidden = false;
}
