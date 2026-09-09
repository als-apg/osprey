// @ts-check
/* OSPREY Web Terminal — Simple-mode operator chat controller.
 *
 * Thin glue between the SSE transport (chat-client.js) and the message-list
 * renderer (chat-render.js). It builds the operator console DOM inside
 * #operator-container, binds it to the tab's session pointer, and owns the
 * streaming/idle UI state, turning user intent (submit, stop, switch view)
 * into transport calls. All rendering and sanitisation live in the renderer;
 * all networking and SSE parsing live in the client — this file holds no
 * innerHTML and no parsing, so the interesting logic stays in the two
 * unit-tested modules.
 *
 * The console does not mint a conversation of its own. Both views are windows
 * onto one session key, held in the browser by session-pointer.js, and this
 * console addresses whatever key the pointer names: it replays that key's
 * transcript when it binds, and a change of pointer rebinds it. Only one view
 * may run the agent at a time, so arriving from the Expert view is a
 * negotiation rather than a connect — see {@link enterFromExpert}.
 *
 * Mode visibility is pure CSS: #operator-container is display-gated off
 * html[data-ui-mode] (operator.css), so the console is built once at boot and
 * simply hidden in expert mode — never torn down or toggled from here.
 */

import { fetchHistory, interrupt, requestHandoff, sendPrompt } from './chat-client.js';
import { createChatRenderer, elem } from './chat-render.js';
import { buildEmptyState, onKindChange, onSettled, renderEmptyStateContent } from './first-contact.js';
import { getPointer, setPointer, subscribe as subscribeToPointer } from './session-pointer.js';
import { notifySessionChange } from './terminal.js';
import {
  HANDOFF_PENDING_MESSAGE,
  HANDOFF_RESTART_BUDGET_MS,
  HANDOFF_RESTARTING_MESSAGE,
} from './terminal-handoff.js';

/** Max textarea height (px) before it scrolls — matches operator.css. */
const MAX_INPUT_HEIGHT = 120;

/** Distance (px) from the bottom within which the log auto-follows new output. */
const STICK_THRESHOLD = 40;

/**
 * User-facing copy keyed on the endpoint's machine-readable `detail.error`
 * slug — the primary table, because the status alone cannot separate two
 * rejections that share one. `POST /api/chat` answers 409 both when a turn is
 * already streaming and when the chat was torn down mid-start by a posture
 * flip; only the second one has a remedy, and it is the operator resending the
 * prompt they just sent.
 * @type {Record<string, string>}
 */
const TRANSPORT_NOTICES_BY_SLUG = {
  chat_terminated: 'Session restarted with the new posture — send your message again.',
  turn_in_progress: 'A turn is already running.',
  chat_capacity: 'Server busy — please retry in a moment.',
};

/**
 * Status-keyed fallback for a rejection that carries no slug — a proxy's own
 * 503, say, or any error shape older than the slug contract. These are
 * distinct from server `error` events (which the renderer paints as red error
 * blocks); a transport failure renders as a centred system notice instead.
 * @type {Record<number, string>}
 */
const TRANSPORT_NOTICES = {
  409: 'A turn is already running.',
  429: 'Server busy — please retry in a moment.',
  503: 'Operator agent unavailable.',
};

/** Fallback copy for a transport failure with no recognised slug or status. */
const TRANSPORT_FALLBACK = 'Connection to the operator agent failed.';

/**
 * What the operator may do about a refused hand-off, keyed on the endpoint's
 * `detail.error` slug. A refusal is not a transport failure: it is the other
 * view still holding the session, so it belongs on the overlay that already
 * has the operator's attention rather than as a line in a log they cannot use.
 *
 * `interrupt` re-sends the same request with the operator's consent to cut a
 * running turn; `retry` re-sends it unchanged; a null action is a refusal
 * retrying does not change, and states the fact alone.
 * @type {Record<string, { message: string, action: 'interrupt'|'retry'|null }>}
 */
const HANDOFF_REFUSALS = {
  session_attached_elsewhere: {
    message: 'This session is in use in another tab or view.',
    action: null,
  },
  handoff_needs_interrupt: {
    message: 'The other view may still be working.',
    action: 'interrupt',
  },
  handoff_superseded: { message: 'Another request took over this session.', action: 'retry' },
  outgoing_still_running: {
    message: 'The previous agent is still shutting down.',
    action: 'retry',
  },
  outgoing_vanished: { message: 'The session changed while switching.', action: 'retry' },
  spawn_not_pooled: { message: 'The session did not start.', action: 'retry' },
  chat_capacity: { message: 'No chat capacity right now.', action: 'retry' },
  sdk_unavailable: { message: 'Operator agent unavailable.', action: null },
};

// `chat_terminated` is deliberately absent above. This table is consulted
// before the notice tables, and that slug's one useful answer is the notice
// it already has: the prompt the operator just sent needs sending again. A
// hand-off refused with it falls to HANDOFF_FALLBACK, which offers the same
// retry the overlay would have given it.

/** Overlay copy for a hand-off that failed with no reason this view knows. */
const HANDOFF_FALLBACK = { message: 'The hand-off failed.', action: /** @type {'retry'} */ ('retry') };

/**
 * Build the operator console interior — session bar, message list, hand-off
 * overlay, and the command-line input row — and return the handles the
 * controller drives.
 */
function buildConsole() {
  const bar = elem('div', 'op-session-bar');
  const led = elem('span', 'op-session-led');
  const newBtn = /** @type {HTMLButtonElement} */ (
    elem('button', 'op-session-new', 'New conversation')
  );
  newBtn.type = 'button';
  bar.append(led, elem('span', 'op-session-label', 'Operator'), newBtn);

  const messages = elem('div', 'op-messages');

  // A state the operator is waiting on rather than one they opened: announced
  // when it appears, never focus-trapped. Built with the console and hidden,
  // because a flip has to paint it in the same frame it disables the input.
  const overlay = elem('div', 'op-handoff');
  overlay.hidden = true;
  overlay.setAttribute('role', 'status');
  overlay.setAttribute('aria-live', 'polite');
  const overlayMessage = elem('p', 'op-handoff-message');
  const overlayAction = /** @type {HTMLButtonElement} */ (elem('button', 'op-handoff-action'));
  overlayAction.type = 'button';
  const overlayCard = elem('div', 'op-handoff-card');
  overlayCard.append(overlayMessage, overlayAction);
  overlay.append(overlayCard);

  const textarea = document.createElement('textarea');
  textarea.placeholder = 'Message the operator agent…';
  textarea.rows = 1;
  textarea.setAttribute('aria-label', 'Operator message');

  const sendBtn = /** @type {HTMLButtonElement} */ (elem('button', 'op-send-btn', 'Send'));
  sendBtn.type = 'button';
  const stopBtn = /** @type {HTMLButtonElement} */ (elem('button', 'op-stop-btn', 'Stop'));
  stopBtn.type = 'button';
  stopBtn.hidden = true;

  // ORDER IS THE LAYOUT. The textarea takes all the slack (`flex: 1`), so this
  // cluster is pinned to the row's right edge and a member's arrival moves
  // everything BEFORE it and nothing after it. Stop is the conditional one —
  // it exists only while a turn streams — so it goes FIRST and Send stays put.
  //
  // The other order is a control-substitution hazard, not just a jump: Stop
  // hides and Send re-enables in the same statement block (see setStreaming),
  // so Send would slide into the exact pixels Stop had occupied at the moment
  // it becomes clickable again — a hand already moving toward Stop as the turn
  // ends lands on Send instead.
  const controls = elem('div', 'op-input-controls');
  controls.append(stopBtn, sendBtn);

  const inputArea = elem('div', 'op-input-area');
  inputArea.append(elem('span', 'op-prompt-char', '›'), textarea, controls);

  return {
    bar,
    led,
    newBtn,
    messages,
    overlay,
    overlayMessage,
    overlayAction,
    inputArea,
    textarea,
    sendBtn,
    stopBtn,
  };
}

/**
 * Extract an HTTP status from a transport error: the `status` the transport
 * attaches, else parsed out of an `HTTP 409: ...` message, else 0 when the
 * failure carries no status at all (a network or parse error).
 * @param {unknown} err
 * @returns {number}
 */
function statusFromError(err) {
  const status = /** @type {{ status?: unknown }} */ (err ?? {}).status;
  if (typeof status === 'number' && Number.isFinite(status)) return status;
  const message = err instanceof Error ? err.message : String(err ?? '');
  const match = /^HTTP (\d+)/.exec(message);
  return match ? Number(match[1]) : 0;
}

/**
 * The notice to show for a transport failure: its slug first, its status
 * second, the generic line last.
 *
 * Slug before status is the whole point — keying on the status alone told an
 * operator whose chat had just been restarted by their own posture flip that a
 * turn was already running, which is both false and unactionable.
 * @param {unknown} err
 * @returns {string}
 */
export function transportNotice(err) {
  const slug = /** @type {{ slug?: unknown }} */ (err ?? {}).slug;
  if (typeof slug === 'string' && slug in TRANSPORT_NOTICES_BY_SLUG) {
    return TRANSPORT_NOTICES_BY_SLUG[slug];
  }
  return TRANSPORT_NOTICES[statusFromError(err)] ?? TRANSPORT_FALLBACK;
}

/**
 * The hand-off refusal *err* describes, or null when it is an ordinary
 * transport failure. Slug-only by design: a bare 409 says nothing about who
 * holds the session, and offering "Stop and switch now" for a rejection that
 * is not about the other view would cut a turn for no reason.
 * @param {unknown} err
 * @returns {{ message: string, action: 'interrupt'|'retry'|null } | null}
 */
export function handoffRefusal(err) {
  const slug = /** @type {{ slug?: unknown }} */ (err ?? {}).slug;
  if (typeof slug === 'string' && slug in HANDOFF_REFUSALS) return HANDOFF_REFUSALS[slug];
  return null;
}

/**
 * The console's hand-off entry point, bound by {@link initChat}. Module-level
 * because the flip that calls it (app.js) has no handle on the console.
 * @type {((options?: { interrupt?: boolean }) => Promise<void>) | null}
 */
let enterFromExpertBinding = null;

/**
 * Take the session over from the Expert view and show it here.
 *
 * Shows the transitional state, asks the server to hand the key's transcript
 * to the Simple view, and replays it once the outgoing agent is dead. `interrupt`
 * says the operator chose to cut a turn that was still running rather than wait
 * for it. Resolves when the console is usable, when the refusal is on screen,
 * or when the server abandoned the request and left the wait as it was; a
 * console that was never mounted resolves immediately.
 * @param {{ interrupt?: boolean }} [options]
 * @returns {Promise<void>}
 */
export function enterFromExpert(options = {}) {
  return enterFromExpertBinding ? enterFromExpertBinding(options) : Promise.resolve();
}

/**
 * Mount the Simple-mode operator chat into #operator-container. Called once in
 * the boot sequence; binds one console to the tab's session pointer. A missing
 * container is a no-op so the rest of the page still boots.
 * @param {string} [containerId]
 * @returns {void}
 */
export function initChat(containerId = 'operator-container') {
  const found = document.getElementById(containerId);
  if (!found) return;
  // Re-bind as non-null: the early-return guard doesn't narrow into the nested
  // handler closures below, so hand them a statically non-null reference.
  const container = /** @type {HTMLElement} */ (found);

  const handles = buildConsole();
  const { bar, led, newBtn, messages, overlay, overlayMessage, overlayAction } = handles;
  const { inputArea, textarea, sendBtn, stopBtn } = handles;
  container.append(bar, messages, overlay, inputArea);

  const renderer = createChatRenderer(messages);

  /** In-flight turn handle, or null when idle. */
  let handle = /** @type {import('./chat-client.js').ChatAbortHandle | null} */ (null);

  /** Whether a turn is streaming. */
  let streaming = false;

  /** Whether a hand-off is in flight or refused — the input stays out of reach. */
  let transitioning = false;

  /** The first-contact block while it is on screen, or null once it is gone. */
  let emptyState = /** @type {HTMLElement | null} */ (null);

  /** Whether first contact has had its moment, so a rebind can re-invite. */
  let settled = false;

  /**
   * The session key this tab was already on, or null for a tab that has none.
   * @type {string|null}
   */
  const adopted = getPointer();

  /**
   * The session key this console is showing. Adopted from the pointer at boot
   * and minted only for a tab that has none — the Expert view resumes the same
   * slot, so a second key here would be a second conversation.
   * @type {string}
   */
  let boundKey = adopted ?? crypto.randomUUID();
  if (!adopted) setPointer(boundKey);

  /** Which hand-off attempt is current; a retry supersedes the one before it. */
  let handoffGeneration = 0;

  /** The in-flight hand-off request, so a retry can abandon it first. */
  let handoffAbort = /** @type {AbortController | null} */ (null);

  /** When the current hand-off began, or null when none is showing. */
  let waitStartedAt = /** @type {number | null} */ (null);

  /** The 1 Hz elapsed-time ticker, or null when nothing is counting. */
  let ticker = /** @type {number | null} */ (null);

  /** The timer that turns a long restart into the wait, or null when none is armed. */
  let escalation = /** @type {number | null} */ (null);

  /** Whether the wait is on screen; it never steps back to a restart. */
  let waiting = false;

  const isSimpleMode = () =>
    document.documentElement.getAttribute('data-ui-mode') === 'simple';

  const isPinned = () =>
    messages.scrollHeight - messages.scrollTop - messages.clientHeight < STICK_THRESHOLD;
  const scrollToBottom = () => {
    messages.scrollTop = messages.scrollHeight;
  };

  /** Offer first contact when the log is empty and there is something to say. */
  function inviteIfEmpty() {
    if (!settled || emptyState || renderer.messageCount() > 0) return;
    emptyState = buildEmptyState();
    messages.prepend(emptyState);
  }

  // First contact, once there is enough known to say something true. A log that
  // already has entries in it is not empty and needs no invitation — a resumed
  // page reaches the settled moment too.
  onSettled(() => {
    settled = true;
    inviteIfEmpty();
  });

  // A switch changes what the sentence may claim. Rebuilt in place, and only
  // while the block is still the whole of the log.
  onKindChange(() => {
    if (emptyState) renderEmptyStateContent(emptyState);
  });

  function autoResize() {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_INPUT_HEIGHT)}px`;
  }

  /** Apply the current streaming/transition state to every control. */
  function applyControls() {
    const blocked = streaming || transitioning;
    textarea.disabled = blocked;
    sendBtn.disabled = blocked;
    newBtn.disabled = blocked;
    stopBtn.hidden = !streaming;
    stopBtn.disabled = false;
  }

  /**
   * Flip the whole console between streaming and idle: the azure top-edge
   * affordance (via the `streaming` class), the session LED, input enablement,
   * and Stop-button visibility.
   * @param {boolean} on
   */
  function setStreaming(on) {
    streaming = on;
    container.classList.toggle('streaming', on);
    led.classList.toggle('active', on);
    applyControls();
    // Return focus to the input when a turn ends, but only while the console is
    // the visible view and reachable — focusing a hidden or disabled textarea
    // is pointless.
    if (!on && !transitioning && isSimpleMode()) textarea.focus();
  }

  /**
   * Put the console into (or out of) the hand-off transition, where the input
   * is out of reach because the session is not this view's yet.
   * @param {boolean} on
   */
  function setTransitioning(on) {
    transitioning = on;
    applyControls();
    if (!on && !streaming && isSimpleMode()) textarea.focus();
  }

  /** @param {string} message */
  function showNotice(message) {
    messages.appendChild(elem('div', 'op-system', message));
    scrollToBottom();
  }

  /** Update the elapsed counter in place. */
  function renderElapsed() {
    const elapsed = overlayMessage.querySelector('.op-handoff-elapsed');
    if (!elapsed || waitStartedAt === null) return;
    const total = Math.max(0, Math.floor((Date.now() - waitStartedAt) / 1000));
    elapsed.textContent = `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
  }

  /** Stop the clock and the escalation, and forget when the hand-off began. */
  function stopTicker() {
    if (ticker !== null) {
      clearInterval(ticker);
      ticker = null;
    }
    disarmEscalation();
    waitStartedAt = null;
    waiting = false;
  }

  /** Cancel a pending restart-to-wait escalation. */
  function disarmEscalation() {
    if (escalation !== null) {
      clearTimeout(escalation);
      escalation = null;
    }
  }

  /**
   * Write the overlay's one line, optionally with the live counter.
   * @param {string} text
   * @param {boolean} withElapsed
   */
  function setOverlayMessage(text, withElapsed) {
    overlayMessage.textContent = text;
    if (!withElapsed) return;
    const elapsed = elem('span', 'op-handoff-elapsed');
    // The card is a polite live region, and a counter inside it would be read
    // out once a second. The sentence is the announcement; the clock is for
    // the eye.
    elapsed.setAttribute('aria-live', 'off');
    overlayMessage.append(elapsed);
  }

  /**
   * Show the overlay's single action, or hide it when there is nothing the
   * operator can do about the state on screen.
   * @param {string} label
   * @param {(() => void) | null} onPress
   */
  function setOverlayAction(label, onPress) {
    overlayAction.textContent = label;
    overlayAction.hidden = onPress === null;
    overlayAction.disabled = false;
    overlayAction.onclick = onPress
      ? () => {
        overlayAction.disabled = true;
        onPress();
      }
      : null;
  }

  /**
   * Show the transitional state. The hand-off route answers only once the
   * session is this view's, so nothing here says up front whether the other
   * view's agent is mid-turn: the restart is shown first — the agent is being
   * stopped there and started here — and the wait, with its clock and its way
   * out, once the restart outlasts its budget. A hand-off that cuts a running
   * turn short is a wait from the start.
   * @param {boolean} cutRunningTurn
   */
  function showHandoffPending(cutRunningTurn) {
    // A retry continues the hand-off the operator is already watching rather
    // than restarting its clock.
    if (waitStartedAt === null) waitStartedAt = Date.now();
    if (cutRunningTurn || waiting) {
      showWait();
    } else {
      setOverlayMessage(HANDOFF_RESTARTING_MESSAGE, false);
      setOverlayAction('', null);
      if (escalation === null) {
        escalation = window.setTimeout(() => {
          escalation = null;
          showWait();
        }, HANDOFF_RESTART_BUDGET_MS);
      }
    }
    overlay.hidden = false;
    setTransitioning(true);
  }

  /**
   * The wait: the other view's agent is finishing its turn and this one is
   * waiting for it. There is no bound on that wait, so the elapsed time is
   * the honest thing to show, and the button is the way out.
   */
  function showWait() {
    waiting = true;
    disarmEscalation();
    setOverlayMessage(HANDOFF_PENDING_MESSAGE, true);
    renderElapsed();
    if (ticker === null) ticker = window.setInterval(renderElapsed, 1000);
    setOverlayAction('Stop and switch now', () => {
      void runHandoff(true);
    });
  }

  /**
   * Show why the server would not hand the session over, and what remains to
   * be done about it. The input stays out of reach: the session is not this
   * view's, and a refusal with no action is final for this view.
   * @param {{ message: string, action: 'interrupt'|'retry'|null }} refusal
   */
  function showHandoffRefused(refusal) {
    stopTicker();
    setOverlayMessage(refusal.message, false);
    if (refusal.action === 'interrupt') {
      setOverlayAction('Stop and switch now', () => {
        void runHandoff(true);
      });
    } else if (refusal.action === 'retry') {
      setOverlayAction('Retry', () => {
        void runHandoff(false);
      });
    } else {
      setOverlayAction('', null);
    }
    overlay.hidden = false;
    setTransitioning(true);
  }

  /** Clear the overlay entirely. */
  function hideOverlay() {
    stopTicker();
    overlay.hidden = true;
    overlayAction.onclick = null;
  }

  /** Clear the log and everything the old conversation left behind. */
  function resetLog() {
    handle?.abort();
    handle = null;
    renderer.reset();
    emptyState = null;
    setStreaming(false);
  }

  /**
   * Point the console at *key*: clear the log, then replay what the session
   * has already said so the Simple view opens on the conversation rather than
   * on an empty console.
   * @param {string} key
   * @returns {Promise<void>}
   */
  async function bindTo(key) {
    boundKey = key;
    resetLog();
    if (isSimpleMode()) notifySessionChange(key);

    /** @type {import('./chat-client.js').ChatTurn[]} */
    let turns = [];
    try {
      turns = await fetchHistory(key);
    } catch (err) {
      if (boundKey === key) showNotice(transportNotice(err));
      return;
    }
    // A newer bind landed while the transcript was in flight; it owns the log.
    if (boundKey !== key) return;
    renderer.replay(turns);
    inviteIfEmpty();
    scrollToBottom();
  }

  /**
   * Ask the server for the session, showing the wait and then either the
   * conversation or the reason it was refused.
   * @param {boolean} cutRunningTurn
   * @returns {Promise<void>}
   */
  async function runHandoff(cutRunningTurn) {
    const generation = ++handoffGeneration;
    // The server frees a key whose requester disconnects, so abandoning the
    // wait in flight is what makes room for this one — two live requests for
    // the same key are two channels, and the second would be refused.
    handoffAbort?.abort();
    const controller = new AbortController();
    handoffAbort = controller;

    const key = boundKey;
    showHandoffPending(cutRunningTurn);
    /** @type {{ state: string, session_id: string } | null} */
    let handed = null;
    try {
      handed = await requestHandoff(key, { interrupt: cutRunningTurn, signal: controller.signal });
    } catch (err) {
      if (generation !== handoffGeneration) return;
      handoffAbort = null;
      const refusal = handoffRefusal(err);
      showHandoffRefused(refusal ?? HANDOFF_FALLBACK);
      return;
    }
    if (generation !== handoffGeneration) return;
    // A 204 with no body: the server saw this request's channel close and
    // abandoned the acquire, so nothing was handed over and nothing failed.
    // The state on screen is still the true one — a wait the operator can
    // still cut short — and replacing it with copy about a request that was
    // never answered would be a claim the server did not make.
    if (!handed) return;
    handoffAbort = null;
    hideOverlay();
    await bindTo(key);
    // The replay is awaited, so a retry can start and put its own wait on
    // screen while this one is still finishing. Enabling the input here would
    // hand the operator a console over the newer attempt's overlay.
    if (generation !== handoffGeneration) return;
    setTransitioning(false);
  }

  /** Submit the current input as a new turn, unless one is already streaming. */
  function submit() {
    if (streaming || transitioning) return;
    const prompt = textarea.value.trim();
    if (!prompt) return;

    const entry = renderer.addUserMessage(prompt);
    // The invitation has been taken; the log is the conversation from here.
    emptyState?.remove();
    emptyState = null;
    textarea.value = '';
    autoResize();
    scrollToBottom();
    setStreaming(true);

    handle = sendPrompt(boundKey, prompt, {
      onEvent: (event) => {
        // Follow the tail only when the operator hasn't scrolled up to read back.
        const pinned = isPinned();
        renderer.handleEvent(event);
        if (pinned) scrollToBottom();
      },
      onError: (err) => {
        const refusal = handoffRefusal(err);
        if (!refusal) {
          showNotice(transportNotice(err));
          return;
        }
        // The other view holds the session, so the prompt never ran. Take the
        // bubble back and return the text to the input the operator typed it
        // in; the overlay carries the one thing left to do.
        entry.remove();
        textarea.value = prompt;
        autoResize();
        showHandoffRefused(refusal);
      },
      onClose: () => {
        handle = null;
        setStreaming(false);
      },
    });
  }

  /** Stop the in-flight turn: interrupt server-side first, then abort locally. */
  async function stop() {
    const current = handle;
    if (!current) return;
    stopBtn.disabled = true;
    // Interrupt the server turn first, then abort the local fetch. Either
    // arrival order is safe server-side; the abort makes the client stop
    // reading immediately. A failed interrupt still falls through to abort.
    try {
      await interrupt(boundKey);
    } catch {
      // Non-2xx or network failure — nothing to surface; abort regardless.
    }
    current.abort();
  }

  /**
   * Start a conversation on a new session key. The pointer is the shared slot,
   * so writing it here is what moves both views; the old key's transcript is
   * left alone — it is a session the operator can still resume.
   */
  function newConversation() {
    const key = crypto.randomUUID();
    // Claim it before the pointer notifies, so the subscription below sees the
    // key the console is already on and does not replay an empty transcript.
    boundKey = key;
    setPointer(key);
    resetLog();
    inviteIfEmpty();
    if (isSimpleMode()) notifySessionChange(key);
  }

  textarea.addEventListener('input', autoResize);
  textarea.addEventListener('keydown', (e) => {
    // Enter sends; Shift+Enter inserts a newline. Skip while an IME is composing.
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    }
  });
  sendBtn.addEventListener('click', submit);
  stopBtn.addEventListener('click', stop);
  newBtn.addEventListener('click', newConversation);

  // The pointer is the tab's session, not this console's: a key chosen
  // anywhere else (the session picker, another view) moves the log here too. A
  // cleared pointer means the key is dead for both views and the page is on
  // its way out — there is no conversation to move to.
  subscribeToPointer((key) => {
    if (!key || key === boundKey) return;
    void bindTo(key);
  });

  enterFromExpertBinding = ({ interrupt: cutRunningTurn = false } = {}) =>
    runHandoff(cutRunningTurn);

  // A page that opens in Simple mode never went through a flip, so nothing
  // else replays the key it adopted above — and nothing else tells the panels
  // which session they are on, because the terminal does not connect here.
  if (isSimpleMode()) {
    if (adopted) void bindTo(adopted);
    else notifySessionChange(boundKey);
  }
}
