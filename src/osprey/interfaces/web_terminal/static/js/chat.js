// @ts-check
/* OSPREY Web Terminal — Simple-mode operator chat controller.
 *
 * Thin glue between the SSE transport (chat-client.js) and the message-list
 * renderer (chat-render.js). It mounts the console that chat-console.js builds
 * inside #operator-container, binds it to the tab's session pointer, and owns
 * the streaming/idle UI state, turning user intent (submit, stop, switch view)
 * into transport calls. All rendering and sanitisation live in the renderer;
 * all networking and SSE parsing live in the client; what the live feed's
 * frames mean for the log lives in chat-follow.js, and the copy for refusals
 * and failures lives in chat-notices.js. This file holds no innerHTML and no
 * parsing, so the interesting logic stays in the unit-tested modules.
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

import { pickUiMode } from '/design-system/js/frame-params.js';
import { uuid4 } from '/design-system/js/uuid.js';

import {
  answerPermission,
  fetchCommands,
  fetchHistory,
  followChat,
  interrupt,
  requestHandoff,
  sendPrompt,
} from './chat-client.js';
import { createChatRenderer, elem } from './chat-render.js';
import { buildConsole } from './chat-console.js';
import { emitChat } from './chat-events.js';
import { createTurnFollower } from './chat-follow.js';
import { initChatFeatures } from './chat-features.js';
import { handoffRefusal, HANDOFF_FALLBACK, transportNotice } from './chat-notices.js';
import {
  buildEmptyState,
  onKindChange,
  onSettled,
  renderEmptyStateContent,
} from './first-contact.js';
import { getPointer, setPointer, subscribe as subscribeToPointer } from './session-pointer.js';
import { attachSlashComplete } from './slash-complete.js';
import { askAboutStartedCommands } from './started-commands.js';
import { notifySessionChange } from './terminal.js';
import {
  HANDOFF_INTERRUPT_LABEL,
  HANDOFF_PENDING_MESSAGE,
  HANDOFF_RESTART_BUDGET_MS,
  HANDOFF_RESTARTING_MESSAGE,
  HANDOFF_RETRY_LABEL,
} from './terminal-handoff.js';

/** Max textarea height (px) before it scrolls — matches operator.css. */
const MAX_INPUT_HEIGHT = 120;

/** Distance (px) from the bottom within which the log auto-follows new output. */
const STICK_THRESHOLD = 40;

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
  const { inputArea, textarea, cmdBtn, sendBtn, stopBtn } = handles;
  container.append(bar, messages, overlay, inputArea);
  const slash = attachSlashComplete(textarea, { load: fetchCommands, mount: inputArea });
  // The "/" button opens the command list as typing a slash would. Its press is
  // swallowed so focus stays in the box: a blur would close the list the click
  // is about to open.
  cmdBtn.addEventListener('mousedown', (e) => e.preventDefault());
  cmdBtn.addEventListener('click', () => {
    if (!textarea.value.trim()) textarea.value = '/';
    textarea.focus();
    textarea.dispatchEvent(new Event('input'));
  });

  const renderer = createChatRenderer(messages, {
    answerPermission: (requestId, body) =>
      answerPermission(boundKey, { request_id: requestId, ...body }),
  });

  /** Transforms the feature modules apply to each prompt as it is sent. */
  const sendTransforms = /** @type {Array<(prompt: string) => string>} */ ([]);

  /** In-flight turn handle, or null when idle. */
  let handle = /** @type {import('./chat-client.js').ChatAbortHandle | null} */ (null);

  /**
   * The live feed of the bound chat (every tab's view of its agent), or null
   * while this tab's own turn streams: that turn's stream already carries every
   * frame the feed would (see {@link pauseFeed}).
   */
  let feed = /** @type {import('./chat-client.js').ChatFeed | null} */ (null);

  /** What the feed's frames mean for the log: the turns this tab did not send. */
  const follower = createTurnFollower({
    renderer,
    draw,
    fetchHistory,
    key: () => boundKey,
    sending: () => handle !== null,
    setStreaming,
    dismissInvitation: () => {
      emptyState?.remove();
      emptyState = null;
    },
    scrollToBottom: () => scrollToBottom(),
    redraw: (turns) => {
      renderer.reset();
      emptyState = null;
      renderer.replay(turns);
      inviteIfEmpty();
    },
  });

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
   * The session key this console is showing. Adopted from the pointer at boot,
   * or minted for a tab that has none — the Expert view resumes the same slot,
   * so a second key here would be a second conversation. A minted key is not
   * written to the pointer here: the pointer names the session the tab is on,
   * and in the Expert view that is the id the terminal's server confirms. The
   * console claims the pointer only where it is the surface running the
   * session — the Simple-mode branch at the end of this function.
   * @type {string}
   */
  let boundKey = adopted ?? uuid4();

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

  const isSimpleMode = () => document.documentElement.getAttribute('data-ui-mode') === 'simple';

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
    cmdBtn.disabled = blocked;
    sendBtn.disabled = blocked;
    newBtn.disabled = blocked;
    sendBtn.hidden = streaming;
    stopBtn.hidden = !streaming;
    stopBtn.disabled = false;
    if (blocked) slash.close();
    emitChat('busy', { busy: blocked });
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
    emitChat('notice', { text: message });
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
    setOverlayAction(HANDOFF_INTERRUPT_LABEL, () => {
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
      setOverlayAction(HANDOFF_INTERRUPT_LABEL, () => {
        void runHandoff(true);
      });
    } else if (refusal.action === 'retry') {
      setOverlayAction(HANDOFF_RETRY_LABEL, () => {
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

  /**
   * Abandon the hand-off and give the session back to the Expert view.
   *
   * Nothing was ended: the agent and the commands it started keep running
   * there. The console stays transitioning, because the session is not this
   * view's.
   */
  function cancelHandoff() {
    ++handoffGeneration;
    handoffAbort?.abort();
    handoffAbort = null;
    hideOverlay();
    pickUiMode('expert');
  }

  /** Clear the log and everything the old conversation left behind. */
  function resetLog() {
    handle?.abort();
    handle = null;
    follower.logChanged();
    renderer.reset();
    emptyState = null;
    setStreaming(false);
  }

  /**
   * Point the console at *key*: clear the log, then replay what the session
   * has already said so the Simple view opens on the conversation rather than
   * on an empty console. A turn the transcript shows still running is left
   * open; the feed's snapshot then says whether it goes on.
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
    }
    // A newer bind landed while the transcript was in flight; it owns the log.
    if (boundKey !== key) return;
    renderer.replay(turns);
    inviteIfEmpty();
    scrollToBottom();
    // Follow the agent from here on, so a turn still running — after a reload,
    // or sent from another tab — keeps drawing.
    follow(key);
  }

  /**
   * Open the live feed for *key*, closing any other.
   * @param {string} key
   */
  function follow(key) {
    feed?.close();
    follower.restart();
    const mine = followChat(key, {
      onEvent: (frame) => {
        // A closed feed can still hand over a frame it had already read.
        if (feed === mine && boundKey === key) follower.frame(frame);
      },
      onNoAgent: () => {
        if (feed === mine && boundKey === key) follower.noAgent();
      },
    });
    feed = mine;
  }

  /**
   * Close the feed while this tab's own turn streams; the turn's `onClose`
   * reopens it. The turn's stream carries every frame the feed would, and each
   * open stream holds one of the six connections a browser gives one host over
   * HTTP/1.1. The panels keep event streams of their own open, so a seventh
   * stream leaves every other request the page makes (panel config reads, panel
   * pages, the gallery's listing) queued until one closes.
   */
  function pauseFeed() {
    feed?.close();
    feed = null;
  }

  /** @param {import('./chat-client.js').ChatEvent} event */
  function draw(event) {
    // Follow the tail only when the operator hasn't scrolled up to read back.
    const pinned = isPinned();
    renderer.handleEvent(event);
    if (pinned) scrollToBottom();
  }

  /**
   * Ask the server for the session, showing the wait and then either the
   * conversation or the reason it was refused.
   *
   * A refusal for commands the outgoing agent started asks the operator
   * whether to end them with it: *Stop both* re-sends this request with
   * `endStarted`, and every other answer abandons the hand-off.
   * @param {boolean} cutRunningTurn
   * @param {boolean} [endStarted] - the operator agreed that ending the
   *   outgoing agent also ends the commands it started
   * @returns {Promise<void>}
   */
  async function runHandoff(cutRunningTurn, endStarted = false) {
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
      handed = await requestHandoff(key, {
        interrupt: cutRunningTurn,
        endStarted,
        signal: controller.signal,
      });
    } catch (err) {
      if (generation !== handoffGeneration) return;
      handoffAbort = null;
      const refusal = handoffRefusal(err);
      showHandoffRefused(refusal ?? HANDOFF_FALLBACK);
      const refused = /** @type {import('./chat-client.js').TransportError} */ (err);
      if (refused?.slug === 'handoff_started_commands') {
        askAboutStartedCommands(refused.commands, {
          onStop: () => void runHandoff(cutRunningTurn, true),
          onCancel: cancelHandoff,
          unread: refused.unread === true,
        });
      }
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
    slash.close();
    autoResize();
    setStreaming(true);
    renderer.beginTurn();
    // After beginTurn: its status line lands at the foot of the log, and a log
    // left above it is no longer pinned, so the reply would not be followed.
    scrollToBottom();

    pauseFeed();
    follower.logChanged();
    const sent = sendTransforms.reduce((text, transform) => transform(text), prompt);
    /** Whether any frame of the turn came back: the agent has the prompt. */
    let answered = false;
    const current = sendPrompt(boundKey, sent, {
      onEvent: (event) => {
        answered = true;
        if (event.type !== 'turn_start') draw(event);
      },
      onError: (err) => {
        // A send that no frame answered never reached the agent. Take the
        // bubble back and return the text to the input the operator typed it
        // in, as the files attached to it return to the tray.
        if (!answered) {
          entry.remove();
          textarea.value = prompt;
          autoResize();
        }
        // The other view holds the session; the overlay carries the one thing
        // left to do.
        const refusal = handoffRefusal(err);
        if (refusal) showHandoffRefused(refusal);
        else showNotice(transportNotice(err));
      },
      onClose: () => {
        // A turn that a reset cut short can close after its successor has started;
        // only the turn still current may touch the console.
        if (handle !== current) return;
        handle = null;
        setStreaming(false);
        renderer.endTurn();
        follow(boundKey);
      },
    });
    handle = current;
  }

  /** Stop the in-flight turn: interrupt server-side first, then abort locally. */
  async function stop() {
    const current = handle;
    if (!current && !follower.following()) return;
    stopBtn.disabled = true;
    // Interrupt the server turn first, then abort the local fetch. Either
    // arrival order is safe server-side; the abort makes the client stop
    // reading immediately. A failed interrupt still falls through to abort.
    // A turn this tab is only following has no fetch of its own: the
    // interrupt's result arrives on the feed and ends it here.
    try {
      await interrupt(boundKey);
    } catch {
      // Non-2xx or network failure — nothing to surface; abort regardless.
    }
    current?.abort();
  }

  /**
   * Start a conversation on a new session key. The pointer is the shared slot,
   * so writing it here is what moves both views; the old key's transcript is
   * left alone — it is a session the operator can still resume.
   */
  function newConversation() {
    const key = uuid4();
    // Claim it before the pointer notifies, so the subscription below sees the
    // key the console is already on and does not replay an empty transcript.
    boundKey = key;
    setPointer(key);
    resetLog();
    inviteIfEmpty();
    if (isSimpleMode()) {
      notifySessionChange(key);
      follow(key);
    }
  }

  textarea.addEventListener('input', autoResize);
  textarea.addEventListener('keydown', (e) => {
    // An open suggestion list owns its keys, Enter included, so picking a
    // command fills the box and never sends. Otherwise Enter sends and
    // Shift+Enter inserts a newline; nothing is taken while an IME is composing.
    if (slash.handleKeydown(e)) return;
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

  initChatFeatures({
    ...handles,
    container,
    renderer,
    getKey: () => boundKey,
    onBeforeSend: (transform) => void sendTransforms.push(transform),
  });

  // A page that opens in Simple mode never went through a flip: the terminal
  // does not connect here, so this console is the surface running the
  // session. That makes it the one to name the session the tab is on — it
  // claims the pointer for a key it minted, replays a key it adopted, and
  // either way is what tells the panels which session they are on.
  if (isSimpleMode()) {
    if (adopted) void bindTo(adopted);
    else {
      setPointer(boundKey);
      notifySessionChange(boundKey);
      follow(boundKey);
    }
  }
}

/**
 * Put a visible failure where the operator console would have been.
 *
 * A console that threw during init leaves an EMPTY container, and in Simple
 * view an empty container is indistinguishable from an agent with nothing to
 * say — the operator waits at a page that is never going to answer. A console
 * log line is not a user-facing surface; this is.
 *
 * @param {string} containerId
 */
export function renderChatBootFailure(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const notice = document.createElement('p');
  notice.className = 'chat-boot-failure';
  notice.textContent =
    'The operator console failed to start. Reload the page, or switch to the Expert view.';
  container.replaceChildren(notice);
}
