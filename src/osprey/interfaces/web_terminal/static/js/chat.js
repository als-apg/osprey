// @ts-check
/* OSPREY Web Terminal — Simple-mode operator chat controller.
 *
 * Thin glue between the SSE transport (chat-client.js) and the message-list
 * renderer (chat-render.js). It builds the operator console DOM inside
 * #operator-container, owns the per-page chat id and the streaming/idle UI
 * state, and turns user intent (submit, stop) into transport calls. All
 * rendering and sanitisation live in the renderer; all networking and SSE
 * parsing live in the client — this file holds no innerHTML and no parsing, so
 * the interesting logic stays in the two unit-tested modules.
 *
 * Mode visibility is pure CSS: #operator-container is display-gated off
 * html[data-ui-mode] (operator.css), so the console is built once at boot and
 * simply hidden in expert mode — never torn down or toggled from here.
 */

import { sendPrompt, interrupt, deleteChat } from './chat-client.js';
import { createChatRenderer, elem } from './chat-render.js';

/** Max textarea height (px) before it scrolls — matches operator.css. */
const MAX_INPUT_HEIGHT = 120;

/** Distance (px) from the bottom within which the log auto-follows new output. */
const STICK_THRESHOLD = 40;

/**
 * User-facing copy for the transport-level HTTP rejections the endpoint can
 * return before any turn streams. These are distinct from server `error`
 * events (which the renderer paints as red error blocks); a transport failure
 * renders as a centred system notice instead.
 * @type {Record<number, string>}
 */
const TRANSPORT_NOTICES = {
  409: 'A turn is already running.',
  429: 'Server busy — please retry in a moment.',
  503: 'Operator agent unavailable.',
};

/** Fallback copy for a transport failure with no recognised HTTP status. */
const TRANSPORT_FALLBACK = 'Connection to the operator agent failed.';

/**
 * Build the operator console interior — session bar, message list, and the
 * command-line input row — and return the handles the controller drives.
 */
function buildConsole() {
  const bar = elem('div', 'op-session-bar');
  const led = elem('span', 'op-session-led');
  bar.append(led, elem('span', 'op-session-label', 'Operator'));

  const messages = elem('div', 'op-messages');

  const textarea = document.createElement('textarea');
  textarea.placeholder = 'Message the operator agent…';
  textarea.rows = 1;
  textarea.setAttribute('aria-label', 'Operator message');

  const sendBtn = /** @type {HTMLButtonElement} */ (elem('button', 'op-send-btn', 'Send'));
  sendBtn.type = 'button';
  const stopBtn = /** @type {HTMLButtonElement} */ (elem('button', 'op-stop-btn', 'Stop'));
  stopBtn.type = 'button';
  stopBtn.hidden = true;

  const controls = elem('div', 'op-input-controls');
  controls.append(sendBtn, stopBtn);

  const inputArea = elem('div', 'op-input-area');
  inputArea.append(elem('span', 'op-prompt-char', '›'), textarea, controls);

  return { bar, led, messages, inputArea, textarea, sendBtn, stopBtn };
}

/**
 * Extract an HTTP status from a transport error message (`HTTP 409: ...`), or 0
 * when the failure carries no status (a network or parse error).
 * @param {unknown} err
 * @returns {number}
 */
function statusFromError(err) {
  const message = err instanceof Error ? err.message : String(err ?? '');
  const match = /^HTTP (\d+)/.exec(message);
  return match ? Number(match[1]) : 0;
}

/**
 * Mount the Simple-mode operator chat into #operator-container. Called once in
 * the boot sequence; mints a single chat id and wires one console. A missing
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

  const chatId = crypto.randomUUID();
  const { bar, led, messages, inputArea, textarea, sendBtn, stopBtn } = buildConsole();
  container.append(bar, messages, inputArea);

  const renderer = createChatRenderer(messages);

  /** In-flight turn handle, or null when idle. Also the streaming/idle flag. */
  let handle = /** @type {import('./chat-client.js').ChatAbortHandle | null} */ (null);

  const isPinned = () =>
    messages.scrollHeight - messages.scrollTop - messages.clientHeight < STICK_THRESHOLD;
  const scrollToBottom = () => {
    messages.scrollTop = messages.scrollHeight;
  };

  function autoResize() {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_INPUT_HEIGHT)}px`;
  }

  /**
   * Flip the whole console between streaming and idle: the azure top-edge
   * affordance (via the `streaming` class), the session LED, input enablement,
   * and Stop-button visibility.
   * @param {boolean} on
   */
  function setStreaming(on) {
    container.classList.toggle('streaming', on);
    led.classList.toggle('active', on);
    textarea.disabled = on;
    sendBtn.disabled = on;
    stopBtn.hidden = !on;
    stopBtn.disabled = false;
    // Return focus to the input when a turn ends, but only while the console is
    // the visible view — focusing a hidden textarea in expert mode is pointless.
    if (!on && document.documentElement.getAttribute('data-ui-mode') === 'simple') {
      textarea.focus();
    }
  }

  /** @param {string} message */
  function showNotice(message) {
    messages.appendChild(elem('div', 'op-system', message));
    scrollToBottom();
  }

  /** Submit the current input as a new turn, unless one is already streaming. */
  function submit() {
    if (handle) return;
    const prompt = textarea.value.trim();
    if (!prompt) return;

    renderer.addUserMessage(prompt);
    textarea.value = '';
    autoResize();
    scrollToBottom();
    setStreaming(true);

    handle = sendPrompt(chatId, prompt, {
      onEvent: (event) => {
        // Follow the tail only when the operator hasn't scrolled up to read back.
        const pinned = isPinned();
        renderer.handleEvent(event);
        if (pinned) scrollToBottom();
      },
      onError: (err) => {
        showNotice(TRANSPORT_NOTICES[statusFromError(err)] ?? TRANSPORT_FALLBACK);
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
      await interrupt(chatId);
    } catch {
      // Non-2xx or network failure — nothing to surface; abort regardless.
    }
    current.abort();
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

  // Best-effort server-side cleanup when the page goes away (keepalive DELETE).
  window.addEventListener('pagehide', () => {
    try {
      deleteChat(chatId);
    } catch {
      // Fire-and-forget: a teardown-time failure has nowhere to surface.
    }
  });
}
