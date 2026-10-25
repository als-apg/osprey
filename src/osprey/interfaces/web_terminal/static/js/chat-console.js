// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode chat: the operator console's DOM.
 *
 * Builds the console interior once, at boot: the session bar, the message
 * list, the hand-off overlay, and the composer (input, slash button, and the
 * send and stop controls). It returns the handles the controller in chat.js
 * drives. Nothing here reads the session pointer or the transport.
 *
 * @module chat-console
 */

import { chatIcon } from './chat-icons.js';
import { elem } from './chat-dom.js';

/**
 * Build the operator console interior — session bar, message list, hand-off
 * overlay, and the command-line input row — and return the handles the
 * controller drives.
 */
export function buildConsole() {
  const bar = elem('div', 'op-session-bar');
  const led = elem('span', 'op-session-led');
  const newBtn = /** @type {HTMLButtonElement} */ (
    elem('button', 'op-session-new', 'New conversation')
  );
  newBtn.type = 'button';
  bar.append(led, elem('span', 'op-session-label', 'Operator'), newBtn);

  const messages = elem('div', 'op-messages');
  // A log by role, but silent by itself: a log is polite by default, and streamed
  // text would be read word by word. chat-announce.js speaks the few things that matter.
  messages.setAttribute('role', 'log');
  messages.setAttribute('aria-live', 'off');
  messages.setAttribute('aria-label', 'Conversation with the agent');

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
  textarea.placeholder = 'Message the agent…';
  textarea.rows = 1;
  textarea.setAttribute('aria-label', 'Operator message');

  // The composer's own controls. Send and Stop share one slot: a turn swaps one
  // for the other, so the button under the pointer is always the one on screen.
  const cmdBtn = /** @type {HTMLButtonElement} */ (elem('button', 'op-cmd-btn', '/'));
  cmdBtn.type = 'button';
  cmdBtn.title = 'Slash commands';
  cmdBtn.setAttribute('aria-label', 'Show slash commands');

  const sendBtn = /** @type {HTMLButtonElement} */ (elem('button', 'op-send-btn'));
  sendBtn.type = 'button';
  sendBtn.title = 'Send';
  sendBtn.setAttribute('aria-label', 'Send');
  sendBtn.append(chatIcon('up'));

  const stopBtn = /** @type {HTMLButtonElement} */ (elem('button', 'op-stop-btn'));
  stopBtn.type = 'button';
  stopBtn.title = 'Stop';
  stopBtn.setAttribute('aria-label', 'Stop');
  stopBtn.append(chatIcon('stop'));
  stopBtn.hidden = true;

  const controls = elem('div', 'op-input-controls');
  controls.append(stopBtn, sendBtn);

  const toolbar = elem('div', 'op-composer-bar');
  toolbar.append(cmdBtn, controls);

  // The composer is one rounded surface: the box and its toolbar. `.op-input-area`
  // is the element the slash list mounts into, and first-contact reaches for it.
  const inputArea = elem('div', 'op-input-area');
  inputArea.append(textarea, toolbar);

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
    cmdBtn,
    sendBtn,
    stopBtn,
  };
}
