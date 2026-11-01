// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode chat: the operator console's DOM.
 *
 * Builds the console interior once, at boot: the Agent bar, the message
 * list, the hand-off overlay, and the composer (input, slash button, and the
 * send and stop controls). It returns the handles the controller in chat.js
 * drives. Nothing here reads the session pointer or the transport.
 *
 * The Agent bar is the Agent pane's one header row. In the dock it does not
 * stay in the console: the Agent tile adopts it as its tile bar in the Simple
 * view (dock-tab.js), so the console body below it is the card interior.
 * Left to right: the status dot, `Agent`, then the icon buttons — New
 * conversation, the notifications bell (chat-attention.js appends it),
 * Export (chat-export.js appends it) and ⋯, the earlier conversations. The
 * ⋯ is returned unattached: chat.js appends it after the feature modules,
 * so it stays last, and sessions.js gives it its menu.
 *
 * @module chat-console
 */

import { chatIcon } from './chat-icons.js';
import { elem } from './chat-dom.js';
import { icon } from '/design-system/js/icons.js';

/**
 * One of the Agent bar's icon buttons: an `.osprey-icon-btn` whose name is
 * its `aria-label`, repeated as the tooltip.
 * @param {string} className
 * @param {string} iconName  a name from design-system icons.js
 * @param {string} label
 * @returns {HTMLButtonElement}
 */
export function barButton(className, iconName, label) {
  const btn = /** @type {HTMLButtonElement} */ (elem('button', `osprey-icon-btn ${className}`));
  btn.type = 'button';
  btn.title = label;
  btn.setAttribute('aria-label', label);
  btn.append(icon(iconName));
  return btn;
}

/**
 * Build the operator console interior — Agent bar, message list, hand-off
 * overlay, and the command-line input row — and return the handles the
 * controller drives.
 */
export function buildConsole() {
  const bar = elem('div', 'op-session-bar');
  // Nothing updates this dot, so it claims no state: decoration only. The
  // label beside it names the pane.
  const led = elem('span', 'osprey-dot ok op-session-led');
  led.setAttribute('aria-hidden', 'true');
  const label = elem('span', 'op-session-label', 'Agent');
  const newBtn = barButton('op-session-new', 'pen', 'New conversation');
  bar.append(led, label, newBtn);
  const moreBtn = barButton('op-session-more', 'more', 'Earlier conversations');
  moreBtn.setAttribute('aria-haspopup', 'menu');
  moreBtn.setAttribute('aria-expanded', 'false');

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
  textarea.setAttribute('aria-label', 'Message the agent');

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
    newBtn,
    moreBtn,
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
    'The agent console failed to start. Reload the page, or switch to the Expert view.';
  container.replaceChildren(notice);
}
