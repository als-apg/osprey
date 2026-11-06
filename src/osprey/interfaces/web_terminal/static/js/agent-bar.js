// @ts-check
/* OSPREY Web Terminal — the Agent pane's bar
 *
 * The Agent pane has one bar, the same on both of its surfaces (the operator
 * chat and the terminal):
 *
 *   [session ▾] ……………… [+] │ then the tile's ⋯ and ×
 *
 * The label is the session switcher (session-switcher.js): the session the
 * tab is on, and the way to the others. `+` starts a new conversation (a new
 * session on the terminal). The spacer between them is bare bar, so the tile
 * still drags from there. The divider ends the pane's own controls: in the
 * dock the Agent tile adopts the bar of the surface showing (dock-tab.js) and
 * puts its ⋯ and × after it. The pane's other actions are rows in that ⋯
 * (agent-menu.js).
 *
 * {@link buildAgentBar} builds the chat console's bar (chat-console.js). The
 * terminal's bar is markup in index.html (`.terminal-header`), whose ids
 * app.js and terminal.js wire; agent-bar.test.mjs holds the two to one shape.
 *
 * The hub's "Ask the agent" control (tile-bar-controls.js) asks for a line in
 * the composer with a document event; {@link listenForCompose} puts it there.
 *
 * The bar imports none of the dock's modules (the tile bar, its ⋯ menu): the
 * dock adopts the bar (dock-workspace.js hands it to dock-tab.js), never the
 * other way round.
 *
 * @module agent-bar
 */

import { icon } from '/design-system/js/icons.js';
import { isTerminalSurface } from './agent-surface.js';
import { onChat } from './chat-events.js';
import { elem } from './chat-dom.js';
import { insertPrompt } from './first-contact.js';

/** The document event of the hub's "Ask the agent" (tile-bar-controls.js). */
const COMPOSE_EVENT = 'osprey-agent-compose';

/**
 * The chat console's Agent bar: the label the session switcher attaches to,
 * the spacer, `+` (New conversation) and the divider. `+` is withheld while
 * the chat is busy (a turn streams or a hand-off is in flight), as the
 * composer is.
 * @param {{ onNew?: () => void }} [options]  what `+` does
 * @returns {{ bar: HTMLElement, label: HTMLElement, newBtn: HTMLButtonElement }}
 */
export function buildAgentBar({ onNew } = {}) {
  const bar = elem('div', 'op-session-bar agent-bar');
  // The session switcher fills it with the session's title and a ▾ once chat.js attaches it.
  const label = elem('span', 'op-session-label agent-bar-label', 'Agent');
  const newBtn = /** @type {HTMLButtonElement} */ (
    elem('button', 'osprey-icon-btn agent-bar-new op-session-new')
  );
  newBtn.type = 'button';
  newBtn.title = 'New conversation';
  newBtn.setAttribute('aria-label', 'New conversation');
  newBtn.append(icon('plus'));
  if (onNew) newBtn.addEventListener('click', () => onNew());
  onChat('busy', ({ busy }) => {
    newBtn.disabled = busy;
  });
  const divider = elem('span', 'tile-tab-divider');
  divider.setAttribute('aria-hidden', 'true');
  bar.append(label, elem('span', 'agent-bar-spacer'), newBtn, divider);
  return { bar, label, newBtn };
}

/**
 * Take the hub's `osprey-agent-compose` `{text}` into the chat composer. The
 * line joins what the operator was writing, the caret waits after it, and
 * nothing is sent; while a turn streams the line waits for the turn to end
 * (insertPrompt, first-contact.js). The terminal has no composer, so an event
 * that arrives while the pane shows it changes nothing (the hub disables its
 * control there). Call once, at boot.
 * @returns {() => void} stop listening
 */
export function listenForCompose() {
  /** @param {Event} event */
  const onCompose = (event) => {
    const text = /** @type {CustomEvent<{ text?: unknown }>} */ (event).detail?.text;
    if (typeof text !== 'string' || !text || isTerminalSurface()) return;
    insertPrompt(text, { append: true });
  };
  document.addEventListener(COMPOSE_EVENT, onCompose);
  return () => document.removeEventListener(COMPOSE_EVENT, onCompose);
}
