// @ts-check
/* OSPREY Web Terminal — the Agent pane's bar
 *
 * The Agent pane has one bar, the same on both of its surfaces (the operator
 * chat and the terminal):
 *
 *   [session ▾] ……………… [+]  then the tile's ⋯ and ×
 *
 * The label is the session switcher (session-switcher.js): the session the
 * tab is on, and the way to the others. `+` starts a new conversation (a new
 * session on the terminal). The spacer between them is bare bar, so the tile
 * still drags from there. In the dock the Agent tile adopts the bar of the
 * surface showing (dock-tab.js) and puts its ⋯ and × after it; in the pane's
 * own window × is Reattach. The pane's other actions are rows in that ⋯
 * (agent-menu.js).
 *
 * {@link buildAgentBar} builds the chat console's bar (chat-console.js). The
 * terminal's bar is markup in index.html (`.terminal-header`), whose ids
 * app.js and terminal.js wire; agent-bar.test.mjs holds the two to one shape.
 *
 * The bar imports none of the dock's modules (the tile bar, its ⋯ menu): the
 * dock adopts the bar (dock-workspace.js hands it to dock-tab.js), never the
 * other way round.
 *
 * @module agent-bar
 */

import { icon } from '/design-system/js/icons.js';
import { onChat } from './chat-events.js';
import { elem } from './chat-dom.js';

/**
 * The chat console's Agent bar: the label the session switcher attaches to,
 * the spacer and `+` (New conversation). `+` is withheld while
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
  bar.append(label, elem('span', 'agent-bar-spacer'), newBtn);
  return { bar, label, newBtn };
}
