// @ts-check
/* OSPREY Web Terminal — Pop out a tile, attached
 *
 * The tile header's Pop out: a tile's content moves into a window the hub
 * drives (attached-window.js) and comes back when that window closes. The
 * tile stays in the dock meanwhile, showing a stand-in — "<panel> is in its
 * own window" and a Bring back button — so the layout, the stored per-session
 * arrangements and the agent's view of the dock never see a pop-out at all.
 *
 *   a service panel — its overlay iframe (dock-iframe.js) moves. The frame
 *     loads again in the window (a browser reloads a frame that changes
 *     document) and is the same element throughout, so every hub reference
 *     to it, its messages and its theme and session broadcasts stay as they
 *     were. A panel at an opaque origin cannot go: it takes its theme only
 *     from its parent window, which would no longer be the hub.
 *   the Agent pane — the `.terminal-panel` subtree (chat console and
 *     terminal) moves without a reload: its SSE and WebSocket clients, its
 *     composer, slash list, queue and attachments are the hub's own objects.
 *     The tile keeps the Agent bar. `html[data-agent="popped"]` marks it
 *     (dock-workspace.js).
 *
 * A node comes back to its tile when its window closes, on Bring back, and
 * when the hub page goes away. A tile closed meanwhile is opened again when
 * the content returns (never on the hub's unload: that page is going).
 *
 * The other pop-out, "Open in new window" (popout.js), opens a panel's
 * standalone page instead; it stays on the panel's menus.
 */

import { TERMINAL_RAIL_ID, titleCase } from './panel-catalog.js';
import { focusAttached, isAttached, openAttached, returnAttached } from './attached-window.js';
import { isPanelDocked, managedIframe, setIframeStandIn } from './dock-iframe.js';
import { agentPaneContent, isTerminalPanelOpen, openTerminalPanel } from './dock-workspace.js';
import { fitTerminal } from './terminal.js';
import { showUndoNotice } from './undo-notice.js';
import { redockPanel } from './popout.js';

export { focusAttached, isAttached };

/** The notice shown when the browser refused the window. */
const BLOCKED_MESSAGE = 'Pop-ups are blocked for this site.';

/**
 * Why the tile's content cannot pop out now, or null when it can: the Agent
 * pane once the page has it, a service panel once its iframe exists and does
 * not run at an opaque origin.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @returns {string | null}
 */
export function attachBlocker(id) {
  if (id === TERMINAL_RAIL_ID) return agentPaneContent() ? null : 'The Agent pane is not ready.';
  const managed = managedIframe(id);
  if (!managed) return 'This panel has not loaded yet.';
  if (managed.iframe.dataset.panelFrame === 'opaque') {
    return 'This panel runs at an opaque origin and cannot be popped out.';
  }
  return null;
}

/**
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @returns {boolean}
 */
export function canAttach(id) {
  return attachBlocker(id) === null;
}

/**
 * Pop a tile's content out into an attached window, or bring its open window
 * to the front. Runs inside the operator's gesture.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @returns {boolean} true when the content is in a window afterwards
 */
export function popOutAttached(id) {
  if (isAttached(id)) return focusAttached(id);
  if (!canAttach(id)) return false;
  const opened = id === TERMINAL_RAIL_ID ? popOutAgent() : popOutService(id);
  if (!opened) showUndoNotice({ message: BLOCKED_MESSAGE });
  return opened;
}

/**
 * Bring a tile's content back and close its window.
 * @param {string} id
 */
export function bringBackAttached(id) {
  returnAttached(id);
}

/** @returns {boolean} */
function popOutAgent() {
  const content = /** @type {HTMLElement} */ (agentPaneContent());
  const win = openAttached({
    key: TERMINAL_RAIL_ID,
    title: 'Agent · OSPREY',
    content,
    standIn: buildStandIn('Agent', TERMINAL_RAIL_ID),
    onResize: () => fitTerminal(),
    onReturn: (reason) => {
      if (reason !== 'unload' && !isTerminalPanelOpen()) openTerminalPanel();
      fitTerminal();
    },
  });
  if (win) fitTerminal();
  return win !== null;
}

/**
 * @param {string} id
 * @returns {boolean}
 */
function popOutService(id) {
  const { iframe, title } = /** @type {{iframe: HTMLIFrameElement, title: string}} */ (
    managedIframe(id)
  );
  const label = titleCase(title);
  const standIn = buildStandIn(label, id);
  const win = openAttached({
    key: id,
    title: `${label} · OSPREY`,
    content: iframe,
    standIn,
    onReturn: (reason) => {
      setIframeStandIn(id, null);
      if (reason !== 'unload' && !isPanelDocked(id)) redockPanel(id);
    },
  });
  if (!win) return false;
  setIframeStandIn(id, standIn);
  return true;
}

/**
 * The card a tile shows while its content is in a window.
 * @param {string} label
 * @param {string} id
 * @returns {HTMLElement}
 */
function buildStandIn(label, id) {
  const card = document.createElement('div');
  card.className = 'attached-standin';
  card.setAttribute('role', 'status');
  const text = document.createElement('p');
  text.className = 'attached-standin-text';
  text.textContent = `${label} is in its own window.`;
  const back = document.createElement('button');
  back.type = 'button';
  back.className = 'osprey-btn attached-standin-back';
  back.textContent = 'Bring back';
  back.addEventListener('click', () => bringBackAttached(id));
  const show = document.createElement('button');
  show.type = 'button';
  show.className = 'osprey-btn attached-standin-show';
  show.textContent = 'Show window';
  show.addEventListener('click', () => focusAttached(id));
  card.append(text, show, back);
  return card;
}
