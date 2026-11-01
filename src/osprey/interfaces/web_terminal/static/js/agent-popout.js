// @ts-check
/* OSPREY Web Terminal — The Agent pane in its own window
 *
 * The Simple view's Agent pane (the chat console) pops out into a popup
 * window on the panel pop-out protocol (popout.js, popout-client.js), under
 * the pane's reserved id `terminal`. The window is the hub page itself,
 * opened as the chat alone: `/?pane=agent&popout=true&theme=<id>&mode=simple`.
 * app.js boots only the chat and the bridges below on that page; index.html
 * stamps `html[data-pane="agent"]` before the first paint, and terminal.css
 * lays the page out as the chat alone.
 *
 * HUB SIDE ({@link initAgentPopout}, called once by app.js)
 * --------
 * The pane is registered as a pop-out surface for its id. While its window is
 * open the hub's Agent tile is closed and the hub's console is DORMANT
 * (chat.js sleepChat): the window's console runs the session, and one key is
 * one agent. The sidebar's Agent row shows the popped-out state and a click on
 * it brings the window to the front (panel-menu-policy.js); no layout path
 * re-docks the tile (dock-workspace.js asks isPoppedOut). When the window
 * goes away the tile comes back and the console wakes, reading the pointer
 * the window may have moved.
 *
 * The window reports three things on top of hello and bye, each taken only
 * from the window the hub holds for `terminal` (popout.js onPopoutMessage):
 *
 *   `osprey-popout-session {session_id}`   the window's console bound a key:
 *       its first bind, New conversation, an earlier conversation picked. The
 *       pointer itself is shared storage, so the hub only tells its panels
 *       and its other windows (notifySessionChange).
 *   `osprey-popout-attention {waiting}`    the window's approval count, put
 *       on the hub's chat bus so the Agent row's badge shows it.
 *   `osprey-popout-navigate {panel, url}`  a link in an answer that opens a
 *       panel; the hub opens it as it opens its own answer links.
 *
 * The hub forwards one thing to the window: a panel's
 * `osprey-paste-to-terminal`, which the hub's own composer cannot take while
 * it is away (app.js initIframePasteBridge → {@link forwardPasteToAgent}).
 *
 * The hub hosts an Agent window in the Simple view only. A flip to Expert
 * takes the window back first ({@link recallAgentPopout}, app.js), and a hub
 * that adopts the window while in Expert (a reload with `?mode=expert`) takes
 * it back as soon as it has adopted it.
 *
 * WINDOW SIDE ({@link initAgentPopoutPage}, called by app.js's agent-only boot)
 * -----------
 * Reports to the hub as above, takes the hub's paste only from its opener at
 * its own origin, and follows the hub's view mode: an `osprey-mode-change` to
 * Expert puts its console to sleep, because the hub is about to close it. It
 * never picks a mode itself.
 */

import { fetchJSON, withPrefix } from './api.js';
import { sleepChat, wakeChat } from './chat.js';
import { emitChat, onChat } from './chat-events.js';
import { closeTerminalPanel, openTerminalPanel } from './dock-workspace.js';
import { insertPrompt } from './first-contact.js';
import { TERMINAL_RAIL_ID } from './panel-catalog.js';
import { buildPopoutSrc } from './panel-iframe-sync.js';
import {
  onPopoutMessage,
  postToPopout,
  recallPanel,
  registerPopoutSurface,
  isPoppedOut,
} from './popout.js';
import { getPointer, subscribe as subscribeToPointer } from './session-pointer.js';
import { notifySessionChange } from './terminal.js';
import { onModeChange } from '/design-system/js/frame-params.js';
import { initPopoutClient, postToOpener } from '/design-system/js/popout-client.js';

/** The query parameter that opens the hub page as the Agent pane alone, and its value. */
const PANE_PARAM = 'pane';
const AGENT_PANE = 'agent';

const SESSION = 'osprey-popout-session';
const ATTENTION = 'osprey-popout-attention';
const NAVIGATE = 'osprey-popout-navigate';

/** The Channel Finder's paste, forwarded under its own type. */
const PASTE = 'osprey-paste-to-terminal';

/** The element the Agent window is sized like: the pane's content box. */
const PANE_SELECTOR = '.terminal-panel';

/* ---- Hub side ---------------------------------------------------------- */

function isSimpleMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple';
}

/**
 * The page the Agent pane pops out to, or null outside the Simple view: the
 * Expert view keeps its terminal docked, so nothing offers the pop-out there
 * (the tile button is not built, the menu and palette rows are withheld, and
 * canPopOut says no).
 * @returns {string | null}
 */
export function agentPopoutUrl() {
  if (!isSimpleMode()) return null;
  return buildPopoutSrc(withPrefix(`/?${PANE_PARAM}=${AGENT_PANE}`));
}

/** @returns {boolean} whether the Agent pane runs in its own window */
export function isAgentPoppedOut() {
  return isPoppedOut(TERMINAL_RAIL_ID);
}

/**
 * The hub adopted the Agent window: the tile goes, and the hub's console
 * sleeps so that only the window's console runs the session. A hub that is
 * not in the Simple view takes the window straight back — after the adoption
 * has finished, so the protocol's own bookkeeping is not re-entered.
 */
function concealAgent() {
  closeTerminalPanel();
  sleepChat();
  if (!isSimpleMode()) queueMicrotask(recallAgentPopout);
}

/**
 * The Agent window went away: forget the count it forwarded, put the tile
 * back where the default layout puts it, and wake the console, which binds
 * to the pointer the window may have moved and replays what it missed.
 */
function redockAgent() {
  emitChat('attention', { waiting: 0 });
  openTerminalPanel();
  wakeChat();
}

/**
 * Close the Agent window and put the pane back now, without waiting for the
 * window's bye: the flip to Expert runs this before its own dock and session
 * steps, which need the pane docked and the console awake.
 * @returns {boolean} true when a window was open
 */
export function recallAgentPopout() {
  return recallPanel(TERMINAL_RAIL_ID);
}

/**
 * Hand a panel's paste to the Agent window, where the composer is while the
 * pane is popped out.
 * @param {string} text
 * @returns {boolean} true when the window took it; false when the pane is docked
 */
export function forwardPasteToAgent(text) {
  return isAgentPoppedOut() && postToPopout(TERMINAL_RAIL_ID, { type: PASTE, text });
}

/**
 * Whether a forwarded panel address may reach the hub's panel frames: empty
 * (open the panel as it is), or root-relative and not protocol-relative. The
 * same rule app.js applies to a panel's own `osprey:navigate`, for the same
 * reason — the address ends up in an iframe `src`.
 * @param {unknown} url
 * @returns {url is string}
 */
function isPanelAddress(url) {
  return typeof url === 'string' && (url === '' || (url.startsWith('/') && !url.startsWith('//')));
}

/**
 * Register the Agent pane as a pop-out surface and take the window's reports.
 * Called once by app.js, after the panel manager has started the protocol.
 * @param {{ navigate: (panelId: string, url: string) => void }} deps - the
 *   hub's human-local panel open (panel-manager.js navigateAndActivatePanel)
 */
export function initAgentPopout({ navigate }) {
  registerPopoutSurface(TERMINAL_RAIL_ID, {
    popoutUrl: agentPopoutUrl,
    conceal: concealAgent,
    redock: redockAgent,
    tile: () => document.querySelector(PANE_SELECTOR),
  });
  onPopoutMessage(SESSION, (panelId, data) => {
    if (panelId !== TERMINAL_RAIL_ID) return;
    if (typeof data.session_id !== 'string' || !data.session_id) return;
    notifySessionChange(data.session_id);
  });
  onPopoutMessage(ATTENTION, (panelId, data) => {
    if (panelId !== TERMINAL_RAIL_ID) return;
    const waiting = Number(data.waiting);
    emitChat('attention', { waiting: Number.isFinite(waiting) ? Math.max(0, waiting) : 0 });
  });
  onPopoutMessage(NAVIGATE, (panelId, data) => {
    if (panelId !== TERMINAL_RAIL_ID) return;
    if (typeof data.panel !== 'string' || !data.panel || !isPanelAddress(data.url)) return;
    navigate(data.panel, data.url);
  });
}

/* ---- Window side ------------------------------------------------------- */

/** @returns {boolean} whether this page was opened as the Agent pane alone */
export function isAgentPopoutPage() {
  return new URLSearchParams(window.location.search).get(PANE_PARAM) === AGENT_PANE;
}

/** The panels the hub hosts, read once from /api/panels. @type {Set<string>} */
let hostedPanels = new Set();

/**
 * Whether the hub hosts a panel, for the answer links: the hub answers with
 * its enabled and custom panels, the same roster its own panel manager seeds
 * from. Until the roster is read, every panel link opens in a new tab, as a
 * link to a panel the hub does not host does.
 * @param {string} panelId
 * @returns {boolean}
 */
export function isHubPanel(panelId) {
  return hostedPanels.has(panelId);
}

/**
 * The opener, while it is still a window of this origin.
 * @returns {Window | null}
 */
function hub() {
  try {
    const opener = window.opener;
    return opener && !opener.closed ? opener : null;
  } catch {
    return null;
  }
}

/**
 * Open a panel in the hub, from a link in an answer shown here: the hub
 * opens it as it opens its own answer links, and comes to the front where
 * the browser allows that from this window's click.
 * @param {string} panelId
 * @param {string} url
 */
export function openPanelInHub(panelId, url) {
  if (!postToOpener({ type: NAVIGATE, panel: panelId, url })) return;
  try {
    hub()?.focus();
  } catch {
    /* the browser declined — the panel is open in the hub all the same */
  }
}

/**
 * Start the window side: report to the hub, forward the console's session
 * and attention, take the hub's paste, follow the hub's view mode and read
 * the hub's panel roster for the answer links. Called by app.js on the
 * agent-only page after the chat is mounted, so the first session report
 * carries the key the console bound.
 */
export function initAgentPopoutPage() {
  initPopoutClient();
  const key = getPointer();
  if (key) postToOpener({ type: SESSION, session_id: key });
  subscribeToPointer((sessionId) => {
    if (sessionId) postToOpener({ type: SESSION, session_id: sessionId });
  });
  onChat('attention', ({ waiting }) => {
    postToOpener({ type: ATTENTION, waiting });
  });
  window.addEventListener('message', (event) => {
    // The paste is taken from the opener alone: another same-origin page has
    // no business putting text in this composer, and a page with no opener
    // has no hub to take one from.
    const opener = hub();
    if (!opener || event.origin !== window.location.origin || event.source !== opener) return;
    const data = event.data;
    if (!data || data.type !== PASTE || typeof data.text !== 'string' || !data.text) return;
    insertPrompt(data.text, { append: true });
    window.focus();
  });
  // The hub closes this window on its flip to Expert; until it has, the
  // composer must not send, or the key would be taken from the terminal.
  onModeChange((mode) => {
    if (mode === 'expert') sleepChat();
    else wakeChat();
  });
  void fetchJSON('/api/panels').then(
    (config) => {
      const enabled = Array.isArray(config?.enabled) ? config.enabled : [];
      const custom = Array.isArray(config?.custom) ? config.custom : [];
      hostedPanels = new Set(
        [...enabled, ...custom.map((/** @type {{ id?: unknown }} */ panel) => panel?.id)].filter(
          (id) => typeof id === 'string'
        )
      );
    },
    (err) => {
      console.warn('Agent window: the panel roster is unavailable:', err);
    }
  );
}
