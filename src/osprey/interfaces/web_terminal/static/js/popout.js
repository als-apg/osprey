// @ts-check
/* OSPREY Web Terminal — Pop-out windows
 *
 * The hub side of the pop-out protocol: a service tile moved into its own
 * browser window and brought back when that window closes. The window is a
 * popup the hub opens and keeps a handle on, not a detached tab — that handle
 * is what lets the hub keep the window's theme, view mode and session in step
 * with its own, bring the window to the front when the operator or the agent
 * asks for the panel, and put the tile back the moment the window goes away.
 *
 * STATE
 * -----
 * One Map, panel id → WindowProxy, is the whole state: a panel is popped out
 * exactly while its id is in the Map. The tile is retired locally the moment
 * the window opens (the panel keeps its rail row, which the sidebar marks), so
 * no layout report, layout restore or arrangement re-docks it while the
 * window is open — those paths all ask {@link isPoppedOut} first.
 *
 * MESSAGES (all same-origin)
 * --------------------------
 *   popup → hub   `osprey-popout-hello {panelId}`  on load, and in answer to
 *                 a roll-call. Adopts `event.source` as the panel's window.
 *   popup → hub   `osprey-popout-bye {panelId}`    on the popup's `pagehide`.
 *   popup → hub   any further type a surface registers through
 *                 {@link onPopoutMessage} (the Agent window's session,
 *                 attention and navigation reports, agent-popout.js).
 *   hub → every   `osprey-popout-roll-call`         once per hub page load, on
 *   popup         the BroadcastChannel named below: the hub's reload loses its
 *                 Map, the popups' `window.opener` survives it, so each popup
 *                 answers with a hello the fresh hub adopts.
 *   hub → popup   the state messages `osprey-theme-change`, `osprey-mode-change`
 *                 and `osprey-session-change`, exactly as the iframes get them,
 *                 on every hello and on every change (panel-manager wires the
 *                 change fan-outs through {@link broadcastToPopouts}), and
 *                 whatever a surface posts through {@link postToPopout}.
 *
 * The hub takes a message only from its own origin. A hello is taken from a
 * window this hub opened under the protocol's window name — read off the
 * source, which the same origin allows — because a hello is how a window is
 * ADOPTED and nothing else may claim a panel. A bye, and every registered
 * type, is taken only from the window the Map already holds for that panel,
 * so no other same-origin page can re-dock a tile or speak for a window.
 *
 * RETURN
 * ------
 * A popup's `pagehide` fires on close, on reload and on navigation alike, so a
 * bye alone cannot tell "gone" from "reloading". While any window is open the
 * hub polls `closed` every {@link CLOSED_POLL_MS}; a closed window is released
 * at the next tick. A window that said bye but is still open is given
 * {@link BYE_GRACE_MS} for its hello to arrive (a reload) before it is treated
 * as having left the protocol (navigated elsewhere) and released. Releasing
 * re-docks the tile through panel-manager, beside the active tile and without
 * a focus report — nobody asked the server for this surfacing.
 *
 * This module imports nothing from the panel stack: panel-manager hands it the
 * verbs it needs at init ({@link initPopouts}), the same seam pattern
 * panel-menu-policy.js and panel-placement.js use, so the sidebar, the tile
 * bar and the dock adapter can all import it without a cycle. A surface that
 * is not a service panel — the Agent pane — registers verbs of its own for
 * its id ({@link registerPopoutSurface}) and rides the same Map and protocol.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { showUndoNotice } from './undo-notice.js';

/**
 * The BroadcastChannel the hub rolls call on. Scoped like a storage key: on a
 * multi-user mount every render shares one origin, and a hub's roll-call is
 * for the popups of its own render.
 */
const CHANNEL = 'osprey-popout';

/** The window name a popup is opened under, followed by its panel id. */
const WINDOW_NAME_PREFIX = 'osprey-popout-';

const HELLO = 'osprey-popout-hello';
const BYE = 'osprey-popout-bye';
const ROLL_CALL = 'osprey-popout-roll-call';

/** The smallest window a tile may pop out into. */
const MIN_WIDTH = 480;
const MIN_HEIGHT = 360;

/** The window a panel pops out into when it holds no tile to measure. */
const DEFAULT_WIDTH = 960;
const DEFAULT_HEIGHT = 720;

/** How often the open windows are checked for `closed`. */
const CLOSED_POLL_MS = 500;

/** How long a window that said bye may take to say hello again (a reload). */
const BYE_GRACE_MS = 1500;

/** The notice shown when the browser refused the window. */
const BLOCKED_MESSAGE = 'Pop-ups are blocked for this site.';

/**
 * The panel-manager verbs this module acts through, registered once at init.
 * @typedef {object} PopoutDeps
 * @property {(id: string) => string | null} popoutUrl - the panel's standalone
 *   URL with the pop-out params, or null while the panel has none (its config
 *   has not resolved, or it runs at an opaque origin)
 * @property {(id: string) => void} conceal - retire the panel's tile locally;
 *   membership and the server are untouched
 * @property {(id: string) => void} redock - put the tile back, without a focus
 *   report
 * @property {(win: Window) => void} sync - send the hub's theme, mode and
 *   session to one window
 */

/**
 * The verbs of a pop-out surface that is not a service panel, registered for
 * its id by the module that owns it; consulted before the panel-manager deps.
 * @typedef {object} PopoutSurface
 * @property {() => string | null} popoutUrl - the page to open, or null while
 *   the surface cannot pop out (the Agent pane outside the Simple view)
 * @property {() => void} conceal - retire the surface's tile locally
 * @property {() => void} redock - put the tile back
 * @property {() => HTMLElement | null} tile - the element the window is sized
 *   like, or null while the surface holds no tile
 */

/** @type {PopoutDeps | null} */
let deps = null;

/** The open windows, by panel id. @type {Map<string, Window>} */
const windows = new Map();

/** The surfaces registered for ids that are not service panels. @type {Map<string, PopoutSurface>} */
const surfaces = new Map();

/** The handlers for the popups' further message types, by type. @type {Map<string, (panelId: string, data: Record<string, unknown>) => void>} */
const messageHandlers = new Map();

/** When each window said bye, by panel id, while its grace runs. @type {Map<string, number>} */
const byeAt = new Map();

/** @type {Set<(panelId: string, poppedOut: boolean) => void>} */
const listeners = new Set();

/** The `closed` poll, while any window is open. @type {ReturnType<typeof setInterval> | null} */
let pollHandle = null;

/**
 * The registered deps, or a hard failure: a pop-out before init is a wiring
 * bug, not a runtime condition to degrade around.
 * @returns {PopoutDeps}
 */
function ctx() {
  if (!deps) throw new Error('popout: initPopouts() has not run');
  return deps;
}

/**
 * Register the panel-manager verbs, start listening for the popups' messages
 * and roll call for the popups a previous page load of this hub opened.
 * Called once from initPanelManager.
 * @param {PopoutDeps} popoutDeps
 */
export function initPopouts(popoutDeps) {
  deps = popoutDeps;
  window.addEventListener('message', onMessage);
  // The channel is only ever written: the popups answer by posting to their
  // opener, which is this window, so there is nothing to keep open.
  const channel = new BroadcastChannel(scopedStorageKey(CHANNEL));
  channel.postMessage({ type: ROLL_CALL });
  channel.close();
}

/**
 * Register the verbs of a surface that is not a service panel, so its id can
 * pop out through every path a service panel does (the tile bar, the row
 * menu, the palette, {@link popOutPanel}). Called once by the module that
 * owns the surface, before or after {@link initPopouts}: the deps decide
 * nothing for a registered id.
 * @param {string} panelId
 * @param {PopoutSurface} surface
 */
export function registerPopoutSurface(panelId, surface) {
  surfaces.set(panelId, surface);
}

/**
 * Handle one further message type from a popup, on top of hello and bye.
 * The handler runs only for a message from this hub's own origin whose
 * source is the window the Map holds for the message's panel id — the same
 * check a bye passes — and gets that id with the message data. One handler
 * per type; a later registration replaces the earlier one.
 * @param {string} type
 * @param {(panelId: string, data: Record<string, unknown>) => void} handler
 */
export function onPopoutMessage(type, handler) {
  messageHandlers.set(type, handler);
}

/**
 * The surface's page to open, or the panel-manager's for a service panel.
 * @param {string} panelId
 * @returns {string | null}
 */
function popoutUrl(panelId) {
  const surface = surfaces.get(panelId);
  return surface ? surface.popoutUrl() : ctx().popoutUrl(panelId);
}

/** @param {string} panelId @returns {boolean} */
export function isPoppedOut(panelId) {
  return windows.has(panelId);
}

/**
 * Whether a panel can pop out: it has a first-party standalone page to open.
 * False before the seam is registered, while the panel's config has not
 * resolved a URL, and for a panel whose frame resolved to an opaque origin —
 * that page would open sandboxed, unable to report to the hub or follow it.
 * The tile bar's button and the menu's row disable on this; the palette's
 * rows are filtered by it.
 * @param {string} panelId
 * @returns {boolean}
 */
export function canPopOut(panelId) {
  return deps !== null && popoutUrl(panelId) !== null;
}

/**
 * Subscribe to pop-out changes. The callback gets the panel id and whether it
 * is now popped out, on every adoption and every release.
 * @param {(panelId: string, poppedOut: boolean) => void} callback
 * @returns {() => void} unsubscribe
 */
export function onPopoutChange(callback) {
  listeners.add(callback);
  return () => {
    listeners.delete(callback);
  };
}

/**
 * Open a panel in its own window, or bring an already open one to the front.
 * Runs inside the operator's gesture, which is what lets the browser open the
 * window at all. A refused window (a pop-up blocker) leaves the tile where it
 * is and says so.
 * @param {string} panelId
 * @returns {boolean} true when a window is open for the panel afterwards
 */
export function popOutPanel(panelId) {
  if (windows.has(panelId)) return focusPopout(panelId);
  const url = popoutUrl(panelId);
  if (url === null) return false;
  const win = window.open(url, WINDOW_NAME_PREFIX + panelId, features(panelId));
  if (!win) {
    showUndoNotice({ message: BLOCKED_MESSAGE });
    return false;
  }
  adopt(panelId, win);
  return true;
}

/**
 * Close a panel's window. The tile comes back through the window's own bye,
 * or the `closed` poll, like any other close.
 * @param {string} panelId
 */
export function bringBackPanel(panelId) {
  const win = windows.get(panelId);
  if (!win) return;
  try {
    win.close();
  } catch {
    /* the window navigated out of reach — the poll releases it */
  }
}

/**
 * Close a panel's window and put its tile back at once, without waiting for
 * the window's bye or the poll. For a caller whose next step needs the tile
 * in place: the flip to Expert takes the Agent back before the hand-off.
 * The window's own bye, if it still arrives, finds no held window and is
 * ignored.
 * @param {string} panelId
 * @returns {boolean} true when a window was open for the panel
 */
export function recallPanel(panelId) {
  const win = windows.get(panelId);
  if (!win) return false;
  try {
    win.close();
  } catch {
    /* the window navigated out of reach — it is no longer the panel's either way */
  }
  release(panelId);
  return true;
}

/**
 * Post one message to a panel's window, addressed to the hub's own origin:
 * the targeted twin of {@link broadcastToPopouts}, for what one surface
 * sends its own window (the Agent window gets a panel's paste).
 * @param {string} panelId
 * @param {Record<string, unknown>} message
 * @returns {boolean} true when the panel has an open window the message went to
 */
export function postToPopout(panelId, message) {
  const win = windows.get(panelId);
  if (!win || win.closed) return false;
  try {
    win.postMessage(message, window.location.origin);
  } catch {
    /* the window navigated out of reach — the poll releases it */
  }
  return true;
}

/**
 * Bring a panel's window to the front. The browser may decline outside an
 * operator gesture (an agent focus, say); the window is still the panel's
 * home either way.
 * @param {string} panelId
 * @returns {boolean} true when the panel has an open window
 */
export function focusPopout(panelId) {
  const win = windows.get(panelId);
  if (!win) return false;
  if (win.closed) {
    release(panelId);
    return false;
  }
  try {
    win.focus();
  } catch {
    /* the window navigated out of reach — the poll releases it */
  }
  return true;
}

/**
 * Send one message to every open window — the fan-out the hub's theme, mode
 * and session broadcasts run alongside their iframe loops.
 * @param {(win: Window) => void} send
 */
export function broadcastToPopouts(send) {
  for (const win of windows.values()) {
    if (!win.closed) send(win);
  }
}

/**
 * The `window.open` features for a panel: a popup sized like the panel's
 * tile — its iframe box, a registered surface's own tile, or a default when
 * it holds no tile — clamped to the window minimum and the screen, and
 * centred on the hub's screen.
 * @param {string} panelId
 * @returns {string}
 */
function features(panelId) {
  const surface = surfaces.get(panelId);
  const tile = surface
    ? surface.tile()
    : [...document.querySelectorAll('.panel-iframe')].find(
        (el) => el instanceof HTMLElement && el.dataset.panelId === panelId
      );
  const box = tile?.getBoundingClientRect();
  const wanted = box && box.width > 0 && box.height > 0 ? box : null;
  const { availWidth, availHeight } = window.screen;
  const availLeft = Number(/** @type {any} */ (window.screen).availLeft) || 0;
  const availTop = Number(/** @type {any} */ (window.screen).availTop) || 0;
  const width = clamp(Math.round(wanted?.width ?? DEFAULT_WIDTH), MIN_WIDTH, availWidth);
  const height = clamp(Math.round(wanted?.height ?? DEFAULT_HEIGHT), MIN_HEIGHT, availHeight);
  const left = availLeft + Math.max(0, Math.round((availWidth - width) / 2));
  const top = availTop + Math.max(0, Math.round((availHeight - height) / 2));
  return `popup=yes,width=${width},height=${height},left=${left},top=${top}`;
}

/**
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function clamp(value, min, max) {
  return Math.min(Math.max(value, min), Math.max(min, max));
}

/**
 * Take a window as a panel's: retire the tile, remember the handle, tell the
 * subscribers and watch for the window closing. A second adoption of the same
 * window is a reload's hello: nothing changes but the state the window needs
 * again.
 * @param {string} panelId
 * @param {Window} win
 */
function adopt(panelId, win) {
  byeAt.delete(panelId);
  if (windows.get(panelId) === win) return;
  windows.set(panelId, win);
  const surface = surfaces.get(panelId);
  if (surface) surface.conceal();
  else ctx().conceal(panelId);
  notify(panelId, true);
  if (pollHandle === null) pollHandle = setInterval(pollClosed, CLOSED_POLL_MS);
}

/**
 * Forget a panel's window and put its tile back.
 * @param {string} panelId
 */
function release(panelId) {
  if (!windows.delete(panelId)) return;
  byeAt.delete(panelId);
  notify(panelId, false);
  const surface = surfaces.get(panelId);
  if (surface) surface.redock();
  else ctx().redock(panelId);
  if (windows.size === 0 && pollHandle !== null) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

/**
 * The `closed` poll: release every window that has closed, and every window
 * whose bye has outlived its grace without a hello.
 */
function pollClosed() {
  const now = Date.now();
  for (const [panelId, win] of [...windows]) {
    const said = byeAt.get(panelId);
    if (win.closed || (said !== undefined && now - said >= BYE_GRACE_MS)) release(panelId);
  }
}

/**
 * @param {string} panelId
 * @param {boolean} poppedOut
 */
function notify(panelId, poppedOut) {
  for (const listener of listeners) {
    try {
      listener(panelId, poppedOut);
    } catch (err) {
      console.error('popout: a change listener threw', err);
    }
  }
}

/**
 * Whether a message source is a window this hub opened for `panelId`: a
 * same-origin window whose opener is this window and whose name is the one
 * {@link popOutPanel} gives it. Both reads need the same origin, which the
 * caller has already checked; a window that navigated elsewhere throws on
 * them and is refused.
 * @param {unknown} source
 * @param {string} panelId
 * @returns {source is Window}
 */
function isOwnPopup(source, panelId) {
  if (!source || typeof source !== 'object') return false;
  const win = /** @type {Window} */ (source);
  try {
    return win.opener === window && win.name === WINDOW_NAME_PREFIX + panelId;
  } catch {
    return false;
  }
}

/** @param {MessageEvent} event */
function onMessage(event) {
  if (event.origin !== window.location.origin) return;
  const data = event.data;
  if (!data || typeof data !== 'object' || typeof data.panelId !== 'string') return;
  const panelId = data.panelId;
  if (data.type === HELLO) {
    if (!isOwnPopup(event.source, panelId)) return;
    adopt(panelId, event.source);
    ctx().sync(event.source);
  } else if (data.type === BYE) {
    if (event.source !== windows.get(panelId)) return;
    byeAt.set(panelId, Date.now());
  } else {
    const handler = messageHandlers.get(String(data.type));
    if (!handler || event.source !== windows.get(panelId)) return;
    handler(panelId, data);
  }
}
