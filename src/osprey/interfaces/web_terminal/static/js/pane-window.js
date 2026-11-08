// @ts-check
/* OSPREY Web Terminal — A pane in a window of its own
 *
 * The one way a pane leaves the dock: its live content moves into a window the
 * hub drives (attached-window.js) and its tile closes, leaving no shell. Callers
 * reach for this module, whichever gesture they serve; the layout holder
 * (dock-apply.js, dock-detached.js) owns what brings a pane back, and this module
 * owns only the windows.
 *
 *   a service panel — its overlay iframe moves. The frame loads again in the
 *     window (a browser reloads a frame that changes document), so its `src` is
 *     set to the page it is on first. It is the same element throughout, so every
 *     hub reference to it, its messages and its session broadcasts stay as they
 *     were; the theme is sent here, because theme-manager reaches only the frames
 *     in the hub's document. A panel at an opaque origin cannot go: it takes its
 *     theme only from its parent window, which would no longer be the hub. A frame
 *     whose kind the hub is still asking about opens its window at once, inside the
 *     gesture, and moves in when the answer comes; an opaque answer closes the window
 *     again and moves nothing.
 *   the Agent pane — the `.terminal-panel` subtree moves without a reload: its
 *     SSE and WebSocket clients, its composer, slash list, queue and attachments
 *     are the hub's own objects.
 *
 * WAYS BACK
 *   - Reattach: the pane docks where it was (R6), and its window closes. With no
 *     room the holder keeps it and the window stays.
 *   - The window closes (R4): the pane is closed. Its content goes back to the hub
 *     out of sight, and a row click opens it by the open rule.
 *   - A reload of the window keeps the pane detached, on its page (R10).
 *   - The hub page goes away (R7): the windows close and nothing else changes, so
 *     a save that follows still writes the panes docked where they were.
 *
 * This module imports no part of the panel stack, so the sidebar, the tile bar and
 * the dock adapter can all import it without a cycle: panel-manager and
 * dock-workspace each hand it what they own through {@link initPaneWindow}.
 */

import {
  closeAllAttached,
  closeAttached,
  focusAttached,
  hasWindow,
  openAttached,
  returnAttached,
} from './attached-window.js';
import { getLayoutHolder } from './dock-apply.js';
import { TERMINAL_RAIL_ID, titleCase } from './panel-catalog.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** The window name prefix; the pane's id follows it, scoped per render. */
const WINDOW_NAME_PREFIX = 'osprey-pane-';

/** The attribute on a window's mount that says the content in it is detached. */
const DETACHED_ATTRIBUTE = 'data-osprey-detached';

/** Why a panel with no frame to move cannot leave yet. */
const NOT_LOADED = 'This panel has not loaded yet.';

/** Why a panel at an opaque origin cannot leave. */
const OPAQUE = 'This panel runs at an opaque origin and cannot open in a new window.';

/** The notice shown when the browser refused the window. */
const BLOCKED_MESSAGE = 'Pop-ups are blocked for this site.';

/**
 * What the panel stack hands this module (all of it through
 * {@link initPaneWindow}, in parts; nothing here imports those modules).
 * @typedef {object} PaneWindowDeps
 * @property {(id: string) => {iframe: HTMLIFrameElement, title: string} | null} managedIframe
 * @property {(id: string) => boolean} canLoadFrame - a panel with no frame can get one now
 * @property {(id: string) => void} loadFrame - build a panel's frame out of sight, with no tile
 * @property {(id: string) => void} concealPanel - take a panel off screen, its frame cached
 * @property {(id: string) => void} panelReturned - a panel's frame is back in the overlay
 * @property {(id: string) => string | null} getPanelPage - the page a frame shows, if readable
 * @property {(page: string) => string} buildEmbedSrc - a page's address as a frame's `src`
 * @property {(iframe: HTMLIFrameElement) => void} sendThemeToIframe
 * @property {(listener: () => void) => unknown} subscribeTheme
 * @property {(id: string) => void} clearAccent - the rail accent leaves the panel, if on it
 * @property {(message: string, doc?: Document) => void} showNotice - in `doc`, or the hub's
 * @property {(id: string, options: {mode: 'detached', onReattach: () => void}) => WindowBar} createTileTab
 * @property {() => HTMLElement | null} agentPaneContent - the Agent's subtree once adopted
 * @property {() => void} fitAgent - refit the Agent's terminal to its box
 * @property {<T>(fn: () => T) => T} withEchoSuppressed - run without reporting a focus
 * @property {(doc: Document) => void} installPlaceMenus - wire a window's document for
 *   the shell's place menus (place-menu.js), idempotent per document
 */

/**
 * The bar of a window: the tile's own header bar (dock-tab.js).
 * @typedef {object} WindowBar
 * @property {HTMLElement} element
 * @property {(params: {title: string}) => void} init
 * @property {() => void} attached - `element` is in the window's document
 * @property {() => void} dispose
 */

/** @type {Partial<PaneWindowDeps>} */
const deps = {};

/** @type {Set<(id: string, detached: boolean) => void>} */
const listeners = new Set();

/**
 * Where the hub sent a detached frame, until the frame's `load` says it arrived.
 * Until then the frame still shows the page it is leaving, or has no document at
 * all: what it reports is not where it is going.
 * @type {Map<string, string>}
 */
const navigating = new Map();

let unloadWired = false;

/**
 * Hand this module part of what it needs. Both panel-manager and dock-workspace
 * call it; each part merges into what is there.
 * @param {Partial<PaneWindowDeps>} part
 */
export function initPaneWindow(part) {
  const followsTheme = deps.subscribeTheme !== undefined;
  Object.assign(deps, part);
  if (!followsTheme) deps.subscribeTheme?.(sendThemeToDetached);
}

/** Theme-manager reaches the frames in the hub's document; these are not. */
function sendThemeToDetached() {
  for (const id of getLayoutHolder()?.detachedKeys() ?? []) {
    const iframe = deps.managedIframe?.(id)?.iframe;
    if (iframe) deps.sendThemeToIframe?.(iframe);
  }
}

/**
 * The page a detached frame is on, or on its way to.
 * @param {string} id
 * @returns {string | null}
 */
function readPage(id) {
  return navigating.get(id) ?? deps.getPanelPage?.(id) ?? null;
}

/**
 * The hub sent a frame to a new page. A detached frame's next document loads
 * what the holder's entry says (a reload of its window, a reattach), so the entry
 * has to know first. A no-op for a pane that is not detached.
 * @param {string} id
 * @param {string} page
 */
export function noteDetachedNavigation(id, page) {
  const holder = getLayoutHolder();
  if (!holder?.isDetached(id)) return;
  navigating.set(id, page);
  holder.setDetachedPage(id, page);
}

/**
 * Why a pane cannot leave for a window now, or null when it can.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @returns {string | null}
 */
export function detachBlocker(id) {
  const holder = getLayoutHolder();
  if (!holder || !deps.withEchoSuppressed) return 'The web terminal is not ready.';
  if (id === TERMINAL_RAIL_ID) {
    if (!deps.agentPaneContent?.() || !deps.fitAgent) return 'The Agent pane is not ready.';
  } else {
    const managed = deps.managedIframe?.(id);
    if (!managed && !deps.canLoadFrame?.(id)) return NOT_LOADED;
    const kind = managed?.iframe.dataset.panelFrame;
    if (kind === 'opaque') return OPAQUE;
  }
  return null;
}

/**
 * Move a pane into a window of its own, or bring its open window to the front.
 * Runs inside the operator's gesture, the one the browser needs to open a window.
 * @param {string} id  service panel id, or TERMINAL_RAIL_ID
 * @returns {boolean} true when the pane has a window afterwards, or is about to
 */
export function detach(id) {
  if (hasWindow(id)) return focusAttached(id);
  if (detachBlocker(id) !== null) return false;
  if (id !== TERMINAL_RAIL_ID && !deps.managedIframe?.(id)) {
    // Never opened: its frame is built now, inside the gesture. What it is may
    // only be known from here on, so it is asked again.
    deps.loadFrame?.(id);
    const blocker = deps.managedIframe?.(id) ? detachBlocker(id) : NOT_LOADED;
    if (blocker !== null) {
      deps.showNotice?.(blocker);
      return false;
    }
  }
  const hadTile = getLayoutHolder()?.holds(id) ?? false;
  const opened = id === TERMINAL_RAIL_ID ? detachAgent(hadTile) : detachService(id, hadTile);
  if (!opened) deps.showNotice?.(BLOCKED_MESSAGE);
  else wireUnload();
  return opened;
}

/**
 * The tile leaves the dock once the window is ready for what is in it. False when
 * the pane had a tile at the gesture and it was closed or replaced during the
 * wait: then nothing moves.
 * @param {string} id
 * @param {string | null} page
 * @param {boolean} hadTile  whether the pane had a tile at the gesture
 * @returns {boolean}
 */
function leaveDock(id, page, hadTile) {
  const holder = getLayoutHolder();
  if (!holder || (hadTile && !holder.holds(id))) return false;
  return /** @type {boolean} */ (deps.withEchoSuppressed?.(() => holder.detach(id, page)));
}

/**
 * Tell the subscribers.
 * @param {string} id
 * @param {boolean} detached
 */
function notify(id, detached) {
  for (const listener of [...listeners]) {
    try {
      listener(id, detached);
    } catch (err) {
      console.error('pane-window: a change listener threw', err);
    }
  }
}

/**
 * The window's end of a detached pane, the same for both kinds.
 * @param {string} id
 * @param {string} label
 * @param {HTMLElement} content
 */
function windowSpec(id, label, content) {
  const anchor = document.createElement('div');
  anchor.hidden = true;
  anchor.dataset.paneAnchor = id;
  return {
    key: id,
    name: scopedStorageKey(WINDOW_NAME_PREFIX + id),
    title: `${label} · OSPREY`,
    content,
    anchor,
    // The window was navigated away from the shell: the pane still has a way home.
    onShellLost: () => {
      if (!reattach(id)) closeDetached(id);
    },
  };
}

/**
 * The window's bar, built again on every dress: a reload of the window brings a
 * new document with nothing on it (R10).
 * @param {string} id
 * @param {string} label
 * @returns {{dress: (mount: HTMLElement) => void, dispose: () => void}}
 */
function windowBar(id, label) {
  /** @type {WindowBar | null} */
  let tab = null;
  const dispose = () => {
    tab?.dispose();
    tab?.element.remove();
    tab = null;
  };
  return {
    dispose,
    dress: (mount) => {
      dispose();
      const doc = mount.ownerDocument;
      tab =
        deps.createTileTab?.(id, {
          mode: 'detached',
          onReattach: () => void reattach(id, doc),
        }) ?? null;
      if (!tab) return;
      tab.init({ title: label });
      mount.prepend(tab.element);
      tab.attached();
    },
  };
}

/**
 * @param {boolean} hadTile
 * @returns {boolean}
 */
function detachAgent(hadTile) {
  const content = deps.agentPaneContent?.();
  if (!content) return false;
  let told = false;
  const fit = () => deps.fitAgent?.();
  const bar = windowBar(TERMINAL_RAIL_ID, 'Agent');
  return (
    openAttached({
      ...windowSpec(TERMINAL_RAIL_ID, 'Agent', content),
      beforeMove: () => leaveDock(TERMINAL_RAIL_ID, null, hadTile),
      onDress: (mount) => {
        mount.setAttribute(DETACHED_ATTRIBUTE, 'true');
        // Again after a reload: a fresh document; once per document otherwise.
        deps.installPlaceMenus?.(mount.ownerDocument);
        bar.dress(mount);
        fit();
        if (!told) notify(TERMINAL_RAIL_ID, true);
        told = true;
      },
      onPageHide: bar.dispose,
      onResize: fit,
      beforeReturn: bar.dispose,
      onReturn: (reason) => {
        if (reason === 'closed') getLayoutHolder()?.forget(TERMINAL_RAIL_ID);
        fit();
        notify(TERMINAL_RAIL_ID, false);
      },
    }) !== null
  );
}

/**
 * @param {string} id
 * @param {boolean} hadTile
 * @returns {boolean}
 */
function detachService(id, hadTile) {
  const managed = deps.managedIframe?.(id);
  if (!managed) return false;
  const { iframe, title } = managed;
  const bar = windowBar(id, titleCase(title));
  /** Where it was, as last read: the holder's entry is gone once it has docked. @type {string | null} */
  let page = null;
  let told = false;
  /** Read the page the frame is on into the holder's entry (and its saved copy). */
  const notePage = () => {
    page = readPage(id) ?? page;
    getLayoutHolder()?.setDetachedPage(id, page);
  };
  /** The frame has arrived: from here on it says where it is. */
  const onFrameLoad = () => {
    navigating.delete(id);
    notePage();
  };
  const loadPage = () => {
    const target = getLayoutHolder()?.detachedEntry(id)?.page ?? page;
    if (target !== null && deps.buildEmbedSrc) iframe.src = deps.buildEmbedSrc(target);
  };
  return (
    openAttached({
      ...windowSpec(id, titleCase(title), iframe),
      // Not known yet to be first-party: it may still turn out opaque.
      holdMove: () => iframe.dataset.panelFrame === 'pending',
      onGiveUp: () => deps.showNotice?.(NOT_LOADED),
      beforeMove: () => {
        // What the frame is may have become known during the wait.
        const blocker = detachBlocker(id);
        if (blocker !== null) {
          deps.showNotice?.(blocker);
          return false;
        }
        page = deps.getPanelPage?.(id) ?? null;
        // A panel with no tile has no live record for the holder to keep.
        if (!hadTile) getLayoutHolder()?.setPanelRecord(id, { title });
        if (!leaveDock(id, page, hadTile)) return false;
        deps.clearAccent?.(id);
        return true;
      },
      beforeDress: loadPage,
      onDress: (mount) => {
        mount.setAttribute(DETACHED_ATTRIBUTE, 'true');
        // Again after a reload: a fresh document; once per document otherwise.
        deps.installPlaceMenus?.(mount.ownerDocument);
        bar.dress(mount);
        if (!told) {
          iframe.addEventListener('load', onFrameLoad);
          notify(id, true);
        }
        told = true;
      },
      onPageHide: () => {
        notePage();
        bar.dispose();
      },
      beforeReturn: () => {
        bar.dispose();
        notePage();
        iframe.removeEventListener('load', onFrameLoad);
        loadPage();
        navigating.delete(id);
      },
      onReturn: (reason) => {
        if (reason === 'closed') {
          // In this order: a detached frame is not the adapter's to conceal.
          getLayoutHolder()?.forget(id);
          deps.concealPanel?.(id);
        } else deps.panelReturned?.(id);
        notify(id, false);
      },
    }) !== null
  );
}

/**
 * Bring a detached pane's window to the front. The browser may decline outside a
 * gesture; the window is still the pane's home either way.
 * @param {string} id
 * @returns {boolean} true when the pane has a window
 */
export function focusWindow(id) {
  return focusAttached(id);
}

/**
 * Dock a detached pane where it was and close its window. With no room for it
 * the operator is told, and the pane and its window stay as they are.
 * @param {string} id
 * @param {Document} [noticeIn]  where to say so, when the gesture came from the
 *   pane's window (default: the hub)
 * @returns {boolean} whether it docked
 */
export function reattach(id, noticeIn) {
  const holder = getLayoutHolder();
  if (!holder?.isDetached(id)) return false;
  // Read while the frame still shows it: the window closes below.
  holder.setDetachedPage(id, readPage(id));
  const back = deps.withEchoSuppressed?.(() => holder.reattach(id));
  if (!back) {
    deps.showNotice?.(`No room to reattach ${paneLabel(id)}.`, noticeIn);
    return false;
  }
  returnAttached(id);
  return true;
}

/**
 * Close a detached pane with its window, the way closing the window does (R4).
 * @param {string} id
 * @returns {boolean} whether it was detached
 */
export function closeDetached(id) {
  if (!getLayoutHolder()?.isDetached(id)) return false;
  closeAttached(id);
  return true;
}

/**
 * @param {string} id
 * @returns {string} the name the operator knows the pane by
 */
function paneLabel(id) {
  if (id === TERMINAL_RAIL_ID) return 'Agent';
  return titleCase(deps.managedIframe?.(id)?.title ?? id);
}

/**
 * Subscribe to every pane arriving in a window and every return from one.
 * @param {(id: string, detached: boolean) => void} callback
 * @returns {() => void} unsubscribe
 */
export function onDetachedChange(callback) {
  listeners.add(callback);
  return () => {
    listeners.delete(callback);
  };
}

/**
 * Close every window and change nothing else: the detached panes stay on the
 * holder, whose stored form keeps them docked (R7). The hub's `pagehide` only.
 */
export function closeAllDetached() {
  closeAllAttached();
}

/** Windows cannot outlive the page that drives them. Wired once, on first use. */
function wireUnload() {
  if (unloadWired) return;
  unloadWired = true;
  window.addEventListener('pagehide', closeAllDetached);
}
