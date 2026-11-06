// @ts-check
/* OSPREY Web Terminal — Attached pop-out windows
 *
 * A pop-out that stays attached to the hub: the hub opens a BLANK same-origin
 * window, mirrors its stylesheets and its <html> attributes (theme, view,
 * Agent state) into it, and moves a live node of its own page into it. The
 * node keeps every listener, every reference the hub's modules hold and every
 * connection behind it, because nothing is rebuilt: the hub still runs it, the
 * window only shows it. Closing the window moves the node back to where it
 * was. This is the mechanic of Canopy's DetachedWindow (react-trace), without
 * React: a node move where Canopy renders a portal.
 *
 * The other pop-out, popout.js, opens a panel's standalone URL as a page of
 * its own ("Open in new window"). That page outlives the hub; this one does
 * not, and must not pretend to.
 *
 * WHERE THE NODE WAS
 * ------------------
 * The caller hands a stand-in, which takes the node's place for as long as the
 * window is open (the "<panel> is in its own window · Bring back" card). The
 * return swaps them back, so the node lands in the same parent at the same
 * position whatever moved around it, and its inline style is restored to the
 * one it left with. The layout is never touched: the tile stays where it was
 * and keeps its place in every stored arrangement.
 *
 * MESSAGES
 * --------
 * An iframe moved into the window posts to its parent, which is now the
 * window. Each such message is re-dispatched on the hub's own `window` with
 * its data, origin and source intact, so the hub's listeners take it exactly
 * as they took it in the tile. {@link queryAllDocuments} lets the hub's
 * "which iframe sent this" lookups see the frames in the windows too.
 *
 * RETURN
 * ------
 * The window's `pagehide` is its close (a blank page has nowhere to
 * navigate), backed by a `closed` poll. The hub's own `pagehide` (a reload, a
 * closed tab) returns every node and closes every window: a blank window
 * cannot outlive the page that drives it, and the stored layout already has
 * the tile, so the reloaded page shows the content in it.
 *
 * Cross-document moves reload an iframe: a browser discards a frame's
 * document when the frame leaves its document. A panel iframe therefore loads
 * again in the window, and again on its way back; whatever the panel keeps
 * (server state, its own storage) comes back with it. Plain DOM — the Agent
 * pane — moves without a reload.
 *
 * This module imports nothing: the tile wiring is attached-popout.js.
 */

/** The window name prefix; the key follows it. */
const NAME_PREFIX = 'osprey-attached-';

/** The smallest window a node may pop out into. */
const MIN_WIDTH = 480;
const MIN_HEIGHT = 360;

/** The window size when the node has no box to measure. */
const DEFAULT_WIDTH = 960;
const DEFAULT_HEIGHT = 720;

/** How often the open windows are checked for `closed`. */
export const CLOSED_POLL_MS = 500;

/** The node's inline style while it lives in a window: it fills the window. */
const FILL_STYLE =
  'flex: 1 1 auto; min-width: 0; min-height: 0; width: 100%; height: 100%; margin: 0; border: 0;';

/** The window's mount: fixed to the viewport, whatever the hub's body rules say. */
const MOUNT_STYLE =
  'position: fixed; inset: 0; display: flex; flex-direction: column; overflow: hidden;' +
  ' background: var(--surface-card);';

/**
 * Why a node came back: its window closed, the hub asked for it, or the hub
 * page is going away.
 * @typedef {'closed' | 'bring-back' | 'unload'} ReturnReason
 */

/**
 * @typedef {object} AttachSpec
 * @property {string} key - one window per key
 * @property {string} title - the window's title
 * @property {HTMLElement} content - the live node that moves into the window
 * @property {HTMLElement} standIn - what takes its place meanwhile
 * @property {(reason: ReturnReason) => void} [onReturn] - after the node is back
 * @property {() => void} [onResize] - the window was resized
 */

/**
 * @typedef {object} Attached
 * @property {AttachSpec} spec
 * @property {Window} win
 * @property {string | null} savedStyle
 * @property {() => void} teardown
 */

/** The open windows, by key. @type {Map<string, Attached>} */
const attached = new Map();

/** @type {Set<(key: string, out: boolean) => void>} */
const listeners = new Set();

/** @type {ReturnType<typeof setInterval> | null} */
let pollHandle = null;

let unloadWired = false;

/** @param {string} key @returns {boolean} */
export function isAttached(key) {
  return attached.has(key);
}

/** @param {string} key @returns {Window | null} */
export function attachedWindow(key) {
  return attached.get(key)?.win ?? null;
}

/**
 * Subscribe to every pop-out and every return.
 * @param {(key: string, out: boolean) => void} callback
 * @returns {() => void} unsubscribe
 */
export function onAttachedChange(callback) {
  listeners.add(callback);
  return () => {
    listeners.delete(callback);
  };
}

/**
 * Move `spec.content` into a window of its own, or bring the key's open
 * window to the front. Runs inside the operator's gesture, which the browser
 * needs to open a window. Null when the browser refused the window or the
 * node has no place to return to; nothing has moved then.
 * @param {AttachSpec} spec
 * @returns {Window | null}
 */
export function openAttached(spec) {
  const open = attached.get(spec.key);
  if (open) {
    focusAttached(spec.key);
    return open.win;
  }
  const { content, standIn } = spec;
  if (!content.parentNode) return null;
  const win = window.open('', NAME_PREFIX + spec.key, features(content));
  if (!win) return null;
  let doc;
  try {
    doc = win.document;
  } catch {
    win.close();
    return null;
  }
  // A window of that name left by an earlier page is reused, not reopened.
  doc.head.replaceChildren();
  doc.body.replaceChildren();
  doc.title = spec.title;
  doc.body.style.margin = '0';
  const base = doc.createElement('base');
  base.href = document.baseURI;
  doc.head.append(base);
  const stopStyles = mirrorStyles(doc);
  const stopRoot = mirrorAttributes(document.documentElement, doc.documentElement);
  const mount = doc.createElement('div');
  mount.className = 'attached-window-root';
  mount.style.cssText = MOUNT_STYLE;
  doc.body.append(mount);

  const savedStyle = content.getAttribute('style');
  content.replaceWith(standIn);
  content.style.cssText = FILL_STYLE;
  mount.append(content);

  /** @param {MessageEvent} event */
  const forward = (event) => {
    if (event.source === win) return;
    window.dispatchEvent(
      new MessageEvent('message', {
        data: event.data,
        origin: event.origin,
        source: event.source,
        ports: [...event.ports],
      })
    );
  };
  const onClose = () => release(spec.key, 'closed');
  const onResize = () => spec.onResize?.();
  win.addEventListener('message', forward);
  win.addEventListener('pagehide', onClose);
  win.addEventListener('resize', onResize);
  const teardown = () => {
    win.removeEventListener('message', forward);
    win.removeEventListener('pagehide', onClose);
    win.removeEventListener('resize', onResize);
    stopStyles();
    stopRoot();
  };
  attached.set(spec.key, { spec, win, savedStyle, teardown });
  wireUnload();
  if (pollHandle === null) pollHandle = setInterval(pollClosed, CLOSED_POLL_MS);
  notify(spec.key, true);
  return win;
}

/**
 * Bring a node back and close its window.
 * @param {string} key
 */
export function returnAttached(key) {
  release(key, 'bring-back');
}

/**
 * Bring a key's window to the front. The browser may decline outside a
 * gesture; the window is still the node's home either way.
 * @param {string} key
 * @returns {boolean} true when the key has a window
 */
export function focusAttached(key) {
  const entry = attached.get(key);
  if (!entry) return false;
  try {
    entry.win.focus();
  } catch {
    /* a closing window — the poll returns the node */
  }
  return true;
}

/**
 * Every element matching `selector` in the hub's document and in each
 * window's: the hub's lookups of "the iframe that sent this" must find the
 * frames that moved.
 * @param {string} selector
 * @returns {Element[]}
 */
export function queryAllDocuments(selector) {
  const found = [...document.querySelectorAll(selector)];
  for (const { win } of attached.values()) {
    try {
      found.push(...win.document.querySelectorAll(selector));
    } catch {
      /* a closing window — the poll returns the node */
    }
  }
  return found;
}

/**
 * Put a node back where its stand-in stands, restore its style, close the
 * window (unless it is already closing) and tell the subscribers. A stand-in
 * someone took off the page leaves the node adopted by the hub's document,
 * unattached, for the caller's `onReturn` to re-home.
 * @param {string} key
 * @param {ReturnReason} reason
 */
function release(key, reason) {
  const entry = attached.get(key);
  if (!entry) return;
  attached.delete(key);
  entry.teardown();
  const { content, standIn } = entry.spec;
  if (entry.savedStyle === null) content.removeAttribute('style');
  else content.setAttribute('style', entry.savedStyle);
  if (standIn.parentNode) standIn.replaceWith(content);
  else document.adoptNode(content);
  if (reason !== 'closed') {
    try {
      entry.win.close();
    } catch {
      /* already gone */
    }
  }
  if (attached.size === 0 && pollHandle !== null) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
  notify(key, false);
  try {
    entry.spec.onReturn?.(reason);
  } catch (err) {
    console.error('attached-window: onReturn threw', err);
  }
}

/** The `closed` poll: return every node whose window has closed. */
function pollClosed() {
  for (const [key, entry] of [...attached]) {
    if (entry.win.closed) release(key, 'closed');
  }
}

/** Return every node when the hub page goes away. Wired once, on first use. */
function wireUnload() {
  if (unloadWired) return;
  unloadWired = true;
  window.addEventListener('pagehide', () => {
    for (const key of [...attached.keys()]) release(key, 'unload');
  });
}

/**
 * @param {string} key
 * @param {boolean} out
 */
function notify(key, out) {
  for (const listener of [...listeners]) {
    try {
      listener(key, out);
    } catch (err) {
      console.error('attached-window: a change listener threw', err);
    }
  }
}

/**
 * Copy the hub's stylesheets into the window, and copy them again whenever
 * the hub's <head> gains or loses one. A `<link>` takes its resolved URL.
 * @param {Document} doc
 * @returns {() => void} stop following
 */
function mirrorStyles(doc) {
  /** @type {Node[]} */
  let clones = [];
  const copy = () => {
    for (const clone of clones) clone.parentNode?.removeChild(clone);
    clones = [...document.head.querySelectorAll('link[rel="stylesheet"], style')].map((node) => {
      const clone = /** @type {HTMLElement} */ (node.cloneNode(true));
      if (node instanceof HTMLLinkElement) clone.setAttribute('href', node.href);
      doc.head.append(clone);
      return clone;
    });
  };
  copy();
  const observer = new MutationObserver(copy);
  observer.observe(document.head, { childList: true });
  return () => observer.disconnect();
}

/**
 * Keep `to`'s attributes equal to `from`'s: the theme, the view and the
 * Agent's state all live on the hub's <html>.
 * @param {Element} from
 * @param {Element} to
 * @returns {() => void} stop following
 */
function mirrorAttributes(from, to) {
  const copy = () => {
    for (const name of to.getAttributeNames()) {
      if (!from.hasAttribute(name)) to.removeAttribute(name);
    }
    for (const name of from.getAttributeNames()) {
      to.setAttribute(name, /** @type {string} */ (from.getAttribute(name)));
    }
  };
  copy();
  const observer = new MutationObserver(copy);
  observer.observe(from, { attributes: true });
  return () => observer.disconnect();
}

/**
 * The `window.open` features: a popup the size of the node, clamped to the
 * window minimum and the screen, and centred on the hub's screen.
 * @param {HTMLElement} content
 * @returns {string}
 */
function features(content) {
  const box = content.getBoundingClientRect();
  const sized = box.width > 0 && box.height > 0;
  const { availWidth, availHeight } = window.screen;
  const availLeft = Number(/** @type {any} */ (window.screen).availLeft) || 0;
  const availTop = Number(/** @type {any} */ (window.screen).availTop) || 0;
  const width = clamp(Math.round(sized ? box.width : DEFAULT_WIDTH), MIN_WIDTH, availWidth);
  const height = clamp(Math.round(sized ? box.height : DEFAULT_HEIGHT), MIN_HEIGHT, availHeight);
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
