// @ts-check
/* OSPREY Web Terminal — Attached windows
 *
 * A window the hub drives: the hub opens it on a shell page of its own
 * (static/attached-window.html — a real same-origin URL, so the address bar
 * shows the hub's address and the window takes the hub's zoom), mirrors its
 * stylesheets and its <html> attributes (theme, view, Agent state) into it, and
 * moves a live node of its own page under them; the caller puts its bar on top
 * from `onDress`. The node keeps every listener, every reference the hub's
 * modules hold and every connection behind it, because nothing is rebuilt: the
 * hub still runs it, the window only shows it. This is the mechanic of Canopy's
 * DetachedWindow (react-trace), without React: a node move where Canopy renders
 * a portal.
 *
 * This module knows windows and nodes. What a window MEANS — the pane leaving
 * the dock, coming back, closing — is pane-window.js, which drives it through
 * the hooks of {@link AttachSpec}.
 *
 * WHERE THE NODE WAS
 * ------------------
 * The caller hands an anchor, a hidden element that takes the node's place for
 * as long as the window is open. The return swaps them back, so the node lands
 * in the same parent at the same position whatever moved around it, and its
 * inline style is restored to the one it left with.
 *
 * MESSAGES
 * --------
 * An iframe moved into the window posts to its parent, which is now the
 * window. Each such message is re-dispatched on the hub's own `window` with
 * its data, origin and source intact, so the hub's listeners take it exactly
 * as they took it in the tile. {@link queryAllDocuments} lets the hub's
 * "which iframe sent this" lookups see the frames in the windows too.
 *
 * THE WINDOW'S LIFE
 * -----------------
 * The node moves once the window has loaded the shell (its marker is polled
 * for), after `beforeMove` agreed; until then nothing has moved, and a window
 * closed, or never loaded, moves nothing. {@link dressWindow} is the one path
 * that puts a loaded shell in order and the node under its bar.
 *
 * `win.closed`, polled, is the only thing that says a window closed: the
 * shell's `pagehide` fires on a close, a reload and a navigation alike. After a
 * `pagehide` the window is watched for a NEW document carrying the marker: a
 * reload, which {@link dressWindow} dresses again with the same node. A window
 * whose marker does not come back was navigated away (`onShellLost`).
 *
 * The node leaves at that `pagehide`, while its listeners are still on it: a
 * browser strips every listener from a document it discards, nodes included, and
 * the node has to work when the next document takes it in. The same goes for
 * whatever the caller put in the window, so `onPageHide` takes its own out.
 *
 * Cross-document moves reload an iframe: a browser discards a frame's
 * document when the frame leaves its document. A panel iframe therefore loads
 * again in the window, and again on its way back; whatever the panel keeps
 * (server state, its own storage) comes back with it. Plain DOM — the Agent
 * pane — moves without a reload.
 */

import { withPrefix } from './api.js';

/** The page every window opens on; its <html> carries {@link SHELL_MARKER}. */
export const SHELL_PATH = '/static/attached-window.html';

/** The attribute that tells a loaded shell from the window's first blank page. */
const SHELL_MARKER = 'data-osprey-attached-shell';

/** How often a window still loading its shell is checked. */
export const READY_POLL_MS = 25;

/** How long a window may take to load its shell before it is given up. */
export const READY_TIMEOUT_MS = 15_000;

/** The smallest window a node may move into. */
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
 * Why a node came back: its window closed, or the hub asked for it.
 * @typedef {'closed' | 'reattach'} ReturnReason
 */

/**
 * @typedef {object} AttachSpec
 * @property {string} key - one window per key
 * @property {string} name - the window's name; a window left under it is reused
 * @property {string} title - the window's title
 * @property {HTMLElement} content - the live node that moves into the window
 * @property {HTMLElement} anchor - hidden; takes the node's place meanwhile
 * @property {() => boolean} [beforeMove] - the shell has loaded, once; false
 *   closes the window and moves nothing
 * @property {() => void} [beforeDress] - before each move into a dressed shell
 * @property {(mount: HTMLElement) => void} [onDress] - the node is in the window
 *   (again, after a reload of the shell); `mount` holds it, ready for a bar above it
 * @property {() => void} [onPageHide] - the shell page is going away; what the
 *   caller keeps and has in the window leaves now
 * @property {() => void} [onShellLost] - the shell did not come back after a
 *   `pagehide`, and the window is still open
 * @property {() => void} [onResize] - the window was resized
 * @property {(reason: ReturnReason) => void} [beforeReturn] - about to move back
 * @property {(reason: ReturnReason) => void} [onReturn] - after the node is back
 */

/**
 * @typedef {object} Attached
 * @property {AttachSpec} spec
 * @property {Window} win
 * @property {boolean} ready - the node has moved; false while the first shell loads
 * @property {Document | null} doc - the document last dressed
 * @property {ParentNode | null} home - the parent the node left
 * @property {string | null} savedStyle
 * @property {() => void} teardown
 */

/** The windows, open or still loading their shell, by key. @type {Map<string, Attached>} */
const attached = new Map();

/** @type {ReturnType<typeof setInterval> | null} */
let pollHandle = null;

/** @param {string} key @returns {boolean} true while the key has a window, even one still loading */
export function hasWindow(key) {
  return attached.has(key);
}

/**
 * Open a window on the shell page for `spec.content`, or bring the key's
 * window to the front. Runs inside the operator's gesture, which the browser
 * needs to open a window. The node moves once the shell has loaded;
 * `spec.onDress` says when. Null when the browser refused the window or the
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
  if (!spec.content.parentNode) return null;
  // A window of that name left by an earlier page is navigated to the shell.
  const win = window.open(withPrefix(SHELL_PATH), spec.name, features(spec.content));
  if (!win) return null;
  /** @type {Attached} */
  const entry = {
    spec,
    win,
    ready: false,
    doc: null,
    home: null,
    savedStyle: null,
    teardown: () => {},
  };
  attached.set(spec.key, entry);
  if (pollHandle === null) pollHandle = setInterval(pollClosed, CLOSED_POLL_MS);
  if (shellLoaded(entry)) moveIn(entry);
  else {
    waitForShell(
      entry,
      () => moveIn(entry),
      () => {
        console.warn(`attached-window: the window for ${spec.key} never loaded its page`);
        release(spec.key, 'closed');
      }
    );
  }
  return win;
}

/**
 * @param {Attached} entry
 * @returns {boolean} true once the window shows a loaded shell page it has not
 *   been dressed on: not the blank page every new window starts on, and not
 *   the document a reload is about to replace
 */
function shellLoaded({ win, doc: dressed }) {
  try {
    if (win.closed) return false;
    const doc = win.document;
    return (
      doc !== dressed &&
      doc.documentElement.hasAttribute(SHELL_MARKER) &&
      doc.readyState !== 'loading'
    );
  } catch {
    return false;
  }
}

/**
 * Poll for the shell until it loads or {@link READY_TIMEOUT_MS} passes. A window
 * that closes meanwhile is the `closed` poll's, which tears this wait down.
 * @param {Attached} entry
 * @param {() => void} onShell
 * @param {() => void} onTimeout
 */
function waitForShell(entry, onShell, onTimeout) {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  const handle = setInterval(() => {
    if (shellLoaded(entry)) {
      entry.teardown();
      onShell();
    } else if (Date.now() >= deadline) {
      entry.teardown();
      onTimeout();
    }
  }, READY_POLL_MS);
  entry.teardown = () => {
    clearInterval(handle);
    entry.teardown = () => {};
  };
}

/**
 * The first shell has loaded: move the node out of the hub, its anchor in its
 * place. A node that lost its place while the shell loaded (its tile closed), or
 * a `beforeMove` that says no, closes the window and moves nothing.
 * @param {Attached} entry
 */
function moveIn(entry) {
  const { spec } = entry;
  const { content, anchor } = spec;
  if (!content.parentNode || (spec.beforeMove && !spec.beforeMove())) {
    release(spec.key, 'closed');
    return;
  }
  entry.savedStyle = content.getAttribute('style');
  entry.home = content.parentNode;
  content.replaceWith(anchor);
  dressWindow(entry);
}

/**
 * Put the loaded shell in order — the hub's stylesheets and <html> attributes,
 * the window's listeners — and move the node into it. The one path for a new window and for a reload of the shell, which brings a new
 * document with nothing on it.
 * @param {Attached} entry
 */
function dressWindow(entry) {
  const { spec, win } = entry;
  const { content } = spec;
  const doc = win.document;
  entry.doc = doc;
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

  runHook(spec.beforeDress);
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
  const onPageHide = () => {
    runHook(spec.onPageHide);
    content.remove();
    entry.teardown();
    waitForShell(
      entry,
      () => dressWindow(entry),
      () => runHook(spec.onShellLost)
    );
  };
  const onResize = () => spec.onResize?.();
  win.addEventListener('message', forward);
  win.addEventListener('pagehide', onPageHide);
  win.addEventListener('resize', onResize);
  entry.teardown = () => {
    win.removeEventListener('message', forward);
    win.removeEventListener('pagehide', onPageHide);
    win.removeEventListener('resize', onResize);
    stopStyles();
    stopRoot();
    entry.teardown = () => {};
  };
  entry.ready = true;
  runHook(() => spec.onDress?.(mount));
}

/**
 * Bring a node back and close its window.
 * @param {string} key
 */
export function returnAttached(key) {
  release(key, 'reattach');
}

/**
 * Close a window the way the operator would: the node comes back with the
 * reason `closed`.
 * @param {string} key
 */
export function closeAttached(key) {
  release(key, 'closed');
}

/**
 * Close every window and move nothing: the hub page is going away, and with it
 * the nodes and whoever would hear of a return.
 */
export function closeAllAttached() {
  for (const entry of attached.values()) {
    entry.teardown();
    closeWindow(entry.win);
  }
  attached.clear();
  stopPollWhenIdle();
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
  for (const { win, ready } of attached.values()) {
    if (!ready) continue;
    try {
      found.push(...win.document.querySelectorAll(selector));
    } catch {
      /* a closing window — the poll returns the node */
    }
  }
  return found;
}

/**
 * Put a node back where its anchor stands, or at the end of the parent it left
 * when the anchor was taken off, restore its style and close the window (unless
 * it is already closing). A window still loading its first shell has moved
 * nothing: it is closed and forgotten, and no one is told.
 * @param {string} key
 * @param {ReturnReason} reason
 */
function release(key, reason) {
  const entry = attached.get(key);
  if (!entry) return;
  attached.delete(key);
  entry.teardown();
  if (!entry.ready) {
    closeWindow(entry.win);
    stopPollWhenIdle();
    return;
  }
  const { spec } = entry;
  const { content, anchor } = spec;
  runHook(() => spec.beforeReturn?.(reason));
  if (entry.savedStyle === null) content.removeAttribute('style');
  else content.setAttribute('style', entry.savedStyle);
  if (anchor.parentNode) anchor.replaceWith(content);
  else entry.home?.append(content);
  closeWindow(entry.win);
  stopPollWhenIdle();
  runHook(() => spec.onReturn?.(reason));
}

/**
 * Run a caller's hook; one that throws must not strand the window's bookkeeping.
 * @param {(() => void) | undefined} hook
 */
function runHook(hook) {
  try {
    hook?.();
  } catch (err) {
    console.error('attached-window: a hook threw', err);
  }
}

/** @param {Window} win */
function closeWindow(win) {
  try {
    if (!win.closed) win.close();
  } catch {
    /* already gone */
  }
}

/** Stop the `closed` poll once no window is left. */
function stopPollWhenIdle() {
  if (attached.size === 0 && pollHandle !== null) {
    clearInterval(pollHandle);
    pollHandle = null;
  }
}

/** The `closed` poll: return every node whose window has closed. */
function pollClosed() {
  for (const [key, entry] of [...attached]) {
    if (entry.win.closed) release(key, 'closed');
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
