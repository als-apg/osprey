// @ts-check
/* OSPREY Web Terminal — Tab shows or hides the sidebar labels
 *
 * A plain Tab (no modifier) flips the sidebar between labels and icons, the
 * same flip as the toggle and Cmd/Ctrl+\ (collapsible.js). It leaves Tab alone
 * where Tab already means something:
 *
 *   - TYPING. Focus in an <input>, <textarea>, <select> or contenteditable:
 *     the message box (slash-command completion), the terminal (xterm's input
 *     is a textarea, so the shell completes), the command palette, any form.
 *   - OPEN LAYERS. Focus inside a popover, menu, dialog or the settings
 *     drawer, where Tab moves between that layer's own controls.
 *   - HANDLED. A keydown another handler already consumed.
 *
 * Shift+Tab still moves focus backwards. A panel is an iframe on this page's
 * origin (`/panel/…`), so the key is heard there too, by the same rules: a
 * click in a panel's empty space must not swallow Tab. A cross-origin frame
 * cannot be listened to and keeps Tab.
 *
 * It is on unless the operator turns it off in ☰ More; the choice is kept
 * per user under `scopedStorageKey('osprey-sidebar-tab-toggle')`.
 *
 * @module sidebar-tab-key
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/** The bare storage key; `'off'` turns the Tab toggle off. */
export const TAB_TOGGLE_KEY = 'osprey-sidebar-tab-toggle';

/** Layers where Tab moves between the layer's own controls. */
const LAYER_SELECTOR = '.osprey-popover, [role="menu"], [role="dialog"], osprey-drawer';

/** @returns {boolean} whether Tab toggles the sidebar (on unless turned off) */
export function tabToggleOn() {
  try {
    return window.localStorage.getItem(scopedStorageKey(TAB_TOGGLE_KEY)) !== 'off';
  } catch {
    return true;
  }
}

/** @param {boolean} on */
export function setTabToggle(on) {
  try {
    window.localStorage.setItem(scopedStorageKey(TAB_TOGGLE_KEY), on ? 'on' : 'off');
  } catch {
    // Storage can be unavailable (private mode, quota); the default stays.
  }
}

/**
 * Whether Tab already means something where focus is. Realm-safe: a panel's
 * elements belong to the panel's window, so no `instanceof` check.
 * @param {Element | null} el
 * @returns {boolean}
 */
export function tabIsTaken(el) {
  if (!el || el.nodeType !== 1) return false;
  if (/** @type {HTMLElement} */ (el).isContentEditable) return true;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)) return true;
  return el.closest(LAYER_SELECTOR) !== null;
}

/**
 * Let a plain Tab flip the sidebar, on this page and in its same-origin
 * panel frames.
 * @param {() => void} flip  the sidebar's toggle (collapsible.js)
 * @returns {() => void} removes the listeners
 */
export function initSidebarTabKey(flip) {
  /** @type {Document[]} */
  const docs = [];

  /** @param {KeyboardEvent} event */
  const onKeydown = (event) => {
    if (event.key !== 'Tab' || event.defaultPrevented) return;
    if (event.shiftKey || event.altKey || event.ctrlKey || event.metaKey) return;
    const doc = /** @type {Document} */ (event.currentTarget);
    if (!tabToggleOn() || tabIsTaken(doc.activeElement)) return;
    event.preventDefault();
    flip();
  };

  /** @param {Document} doc */
  const listen = (doc) => {
    if (docs.includes(doc)) return;
    docs.push(doc);
    doc.addEventListener('keydown', onKeydown);
  };

  /** @param {HTMLIFrameElement} frame */
  const hookFrame = (frame) => {
    try {
      if (frame.contentDocument) listen(frame.contentDocument);
    } catch {
      // Cross-origin: the frame keeps Tab.
    }
  };

  /** A frame's document is replaced on every navigation; hook each one. @param {Event} event */
  const onLoad = (event) => {
    const target = /** @type {Element | null} */ (event.target);
    if (target?.tagName === 'IFRAME') hookFrame(/** @type {HTMLIFrameElement} */ (target));
  };

  listen(document);
  document.querySelectorAll('iframe').forEach(hookFrame);
  // `load` does not bubble, but a capture listener on the document sees it.
  document.addEventListener('load', onLoad, true);
  return () => {
    document.removeEventListener('load', onLoad, true);
    for (const doc of docs.splice(0)) doc.removeEventListener('keydown', onKeydown);
  };
}
