// @ts-check
/**
 * ARIEL panel keyboard shortcuts. Every key writes the store; nothing here
 * touches the panes directly.
 *
 *   j / k   move `sel` through the rendered list rows, clearing `att` (the
 *           reader follows)
 *   Enter   open the selection in the reader (`narrowView:'reader'`)
 *   /       focus the search box (standalone only; embedded it is the host's)
 *   [       toggle `railCollapsed`
 *   c       open the compose sheet on a new entry (`compose:{draft:null}`)
 *   r       open the compose sheet on a follow-up of the selection
 *           (`compose:{draft:null, followsUp:sel}`); nothing without a `sel`
 *   a       embedded only: cite the shown entry in the host's assistant prompt
 *           (`ask-agent.js`, a reader-seam key routed by `app.js`, not here)
 *   e       open the reader's "Enhancements" tab (`inspector.js`, a reader-seam
 *           tab key routed by `app.js`, not here)
 *   Esc     close one layer: the lightbox (`att:null`), else the compose sheet
 *           (`compose:null`; its autosaved draft is kept), else Back in the
 *           stacked layout
 *
 * The design system already closes popovers and drawers on `Esc` from its own
 * bubble-phase listeners, without `preventDefault`. So whether a popover or
 * drawer was open is read from a capture-phase snapshot taken before those
 * listeners run, and an `Esc` that arrived while one was open (or was
 * `defaultPrevented`) is left alone: one `Esc` closes one layer.
 *
 * Keys are ignored while typing in an input, textarea, select or editable
 * element, except `Esc`. Only keydowns of the panel's own document are
 * handled: an embedded host owns `Esc` inside its search box.
 */

import { isEmbedded } from '/design-system/js/frame-params.js';
import { focusSearch } from './list.js';
import { getState, setState } from './store.js';

const ROW_SELECTOR = '#ariel-list [data-entry-id]';
const OPEN_LAYER_SELECTOR = '.osprey-popover:not([hidden]), osprey-drawer[open]';

/**
 * @param {EventTarget|null} target
 * @returns {boolean} whether the event target is a text-entry control
 */
function isTyping(target) {
  const el = /** @type {HTMLElement|null} */ (target);
  if (!el || typeof el.closest !== 'function') return false;
  if (el.isContentEditable) return true;
  return (
    el.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])') !== null
  );
}

/**
 * Move the selection one rendered row down (+1) or up (-1).
 * @param {Document} doc
 * @param {1|-1} step
 * @returns {boolean} whether there were rows to move through
 */
function moveSelection(doc, step) {
  const rows = /** @type {HTMLElement[]} */ (Array.from(doc.querySelectorAll(ROW_SELECTOR)));
  if (rows.length === 0) return false;
  const ids = rows.map((row) => row.dataset.entryId || '');
  const at = ids.indexOf(getState().sel ?? '');
  const next = at < 0 ? 0 : Math.min(Math.max(at + step, 0), ids.length - 1);
  setState({ sel: ids[next], att: null });
  rows[next].scrollIntoView?.({ block: 'nearest' });
  return true;
}

/**
 * Close the innermost store-owned layer `Esc` is responsible for.
 * @returns {boolean} whether a layer was closed
 */
function closeOneLayer() {
  const state = getState();
  if (state.att !== null) {
    setState({ att: null });
    return true;
  }
  if (state.compose !== null) {
    setState({ compose: null });
    return true;
  }
  if (state.stacked && state.narrowView === 'reader') {
    setState({ narrowView: 'list' });
    return true;
  }
  return false;
}

/**
 * Install the panel's keyboard shortcuts.
 * @param {Document} [doc] the panel document
 * @param {{embedded?: boolean}} [options] `embedded` defaults to `isEmbedded()`
 * @returns {{destroy(): void}}
 */
export function initKeyboard(doc = document, options = {}) {
  const embedded = options.embedded ?? isEmbedded();
  const win = doc.defaultView;
  if (!win) return { destroy() {} };

  /** @type {WeakSet<Event>} keydowns that arrived while a popover or drawer was open */
  const layerWasOpen = new WeakSet();

  /** @param {KeyboardEvent} event */
  const snapshot = (event) => {
    if (event.key === 'Escape' && doc.querySelector(OPEN_LAYER_SELECTOR)) layerWasOpen.add(event);
  };

  /** @param {KeyboardEvent} event */
  const onKeydown = (event) => {
    const target = event.target;
    const foreign = /** @type {Node|null} */ (target)?.ownerDocument !== doc;
    if (target && target !== win && foreign && target !== doc) return;
    if (event.defaultPrevented) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    if (event.key === 'Escape') {
      if (layerWasOpen.has(event)) return;
      if (closeOneLayer()) event.preventDefault();
      return;
    }
    if (isTyping(event.target)) return;

    switch (event.key) {
      case 'j':
      case 'k':
        if (moveSelection(doc, event.key === 'j' ? 1 : -1)) event.preventDefault();
        break;
      case 'Enter': {
        const el = /** @type {HTMLElement|null} */ (event.target);
        if (el && typeof el.closest === 'function' && el.closest('button, a[href], summary'))
          return;
        if (getState().sel === null) return;
        setState({ narrowView: 'reader' });
        event.preventDefault();
        break;
      }
      case '/':
        if (embedded) return;
        event.preventDefault();
        focusSearch();
        break;
      case '[':
        setState({ railCollapsed: !getState().railCollapsed });
        event.preventDefault();
        break;
      case 'c':
        setState({ compose: { draft: null } });
        event.preventDefault();
        break;
      case 'r': {
        const sel = getState().sel;
        if (sel === null) return;
        setState({ compose: { draft: null, followsUp: sel } });
        event.preventDefault();
        break;
      }
      default:
    }
  };

  win.addEventListener('keydown', snapshot, true);
  win.addEventListener('keydown', onKeydown);
  return {
    destroy() {
      win.removeEventListener('keydown', snapshot, true);
      win.removeEventListener('keydown', onKeydown);
    },
  };
}
