// @ts-check
/* OSPREY Design System — collapsible region (the sidebar's labels toggle).
 *
 * The state lives in ONE place: an attribute on <html> (`data-sidebar` by
 * default) whose value is 'collapsed' or 'expanded'; an absent attribute means
 * expanded. CSS keys every collapsed rule off that attribute, and
 * sidebar-boot.js stamps it before first paint from the same storage key this
 * module writes, so a reload never flashes the other width.
 *
 * `initCollapsible()` wires three inputs to that attribute: the toggle button,
 * the Cmd+\ (macOS) / Ctrl+\ shortcut, and `set()` for callers. On every
 * change it stores the state under `scopedStorageKey(storageKey)` (see
 * storage-scope.js for why the key is per-user on a multi-user mount),
 * updates the toggle's `aria-pressed`, `aria-label` and `title`, and
 * dispatches `osprey-collapsible-change` with `{attribute, state}` on
 * `document`.
 *
 * A keydown that another handler already consumed (`defaultPrevented`, as
 * xterm does for the keys it sends to the shell) does not toggle. One
 * shortcut drives one region: initialising again with the same shortcut
 * (a re-rendered shell) rebinds it instead of toggling twice per press.
 *
 * @module collapsible
 */

import { isMacPlatform } from './platform.js';
import { scopedStorageKey } from './storage-scope.js';

/** @typedef {'collapsed' | 'expanded'} CollapsibleState */

/**
 * @typedef {object} CollapsibleOptions
 * @property {string} [attribute] the <html> attribute; default 'data-sidebar'
 * @property {string} storageKey the bare localStorage key, e.g. 'osprey-sidebar'
 * @property {HTMLElement | null} [toggle] the button that flips the state
 * @property {string} [shortcut] the key pressed with Cmd/Ctrl; default '\\'
 */

/**
 * @typedef {object} Collapsible
 * @property {() => boolean} isCollapsed
 * @property {() => void} toggle
 * @property {(state: CollapsibleState) => void} set
 */

/** The event dispatched on `document` after every change. */
const COLLAPSIBLE_CHANGE_EVENT = 'osprey-collapsible-change';

/** @type {Map<string, () => void>} shortcut key → the flip it triggers */
const shortcuts = new Map();
let listening = false;

/** @param {KeyboardEvent} event */
function onKeydown(event) {
  if (event.defaultPrevented || event.altKey || event.shiftKey) return;
  if (!(event.metaKey || event.ctrlKey)) return;
  const flip = shortcuts.get(event.key);
  if (!flip) return;
  event.preventDefault();
  flip();
}

/**
 * Wire a collapsible region to its <html> attribute.
 *
 * @param {CollapsibleOptions} options
 * @returns {Collapsible}
 */
export function initCollapsible({ attribute = 'data-sidebar', storageKey, toggle = null, shortcut = '\\' }) {
  if (!storageKey) throw new TypeError('initCollapsible: storageKey is required');
  const root = document.documentElement;
  const key = scopedStorageKey(storageKey);
  const chord = `${isMacPlatform() ? '⌘' : 'Ctrl+'}${shortcut}`;

  const isCollapsed = () => root.getAttribute(attribute) === 'collapsed';

  function syncToggle() {
    if (!toggle) return;
    const collapsed = isCollapsed();
    toggle.setAttribute('aria-pressed', String(collapsed));
    toggle.setAttribute('aria-label', collapsed ? 'Show sidebar labels' : 'Hide sidebar labels');
    toggle.title = `${collapsed ? 'Show' : 'Hide'} labels (${chord})`;
  }

  /** @param {CollapsibleState} state */
  function set(state) {
    if (state !== 'collapsed' && state !== 'expanded') {
      throw new TypeError(`initCollapsible: unknown state "${state}"`);
    }
    const changed = isCollapsed() !== (state === 'collapsed');
    root.setAttribute(attribute, state);
    try {
      window.localStorage.setItem(key, state);
    } catch {
      // Storage can be unavailable (private mode, quota); the attribute still
      // holds the state for this page.
    }
    syncToggle();
    if (changed) {
      document.dispatchEvent(
        new CustomEvent(COLLAPSIBLE_CHANGE_EVENT, { detail: { attribute, state } })
      );
    }
  }

  function flip() {
    set(isCollapsed() ? 'expanded' : 'collapsed');
  }

  toggle?.addEventListener('click', flip);
  shortcuts.set(shortcut, flip);
  if (!listening) {
    listening = true;
    document.addEventListener('keydown', onKeydown);
  }

  syncToggle();
  return { isCollapsed, toggle: flip, set };
}
