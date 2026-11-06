// @ts-check
/*
 * OSPREY Design System — the content strip.
 *
 * The controls that act on the shown content (a plot's zoom and pan, a
 * table's copy and export) sit in one small strip at the top-right corner of
 * the content, not in the toolbar (DESIGN.md §12, "Panel chrome standard").
 * Groups are split by a hairline; order them modes, then view, then output.
 *
 * The strip is out of the way until it is wanted. It shows while the pointer
 * is over the content or focus is inside it, for about 3 s after `show()`
 * (call it when content opens), and while a mode other than its group's
 * default is pressed. The visual half is `.osprey-content-strip` in
 * primitives.css: `[data-visible]` is the timed show, `[data-pinned]` the
 * pressed mode, and the host's class `.osprey-content-strip-host` carries the
 * hover and focus rules.
 *
 * Modes carry `aria-pressed`, and a group has at most one pressed. A group
 * whose items mark one mode `pressed` is a radio set with that mode as its
 * default: one mode is always pressed there. In a group without one, a mode
 * toggles. Every other item is a plain action.
 *
 * @module content-strip
 */

import { icon as buildIcon } from './icons.js';

/** How long `show()` keeps the strip visible. */
const SHOW_FOR_MS = 3000;
const HOST_CLASS = 'osprey-content-strip-host';

/**
 * @typedef {object} ContentStripItem
 * @property {string} id passed to `onAction`
 * @property {string} icon an `icons.js` name
 * @property {string} title the tooltip and the accessible name
 * @property {string} [detail] one sentence for the tooltip's second line
 * @property {boolean} [mode] a mode: it toggles `aria-pressed`
 * @property {boolean} [pressed] the mode starts pressed and is its group's default
 */

/**
 * @typedef {object} ContentStripGroup
 * @property {string} id
 * @property {ContentStripItem[]} items
 */

/**
 * @typedef {object} ContentStripOptions
 * @property {ContentStripGroup[]} groups
 * @property {(id: string, pressed?: boolean) => void} onAction a mode passes
 *   its new pressed state; an action passes only its id
 * @property {string} [label] the strip's accessible name. Default: "Content controls"
 */

/**
 * @typedef {object} ContentStripApi
 * @property {() => void} show Show the strip for about 3 s.
 * @property {() => void} hide End that show at once.
 * @property {(id: string, pressed: boolean) => void} setPressed Set a mode
 *   from the content's own state; `onAction` is not called.
 * @property {() => void} destroy Remove the strip and its timer.
 */

/**
 * @typedef {object} GroupState
 * @property {HTMLButtonElement[]} modes
 * @property {HTMLButtonElement | null} fallback the default mode, if any
 */

/** @param {HTMLButtonElement} button */
const isPressed = (button) => button.getAttribute('aria-pressed') === 'true';

/**
 * Render a content strip into `host`, which gets `position: relative` when it
 * has no position of its own.
 *
 * @param {HTMLElement} host the element that holds the content
 * @param {ContentStripOptions} options
 * @returns {ContentStripApi}
 * @throws {Error} on an unknown icon name (from `icons.js`)
 */
export function createContentStrip(host, { groups, onAction, label = 'Content controls' }) {
  const strip = document.createElement('div');
  strip.className = 'osprey-content-strip';
  strip.setAttribute('role', 'toolbar');
  strip.setAttribute('aria-label', label);

  /** @type {Map<string, {button: HTMLButtonElement, item: ContentStripItem, group: GroupState}>} */
  const entries = new Map();
  /** @type {GroupState[]} */
  const states = [];

  const updatePinned = () => {
    const pinned = states.some((state) => (state.modes.find(isPressed) ?? null) !== state.fallback);
    strip.toggleAttribute('data-pinned', pinned);
  };

  /**
   * @param {string} id
   * @param {boolean} pressed
   */
  const press = (id, pressed) => {
    const entry = entries.get(id);
    if (!entry) return;
    if (pressed) {
      for (const other of entry.group.modes) other.setAttribute('aria-pressed', 'false');
    }
    entry.button.setAttribute('aria-pressed', String(pressed));
    updatePinned();
  };

  /** @param {string} id */
  const activate = (id) => {
    const entry = entries.get(id);
    if (!entry) return;
    if (!entry.item.mode) {
      onAction(id);
      return;
    }
    const pressed = isPressed(entry.button);
    // A radio set keeps its pressed mode.
    if (pressed && entry.group.fallback) return;
    press(id, !pressed);
    onAction(id, !pressed);
  };

  for (const group of groups) {
    const box = document.createElement('div');
    box.className = 'osprey-content-strip-group';
    box.setAttribute('role', 'group');
    /** @type {GroupState} */
    const state = { modes: [], fallback: null };
    for (const item of group.items) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'osprey-icon-btn';
      button.title = item.title;
      button.setAttribute('aria-label', item.title);
      button.setAttribute('data-tip-place', 'below');
      if (item.detail) button.setAttribute('data-tip-detail', item.detail);
      button.append(buildIcon(item.icon));
      if (item.mode) {
        const pressed = Boolean(item.pressed) && !state.fallback;
        button.setAttribute('aria-pressed', String(pressed));
        if (pressed) state.fallback = button;
        state.modes.push(button);
      }
      button.addEventListener('click', () => activate(item.id));
      entries.set(item.id, { button, item, group: state });
      box.append(button);
    }
    states.push(state);
    strip.append(box);
  }

  const hostPosition = getComputedStyle(host).position;
  const positioned = hostPosition !== '' && hostPosition !== 'static';
  if (!positioned) host.style.position = 'relative';
  host.classList.add(HOST_CLASS);
  host.append(strip);
  updatePinned();

  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let timer;
  const hide = () => {
    clearTimeout(timer);
    strip.removeAttribute('data-visible');
  };

  return {
    show() {
      clearTimeout(timer);
      strip.setAttribute('data-visible', '');
      timer = setTimeout(hide, SHOW_FOR_MS);
    },
    hide,
    setPressed(id, pressed) {
      const entry = entries.get(id);
      if (!entry?.item.mode) throw new Error(`content-strip: "${id}" is not a mode`);
      press(id, pressed);
    },
    destroy() {
      clearTimeout(timer);
      strip.remove();
      host.classList.remove(HOST_CLASS);
      if (!positioned) host.style.position = '';
    },
  };
}
