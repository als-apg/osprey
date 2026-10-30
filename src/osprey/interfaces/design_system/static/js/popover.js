// @ts-check
/* OSPREY Design System — shared popover and flyout.
 *
 * Popovers are `.osprey-popover[hidden]` elements that already sit at <body>
 * level (primitives.css skins them: `position: fixed`, card surface). This
 * module places, opens and closes them, so a scrolled container never cuts a
 * popover and every popover in the shell dismisses the same way.
 *
 * Placement (`place`), in viewport pixels:
 *   - 'beside-right': left = the right edge + 8, top = the anchor's top
 *     (moved up when the popover would leave the bottom of the window);
 *   - 'beside-right-bottom': left = the right edge + 8, bottom edge level with
 *     the anchor's bottom edge (for an opener at the foot of the sidebar),
 *     moved down when the popover would leave the top of the window and up
 *     when the anchor is below the bottom;
 *   - 'below-end': 4 px under the anchor, right edges aligned.
 * "The right edge" is the anchor's, or that of `edge` when given: pass the
 * sidebar as `edge` so a popover opened from any sidebar row starts beside the
 * sidebar, not over it.
 *
 * Open: the popover shows, the opener (the anchor) gets
 * `aria-expanded="true"` and `aria-controls` = the popover's id (an opener
 * that controls different elements in different states, such as the Artifacts
 * row, is kept in step), and focus moves to the first
 * `.osprey-popover-item, .osprey-mini, input` inside. Every other open popover
 * closes, except the popovers that contain the opener: a popover opened from
 * inside another keeps its parent open.
 *
 * Close: on a pointer press outside every open popover and its opener, on
 * Escape, on window resize, and on window blur (a click into a panel iframe
 * never reaches this document). Closing a popover also closes the popovers
 * opened from it. The opener gets `aria-expanded="false"`, and when focus was
 * inside the popover it returns to the opener, so it never stays on a hidden
 * element. (A click into an iframe has already made the iframe the active
 * element, so the blur close does not pull focus back out of it.)
 *
 * Menu: in a popover that is, or whose first child is, a `[role="menu"]`,
 * ArrowUp / ArrowDown (wrapping), Home and End move focus between its visible,
 * enabled `menuitem`, `menuitemradio` and `menuitemcheckbox` rows.
 *
 * Only popovers opened through this module are tracked and closed: a
 * `.osprey-popover` shown statically (a specimen, a popover another module
 * drives) is left alone.
 *
 * @module popover
 */

import { hideTooltip } from './tooltip.js';

/** @typedef {'beside-right' | 'beside-right-bottom' | 'below-end'} Placement */

/**
 * @typedef {object} PopoverOptions
 * @property {Placement} [place] where to put the popover; default 'below-end'
 * @property {Element} [edge] the element whose right edge 'beside-*' uses
 */

const GAP_BESIDE = 8;
const GAP_BELOW = 4;
const VIEWPORT_MARGIN = 8;
const FOCUS_TARGETS = '.osprey-popover-item, .osprey-mini, input';
const MENU_ITEMS = '[role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"]';
const MENU_KEYS = ['ArrowDown', 'ArrowUp', 'Home', 'End'];

/** @type {WeakMap<HTMLElement, HTMLElement>} popover → the element that opened it */
const openers = new WeakMap();

/** @type {Set<HTMLElement>} popovers opened here and not yet closed */
const opened = new Set();

let listening = false;

/** @returns {HTMLElement[]} the open popovers, in the order they opened */
function openPopovers() {
  for (const pop of opened) {
    if (pop.hidden || !pop.isConnected) opened.delete(pop);
  }
  return Array.from(opened);
}

/**
 * The popover `pop` and the chain of popovers it was opened from.
 * @param {HTMLElement} pop
 * @returns {HTMLElement[]}
 */
function withAncestors(pop) {
  /** @type {HTMLElement[]} */
  const chain = [];
  /** @type {HTMLElement | null} */
  let current = pop;
  while (current && !chain.includes(current)) {
    chain.push(current);
    const opener = openers.get(current);
    current = opener ? /** @type {HTMLElement | null} */ (opener.closest('.osprey-popover')) : null;
  }
  return chain;
}

/**
 * Set top/left/bottom/right for `place`.
 * @param {HTMLElement} pop
 * @param {Element} anchor
 * @param {PopoverOptions} options
 */
function position(pop, anchor, { place = 'below-end', edge } = {}) {
  const rect = anchor.getBoundingClientRect();
  const style = pop.style;
  style.top = style.bottom = style.left = style.right = '';
  if (place === 'below-end') {
    style.top = `${rect.bottom + GAP_BELOW}px`;
    style.right = `${window.innerWidth - rect.right}px`;
    return;
  }
  const right = (edge ?? anchor).getBoundingClientRect().right;
  style.left = `${right + GAP_BESIDE}px`;
  if (place === 'beside-right-bottom') {
    const maxBottom = window.innerHeight - VIEWPORT_MARGIN;
    const minBottom = VIEWPORT_MARGIN + pop.offsetHeight;
    const bottom = Math.max(minBottom, Math.min(rect.bottom, maxBottom));
    style.bottom = `${window.innerHeight - bottom}px`;
    return;
  }
  const maxTop = window.innerHeight - pop.offsetHeight - VIEWPORT_MARGIN;
  style.top = `${Math.max(VIEWPORT_MARGIN, Math.min(rect.top, maxTop))}px`;
}

/**
 * The `[role="menu"]` element of `pop`: the popover itself or its first child.
 * @param {HTMLElement} pop
 * @returns {Element | null}
 */
function menuOf(pop) {
  if (pop.getAttribute('role') === 'menu') return pop;
  const first = pop.firstElementChild;
  return first?.getAttribute('role') === 'menu' ? first : null;
}

/**
 * Move focus between the rows of the open menu that holds the event's target.
 * @param {KeyboardEvent} event
 */
function moveInMenu(event) {
  const target = /** @type {Node | null} */ (event.target);
  if (!target) return;
  const pop = openPopovers().find((p) => p.contains(target));
  const menu = pop ? menuOf(pop) : null;
  if (!menu || !menu.contains(target)) return;
  // A hidden group or a disabled row cannot take focus: stepping onto one
  // would leave focus where it was.
  const rows = /** @type {HTMLElement[]} */ (
    Array.from(menu.querySelectorAll(MENU_ITEMS)).filter(
      (el) => !el.closest('[hidden]') && !el.hasAttribute('disabled')
    )
  );
  if (!rows.length) return;
  event.preventDefault();
  const at = rows.indexOf(/** @type {HTMLElement} */ (document.activeElement));
  let next = 0;
  if (event.key === 'End') next = rows.length - 1;
  else if (event.key === 'ArrowDown') next = (at + 1) % rows.length;
  else if (event.key === 'ArrowUp') next = (at - 1 + rows.length) % rows.length;
  rows[next].focus();
}

/**
 * Open `pop` at `anchor`.
 *
 * @param {HTMLElement} pop a body-level `.osprey-popover`
 * @param {HTMLElement} anchor the opener; it receives aria-expanded/aria-controls
 * @param {PopoverOptions} [options]
 * @returns {void}
 */
export function openPopover(pop, anchor, options = {}) {
  listen();
  hideTooltip();
  const parent = /** @type {HTMLElement | null} */ (anchor.closest('.osprey-popover'));
  closeAllPopovers(parent ? withAncestors(parent) : []);
  openers.set(pop, anchor);
  opened.add(pop);
  if (pop.id) anchor.setAttribute('aria-controls', pop.id);
  anchor.setAttribute('aria-expanded', 'true');
  pop.hidden = false;
  position(pop, anchor, options);
  const first = /** @type {HTMLElement | null} */ (pop.querySelector(FOCUS_TARGETS));
  first?.focus({ preventScroll: true });
}

/**
 * Close `pop` and every popover opened from inside it.
 *
 * @param {HTMLElement} pop
 * @returns {void}
 */
export function closePopover(pop) {
  for (const other of openPopovers()) {
    if (other !== pop && withAncestors(other).includes(pop)) closePopover(other);
  }
  // Read before hiding: a hidden subtree drops its focus to <body>.
  const active = document.activeElement;
  const hadFocus = !!active && pop.contains(active);
  pop.hidden = true;
  opened.delete(pop);
  const opener = openers.get(pop);
  if (!opener) return;
  opener.setAttribute('aria-expanded', 'false');
  if (hadFocus) opener.focus({ preventScroll: true });
}

/**
 * Close every open popover except those in `except`.
 *
 * @param {HTMLElement[]} [except]
 * @returns {void}
 */
export function closeAllPopovers(except = []) {
  for (const pop of openPopovers()) {
    if (!except.includes(pop) && !pop.hidden) closePopover(pop);
  }
}

/**
 * Open `pop` from `opener`, or close it when it is open.
 *
 * @param {HTMLElement} opener
 * @param {HTMLElement} pop
 * @param {PopoverOptions} [options]
 * @returns {void}
 */
export function togglePopover(opener, pop, options = {}) {
  if (pop.hidden) openPopover(pop, opener, options);
  else closePopover(pop);
}

/** Install the document-level dismissal listeners, once. */
function listen() {
  if (listening) return;
  listening = true;
  document.addEventListener(
    'pointerdown',
    (event) => {
      const target = /** @type {Node | null} */ (event.target);
      /** @type {HTMLElement[]} */
      const keep = [];
      for (const pop of openPopovers()) {
        const opener = openers.get(pop);
        const hit = !!target && (pop.contains(target) || (!!opener && opener.contains(target)));
        if (hit) keep.push(...withAncestors(pop));
      }
      closeAllPopovers(keep);
    },
    true
  );
  document.addEventListener('keydown', (event) => {
    if (MENU_KEYS.includes(event.key)) {
      moveInMenu(event);
      return;
    }
    if (event.key !== 'Escape') return;
    // closePopover hands focus back up the chain, so focus inside a nested
    // popover ends on the opener of the outermost one.
    closeAllPopovers();
  });
  window.addEventListener('resize', () => closeAllPopovers());
  window.addEventListener('blur', () => closeAllPopovers());
}
