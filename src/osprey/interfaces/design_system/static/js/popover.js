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
 *     the anchor's bottom edge (for an opener at the foot of the sidebar);
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
 * Escape (focus returns to the opener when it was inside the popover), on
 * window resize, and on window blur (a click into a panel iframe never reaches
 * this document). Closing a popover also closes the popovers opened from it.
 * The opener gets `aria-expanded="false"`.
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
    style.bottom = `${window.innerHeight - rect.bottom}px`;
    return;
  }
  const maxTop = window.innerHeight - pop.offsetHeight - VIEWPORT_MARGIN;
  style.top = `${Math.max(VIEWPORT_MARGIN, Math.min(rect.top, maxTop))}px`;
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
  pop.hidden = true;
  opened.delete(pop);
  const opener = openers.get(pop);
  if (opener) opener.setAttribute('aria-expanded', 'false');
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
    if (event.key !== 'Escape') return;
    const open = openPopovers();
    if (open.length === 0) return;
    const active = document.activeElement;
    const focused = open.find((pop) => active && pop.contains(active));
    const returnTo = focused ? openers.get(withAncestors(focused).at(-1) ?? focused) : null;
    closeAllPopovers();
    returnTo?.focus({ preventScroll: true });
  });
  window.addEventListener('resize', () => closeAllPopovers());
  window.addEventListener('blur', () => closeAllPopovers());
}
