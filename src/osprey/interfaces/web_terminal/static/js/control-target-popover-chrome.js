// @ts-check
/* OSPREY Web Terminal — Control-target popover chrome
 *
 * Where the control-target popover sits and what dismisses it.
 * control-target-popover.js owns the card, the rows and the actions; this
 * module owns the two things around them: the placement beside the sidebar
 * (fixed at the sidebar's right edge, level with the chip) and the document-
 * level dismissals — an outside click, Escape, and the sidebar's edge moving
 * under a fixed popover (a window resize or the sidebar collapsing).
 */

import { dismissConfirm, isConfirmUp } from './posture-confirm.js';

/** The gap between the sidebar's right edge and the popover. */
const BESIDE_SIDEBAR_GAP_PX = 8;
/** The least room the popover keeps from the window's edges. */
const VIEWPORT_MARGIN_PX = 8;

/**
 * Place the popover when its chip sits in the sidebar: fixed, 8 px right of
 * the sidebar's edge, its top level with the chip and moved up as far as the
 * window needs. Under a chip on a page bar (JupyterLab) the stylesheet hangs
 * it below its trigger instead, and the inline position is cleared.
 * @param {HTMLElement} popover
 * @param {HTMLElement | null} chip
 * @param {Element | null | undefined} sidebar the `#sidebar` the chip sits in, if any
 */
export function placeBesideSidebar(popover, chip, sidebar) {
  if (!sidebar || !chip) {
    popover.style.left = '';
    popover.style.top = '';
    return;
  }
  const maxTop = window.innerHeight - popover.offsetHeight - VIEWPORT_MARGIN_PX;
  const top = Math.max(VIEWPORT_MARGIN_PX, Math.min(chip.getBoundingClientRect().top, maxTop));
  popover.style.left = `${sidebar.getBoundingClientRect().right + BESIDE_SIDEBAR_GAP_PX}px`;
  popover.style.top = `${top}px`;
}

/**
 * Wire what dismisses an open popover, and return the function that unwires it.
 *
 * - A click anywhere that is not inside the anchor (chip + popover) or inside a
 *   confirm the popover raised closes it. Capture phase, so it dismisses
 *   before the click does anything else. The confirm mounts on `document.body`,
 *   outside the anchor, as a full layer ABOVE the popover: clicking inside it
 *   is the operator answering the question the popover asked, not leaving it.
 * - Escape, in the order the things on screen were opened: a confirm first,
 *   the popover only when none is up. Cancelling a confirm and losing the rows
 *   it was about in the same keystroke would undo the point of leaving the
 *   popover open beneath it. Closing by Escape hands focus back to the chip.
 * - A fixed popover cannot follow the sidebar's edge when the window or the
 *   sidebar's width changes, so either change closes it.
 *
 * @param {object} deps
 * @param {() => Element | null} deps.anchor the chip's anchor, read per event
 * @param {() => void} deps.close
 * @param {() => void} deps.focusChip
 * @returns {() => void}
 */
export function attachDismissals({ anchor, close, focusChip }) {
  /** @param {MouseEvent} event */
  const onClick = (event) => {
    const target = /** @type {Node|null} */ (event.target);
    if (!target) return;
    if (anchor()?.contains(target)) return;
    if (target instanceof Element && target.closest('.posture-modal-overlay')) return;
    close();
  };
  /** @param {KeyboardEvent} event */
  const onKeydown = (event) => {
    if (event.key !== 'Escape') return;
    if (isConfirmUp()) {
      event.stopPropagation();
      dismissConfirm();
      return;
    }
    close();
    focusChip();
  };
  document.addEventListener('click', onClick, true);
  document.addEventListener('keydown', onKeydown, true);
  window.addEventListener('resize', close);
  document.addEventListener('osprey-collapsible-change', close);
  return () => {
    document.removeEventListener('click', onClick, true);
    document.removeEventListener('keydown', onKeydown, true);
    window.removeEventListener('resize', close);
    document.removeEventListener('osprey-collapsible-change', close);
  };
}
