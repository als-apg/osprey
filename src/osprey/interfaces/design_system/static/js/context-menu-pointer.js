// @ts-check
/* OSPREY Design System — the pointer half of the context menu
 *
 * context-menu.js gives focus to a row so the keyboard can act on it. This
 * module keeps that focus, and an open submenu, from staying behind the
 * pointer: nothing is lit or open where the pointer is not.
 */

/** How long the pointer may be off every level before the submenus close, in ms. */
export const LEAVE_DELAY = 140;

/** @type {ReturnType<typeof setTimeout> | undefined} pending close after the pointer left */
let leaveTimer;

/**
 * @param {EventTarget | null} target
 * @returns {HTMLElement | null} `target` when it is an enabled menu row
 */
function enabledRow(target) {
  const el = /** @type {HTMLElement | null} */ (target);
  if (el?.tagName !== 'BUTTON' || el.getAttribute('aria-disabled') === 'true') return null;
  return el;
}

/**
 * Make one menu level follow the pointer.
 *
 * Focus: an enabled row takes focus when the pointer enters it, so no other
 * row is lit beside it. When the pointer leaves the row, the level takes the
 * focus, so no row is lit and the keys still reach the menu.
 *
 * Submenus: `closeSubmenus` runs a moment after the pointer leaves the level,
 * unless the pointer enters a level again (a submenu is a separate element
 * beside its parent). A touch has no hover, so a tap never closes what it
 * opened.
 *
 * The listeners are on the level in the capture phase: `pointerenter` and
 * `pointerleave` do not bubble, but they are captured, so rows added later are
 * covered.
 * @param {HTMLElement} menu - a menu level; its rows are `<button>` elements
 * @param {() => void} closeSubmenus
 */
export function followPointer(menu, closeSubmenus) {
  menu.tabIndex = -1;
  menu.addEventListener(
    'pointerenter',
    (e) => {
      clearTimeout(leaveTimer);
      enabledRow(e.target)?.focus({ preventScroll: true });
    },
    true
  );
  menu.addEventListener(
    'pointerleave',
    (e) => {
      if (e.target !== menu) {
        const row = enabledRow(e.target);
        if (row && menu.ownerDocument.activeElement === row) menu.focus({ preventScroll: true });
        return;
      }
      if (e.pointerType === 'touch') return;
      clearTimeout(leaveTimer);
      leaveTimer = setTimeout(() => {
        if (menu.isConnected) closeSubmenus();
      }, LEAVE_DELAY);
    },
    true
  );
}
