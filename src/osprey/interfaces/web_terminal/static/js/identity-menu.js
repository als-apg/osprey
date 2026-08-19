/**
 * OSPREY Web Terminal — header identity menu.
 *
 * The header's identity chip ("Control Room (Alice)") is the trigger for a
 * small popover holding who you are and Log out. Operators read the name in the
 * corner as "who am I / where am I" and reach for it when they want to leave,
 * so that is where leaving lives.
 *
 * The popover's markup is in index.html, not built here: `#logout-btn` is
 * resolved BY ID by app.js's initLogoutButton() and by palette-boot.js's
 * "Log out" command, so it must be in the DOM while the menu is closed. This
 * module only opens and closes it.
 *
 * Same interaction contract as display-menu.js — capture-phase outside-click
 * and Escape close, `aria-expanded` on the trigger, focus returned to the
 * trigger on Escape.
 */

const TRIGGER_ID = 'header-identity-trigger';
const MENU_ID = 'header-identity-menu';

/**
 * Wire the header identity menu, if this deployment renders one.
 *
 * Single-user terminals know no user and render no chip, so this is a no-op
 * there rather than an error.
 *
 * @returns {{ open: () => void, close: () => void, isOpen: () => boolean } | null}
 */
export function initIdentityMenu() {
  const trigger = /** @type {HTMLButtonElement | null} */ (document.getElementById(TRIGGER_ID));
  const menu = document.getElementById(MENU_ID);
  if (!trigger || !menu) return null;

  // Arrow consts rather than hoisted `function` declarations: these close over
  // `trigger`/`menu` after the null guard above, and a hoisted declaration
  // would discard that narrowing (it could, in principle, run before it).
  const isOpen = () => menu.classList.contains('open');

  /** @param {MouseEvent} e */
  const onDocClick = (e) => {
    const target = /** @type {Node | null} */ (e.target);
    if (target && (menu.contains(target) || trigger.contains(target))) return;
    closeMenu();
  };

  /** @param {KeyboardEvent} e */
  const onKeydown = (e) => {
    if (e.key === 'Escape') {
      closeMenu();
      trigger.focus();
    }
  };

  const openMenu = () => {
    menu.classList.add('open');
    trigger.setAttribute('aria-expanded', 'true');
    // Capture phase, so an outside click closes before it does anything else.
    document.addEventListener('click', onDocClick, true);
    document.addEventListener('keydown', onKeydown, true);
  };

  const closeMenu = () => {
    menu.classList.remove('open');
    trigger.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', onDocClick, true);
    document.removeEventListener('keydown', onKeydown, true);
  };

  trigger.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isOpen()) closeMenu();
    else openMenu();
  });

  // Log out: app.js's initLogoutButton() owns the action (and its in-flight
  // lock). This module only dismisses the menu, so a logout that is refused or
  // still in flight does not leave the popover hanging over the page.
  menu.querySelector('#logout-btn')?.addEventListener('click', () => closeMenu());

  return { open: openMenu, close: closeMenu, isOpen };
}
