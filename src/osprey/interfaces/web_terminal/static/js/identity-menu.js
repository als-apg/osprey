// @ts-check
/* OSPREY Web Terminal — Identity Menu (the header user chip)
 *
 * One chip in the header states who you are; clicking it opens a popover
 * holding the deployment line and the Log out control. It replaces the old
 * pair — a bare username badge beside a standalone door button — which read
 * as two unrelated controls despite being one fact and one action on it.
 *
 * Division of labour:
 *   - This module owns ONLY the popover (open/close, outside-click, Escape).
 *   - The logout BEHAVIOUR is not this module's: the button keeps its
 *     `#logout-btn` id and `data-landing-url` attribute, so app.js's
 *     initLogoutButton() still owns the safe-URL check, the POST, the
 *     disabled/aria-busy lock and the navigation, and palette-boot.js's
 *     "Log out" command still finds the button by id. Moving the control
 *     into a popover changed its placement, not its contract.
 *
 * Deliberately no close-on-logout-click: every path out of that handler
 * navigates away, and closing the card first would flash the chip back to its
 * resting state while the POST is still in flight, hiding the button's own
 * aria-busy state from the operator (and from assistive tech) at exactly the
 * moment it matters.
 *
 * The popover grammar (capture-phase outside-click + Escape close;
 * aria-expanded mirrored on the trigger) matches display-menu.js and
 * panel-add-menu.js, the shell's other header popovers.
 */

/**
 * Wire the header identity chip's popover.
 *
 * No-op when the chip is absent — a single-user deployment renders no
 * `OSPREY_TERMINAL_USER`, so there is no identity to show and the header
 * carries search + display menu alone.
 */
export function initIdentityMenu() {
  const root = document.getElementById('identity-menu');
  const button = /** @type {HTMLButtonElement | null} */ (
    document.getElementById('identity-menu-btn')
  );
  const card = document.getElementById('identity-menu-card');
  if (!root || !button || !card) return;

  const isOpen = () => card.classList.contains('open');

  /** @param {MouseEvent} e */
  function onDocClick(e) {
    if (e.target instanceof Node && root && !root.contains(e.target)) closeMenu();
  }

  /** @param {KeyboardEvent} e */
  function onKeydown(e) {
    if (e.key === 'Escape') {
      closeMenu();
      button?.focus();
    }
  }

  function openMenu() {
    if (!card || !button) return;
    card.classList.add('open');
    button.setAttribute('aria-expanded', 'true');
    // Capture-phase so an outside click closes before it does anything else.
    document.addEventListener('click', onDocClick, true);
    document.addEventListener('keydown', onKeydown, true);
  }

  function closeMenu() {
    if (!card || !button) return;
    card.classList.remove('open');
    button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', onDocClick, true);
    document.removeEventListener('keydown', onKeydown, true);
  }

  button.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isOpen()) closeMenu();
    else openMenu();
  });
}
