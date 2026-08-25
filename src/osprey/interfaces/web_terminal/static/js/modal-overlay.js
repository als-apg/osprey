/* OSPREY Web Terminal — shared modal overlay lifecycle */

/**
 * Milliseconds after which a fading overlay is removed regardless of the
 * transition. Long enough for the CSS fade, short enough that a dialog never
 * appears stuck; the removal is idempotent, so a transition that *does* fire
 * simply gets there first.
 */
const FADE_FALLBACK_MS = 300;

/**
 * Put a built overlay on screen and start its fade-in.
 *
 * The `visible` class is added on the next frame rather than immediately so the
 * browser has laid the node out at its starting opacity — set in the same tick,
 * the transition has nothing to animate from and the dialog snaps in.
 *
 * @param {HTMLElement} overlay Fully built overlay node, not yet in the document.
 */
export function mountOverlay(overlay) {
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add('visible'));
}

/**
 * Fade an overlay out and remove it, whether or not the transition fires.
 *
 * Removal is driven by `transitionend` *and* by a fallback timer, because
 * reduced-motion settings and test environments never fire the event — and a
 * dismissed dialog that stays in the DOM keeps its backdrop over the page.
 * Both paths are guarded (`{ once: true }`, and a `parentNode` check), so
 * whichever arrives second is a no-op.
 *
 * Callers own everything that is not the fade: clearing their own listeners
 * and flags, and any state they mark on the node before it goes.
 *
 * @param {HTMLElement} overlay Overlay currently in the document.
 */
export function fadeOutOverlay(overlay) {
  overlay.classList.remove('visible');
  overlay.addEventListener('transitionend', () => overlay.remove(), { once: true });
  setTimeout(() => {
    if (overlay.parentNode) overlay.remove();
  }, FADE_FALLBACK_MS);
}
