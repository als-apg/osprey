// @ts-check
/**
 * Agent-activity highlight helper for the design-system front-end.
 *
 * Pairs with css/highlight.css: `flashElement(el)` fires the one-shot
 * `.agent-flash` glow on any element. Hand-written ES module (mirrors
 * dom.js) with JSDoc types so it type-checks under `tsc --noEmit --strict`.
 *
 * The persistent `.agent-attention` badge from the same stylesheet needs no
 * JS beyond `classList.toggle`, so consumers drive it directly.
 *
 * @module highlight
 */

/** Class name applied for the duration of the flash animation. */
const FLASH_CLASS = 'agent-flash';

/** Keyframes name the flash class animates (see css/highlight.css). */
const FLASH_ANIMATION = 'agent-flash-glow';

/**
 * Flash `el` with the one-shot agent-activity glow.
 *
 * Reflow-restart pattern: remove the class, force a style/layout flush by
 * reading `offsetWidth`, then re-add it — so two rapid calls on the same
 * element re-fire the animation instead of the second `add` being a no-op.
 * The class is removed again on `animationend` (self-cleaning), keeping the
 * element's class list stable between flashes.
 *
 * @param {HTMLElement} el
 * @returns {void}
 */
export function flashElement(el) {
  el.classList.remove(FLASH_CLASS);
  void el.offsetWidth; // force reflow so re-adding the class restarts the animation
  el.classList.add(FLASH_CLASS);
  el.addEventListener(
    'animationend',
    /** @param {Event} ev */
    (ev) => {
      // Ignore bubbled ends of unrelated child animations; tolerate synthetic
      // events (tests, older engines) that carry no animationName.
      if (ev.target !== el) return;
      const name = /** @type {AnimationEvent} */ (ev).animationName;
      if (name && name !== FLASH_ANIMATION) return;
      el.classList.remove(FLASH_CLASS);
    },
    { once: true },
  );
}
