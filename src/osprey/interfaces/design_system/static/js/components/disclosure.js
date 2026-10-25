// @ts-check
/**
 * Disclosure behaviour for `.disclosure-btn[aria-expanded]` (components.css).
 *
 * A disclosure button names the panel it opens with `aria-controls`. Clicking
 * it flips `aria-expanded`, shows or hides that panel (`hidden`), marks the
 * enclosing `.step` `.is-open` (the step-row options pattern), and fires a
 * `disclosure:toggle` event with `{ open }` on the button. A button with no
 * `aria-controls` only flips `aria-expanded`.
 *
 *     import { wireDisclosures } from '/design-system/js/components/disclosure.js';
 *     const dispose = wireDisclosures(document.getElementById('step-list'));
 *
 * One delegated listener per root, so rows rendered later are covered too.
 * Native `<details class="folder">` needs none of this; it is pure CSS.
 *
 * @module components/disclosure
 */

/**
 * Open or close one disclosure.
 *
 * @param {Element} button  The `.disclosure-btn`.
 * @param {boolean} open
 */
export function setDisclosure(button, open) {
  button.setAttribute('aria-expanded', open ? 'true' : 'false');
  const id = button.getAttribute('aria-controls');
  const panel = id ? button.ownerDocument.getElementById(id) : null;
  if (panel) panel.hidden = !open;
  const step = button.closest('.step');
  if (step) step.classList.toggle('is-open', open);
}

/**
 * Wire every disclosure button under `root`, now and later.
 *
 * @param {Element|Document} [root]
 * @returns {() => void} Removes the listener.
 */
export function wireDisclosures(root = document) {
  /** @param {Event} event */
  const onClick = (event) => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest('.disclosure-btn');
    if (!button || !root.contains(button)) return;
    const open = button.getAttribute('aria-expanded') !== 'true';
    setDisclosure(button, open);
    button.dispatchEvent(new CustomEvent('disclosure:toggle', { detail: { open }, bubbles: true }));
  };
  root.addEventListener('click', onClick);
  return () => root.removeEventListener('click', onClick);
}
