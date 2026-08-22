/**
 * Shared Tab/Shift+Tab focus trap for modal surfaces.
 *
 * Extracted verbatim from `<osprey-drawer>`, which was the first surface to
 * need it and is still the reference consumer. Any modal that covers the page
 * — drawer, dialog, feedback modal — owes the keyboard the same contract:
 * while it is open, Tab cycles inside it and never escapes to the inert
 * background behind it.
 *
 * The trap is deliberately small and DOM-only: one bubble-phase `keydown`
 * listener on the root element, no dependencies, no CSS, no globals. It
 * handles the wraparound only; showing/hiding the surface, moving focus in on
 * open, restoring it on close, and marking the background inert all stay with
 * the consumer.
 *
 * Visibility is judged by `offsetParent !== null`, which is the cheap layout
 * read that catches `display: none` subtrees and detached nodes. It is null
 * for *every* element under happy-dom (no layout engine), so unit tests have
 * to define an `offsetParent` on their fixture elements.
 *
 * @module focus-trap
 */

// ---- Module state ----

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]',
].join(',');

// The installed listener per trapped root. A WeakMap rather than a property on
// the element so the trap leaves no marks on its consumer's public surface,
// and so a discarded root is collectable without an explicit removal.
/** @type {WeakMap<HTMLElement, (event: KeyboardEvent) => void>} */
const _handlers = new WeakMap();

// ---- Public API ----

/**
 * Every focusable, actually-rendered descendant of `rootEl`, in document order.
 *
 * Useful on its own for moving focus into a surface as it opens (focus the
 * first entry) — the trap itself uses it on every Tab.
 *
 * @param {HTMLElement} rootEl - the trapped surface
 * @returns {HTMLElement[]} focusable descendants, first to last
 */
export function focusableElements(rootEl) {
  return Array.from(rootEl.querySelectorAll(FOCUSABLE_SELECTOR)).filter(
    /** @returns {el is HTMLElement} */
    (el) => el instanceof HTMLElement && el.offsetParent !== null
  );
}

/**
 * Install the Tab/Shift+Tab wraparound on `rootEl`. Idempotent — a second
 * install on an already-trapped root does nothing, so one `removeFocusTrap`
 * always disarms it.
 *
 * Tab past the last focusable wraps to the first, Shift+Tab past the first
 * wraps to the last, and either one with focus outside the root pulls it back
 * in. A root with nothing focusable swallows Tab entirely rather than letting
 * focus escape to the background.
 *
 * @param {HTMLElement} rootEl - the surface to trap focus inside
 * @returns {void}
 */
export function installFocusTrap(rootEl) {
  if (_handlers.has(rootEl)) return;
  /** @param {KeyboardEvent} event */
  const handler = (event) => {
    if (event.key !== 'Tab') return;
    const focusable = focusableElements(rootEl);
    if (focusable.length === 0) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const current = document.activeElement;
    const wrapsBackward = event.shiftKey && (current === first || !rootEl.contains(current));
    const wrapsForward = !event.shiftKey && (current === last || !rootEl.contains(current));
    if (wrapsBackward) {
      event.preventDefault();
      last.focus();
    } else if (wrapsForward) {
      event.preventDefault();
      first.focus();
    }
  };
  _handlers.set(rootEl, handler);
  rootEl.addEventListener('keydown', handler);
}

/**
 * Remove the Tab/Shift+Tab wraparound from `rootEl`. Idempotent — a no-op if
 * no trap is installed, so close paths can call it unconditionally.
 *
 * @param {HTMLElement} rootEl - the surface to release
 * @returns {void}
 */
export function removeFocusTrap(rootEl) {
  const handler = _handlers.get(rootEl);
  if (!handler) return;
  rootEl.removeEventListener('keydown', handler);
  _handlers.delete(rootEl);
}
