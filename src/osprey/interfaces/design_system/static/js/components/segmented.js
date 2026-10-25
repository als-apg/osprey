// @ts-check
/**
 * Segmented-control behaviour for `.seg` (components.css).
 *
 * A segmented control is a `role="group"` of buttons, each with a
 * `data-value`, exactly one of them `aria-pressed="true"`. A click (or the
 * left / right arrow keys) presses another one and reports its value; the
 * pressed button and a disabled one (`disabled` or `aria-disabled="true"`)
 * report nothing.
 *
 *     import { wireSegmented } from '/design-system/js/components/segmented.js';
 *     const dispose = wireSegmented(document.getElementById('mode'), (value) => setMode(value));
 *
 * @module components/segmented
 */

/**
 * @param {Element} seg
 * @returns {HTMLButtonElement[]}
 */
function buttons(seg) {
  return /** @type {HTMLButtonElement[]} */ ([...seg.querySelectorAll('button[data-value]')]);
}

/** @param {HTMLButtonElement} button */
function isDisabled(button) {
  return button.disabled || button.getAttribute('aria-disabled') === 'true';
}

/**
 * Press the button whose `data-value` is `value`, without reporting it.
 *
 * @param {Element} seg
 * @param {string} value
 */
export function setSegmented(seg, value) {
  for (const button of buttons(seg)) {
    button.setAttribute('aria-pressed', button.dataset.value === value ? 'true' : 'false');
  }
}

/**
 * Wire one segmented control.
 *
 * @param {Element} seg
 * @param {(value: string, button: HTMLButtonElement) => void} onChange
 * @returns {() => void} Removes the listeners.
 */
export function wireSegmented(seg, onChange) {
  /** @param {HTMLButtonElement|undefined} button */
  const press = (button) => {
    if (!button || isDisabled(button) || button.getAttribute('aria-pressed') === 'true') return;
    setSegmented(seg, button.dataset.value ?? '');
    onChange(button.dataset.value ?? '', button);
  };
  /** @param {Event} event */
  const onClick = (event) => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest('button[data-value]');
    if (button instanceof HTMLButtonElement && seg.contains(button)) press(button);
  };
  /** @param {Event} event */
  const onKeydown = (event) => {
    const key = /** @type {KeyboardEvent} */ (event).key;
    if (key !== 'ArrowRight' && key !== 'ArrowLeft') return;
    const enabled = buttons(seg).filter((button) => !isDisabled(button));
    const current = enabled.findIndex((button) => button.getAttribute('aria-pressed') === 'true');
    if (enabled.length === 0) return;
    event.preventDefault();
    const step = key === 'ArrowRight' ? 1 : -1;
    const next = enabled[(current + step + enabled.length) % enabled.length];
    press(next);
    next.focus();
  };
  seg.addEventListener('click', onClick);
  seg.addEventListener('keydown', onKeydown);
  return () => {
    seg.removeEventListener('click', onClick);
    seg.removeEventListener('keydown', onKeydown);
  };
}
