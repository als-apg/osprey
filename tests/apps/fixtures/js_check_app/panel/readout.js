// @ts-check
import { el } from '/design-system/js/dom.js';

/**
 * Show a value and remember it, the browser surfaces a panel test needs set up.
 * @param {string} value
 * @returns {HTMLElement}
 */
export function readout(value) {
  localStorage.setItem('readout', value);
  const node = el('span');
  node.textContent = value;
  return node;
}
