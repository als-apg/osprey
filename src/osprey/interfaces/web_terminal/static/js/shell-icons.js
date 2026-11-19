// @ts-check
/* OSPREY Web Terminal — the shell's glyphs and key hints
 *
 * The page ships its own glyphs as empty `svg[data-icon]` placeholders, sized
 * by `.osprey-ic` from the first paint, and its key hints as `[data-chord]`
 * elements. This module fills both. It imports icons.js and platform.js and
 * nothing else, so the page's first module (boot-shell.js) can draw the chrome
 * long before the application's own module graph has arrived.
 */

import { icon } from '/design-system/js/icons.js';
import { isMacPlatform } from '/design-system/js/platform.js';

/**
 * Fill every empty `svg[data-icon]` under `root` with the named design-system
 * glyph, in place, so the element (and any reference to it) is kept.
 * @param {ParentNode} [root]
 * @returns {void}
 */
export function hydrateIcons(root = document) {
  for (const svg of root.querySelectorAll('svg[data-icon]')) {
    if (svg.childElementCount > 0) continue;
    const name = svg.getAttribute('data-icon') ?? '';
    const glyph = icon(name);
    for (const attr of ['viewBox', 'aria-hidden', 'focusable']) {
      svg.setAttribute(attr, glyph.getAttribute(attr) ?? '');
    }
    svg.replaceChildren(...glyph.childNodes);
  }
}

/**
 * Write each `[data-chord]` key hint under `root` for this platform: `⌘` on
 * macOS, `Ctrl+` elsewhere, then the key the attribute names.
 * @param {ParentNode} [root]
 * @returns {void}
 */
export function paintChords(root = document) {
  const prefix = isMacPlatform() ? '⌘' : 'Ctrl+';
  for (const el of root.querySelectorAll('[data-chord]')) {
    el.textContent = `${prefix}${/** @type {HTMLElement} */ (el).dataset.chord}`;
  }
}
