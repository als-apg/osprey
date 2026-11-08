// @ts-check
/**
 * The plan bundle's element builder.
 *
 * Every node in this panel is built through `h` — `createElement` plus
 * `textContent`/property assignment — so plan-authored strings (names,
 * descriptions, source, enum values) reach the DOM as text and are never
 * interpreted as markup. There is no innerHTML sink anywhere in the bundle,
 * and this module is what makes that affordable: the safe way to build a node
 * is also the short way.
 *
 * Shared by plans-view.js, plan-browser.js and schema-form.js so all three build
 * nodes the same way rather than each carrying its own copy.
 *
 * @module hyperscript
 */

import { icon } from '/design-system/js/icons.js';

/**
 * Create an element, apply props, append children.
 *
 * Props: `class` sets className, `text` sets textContent, anything else that
 * names an existing property is assigned as a property (value, checked,
 * disabled, min, max, step, …) and the rest become attributes. `null`,
 * `undefined` and `false` values are skipped, so optional props can be passed
 * inline. Children that are strings or numbers become text nodes.
 *
 * @param {string} tag
 * @param {Record<string, unknown>} [props]
 * @param {...(Node|string|number|null|undefined)} children
 * @returns {HTMLElement}
 */
export function h(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') {
        node.className = String(value);
      } else if (key === 'text') {
        node.textContent = String(value);
      } else if (key in node) {
        /** @type {any} */ (node)[key] = value;
      } else {
        node.setAttribute(key, String(value));
      }
    }
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === 'object' ? child : document.createTextNode(String(child)));
  }
  return node;
}

/**
 * A small Info icon whose tooltip is `text` (`data-tip`, the design system's
 * tooltip layer): how field help reads beside its label instead of as a line
 * of text on the page. Focusable, so the tip shows on keyboard focus too, and
 * `text` is its accessible name.
 *
 * @param {string} text
 * @returns {HTMLElement}
 */
export function infoTip(text) {
  return h(
    'span',
    { class: 'field-help', tabindex: '0', role: 'img', 'aria-label': text, 'data-tip': text },
    icon('info', { size: 14 })
  );
}
