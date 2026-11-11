// @ts-check
/**
 * The OPTIMIZER bundle's element builder.
 *
 * Every dynamic node in this panel is built through `h` — `createElement`
 * plus `textContent`/property assignment — so service-supplied strings
 * (variable names, error details, preset descriptions) reach the DOM as text
 * and are never interpreted as markup. There is no innerHTML sink anywhere in
 * the bundle.
 *
 * @module hyperscript
 */

/**
 * Create an element, apply props, append children.
 *
 * Props: `class` sets className, `text` sets textContent, `on<event>` with a
 * function value adds a listener, anything else that names an existing
 * property is assigned as a property (value, checked, disabled, …) and the
 * rest become attributes. `null`, `undefined` and `false` values are skipped,
 * so optional props can be passed inline. String and number children become
 * text nodes.
 *
 * @param {string} tag
 * @param {Record<string, unknown>|null} [props]
 * @param {...(Node|string|number|null|undefined|false)} children
 * @returns {HTMLElement}
 */
export function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') {
      node.className = String(value);
    } else if (key === 'text') {
      node.textContent = String(value);
    } else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2), /** @type {EventListener} */ (value));
    } else if (key in node) {
      /** @type {any} */ (node)[key] = value;
    } else {
      node.setAttribute(key, String(value));
    }
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === 'object' ? child : document.createTextNode(String(child)));
  }
  return node;
}

/**
 * Replace every child of `parent` with `children`.
 *
 * @param {Element} parent
 * @param {...(Node|null|undefined|false)} children
 */
export function replaceChildren(parent, ...children) {
  parent.replaceChildren(...children.filter((c) => c !== null && c !== undefined && c !== false));
}

/**
 * Format a number for display: finite numbers to at most `digits` significant
 * figures, everything else as an em dash.
 *
 * @param {unknown} value
 * @param {number} [digits]
 * @returns {string}
 */
export function fmt(value, digits = 6) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return String(Number(value.toPrecision(digits)));
}
