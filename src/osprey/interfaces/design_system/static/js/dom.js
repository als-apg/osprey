// @ts-check
/**
 * DOM utility helpers for the design-system front-end.
 *
 * Small, dependency-free primitives shared across the web-terminal UI:
 * element creation, HTML escaping, and trailing-edge debouncing. Hand-written
 * ES module (mirrors theme-manager.js) with JSDoc types so it type-checks under
 * `tsc --noEmit --strict`.
 *
 * @module dom
 */

/**
 * Create an element and optionally assign it a class name.
 *
 * @param {string} tag
 * @param {string} [className]
 * @returns {HTMLElement}
 */
export function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

/**
 * HTML-escape a value using the textContent -> innerHTML trick, then also
 * escape double- and single-quotes.
 *
 * Nullish input yields "" (not "undefined"/"null"). The result is safe to
 * interpolate both into element text/HTML content AND into a quoted
 * attribute value (single- or double-quoted), since `"` and `'` are escaped
 * in addition to `&`, `<`, and `>`.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value ?? "");
  return div.innerHTML.replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

/**
 * The reverse-proxy path prefix a panel is mounted under when reached through
 * the web-terminal proxy at ``/panel/{id}/…`` — everything up to and including
 * the ``/panel/{id}`` segment. Multi-user deployments mount each user's whole
 * terminal under an outer prefix (``/u/<user>/panel/{id}/…``), so the match is
 * a lazy scan to the FIRST ``/panel/`` segment rather than anchored at the
 * root. A panel issues its API fetches prefixed with this (the proxy does not
 * rewrite root-absolute API paths), so they resolve back to the panel's own
 * sidecar. Returns ``""`` when the panel is reached directly — mounted
 * standalone by its sidecar, or opened in a visual-test harness — so those
 * fetches fall back to root-relative.
 *
 * @returns {string}
 */
export function panelApiPrefix() {
  return (location.pathname.match(/^.*?\/panel\/[^/]+/) || [''])[0];
}

/**
 * Wrap `fn` so it only fires once `ms` has elapsed since the last call
 * (trailing edge). Preserves `this` and forwards arguments.
 *
 * @template {(...args: any[]) => any} F
 * @param {F} fn
 * @param {number} ms
 * @returns {(...args: Parameters<F>) => void}
 */
export function debounce(fn, ms) {
  /** @type {ReturnType<typeof setTimeout>|undefined} */
  let t;
  /**
   * @this {any}
   * @param {Parameters<F>} args
   * @returns {void}
   */
  return function (...args) {
    const self = this;
    clearTimeout(t);
    t = setTimeout(() => fn.apply(self, args), ms);
  };
}
