// @ts-check
/**
 * OSPREY Web Terminal — operator chat: DOM and text helpers.
 *
 * The element and button builders every chat block is made from, and the
 * duration and JSON formatters the blocks print. Text goes in with textContent;
 * nothing here reaches innerHTML.
 *
 * @module chat-dom
 */

/** An SSE event as decoded off the wire; the builders read fields by name. @typedef {Record<string, any>} WireEvent */

/**
 * @param {string | undefined} tag
 * @param {string} className
 * @param {string} [text] - assigned via textContent (never innerHTML)
 * @returns {HTMLElement}
 */
export function elem(tag, className, text) {
  const node = document.createElement(tag ?? 'div');
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLButtonElement}
 */
export function button(className, text) {
  const node = /** @type {HTMLButtonElement} */ (elem('button', className, text));
  node.type = 'button';
  return node;
}

/**
 * A duration as the chat states it: `12s` in the footer, one decimal on the
 * live loader, and minutes once past a minute.
 * @param {number} ms
 * @param {boolean} [precise]
 * @returns {string}
 */
export function formatElapsed(ms, precise = false) {
  const seconds = Math.max(0, ms) / 1000;
  if (seconds < 60) return precise ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`;
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, '0')}s`;
}

/**
 * @param {unknown} value
 * @returns {string}
 */
export function prettyJson(value) {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return String(value);
  }
}
