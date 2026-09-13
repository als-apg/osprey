// @ts-check
/*
 * System Health dashboard — pure, framework-free helpers.
 *
 * Every function defined here is self-contained and side-effect-free apart
 * from DOM node construction. Keep it that way so the module stays trivially
 * testable in isolation. Ported from an earlier integration-status dashboard,
 * with the DOM builder aligned to the okf panel's el(tag, attrs, children)
 * signature and every status mapped to a CSS class name — never an inline
 * color literal.
 *
 * `fmtName` is the one exception: the web terminal's health bar item shows the
 * same check names, and this bundle is served through the panel proxy rather
 * than to that page, so the rule lives in the design system and is re-exported
 * here for this dashboard's own importers.
 */

export { fmtName } from "/design-system/js/check-name.js";

/**
 * Status severity, highest wins. `worst()` reduces a set of statuses to the
 * most severe present; an unknown status is treated as 0 (ok-like).
 *
 * @type {Record<string, number>}
 */
export const STATUS_PRIORITY = { error: 3, warning: 2, skip: 1, ok: 0 };

/** @type {HTMLDivElement | null} */
let escDiv = null;

/**
 * HTML-escape an arbitrary server value by round-tripping it through a detached
 * element's `textContent`. `null`/`undefined` become the empty string. The
 * scratch element is created lazily and reused across calls.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function esc(value) {
  if (value == null) return "";
  if (!escDiv) escDiv = document.createElement("div");
  escDiv.textContent = String(value);
  return escDiv.innerHTML;
}

/**
 * Create a DOM element with a flat string-valued attribute map and a list of
 * child nodes. Signature-compatible with the okf panel's `el`.
 *
 * @param {string} tag
 * @param {Record<string, string>|null} [attrs] String-valued attribute map.
 *   The key `"class"` sets `className`; `"text"` sets `textContent`; every other
 *   key is set via `setAttribute`.
 * @param {(Node|null|undefined)[]} [children] Falsy entries are skipped.
 * @returns {HTMLElement}
 */
export function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const k in attrs) {
      if (k === "class") node.className = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
  }
  if (children) {
    for (const child of children) {
      if (child) node.appendChild(child);
    }
  }
  return node;
}

/**
 * Reduce a list of check results to the single worst status present. An empty
 * list is `"ok"`.
 *
 * @param {{status: string}[]} results
 * @returns {string}
 */
export function worst(results) {
  return results.reduce(
    (w, c) => ((STATUS_PRIORITY[c.status] || 0) > (STATUS_PRIORITY[w] || 0) ? c.status : w),
    "ok",
  );
}

/**
 * Format a latency in milliseconds for display. Zero, negative, or missing
 * values yield the empty string — matching `CheckResult.to_dict()`, which omits
 * `latency_ms` unless it is greater than zero.
 *
 * @param {number} [ms]
 * @returns {string}
 */
export function fmtMs(ms) {
  if (!ms || ms <= 0) return "";
  return ms < 1000 ? Math.round(ms) + "ms" : (ms / 1000).toFixed(1) + "s";
}

/**
 * Map a latency to a bucket CSS class — fast (`ms-f`), medium (`ms-m`), or slow
 * (`ms-s`). Zero, negative, or missing latency yields the empty string.
 *
 * @param {number} [ms]
 * @returns {string}
 */
export function msCls(ms) {
  if (!ms || ms <= 0) return "";
  if (ms < 100) return "ms-f";
  if (ms < 500) return "ms-m";
  return "ms-s";
}

/**
 * Group check results by their `category`, preserving first-seen order.
 *
 * @template {{category: string}} T
 * @param {T[]} results
 * @returns {Map<string, T[]>}
 */
export function byCategory(results) {
  /** @type {Map<string, T[]>} */
  const m = new Map();
  for (const r of results) {
    const bucket = m.get(r.category);
    if (bucket) bucket.push(r);
    else m.set(r.category, [r]);
  }
  return m;
}
