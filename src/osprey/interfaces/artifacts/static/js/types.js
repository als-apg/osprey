// @ts-check
/**
 * OSPREY Artifact Gallery — type registry, formatting, and color-pass utilities.
 *
 * Stateless/self-contained: the only mutable state is the type registry
 * fetched from `/api/type-registry` (behind a getter, since ES modules only
 * give importers a read-only live view of an exported binding). Everything
 * else here is a pure function of its arguments.
 *
 * @module types
 */

import { fileUrl } from "./state.js";

// ---- Type Registry ---- //

/** @type {any} */
let typeRegistry = {};

/** @returns {any} */
export function getTypeRegistry() { return typeRegistry; }

/**
 * Fetch the type registry from the API. Silent on failure (matches the
 * original: console-only), leaving `typeRegistry` at its previous value.
 * @returns {Promise<void>}
 */
export async function initTypeRegistry() {
  try {
    const resp = await fetch("/api/type-registry");
    typeRegistry = await resp.json();
  } catch (err) {
    console.error("Failed to load type registry:", err);
  }
}

/**
 * Human-readable label for an artifact/category type.
 * @param {string} type
 * @returns {string}
 */
export function typeBadge(type) {
  const info =
    (typeRegistry.categories && typeRegistry.categories[type]) ||
    (typeRegistry.artifact_types && typeRegistry.artifact_types[type]) || {};
  return info.label || type.replace(/_/g, " ");
}

/**
 * SVG icon markup for an artifact/category type.
 * @param {string} type
 * @returns {string}
 */
export function typeIcon(type) {
  /** @type {Record<string, string>} */
  const icons = {
    // Artifact types
    plot_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    plot_png: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
    table_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/></svg>',
    html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
    markdown: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/></svg>',
    text: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
    json: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/></svg>',
    image: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
    notebook: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
    dashboard_html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    // Category types
    archiver_data: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 5v14c0 1.66-4.03 3-9 3s-9-1.34-9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
    channel_values: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    write_results: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>',
    code_output: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
    visualization: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><polyline points="7 14 11 10 15 14 19 8"/></svg>',
    dashboard: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    document: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
    screenshot: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>',
    channel_finder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    search_results: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    logbook_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
    literature_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
    wiki_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
    mml_research: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    agent_response: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>',
    user_artifact: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    diagnostic_report: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M16 4h2a2 2 0 012 2v14a2 2 0 01-2 2H6a2 2 0 01-2-2V6a2 2 0 012-2h2"/><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M9 14l2 2 4-4"/></svg>',
  };
  return icons[type] || icons.text;
}

/**
 * CSS color for an artifact/category type, with a theme-invariant fallback.
 * @param {string} type
 * @returns {string}
 */
export function typeColor(type) {
  const info =
    (typeRegistry.categories && typeRegistry.categories[type]) ||
    (typeRegistry.artifact_types && typeRegistry.artifact_types[type]) || {};
  return info.color || "#64748b"; // hygiene-allow-color: matches --text-muted exactly, theme-invariant fallback
}

/**
 * Thumbnail markup for an artifact card: an image/iframe preview for
 * displayable types, a summary-field dump, or a generic type icon fallback.
 * @param {any} a
 * @returns {string}
 */
export function thumbnailHtml(a) {
  const url = fileUrl(a);
  switch (a.artifact_type) {
    case "plot_png":
    case "image":
      return `<img src="${url}" alt="" loading="lazy"
               onerror="this.parentElement.classList.add('img-error')" />`;
    case "plot_html":
    case "table_html":
    case "dashboard_html":
    case "html":
      return `<iframe src="${url}" sandbox="allow-scripts allow-same-origin"
               loading="lazy" tabindex="-1"></iframe>`;
    case "notebook":
      return `<iframe src="/api/notebooks/${a.id}/rendered"
               sandbox="allow-scripts allow-same-origin"
               loading="lazy" tabindex="-1"></iframe>`;
    default:
      if (a.summary && Object.keys(a.summary).length > 0) {
        const text = Object.entries(a.summary)
          .map(([k, v]) => `${k}: ${v}`)
          .join("\n");
        return `<div class="thumb-summary">${escapeHtml(text)}</div>`;
      }
      return `<div class="thumb-placeholder">${typeIcon(a.artifact_type)}<span>${typeBadge(a.artifact_type)}</span></div>`;
  }
}

// ---- Utilities ---- //

/**
 * HTML-escape a value using the textContent -> innerHTML trick. Nullish
 * input yields "" (not "undefined"/"null").
 * @param {unknown} str
 * @returns {string}
 */
export function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = /** @type {any} */ (str) || "";
  return d.innerHTML;
}

/**
 * Human-readable byte size (B/KB/MB/GB).
 * @param {number} bytes
 * @returns {string}
 */
export function formatSize(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let size = bytes;
  while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
  return `${size.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/**
 * Locale time-of-day (e.g. "3:45 PM"). Empty string for falsy input.
 * @param {string} [iso]
 * @returns {string}
 */
export function formatTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/**
 * Full locale date + time (e.g. "Jul 3, 2026, 3:45 PM"). Empty string for
 * falsy input.
 * @param {string} [iso]
 * @returns {string}
 */
export function formatFullTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

/**
 * "Today" / "Yesterday" / a short locale date, for grouping the activity
 * timeline. "Unknown" for falsy input.
 * @param {string} [iso]
 * @returns {string}
 */
export function formatDate(iso) {
  if (!iso) return "Unknown";
  const d = new Date(iso);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) return "Today";
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * URL for "Open in new tab" — uses rendered endpoints for types that
 * browsers can't display natively (markdown, notebook).
 * @param {any} a
 * @returns {string}
 */
export function openUrl(a) {
  switch (a.artifact_type) {
    case "markdown": return `/api/markdown/${a.id}/rendered`;
    case "notebook": return `/api/notebooks/${a.id}/rendered`;
    default:         return fileUrl(a);
  }
}

/**
 * Whether an artifact was created during the current gallery session.
 * `sessionStart` is passed in explicitly (gallery.js's `_sessionStart`, set
 * once at page load) rather than held here, keeping this module stateless.
 * @param {{timestamp?: string}} a
 * @param {string} sessionStart
 * @returns {boolean}
 */
export function isNewThisSession(a, sessionStart) {
  return !!(a.timestamp && a.timestamp >= sessionStart);
}

/**
 * Forward text to the parent embedder (e.g. a terminal panel) for paste,
 * via postMessage. No-op outside an embedded/iframed context.
 * @param {string} text
 * @returns {void}
 */
export function sendToTerminal(text) {
  try {
    if (window.parent && window.parent !== window) {
      // Intentional '*' (same-origin contract exception): parent embedder may be cross-origin.
      window.parent.postMessage({ type: "osprey-paste-to-terminal", text }, "*");
    }
  } catch { /* cross-origin */ }
}

// ---- Color pass: color badges by type ---- //

/**
 * @param {HTMLElement} el
 * @param {string} color
 */
function _setTypeColorVars(el, color) {
  el.style.setProperty("--type-color", color);
  el.style.setProperty("--type-bg", color + "14");
  el.style.setProperty("--type-border", color + "40");
}

/**
 * Re-color every type badge/section/card on screen from the current
 * registry, on the next animation frame.
 * @returns {void}
 */
export function requestColorPass() {
  requestAnimationFrame(() => {
    document.querySelectorAll("[class*='badge-']").forEach((el) => {
      const cls = [...el.classList].find((c) => c.startsWith("badge-"));
      if (cls) {
        const type = cls.replace("badge-", "");
        const color = typeColor(type);
        /** @type {HTMLElement} */ (el).style.color = color;
        /** @type {HTMLElement} */ (el).style.borderColor = color;
      }
    });
    document.querySelectorAll(".tree-section[data-type]").forEach((el) => {
      const type = /** @type {HTMLElement} */ (el).dataset.type;
      if (type) _setTypeColorVars(/** @type {HTMLElement} */ (el), typeColor(type));
    });
    document.querySelectorAll(".gallery-card[data-type]").forEach((el) => {
      const type = /** @type {HTMLElement} */ (el).dataset.type;
      if (type) _setTypeColorVars(/** @type {HTMLElement} */ (el), typeColor(type));
    });
    document.querySelectorAll(".timeline-item[data-type]").forEach((el) => {
      const type = /** @type {HTMLElement} */ (el).dataset.type;
      if (type) _setTypeColorVars(/** @type {HTMLElement} */ (el), typeColor(type));
    });
  });
}
