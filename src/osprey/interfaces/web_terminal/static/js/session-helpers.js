// @ts-check
/* OSPREY Web Terminal — Session Activity Log: shared render helpers
 *
 * Small formatting/lookup helpers (server-color class, timestamp
 * formatting, byte formatting, empty-state markup, artifact type icons)
 * shared by two or more of session-views.js's Activity-log view renderers.
 * Kept separate from session-views.js to hold that module under the
 * 450-line cap -- these five functions have no view-specific state or DOM
 * wiring, so they're a clean, independently-testable seam.
 *
 * @module session-helpers
 */

import { escapeHtml as esc } from '/design-system/js/dom.js';

/** @type {Record<string, string>} */
const SERVER_COLORS = {
  controls: 'srv-controls', python: 'srv-python', workspace: 'srv-workspace',
  ariel: 'srv-ariel', 'channel-finder': 'srv-channel-finder',
};

/**
 * @param {string|null|undefined} name
 * @returns {string}
 */
export function serverClass(name) {
  if (!name) return 'srv-unknown';
  const lower = name.toLowerCase();
  return SERVER_COLORS[lower] || 'srv-unknown';
}

/**
 * @param {string|null|undefined} isoStr
 * @returns {string}
 */
export function ts(isoStr) {
  if (!isoStr) return '';
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return ''; }
}

/**
 * @param {number|null|undefined} n
 * @returns {string}
 */
export function fmtBytes(n) {
  if (!n || n === 0) return '0 B';
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

/**
 * @param {string} title
 * @param {string} sub
 * @returns {string}
 */
export function emptyState(title, sub) {
  return `<div class="empty"><div class="empty-title">${esc(title)}</div><div class="empty-sub">${esc(sub)}</div></div>`;
}

/** @type {Record<string, string>} */
const TYPE_ICONS = {
  plot: '\u{1F4C8}', table: '\u{1F4CA}', text: '\u{1F4C4}', code: '\u{1F4BB}',
  json: '\u{1F4CB}', csv: '\u{1F4C3}', image: '\u{1F5BC}', notebook: '\u{1F4D3}',
  markdown: '\u{270F}', html: '\u{1F310}', log: '\u{1F4DC}',
};
/**
 * @param {string|null|undefined} t
 * @returns {string}
 */
export function typeIcon(t) {
  return TYPE_ICONS[(t || '').toLowerCase()] || '\u{1F4E6}';
}
