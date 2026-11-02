// @ts-check
/**
 * ARIEL Entries Helpers
 *
 * Pure helpers shared between entries.js and components.js.
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { facilityReferenceView } from '/design-system/js/facility-references.js';

/**
 * Format file size in human-readable format.
 * @param {number} bytes - Size in bytes
 * @returns {string} Formatted size
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * Split an entry's raw_text into a subject (first line) and details (the
 * remainder, falling back to the full text when there is no remainder).
 * `detailsStart` is the code-point offset at which `details` begins in the text.
 * @param {string} [rawText] - Raw entry text
 * @returns {{subject: string, details: string, detailsStart: number}}
 */
export function parseEntryText(rawText) {
  const text = rawText || '';
  const lines = text.split('\n');
  const subject = lines[0] || 'Untitled';
  const rest = lines.slice(1).join('\n');
  const trimmed = rest.trim();
  if (!trimmed) return { subject, details: text, detailsStart: 0 };
  const leading = rest.length - rest.trimStart().length;
  const detailsStart = Array.from(lines[0]).length + 1 + Array.from(rest.slice(0, leading)).length;
  return { subject, details: trimmed, detailsStart };
}

/**
 * Render the code points `[from, to)` of an entry's text as HTML, every character
 * escaped, each facility reference lying inside the window wrapped in a span.
 * @param {string} text - The entry's raw text
 * @param {Array<{start: number, end: number, target: string, records?: Array<*>|null}>|null} [references]
 * @param {number} [from]
 * @param {number} [to]
 * @returns {string} HTML string
 */
export function renderEntryText(text, references, from = 0, to = Infinity) {
  const points = Array.from(text);
  const end = Math.min(to, points.length);
  const plain = (/** @type {number} */ a, /** @type {number} */ b) =>
    escapeHtml(points.slice(a, b).join(''));
  const inside = (references || [])
    .filter((reference) => reference.start >= from && reference.end <= end)
    .sort((a, b) => a.start - b.start);
  let html = '';
  let cursor = from;
  for (const reference of inside) {
    if (reference.start < cursor) continue;
    const { title, dangling } = facilityReferenceView(reference.target, reference.records);
    const cls = dangling ? 'entry-facility-link--dangling' : 'entry-facility-link';
    html += plain(cursor, reference.start);
    html += `<span class="${cls}" title="${escapeHtml(title)}">`;
    html += `${plain(reference.start, reference.end)}</span>`;
    cursor = reference.end;
  }
  return html + plain(cursor, end);
}
