// @ts-check
/**
 * ARIEL Reader: facility references
 *
 * The reader's title and body with every quoted channel or device id marked
 * (`renderEntryText`), and the sidebar card that lists each referenced id
 * once with its facility records. References carry code-point offsets into
 * `raw_text`, so a stored subject that is not raw_text's first line shows
 * plain.
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { facilityReferenceView } from '/design-system/js/facility-references.js';
import { parseEntryText, renderEntryText } from './entries-helpers.js';

/**
 * @typedef {{start: number, end: number, target: string, records?: Array<*>|null}} Reference
 */

/**
 * The reader's title and body as HTML. The title is the stored/derived
 * subject, else raw_text's first line, and is marked only when it is that
 * first line.
 * @param {{raw_text?: string, subject?: string|null, facility_references?: Reference[]|null}} entry
 * @returns {{titleHtml: string, bodyHtml: string}}
 */
export function renderEntryTextHtml(entry) {
  const rawText = entry.raw_text || '';
  const references = entry.facility_references;
  const { subject, details, detailsStart } = parseEntryText(rawText);
  const firstLine = rawText.split('\n')[0];
  const subjectEnd = Array.from(firstLine).length;
  const titleHtml =
    entry.subject && entry.subject !== firstLine
      ? escapeHtml(entry.subject)
      : subjectEnd
        ? renderEntryText(rawText, references, 0, subjectEnd)
        : escapeHtml(entry.subject || subject);
  const detailsEnd = detailsStart + Array.from(details).length;
  return { titleHtml, bodyHtml: renderEntryText(rawText, references, detailsStart, detailsEnd) };
}

/**
 * The body of the sidebar's "Facility references" card: each target once, in
 * text order, with one line per record (`kind · unit · description`), or the
 * reason it has none. Empty when the entry has no references.
 * @param {Reference[]|null|undefined} references
 * @returns {string} HTML string
 */
export function renderFacilityReferencesList(references) {
  /** @type {Map<string, Reference>} */
  const byTarget = new Map();
  for (const reference of references || []) {
    if (!byTarget.has(reference.target)) byTarget.set(reference.target, reference);
  }
  if (byTarget.size === 0) return '';
  const items = [...byTarget.values()].map(({ target, records }) => {
    const { title } = facilityReferenceView(target, records);
    const resolved = Array.isArray(records) && records.length > 0;
    // The view's wording: one line per record, else "<target> · <reason>".
    const lines = resolved
      ? title
          .split('\n')
          .map((line) => `<div class="facility-ref-record">${escapeHtml(line)}</div>`)
      : [
          `<div class="facility-ref-record facility-ref-none">${escapeHtml(title.slice(target.length + ' · '.length))}</div>`,
        ];
    return `<div class="facility-ref"><div class="facility-ref-target font-mono">${escapeHtml(target)}</div>${lines.join('')}</div>`;
  });
  return `<div class="facility-ref-list">${items.join('')}</div>`;
}
