// @ts-check
/**
 * ARIEL Reader: facility references
 *
 * The reader's title and body with every quoted channel or device id marked
 * (`renderEntryText`), and the sidebar card that lists each referenced id
 * once with its facility records. References carry code-point offsets into
 * `raw_text`, so a stored subject that is not raw_text's first line shows
 * plain. A body whose entry declares `metadata.body_format: markdown` is
 * rendered as formatted text instead (`entry-markdown.js`); every other entry
 * shows its text as stored.
 */

import { escapeHtml } from '/design-system/js/dom.js';
import { facilityReferenceView } from '/design-system/js/facility-references.js';
import { parseEntryText, renderEntryText } from './entries-helpers.js';
import { renderEntryMarkdown } from './entry-markdown.js';

/**
 * @typedef {{start: number, end: number, target: string, records?: Array<*>|null}} Reference
 */

/**
 * The reader's title and body as HTML. The title is the stored/derived
 * subject, else raw_text's first line, and is marked only when it is that
 * first line. The body is rendered as Markdown (`rendered: true`) only when
 * the entry's `metadata.body_format` is `markdown`, in any letter case; with
 * no format, any other format, or a render that fails, it is the stored text,
 * escaped. The title is always plain.
 * @param {{raw_text?: string, subject?: string|null, facility_references?: Reference[]|null,
 *   metadata?: {body_format?: unknown}|null, attachments?: Array<*>|null}} entry
 * @returns {{titleHtml: string, bodyHtml: string, rendered: boolean}}
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
  const format = entry.metadata?.body_format;
  const markdownHtml =
    typeof format === 'string' && format.toLowerCase() === 'markdown'
      ? renderEntryMarkdown(details, { references, attachments: entry.attachments })
      : null;
  if (markdownHtml !== null) return { titleHtml, bodyHtml: markdownHtml, rendered: true };
  const detailsEnd = detailsStart + Array.from(details).length;
  return {
    titleHtml,
    bodyHtml: renderEntryText(rawText, references, detailsStart, detailsEnd),
    rendered: false,
  };
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
