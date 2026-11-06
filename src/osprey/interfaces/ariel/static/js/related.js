// @ts-check
/**
 * ARIEL reader "Related" section.
 *
 * An after-body reader-seam section, below the thread. Once the body has
 * painted it fetches `/entries/{id}/related` and shows up to five rows as the
 * same `data-thread-entry` buttons the thread uses, so the reader routes a
 * click to its `onSelectEntry` callback with no handler of its own.
 *
 * The server's `state` decides what shows: `ok` lists the rows (nothing when
 * there are none), `no_embedding` shows a one-line note, `disabled` hides the
 * section. A failed fetch leaves the section out and logs the error; it never
 * raises a toast, because the entry itself is unaffected.
 */

import { entriesApi } from './api.js';
import { escapeHtml } from './components.js';
import { registerReaderSection } from './reader.js';
import { renderThreadLink } from './reader-thread.js';

/** The most related rows the section shows. */
export const RELATED_LIMIT = 5;

/**
 * The section's frame around `body`.
 * @param {string} body - HTML string
 * @returns {string} HTML string
 */
function frame(body) {
  return `<div class="entry-detail-content entry-thread entry-related" data-entry-related>
      <h4>Related</h4>
      ${body}
    </div>`;
}

/**
 * The section's markup for a `/related` response.
 * @param {import('./api.js').RelatedResponse} related
 * @returns {string} HTML string, `''` when nothing shows
 */
export function renderRelatedSection(related) {
  const state = related?.state;
  if (state === 'no_embedding') {
    return frame(
      `<div class="entry-related-note">${escapeHtml('Related entries appear once this entry is embedded.')}</div>`
    );
  }
  if (state !== 'ok') return '';
  const rows = Array.isArray(related.entries) ? related.entries.slice(0, RELATED_LIMIT) : [];
  if (rows.length === 0) return '';
  return frame(`<ul class="entry-thread-list">${rows.map(renderThreadLink).join('')}</ul>`);
}

/** @type {import('./reader-seam.js').ReaderSection} */
export const relatedSection = {
  id: 'related',
  async render({ entryId }) {
    try {
      return renderRelatedSection(await entriesApi.related(entryId, RELATED_LIMIT));
    } catch (error) {
      console.error('Related entries failed to load:', error);
      return '';
    }
  },
};

/** Register the section after the thread. */
export function registerRelated() {
  registerReaderSection(relatedSection);
}
