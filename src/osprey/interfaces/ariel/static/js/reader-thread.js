// @ts-check
/**
 * ARIEL reader thread section (C-THREAD).
 *
 * The reader's first after-body section: "Follows up" (the parent link, or
 * the bare id when the parent is not held) and "Follow-ups (n)". The reader
 * seeds it into its section slot, so it renders wherever `showInReader` does.
 * A failed or empty thread fetch renders nothing and leaves the entry as it
 * is. Thread links carry `data-thread-entry`, which the reader routes to its
 * `onSelectEntry` callback.
 */

import { entriesApi } from './api.js';
import { formatTimestamp, escapeHtml } from './components.js';

/**
 * One row as a link that selects the entry (`data-thread-entry`).
 * @param {any} row - A C-THREAD `RowResponse`
 * @returns {string} HTML string
 */
export function renderThreadLink(row) {
  const id = String(row.entry_id ?? '');
  const subject = row.subject || id;
  return `
    <li>
      <button type="button" class="entry-thread-link" data-thread-entry="${escapeHtml(id)}">
        <span class="entry-thread-subject">${escapeHtml(subject)}</span>
        <span class="entry-thread-meta font-mono">${escapeHtml(formatTimestamp(row.timestamp))}</span>
        <span class="entry-thread-meta">${escapeHtml(row.author || '')}</span>
      </button>
    </li>`;
}

/**
 * The C-THREAD section. Empty when the entry has neither a parent nor
 * follow-ups.
 * @param {any} thread - `{parent_id, parent, children, truncated?}`
 * @returns {string} HTML string
 */
export function renderThreadSection(thread) {
  const parentId = thread?.parent_id;
  const children = Array.isArray(thread?.children) ? thread.children : [];
  const parts = [];
  if (parentId) {
    const parent = thread.parent
      ? `<ul class="entry-thread-list">${renderThreadLink(thread.parent)}</ul>`
      : `<div class="entry-thread-missing font-mono">${escapeHtml(parentId)}</div>`;
    parts.push(`<h4>Follows up</h4>${parent}`);
  }
  if (children.length > 0) {
    // A17: the server returns only the newest follow-ups and flags the rest.
    const older = thread.truncated
      ? '<div class="entry-thread-missing" data-thread-truncated>Older follow-ups not shown</div>'
      : '';
    parts.push(
      `<h4>Follow-ups (${children.length})</h4>${older}
      <ul class="entry-thread-list">${children.map(renderThreadLink).join('')}</ul>`
    );
  }
  if (parts.length === 0) return '';
  return `<div class="entry-detail-content entry-thread" data-entry-thread>${parts.join('')}</div>`;
}

/** @type {import('./reader-seam.js').ReaderSection} */
export const threadSection = {
  id: 'thread',
  async render({ entryId }) {
    /** @type {any} */
    let thread;
    try {
      thread = await entriesApi.thread(entryId);
    } catch {
      return '';
    }
    if (!thread || typeof thread !== 'object') return '';
    return renderThreadSection(thread);
  },
};
