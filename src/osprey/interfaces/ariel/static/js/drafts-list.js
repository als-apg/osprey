// @ts-check
/**
 * ARIEL Drafts Mailbox
 *
 * The Drafts mailbox's rows: the staged agent drafts, newest first, each with
 * its subject, author, who staged it and its age, or an empty state when
 * there are none. list.js hands its body over with `renderDrafts` when the
 * mailbox is Drafts; this module fetches `/api/drafts` and paints.
 *
 * It fetches on entering Drafts and again on every `draftsRev` change (a draft
 * was published or discarded). A response paints only while it is the latest
 * drafts fetch, the mailbox is still Drafts, and the list has started no
 * refresh since (list.js's `isCurrent`).
 *
 * Rows carry `data-draft-id`, never `data-entry-id`, so the list's row click
 * and `j`/`k` never select a draft. Selecting a row writes
 * `compose:{draft:id}`, which opens the draft in the compose sheet.
 */

import { draftsApi } from './api.js';
import {
  escapeHtml,
  formatRelativeTime,
  renderEmptyState,
  renderErrorState,
  renderLoading,
  stagedByText,
} from './components.js';
import { getState, setState, subscribe } from './store.js';

/** @typedef {import('./store.js').State} State */

/**
 * @typedef {Object} DraftSummary
 * @property {string} draft_id
 * @property {string} subject
 * @property {?string} author
 * @property {?string} logbook
 * @property {string} created_at
 * @property {?string} staged_by
 * @property {number} attachment_count
 */

/** The Drafts mailbox's empty state. */
export const DRAFTS_NONE = {
  title: 'No Drafts',
  text: 'Drafts an agent stages for you appear here.',
};

/** @type {HTMLElement|null} Where the rows are painted (the list body). */
let host = null;

/** @type {() => boolean} Whether the list run that handed over the body is still current. */
let listCurrent = () => false;

// Monotonic id of the newest drafts fetch; an older response paints nothing.
let fetchSeq = 0;

/** @type {(() => void)|null} The draftsRev subscription, installed once. */
let unsubscribe = null;

/** @type {WeakSet<HTMLElement>} Hosts that already carry the row-click listener. */
const bound = new WeakSet();

/**
 * Render one draft row. Every value is escaped; the id rides in `data-draft-id`.
 * @param {DraftSummary} draft
 * @param {string|null} [selected] - The draft open in the compose sheet
 * @returns {string}
 */
export function renderDraftRow(draft, selected = null) {
  const id = String(draft.draft_id ?? '');
  const created = draft.created_at || '';
  const staged = draft.staged_by ? stagedByText(draft.staged_by) : '';
  const count = Number(draft.attachment_count) || 0;
  return `
    <div class="osprey-list-row multiline ariel-row ariel-draft-row" role="option" tabindex="-1"
         data-draft-id="${escapeHtml(id)}" aria-selected="${selected !== null && selected === id}">
      <div class="ariel-row-top">
        <span class="osprey-list-title ariel-row-subject">${escapeHtml(draft.subject || '(no subject)')}</span>
        <time class="ariel-row-time" datetime="${escapeHtml(created)}">${escapeHtml(formatRelativeTime(created))}</time>
      </div>
      <div class="ariel-row-meta">
        <span class="ariel-row-author">${escapeHtml(draft.author || 'Unknown')}</span>
        <span class="text-muted ariel-row-staged">${escapeHtml(staged)}</span>
        ${count > 0 ? `<span class="text-muted" data-attachment-count="${count}">📎 ${count}</span>` : ''}
      </div>
    </div>
  `;
}

/**
 * The drafts sorted newest first (the server already sends them so; this keeps
 * the order whatever it sends).
 * @param {DraftSummary[]} drafts
 * @returns {DraftSummary[]}
 */
export function newestFirst(drafts) {
  const time = (/** @type {DraftSummary} */ d) => Date.parse(d.created_at) || 0;
  return [...drafts].sort((a, b) => time(b) - time(a));
}

/** Whether a drafts response may still paint. */
function mayPaint(/** @type {number} */ seq) {
  return seq === fetchSeq && host !== null && getState().mailbox === 'drafts' && listCurrent();
}

/** Fetch the drafts and paint them, under the guard. */
async function load() {
  const seq = ++fetchSeq;
  const target = host;
  if (!target) return;
  target.setAttribute('aria-busy', 'true');
  try {
    const body = await draftsApi.list();
    if (!mayPaint(seq)) return;
    const drafts = newestFirst(Array.isArray(body?.drafts) ? body.drafts : []);
    const open = getState().compose?.draft ?? null;
    target.innerHTML = drafts.length
      ? drafts.map((d) => renderDraftRow(d, open)).join('')
      : renderEmptyState(DRAFTS_NONE.title, DRAFTS_NONE.text);
  } catch (error) {
    if (!mayPaint(seq)) return;
    console.error('Drafts failed:', error);
    target.innerHTML = renderErrorState('Drafts Failed', error);
  }
  target.setAttribute('aria-busy', 'false');
}

/**
 * A row click opens that draft in the compose sheet.
 * @param {MouseEvent} e
 */
function onClick(e) {
  const target = /** @type {HTMLElement|null} */ (e.target);
  const row = /** @type {HTMLElement|null} */ (target?.closest?.('[data-draft-id]') ?? null);
  if (!row || getState().mailbox !== 'drafts') return;
  const id = row.dataset.draftId;
  if (id) setState({ compose: { draft: id } });
}

/**
 * The store subscriber: a `draftsRev` change refetches while Drafts is shown.
 * @param {State} state
 * @param {State} _prev
 * @param {Set<string>} changed
 */
function onStoreChange(state, _prev, changed) {
  if (changed.has('draftsRev') && state.mailbox === 'drafts' && host && listCurrent()) load();
}

/**
 * Paint the Drafts mailbox into `body` (list.js's drafts branch).
 * @param {HTMLElement} body - The list body
 * @param {() => boolean} isCurrent - False once the list has started another run
 */
export function renderDrafts(body, isCurrent) {
  host = body;
  listCurrent = isCurrent;
  if (!bound.has(body)) {
    body.addEventListener('click', onClick);
    bound.add(body);
  }
  unsubscribe ??= subscribe(onStoreChange);
  body.innerHTML = renderLoading('Loading drafts...');
  load();
}
