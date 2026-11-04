// @ts-check
/**
 * ARIEL Reader Module
 *
 * Renders one entry into a container the caller hands in (loading, the
 * entry, or the error state), and the image lightbox over it. The module
 * holds no application state: the caller learns about the lightbox opening
 * and closing through callbacks and closes it from outside with
 * `closeLightbox`. The lightbox installs no `Esc` listener of its own.
 *
 * Features extend the reader through one seam instead of editing it:
 * `registerReaderSection` (after the body, in registration order),
 * `registerReaderTab` (beside "Entry", rendered when opened),
 * `registerReaderAction` (a header button) and `registerReaderKey`, with the
 * registry in `reader-seam.js`. The caller routes keydowns through
 * `runReaderKey`. Every render is handed a `ReaderContext` and placed only
 * while its entry is still the newest call's, so a stale entry's work is
 * discarded. The thread section (`reader-thread.js`) is the first section.
 */

import { entriesApi, apiUrl } from './api.js';
import {
  formatTimestamp,
  renderLoading,
  renderTags,
  renderErrorState,
  escapeHtml,
  sourceName,
} from './components.js';
import { parseEntryText } from './entries-helpers.js';
import { seam, place, renderActionButtons, renderTabStrip } from './reader-seam.js';

export {
  registerReaderSection,
  registerReaderTab,
  registerReaderAction,
  registerReaderKey,
} from './reader-seam.js';

/**
 * One of this API's attachment routes, relative to the API base: the original
 * (`/attachments/<id>`) or its rendition (`/attachments/<id>/rendition`). The
 * id is one plain path segment, never `.` or `..`.
 */
const ATTACHMENT_ROUTE = /^\/attachments\/(?!\.\.?(?:\/|$))[A-Za-z0-9._~-]+(?:\/rendition)?$/;

/**
 * Turn an attachment's `display_url` into a URL safe to place in an href or
 * src, else null. An absolute http(s) URL is kept unchanged; an attachment
 * route relative to the API base is joined to the page's API base, so it
 * resolves under whatever prefix the page is served from. Every other value
 * (javascript:, data:, vbscript:, protocol-relative //host, any other path)
 * is refused.
 * @param {unknown} u - Candidate URL
 * @returns {string|null} The URL to use, or null when it is not safe
 */
export function safeHref(u) {
  if (typeof u !== 'string') return null;
  if (/^https?:\/\//i.test(u)) return u;
  if (ATTACHMENT_ROUTE.test(u)) return apiUrl(u);
  return null;
}

/**
 * The URL of the stored original behind a rendition `display_url`
 * (`/attachments/<id>/rendition` -> `/attachments/<id>`), made safe as by
 * `safeHref`, else null.
 * @param {unknown} displayUrl - The attachment's `display_url`
 * @returns {string|null} The original's URL, or null when there is none
 */
export function originalHref(displayUrl) {
  const suffix = '/rendition';
  if (typeof displayUrl !== 'string' || !ATTACHMENT_ROUTE.test(displayUrl)) return null;
  if (!displayUrl.endsWith(suffix)) return null;
  return safeHref(displayUrl.slice(0, -suffix.length));
}

/**
 * Render one attachment as a thumbnail card or a file card. A thumbnail is
 * drawn only when the server marks the attachment `viewable: true` and its
 * `display_url` is safe; everything else is a file card, linked to the
 * download when `display_url` is safe and link-less otherwise. A file card
 * whose attachment carries a skip reason shows the reason's text.
 * @param {any} att - Attachment from the entry response
 * @returns {string} HTML string
 */
function renderAttachmentCard(att) {
  const href = safeHref(att.display_url);
  const escapedName = escapeHtml(att.filename || 'attachment');
  if (att.viewable === true && href) {
    const escapedUrl = escapeHtml(href);
    return `
    <div class="card" style="width: 150px; cursor: pointer;"
         data-lightbox-url="${escapeHtml(att.display_url)}" data-lightbox-name="${escapedName}"
         data-attachment-id="${escapeHtml(att.attachment_id || '')}">
      <div class="card-body" style="padding: 12px; text-align: center;">
        <img src="${escapedUrl}" alt="${escapedName}"
             style="width: 126px; height: 100px; object-fit: cover; border-radius: var(--radius-md); margin-bottom: 8px;">
        <div class="truncate text-sm">${escapedName}</div>
        <div class="text-xs text-muted">${escapeHtml(att.mime_type || 'image')}</div>
      </div>
    </div>`;
  }
  const reason =
    typeof att.skip_reason_text === 'string' && att.skip_reason_text
      ? `
      <div class="text-xs text-muted" data-skip-reason style="margin-top: 4px; text-align: left; overflow-wrap: anywhere;">${escapeHtml(
        `${att.copy_status === 'skipped' ? 'Not copied' : 'Not shown'}: ${att.skip_reason_text}`
      )}</div>`
      : '';
  const body = `
    <div class="card-body" style="padding: 12px; text-align: center;">
      <div style="font-size: var(--text-4xl); margin-bottom: 8px;">\u{1F4CE}</div>
      <div class="truncate text-sm">${escapedName}</div>
      <div class="text-xs text-muted">${escapeHtml(att.mime_type || 'file')}</div>${reason}
    </div>`;
  if (!href) {
    return `
  <div class="card" style="width: 150px; color: inherit;">${body}
  </div>`;
  }
  return `
  <a href="${escapeHtml(href)}" target="_blank" rel="noopener"
     class="card" style="width: 150px; text-decoration: none; color: inherit; cursor: pointer;">${body}
  </a>`;
}

/**
 * @typedef {object} ReaderCallbacks
 * @property {(attachmentId: string|null) => void} [onAttachmentOpen] - Called
 *   when the lightbox opens, with the attachment it shows
 * @property {() => void} [onAttachmentClose] - Called when the user dismisses
 *   the lightbox (never by `closeLightbox`, whose caller already knows)
 * @property {(attachmentId: string) => void} [onAttachmentMissing] - Called
 *   when `showInReader`'s `attachmentId` names no viewable, safe picture of
 *   the entry, so nothing opened
 * @property {(entryId: string) => void} [onSelectEntry] - Called when a
 *   thread link (the parent or a follow-up) is clicked, with its entry id
 * @property {(entryId: string) => void} [onFollowUp] - Called when the
 *   entry's Follow up action is clicked, with the shown entry's id
 */

/**
 * @typedef {object} ReaderState
 * @property {number} seq - Bumped by every `showInReader`/`clearReader` call;
 *   only the newest call may render or open an attachment
 * @property {boolean} delegated - Whether the click delegation is bound
 * @property {ReaderCallbacks} callbacks - Callbacks of the newest call
 * @property {HTMLElement|null} lightbox - The lightbox this container opened
 * @property {ReaderContext|null} ctx - The shown entry's seam context
 * @property {number} tabs - Bumped by every tab switch; only the newest renders
 */

/** @type {WeakMap<HTMLElement, ReaderState>} */
const readers = new WeakMap();

/**
 * The per-container state, created on first use.
 * @param {HTMLElement} container
 * @returns {ReaderState}
 */
function readerState(container) {
  let state = readers.get(container);
  if (!state) {
    state = { seq: 0, delegated: false, callbacks: {}, lightbox: null, ctx: null, tabs: 0 };
    readers.set(container, state);
  }
  return state;
}

/**
 * Open the lightbox for an attachment on behalf of a container, recording it
 * so `closeLightbox(container)` can close it. A `display_url` that is not safe
 * opens nothing.
 * @param {HTMLElement} container
 * @param {unknown} displayUrl - The attachment's `display_url`
 * @param {string} filename
 * @param {string|null} attachmentId
 * @returns {boolean} whether the lightbox opened
 */
function openLightboxFor(container, displayUrl, filename, attachmentId) {
  const url = safeHref(displayUrl);
  if (!url) return false;
  const state = readerState(container);
  /** @type {HTMLElement|null} */
  let overlay = null;
  overlay = showImageLightbox(url, filename, originalHref(displayUrl), attachmentId, {
    onAttachmentOpen: (id) => state.callbacks.onAttachmentOpen?.(id),
    onAttachmentClose: () => {
      if (state.lightbox === overlay) state.lightbox = null;
      state.callbacks.onAttachmentClose?.();
    },
  });
  state.lightbox = overlay;
  return true;
}

/**
 * Bind delegated thumbnail clicks to the container, once. The container's
 * innerHTML is replaced on every render, so the listener lives on the stable
 * container rather than on the thumbnails.
 * @param {HTMLElement} container
 */
function bindDelegation(container) {
  const state = readerState(container);
  if (state.delegated) return;
  state.delegated = true;
  container.addEventListener('click', (e) => {
    const target = /** @type {HTMLElement|null} */ (e.target);
    if (handleSeamClick(container, target)) return;
    const thumb = /** @type {HTMLElement|null} */ (
      target?.closest?.('[data-lightbox-url]') ?? null
    );
    if (!thumb || !container.contains(thumb)) return;
    openLightboxFor(
      container,
      thumb.dataset.lightboxUrl,
      thumb.dataset.lightboxName || '',
      thumb.dataset.attachmentId || null
    );
  });
}

/** @typedef {import('./reader-seam.js').ReaderContext} ReaderContext */

/**
 * Run the key registered for `key` on the container's shown entry.
 * @param {HTMLElement} container
 * @param {string} key - `KeyboardEvent.key`
 * @returns {boolean} whether one ran (never while loading or cleared)
 */
export function runReaderKey(container, key) {
  const { ctx } = readerState(container);
  const tab = seam.tabs.find((t) => t.key === key);
  const run = seam.actions.find((a) => a.key === key)?.run ?? seam.keys.get(key);
  if (!ctx?.isCurrent() || !(tab || run)) return false;
  if (tab) openTab(container, tab.id);
  else run?.(ctx);
  return true;
}

/**
 * Show one tab of the shown entry (`''` is the entry itself) and render it;
 * a later tab switch or entry discards that render.
 * @param {HTMLElement} container
 * @param {string} id
 */
function openTab(container, id) {
  const state = readerState(container);
  const { ctx } = state;
  const tab = seam.tabs.find((t) => t.id === id);
  const panel = /** @type {HTMLElement|null} */ (
    container.querySelector('[data-reader-tab-panel]')
  );
  const grid = /** @type {HTMLElement|null} */ (container.querySelector('.entry-detail-grid'));
  if (!ctx?.isCurrent() || !panel || !grid) return;
  const open = ++state.tabs;
  for (const b of container.querySelectorAll('[data-reader-tab]')) {
    b.setAttribute('aria-selected', String(b.getAttribute('data-reader-tab') === (tab?.id ?? '')));
  }
  // `.entry-detail-grid { display: grid }` outranks `[hidden]`, so hide it inline too.
  grid.hidden = Boolean(tab);
  grid.style.display = tab ? 'none' : '';
  panel.hidden = !tab;
  if (!tab) return;
  panel.innerHTML = renderLoading(`Loading ${tab.label}...`);
  place(ctx, tab.render, panel, () => state.tabs === open);
}

/**
 * Route a click on a thread link, the Follow up action, a seam tab or a seam
 * action to the newest call.
 * @param {HTMLElement} container
 * @param {HTMLElement|null} target - The click target
 * @returns {boolean} whether the click was one of these
 */
function handleSeamClick(container, target) {
  const el = /** @type {HTMLElement|null} */ (
    target?.closest?.(
      '[data-thread-entry], [data-follow-up], [data-reader-tab], [data-reader-action]'
    ) ?? null
  );
  if (!el || !container.contains(el)) return false;
  const { callbacks, ctx } = readerState(container);
  const data = el.dataset;
  if (data.threadEntry !== undefined) callbacks.onSelectEntry?.(data.threadEntry);
  else if (data.followUp !== undefined) callbacks.onFollowUp?.(data.followUp);
  else if (data.readerTab !== undefined) openTab(container, data.readerTab);
  else if (ctx?.isCurrent()) seam.actions.find((a) => a.id === data.readerAction)?.run(ctx);
  return true;
}

/**
 * Load an entry into the container: the loading state first, then the entry,
 * or the error state when the id is unknown or the request fails. When
 * `attachmentId` names one of the entry's viewable pictures, that picture
 * opens in the lightbox; otherwise `onAttachmentMissing` is called. Only the newest call per container renders or opens
 * an attachment; an older call that resolves later does nothing.
 * @param {HTMLElement} container - Element the entry is rendered into
 * @param {string} entryId - Entry ID
 * @param {ReaderCallbacks & {attachmentId?: string|null}} [options]
 * @returns {Promise<any>} The entry shown, or null when it was not (failed or
 *   superseded)
 */
export async function showInReader(container, entryId, options = {}) {
  const {
    attachmentId = null,
    onAttachmentOpen,
    onAttachmentClose,
    onAttachmentMissing,
    onSelectEntry,
    onFollowUp,
  } = options;
  const state = readerState(container);
  const seq = ++state.seq;
  state.callbacks = { onAttachmentOpen, onAttachmentClose, onSelectEntry, onFollowUp };
  bindDelegation(container);
  container.innerHTML = renderLoading('Loading entry...');

  /** @type {any} */
  let entry;
  try {
    entry = await entriesApi.get(entryId);
    if (!entry) throw new Error(`Entry ${entryId} not found`);
  } catch (error) {
    if (seq !== state.seq) return null;
    console.error('Failed to load entry:', error);
    container.innerHTML = renderErrorState('Failed to Load Entry', error);
    return null;
  }
  if (seq !== state.seq) return null;
  const id = String(entry.entry_id || entryId);
  const isCurrent = () => seq === state.seq;
  const ctx = { container, entry, entryId: id, callbacks: state.callbacks, isCurrent };
  state.ctx = ctx;
  renderEntryDetail(container, entry);

  if (attachmentId) {
    const att = (entry.attachments || []).find(
      (/** @type {any} */ a) => a && a.attachment_id === attachmentId
    );
    const opened =
      att?.viewable === true &&
      openLightboxFor(
        container,
        att.display_url,
        att.filename || 'attachment',
        att.attachment_id || null
      );
    if (!opened) onAttachmentMissing?.(attachmentId);
  }
  const slot = (/** @type {string} */ id) => container.querySelector(`[data-reader-slot="${id}"]`);
  await Promise.all(seam.sections.map((s) => place(ctx, s.render, slot(s.id))));
  return entry;
}

/**
 * Empty the container and cancel any `showInReader` call still in flight.
 * @param {HTMLElement} container
 */
export function clearReader(container) {
  const state = readerState(container);
  state.seq += 1;
  container.innerHTML = '';
}

/**
 * Close the lightbox the container opened, if it is still open. This is the
 * outside direction (the caller's state already says "no attachment"), so it
 * does not call `onAttachmentClose`.
 * @param {HTMLElement} container
 */
export function closeLightbox(container) {
  const state = readerState(container);
  const overlay = state.lightbox;
  state.lightbox = null;
  overlay?.remove();
}

/**
 * One label/value row of a metadata card.
 * @param {string} label
 * @param {unknown} value
 * @param {boolean} [mono]
 * @returns {string} HTML string
 */
function renderMetaItem(label, value, mono = false) {
  return `
    <div class="metadata-item">
      <span class="metadata-label">${escapeHtml(label)}</span>
      <span class="metadata-value${mono ? ' font-mono' : ''}">${escapeHtml(value)}</span>
    </div>
  `;
}

/**
 * A metadata card in the reader's sidebar.
 * @param {string} title
 * @param {string} bodyHtml - Already-escaped card body
 * @returns {string} HTML string
 */
function renderMetaCard(title, bodyHtml) {
  return `
    <div class="metadata-card">
      <h4>${title}</h4>
      ${bodyHtml}
    </div>
  `;
}

/**
 * A titled section below the entry's content.
 * @param {string} title - Already-escaped title
 * @param {string} bodyHtml - Already-escaped section body
 * @returns {string} HTML string
 */
function renderDetailSection(title, bodyHtml) {
  return `
    <div class="entry-detail-content" style="margin-top: 24px;">
      <h3>${title}</h3>
      ${bodyHtml}
    </div>
  `;
}

/**
 * The Session Context card, or nothing when the session carries no known field.
 * @param {any} session - The entry's `metadata.session_metadata`
 * @returns {string} HTML string
 */
function renderSessionCard(session) {
  if (!session) return '';
  /** @type {Array<[string, unknown, boolean?]>} */
  const fields = [
    ['Operator', session.operator],
    ['Session', session.session_id, true],
    ['Branch', session.git_branch, true],
    ['Model', session.model_name || session.model],
    ['Source', session.created_via && sourceName(String(session.created_via))],
    ['Started', session.session_start_time],
  ];
  const items = fields
    .filter(([, value]) => value)
    .map(([label, value, mono]) => renderMetaItem(label, String(value), mono));
  if (items.length === 0) return '';
  return renderMetaCard('Session Context', `<div class="metadata-list">${items.join('')}</div>`);
}

/**
 * Render entry detail view.
 * @param {HTMLElement} container - Container element
 * @param {any} entry - Entry data
 */
export function renderEntryDetail(container, entry) {
  bindDelegation(container);
  const metadata = entry.metadata || {};
  const keywords = entry.keywords || [];
  const attachments = entry.attachments || [];
  const time = formatTimestamp(entry.timestamp);

  // The title is the row's subject: the stored/derived one, else raw_text's first line
  const parsed = parseEntryText(entry.raw_text);
  const subject = entry.subject || parsed.subject;
  const { details } = parsed;

  container.innerHTML = `
    <div class="entry-detail">
      <div class="entry-detail-header">
        <div class="entry-detail-title-row">
          <h2 class="entry-detail-title">${escapeHtml(subject)}</h2>
          <button type="button" class="btn btn-secondary btn-sm entry-follow-up"
                  data-follow-up="${escapeHtml(entry.entry_id)}">\u21a9 Follow up</button>
          ${renderActionButtons()}
        </div>
        <div class="entry-detail-meta">
          <span class="entry-id font-mono text-accent-secondary">${escapeHtml(entry.entry_id)}</span>
          <span class="timestamp font-mono">${escapeHtml(time)}</span>
          <span>${escapeHtml(entry.author || 'Unknown')}</span>
          <span class="text-muted">${escapeHtml(entry.source_system)}</span>
        </div>
      </div>
      ${renderTabStrip()}

      <div class="entry-detail-grid">
        <div class="entry-detail-main">
          <div class="entry-detail-content">
            <h3>Content</h3>
            <div class="entry-detail-text">${escapeHtml(details)}</div>
          </div>

          ${
            attachments.length > 0
              ? renderDetailSection(
                  `Attachments (${attachments.length})`,
                  `<div style="display: flex; flex-wrap: wrap; gap: 16px;">
                    ${attachments.map(renderAttachmentCard).join('')}
                  </div>`
                )
              : ''
          }

          ${
            entry.summary
              ? renderDetailSection(
                  'AI Summary',
                  `<div class="text-secondary">${escapeHtml(entry.summary)}</div>`
                )
              : ''
          }
          ${seam.sections.map((s) => `<div data-reader-slot="${s.id}" style="display: contents;"></div>`).join('')}
        </div>

        <div class="entry-detail-sidebar">
          ${renderMetaCard(
            'Metadata',
            `<div class="metadata-list">
              ${renderMetaItem('ID', entry.entry_id)}
              ${renderMetaItem('Source', entry.source_system)}
              ${renderMetaItem('Author', entry.author || 'Unknown')}
              ${renderMetaItem('Timestamp', time)}
              ${metadata.logbook ? renderMetaItem('Logbook', metadata.logbook) : ''}
              ${metadata.shift ? renderMetaItem('Shift', metadata.shift) : ''}
            </div>`
          )}

          ${
            keywords.length > 0
              ? renderMetaCard(
                  'Keywords',
                  `<div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    ${renderTags(keywords, 'accent')}
                  </div>`
                )
              : ''
          }

          ${renderSessionCard(metadata.session_metadata)}
        </div>
      </div>
      <div class="entry-detail-tab-panel" data-reader-tab-panel hidden></div>
    </div>
  `;

  // Broken-image fallback for attachment thumbnails, bound here rather than
  // as an inline onerror: the `error` event doesn't bubble, so it can't be
  // handled by the container delegation above, but these <img> elements are
  // freshly created by the innerHTML assignment on every render, so binding
  // directly (once each) can never double-bind.
  container.querySelectorAll('[data-lightbox-url] img').forEach((img) => {
    img.addEventListener(
      'error',
      () => {
        img.outerHTML =
          '<div style="font-size: var(--text-4xl); margin-bottom: 8px;">\u{1F4CE}</div>';
      },
      { once: true }
    );
  });
}

/**
 * Show a lightbox overlay for an image attachment.
 * @param {string} url - Image URL
 * @param {string} filename - Display filename
 * @param {string|null} [originalUrl] - The full-resolution original, linked
 *   beside the picture when given
 * @param {string|null} [attachmentId] - Stored attachment identifier for the
 *   local Workspace handoff; an external attachment URL is never used for it
 * @param {ReaderCallbacks} [callbacks] - `onAttachmentOpen(attachmentId)` once
 *   the lightbox is shown; `onAttachmentClose()` when the user dismisses it
 * @returns {HTMLElement} The lightbox overlay
 */
export function showImageLightbox(
  url,
  filename,
  originalUrl = null,
  attachmentId = null,
  callbacks = {}
) {
  // Remove existing lightbox if any
  const existing = document.getElementById('image-lightbox');
  if (existing) existing.remove();

  const overlay = document.createElement('div');
  overlay.id = 'image-lightbox';
  // A lightbox scrim is conventionally dark regardless of the active site
  // theme (it exists to make the image itself readable), so its backdrop
  // and text use theme-invariant black/white composites rather than
  // themed tokens that would go light-on-light in light mode.
  overlay.style.cssText =
    'position: fixed; inset: 0; background: color-mix(in srgb, black 85%, transparent); display: flex; ' +
    'flex-direction: column; align-items: center; justify-content: center; z-index: 10000; ' +
    'cursor: pointer;';

  overlay.innerHTML = `
    <img src="${escapeHtml(url)}" alt="${escapeHtml(filename)}"
         style="max-width: 90vw; max-height: 80vh; object-fit: contain; border-radius: var(--radius-xl); cursor: default;">
    <div style="margin-top: 16px; display: flex; align-items: center; gap: 16px;">
      <span style="color: silver; font-size: var(--text-xl);">${escapeHtml(filename)}</span>
      <a href="${escapeHtml(url)}" target="_blank" rel="noopener"
         style="color: var(--color-accent-secondary); font-size: var(--text-xl); text-decoration: none;">Open in new tab &#x2197;</a>
      ${
        originalUrl
          ? `
      <a href="${escapeHtml(originalUrl)}" target="_blank" rel="noopener" data-original
         title="The picture as stored, at full resolution"
         style="color: var(--color-accent-secondary); font-size: var(--text-xl); text-decoration: none;">Open original &#x2197;</a>
      `
          : ''
      }
      ${
        attachmentId
          ? `
      <button type="button" data-use-in-workspace data-attachment-id="${escapeHtml(attachmentId)}"
         style="color: var(--color-accent-secondary); background: transparent; border: 0; font: inherit; font-size: var(--text-xl); cursor: pointer;">Use in Workspace</button>
      <span data-workspace-status style="color: silver; font-size: var(--text-sm);" aria-live="polite"></span>
      `
          : ''
      }
    </div>
  `;

  // Clicking the image or one of its links must not bubble to the
  // overlay's own click handler (which dismisses the lightbox), and a broken
  // image needs its fallback markup — all bound directly here (no
  // delegation) rather than as inline attributes, since these elements are
  // created fresh on every call, so this can never double-bind. (The `error`
  // event doesn't bubble, so delegation wouldn't reach it anyway.)
  const lightboxImg = overlay.querySelector('img');
  if (lightboxImg) {
    lightboxImg.addEventListener('click', (e) => e.stopPropagation());
    lightboxImg.addEventListener(
      'error',
      () => {
        lightboxImg.outerHTML =
          '<div style="color:white;font-size:var(--text-3xl);">Failed to load image</div>';
      },
      { once: true }
    );
  }
  overlay
    .querySelectorAll('a')
    .forEach((a) => a.addEventListener('click', (e) => e.stopPropagation()));

  const workspaceButton = /** @type {HTMLButtonElement|null} */ (
    overlay.querySelector('[data-use-in-workspace]')
  );
  const workspaceStatus = /** @type {HTMLElement|null} */ (
    overlay.querySelector('[data-workspace-status]')
  );
  workspaceButton?.addEventListener('click', async (e) => {
    e.stopPropagation();
    const id = workspaceButton.dataset.attachmentId;
    if (!id) return;
    workspaceButton.disabled = true;
    if (workspaceStatus) workspaceStatus.textContent = 'Copying…';
    try {
      const response = await fetch(apiUrl(`/attachments/${encodeURIComponent(id)}/artifact`), {
        method: 'POST',
      });
      if (!response.ok) throw new Error(`Workspace copy failed (${response.status})`);
      if (workspaceStatus) workspaceStatus.textContent = 'Copied to Workspace';
      // This is deliberately a local browser action. The host verifies that
      // this message came from its registered ARIEL iframe; it activates no
      // agent panel and sends no cross-client focus event.
      if (window.parent !== window) {
        window.parent.postMessage(
          { type: 'osprey:activate', panel: 'artifacts' },
          window.location.origin
        );
      }
    } catch (error) {
      console.error('Failed to copy picture to Workspace:', error);
      workspaceButton.disabled = false;
      if (workspaceStatus) workspaceStatus.textContent = 'Could not copy picture';
    }
  });

  // Dismissing by a click on the scrim is the only close inside the module;
  // `Esc` belongs to keyboard.js, and `closeLightbox` closes from outside.
  overlay.addEventListener('click', () => {
    if (!overlay.isConnected) return;
    overlay.remove();
    callbacks.onAttachmentClose?.();
  });

  document.body.appendChild(overlay);
  callbacks.onAttachmentOpen?.(attachmentId);
  return overlay;
}
