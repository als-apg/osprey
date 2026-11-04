// @ts-check
/**
 * ARIEL Entries Module — the compose host.
 *
 * Hosts the create form in the `#ariel-compose` sheet and owns its init and
 * controls. Store keys:
 *   - reads `compose`: set shows the sheet, titled by kind, and opens it
 *     through the compose lifecycle (compose-lifecycle.js: reset, fields,
 *     kind fill, autosave restore); null hides it and closes the lifecycle,
 *     which keeps the context's snapshot;
 *   - writes `compose:null` on Close and Discard. Discard first finishes the
 *     sheet (its snapshot is cleared) and, on a draft, deletes the draft
 *     (Decision 10); Close keeps the snapshot. It never writes
 *     `narrowView` (C-COMPOSE): the sheet overlays the tile when stacked, so
 *     closing shows whichever pane was there. Maximise is per-open UI state
 *     and is not stored. A successful create writes its own keys
 *     (entries-form.js). Close, Discard and Esc return focus to the element
 *     that opened the sheet, or to the selected row when that is gone (the
 *     lifecycle's `scope` and `fallbackFocus`).
 * The sheet keeps itself over the list column (placeSheet), and its footer
 * names where the entry goes (adaptPublishingSection).
 * The entry list lives in list.js and the reader in reader.js.
 */

import { entriesApi } from './api.js';
import { createComposeLifecycle } from './compose-lifecycle.js';
import {
  applySnapshot,
  deleteDraft,
  handleCreateEntry,
  handleTagInput,
  handleFilePreview,
  loadDraft,
  loadFollowUp,
  initEntryTags,
  onComposeFinish,
  readSnapshot,
  resetCreateForm,
} from './entries-form.js';
import { entryFieldsRendered, renderEntryFields } from './entry-fields.js';
import { formatFacilityTime } from '/design-system/js/facility-time.js';
import { getState, setState, subscribe } from './store.js';

/** @typedef {import('./store.js').State} State */

/**
 * The sheet's title for a compose kind.
 * @param {NonNullable<State['compose']>} compose
 * @returns {string}
 */
function composeTitle(compose) {
  if (compose.draft) return 'Draft';
  return compose.followsUp ? 'Follow up' : 'New entry';
}

/**
 * Maximise (span the panel) or restore the sheet.
 * @param {HTMLElement|null} host
 * @param {boolean} max
 */
function setMax(host, max) {
  if (!host) return;
  host.classList.toggle('max', max);
  const btn = host.querySelector('#ariel-compose-max');
  if (!btn) return;
  btn.setAttribute('aria-pressed', String(max));
  const label = max ? 'Restore' : 'Expand';
  btn.setAttribute('aria-label', label);
  btn.setAttribute('title', label);
}

/**
 * Write the footer's autosave stamp (C-SHEET-LIFECYCLE Saving): the time of
 * a landed save, "not saved locally" after a failed one, empty on open.
 * @param {boolean|null} ok
 * @param {number} [savedAt]
 */
function setSavedStamp(ok, savedAt) {
  const stamp = document.getElementById('ariel-compose-saved');
  if (!stamp) return;
  if (ok === null) stamp.textContent = '';
  else if (!ok) stamp.textContent = 'not saved locally';
  else {
    const time = formatFacilityTime(savedAt, { hour: '2-digit', minute: '2-digit', hour12: false });
    stamp.textContent = `saved locally ${time}`;
  }
}

/**
 * The sheet's lifecycle over the create form. The follow filler prefills from
 * the parent entry; the draft filler loads the draft. Focus leaving the sheet
 * falls back to the list's selected row.
 * @param {ParentNode} root
 * @param {AbortSignal} signal
 */
function createLifecycle(root, signal) {
  const form = /** @type {HTMLFormElement|null} */ (root.querySelector('#create-entry-form'));
  return createComposeLifecycle(
    {
      reset() {
        resetCreateForm();
        setSavedStamp(null);
      },
      renderFields: () => entryFieldsRendered(),
      fillers: { follow: loadFollowUp, draft: loadDraft },
      readSnapshot,
      applySnapshot,
      onSaved: (ok, savedAt) => setSavedStamp(ok, savedAt),
    },
    {
      form,
      gated: [
        form?.querySelector('button[type="submit"]') ?? null,
        /** @type {HTMLButtonElement|null} */ (root.querySelector('#create-cancel-btn')),
      ],
      scope: /** @type {HTMLElement|null} */ (root.querySelector('#ariel-compose')),
      fallbackFocus: () =>
        /** @type {HTMLElement|null} */ (
          root.querySelector('[data-entry-id][aria-selected="true"]')
        ),
      signal,
    }
  );
}

/**
 * Show or hide the compose sheet for `compose`, opening or closing the
 * lifecycle. Every open and close starts the sheet restored.
 * @param {HTMLElement|null} host
 * @param {State['compose']} compose
 * @param {ReturnType<typeof createComposeLifecycle>} lifecycle
 */
function applyCompose(host, compose, lifecycle) {
  setMax(host, false);
  if (host) host.hidden = !compose;
  if (!compose) {
    lifecycle.close();
    return;
  }
  const title = composeTitle(compose);
  const titleEl = host?.querySelector('#ariel-compose-title');
  if (titleEl) titleEl.textContent = title;
  host?.setAttribute('aria-label', title);
  lifecycle.open(compose);
}

/**
 * Keep the sheet's default position over the list column. The sheet is not
 * inside `.ariel-list-col` (Decision 10), so it reads the column's box into
 * two custom properties; the stacked and maximised styles ignore them.
 * @param {HTMLElement|null} host
 * @param {ParentNode} root
 * @param {AbortSignal} signal
 */
function placeSheet(host, root, signal) {
  const col = /** @type {HTMLElement|null} */ (root.querySelector('#ariel-list-col'));
  if (!host || !col) return;
  const place = () => {
    if (!col.offsetWidth) return; // hidden while stacked: keep the last box
    host.style.setProperty('--ariel-compose-left', `${col.offsetLeft}px`);
    host.style.setProperty('--ariel-compose-width', `${col.offsetWidth}px`);
  };
  place();
  if (typeof ResizeObserver === 'undefined') return;
  const observer = new ResizeObserver(place);
  // The rail moves the column without resizing it, so watch both.
  for (const el of [col, root.querySelector('#ariel-rail')]) if (el) observer.observe(el);
  signal.addEventListener('abort', () => observer.disconnect());
}

/**
 * Initialize the create form and the compose host.
 * @param {ParentNode} [root=document]
 * @returns {{destroy(): void}}
 */
export function initEntries(root = document) {
  const controller = new AbortController();
  const { signal } = controller;

  root.querySelector('#create-entry-form')?.addEventListener('submit', handleCreateEntry, {
    signal,
  });
  root
    .querySelector('#entry-tags-input')
    ?.addEventListener('keydown', /** @type {EventListener} */ (handleTagInput), { signal });
  root.querySelector('#entry-files')?.addEventListener('change', handleFilePreview, { signal });
  const lifecycle = createLifecycle(root, signal);
  onComposeFinish(() => lifecycle.finish());
  const close = () => setState({ compose: null });
  // Discard: finish (drop the snapshot), close, and delete a draft outright.
  const discard = () => {
    const draft = getState().compose?.draft;
    lifecycle.finish();
    close();
    if (draft) deleteDraft(draft, 'Draft discarded');
  };
  root.querySelector('#create-cancel-btn')?.addEventListener('click', discard, { signal });
  root.querySelector('#ariel-compose-close')?.addEventListener('click', close, { signal });
  const host = /** @type {HTMLElement|null} */ (root.querySelector('#ariel-compose'));
  root
    .querySelector('#ariel-compose-max')
    ?.addEventListener('click', () => setMax(host, !host?.classList.contains('max')), {
      signal,
    });
  placeSheet(host, root, signal);

  initEntryTags();

  // Adapt the publishing section and the declared entry fields to the
  // configured logbook adapter.
  adaptCreateForm();

  const unsubscribe = subscribe((next, _prev, changed) => {
    if (changed.has('compose')) applyCompose(host, next.compose, lifecycle);
  });
  applyCompose(host, getState().compose, lifecycle);

  return {
    destroy() {
      unsubscribe();
      onComposeFinish(() => {});
      controller.abort();
    },
  };
}

/**
 * Adapt the create form to the configured adapter, from one publish-info read.
 *
 * The adapter declares whether publishing needs credentials, so the form shows
 * the credential fields only when they can actually be used and tells the
 * operator what leaving the form will do — instead of fixed, possibly-wrong text.
 * It also declares the entry fields the form renders; when it declares none, or
 * publish-info cannot be read, the form keeps its built-in inputs unchanged.
 * @returns {Promise<void>}
 */
export async function adaptCreateForm() {
  let info;
  try {
    info = await entriesApi.getPublishInfo();
  } catch {
    // Service/DB unavailable — keep the neutral default text and built-in inputs.
    setDestination('Destination unknown');
    await renderEntryFields([]);
    return;
  }

  adaptPublishingSection(info);
  await renderEntryFields(Array.isArray(info.entry_fields) ? info.entry_fields : []);
}

/**
 * Write the sheet footer's destination line.
 * @param {string} text
 */
function setDestination(text) {
  const dest = document.getElementById('ariel-compose-dest');
  if (dest) dest.textContent = text;
}

/**
 * Adapt the "Logbook Publishing" section and the footer's destination line
 * (Decision 11) to the adapter's publish-info.
 * @param {{supports_write: boolean, requires_auth: boolean, source_system: ?string}} info
 */
function adaptPublishingSection(info) {
  const book = info.source_system ? `${info.source_system} logbook` : 'facility logbook';
  if (!info.supports_write) setDestination('→ Local ARIEL only (logbook read-only)');
  else setDestination(`→ ${book}${info.requires_auth ? ' (credentials required)' : ''}`);

  const helper = document.getElementById('publish-helper');
  const credentials = document.getElementById('publish-credentials');
  if (!helper) return;

  const where = info.source_system ? ` to ${info.source_system}` : '';
  if (!info.supports_write) {
    helper.textContent = 'Entries are saved to ARIEL only — this logbook is read-only.';
    if (credentials) credentials.style.display = 'none';
  } else if (info.requires_auth) {
    helper.textContent = `Enter your logbook credentials to publish${where}.`;
    if (credentials) credentials.style.display = '';
  } else {
    helper.textContent = `Publishes${where} — no credentials required.`;
    if (credentials) credentials.style.display = 'none';
  }
}

export default { initEntries, adaptCreateForm };
