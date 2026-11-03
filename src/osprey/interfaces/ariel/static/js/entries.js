// @ts-check
/**
 * ARIEL Entries Module — the compose host.
 *
 * Hosts the create form in `#ariel-compose` and owns its init. Store keys:
 *   - reads `compose`: set shows the host (empty, or with `compose.draft`
 *     loaded via `loadDraft`), null hides it;
 *   - writes `narrowView:'list'` when compose opens, and `compose:null` on
 *     Cancel. A successful create writes its own keys (entries-form.js).
 * The entry list lives in list.js and the reader in reader.js.
 */

import { entriesApi } from './api.js';
import {
  handleCreateEntry,
  handleTagInput,
  handleFilePreview,
  loadDraft,
  initEntryTags,
  resetCreateForm,
} from './entries-form.js';
import { renderEntryFields } from './entry-fields.js';
import { getState, setState, subscribe } from './store.js';

/** @typedef {import('./store.js').State} State */

/**
 * Show or hide the compose host for `compose`, filling the form on open.
 * @param {HTMLElement|null} host
 * @param {State['compose']} compose
 */
function applyCompose(host, compose) {
  if (host) host.hidden = !compose;
  if (!compose) return;
  setState({ narrowView: 'list' });
  resetCreateForm();
  if (compose.draft) loadDraft(compose.draft);
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
  root
    .querySelector('#create-cancel-btn')
    ?.addEventListener('click', () => setState({ compose: null }), { signal });

  initEntryTags();

  // Adapt the publishing section and the declared entry fields to the
  // configured logbook adapter.
  adaptCreateForm();

  const host = /** @type {HTMLElement|null} */ (root.querySelector('#ariel-compose'));
  const unsubscribe = subscribe((next, _prev, changed) => {
    if (changed.has('compose')) applyCompose(host, next.compose);
  });
  applyCompose(host, getState().compose);

  return {
    destroy() {
      unsubscribe();
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
    await renderEntryFields([]);
    return;
  }

  adaptPublishingSection(info);
  await renderEntryFields(Array.isArray(info.entry_fields) ? info.entry_fields : []);
}

/**
 * Adapt the "Logbook Publishing" section to the adapter's publish-info.
 * @param {{supports_write: boolean, requires_auth: boolean, source_system: ?string}} info
 */
function adaptPublishingSection(info) {
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
