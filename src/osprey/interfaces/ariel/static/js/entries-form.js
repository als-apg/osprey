// @ts-check
/**
 * ARIEL Entry Creation & Draft Module
 *
 * Entry creation form: submission, tags, file preview, and the draft and
 * follow-up fillers. A follow-up submits its parent as `metadata.follows_up`.
 *
 * A successful create finishes the sheet (its snapshot is cleared), hands
 * the new entry to the reader through the store (writes `compose:null`, `sel`,
 * `narrowView:'reader'`), remembers the author for the Mine mailbox and
 * deletes a draft whose load succeeded (Decision 9). Failures keep the form
 * open and populated and say why in a toast (C-TOAST).
 */

import { entriesApi, draftsApi, ApiError } from './api.js';
import { escapeHtml } from './components.js';
import { getState, setState } from './store.js';
import { rememberAuthor } from './rail.js';
import {
  clearEntryFieldErrors,
  collectEntryFieldValues,
  entryFieldsRendered,
  markEntryFieldError,
} from './entry-fields.js';
import { formatFileSize } from './entries-helpers.js';
import { messageOf } from './utils.js';
import { loadCompose } from './compose-autosave.js';
import { showToast } from '/design-system/js/toast.js';

// Draft metadata (populated when loading a draft)
/** @type {any} */
let draftMetadata = null;

/** A loaded draft's staged attachments, fetched into memory at open (Decision 8). */
/** @type {File[]} */
let draftFiles = [];

/** The id of the draft whose load succeeded, or null (nothing to delete after submit). */
/** @type {string|null} */
let loadedDraft = null;

/**
 * The draft whose load succeeded into the open sheet, or null when none did.
 * @returns {string|null}
 */
export function loadedDraftId() {
  return loadedDraft;
}

/** Finishes the open sheet (C-SHEET-LIFECYCLE `finish()`); set by the compose host. */
let finishCompose = () => {};

/**
 * Register the sheet's `finish()`, called on submit success before closing.
 * @param {() => void} finish
 */
export function onComposeFinish(finish) {
  finishCompose = finish;
}

/** Toast options for a warning (C-TOAST). */
const WARN = /** @type {const} */ ({ kind: 'warning' });

/**
 * Delete a draft after a successful submit or on Discard (Decisions 9, 10).
 * A 404 means it is already gone; any other failure warns that it is still
 * listed. `draftsRev` is bumped once the DELETE settles, whatever its outcome.
 * @param {string} draftId
 * @param {string|null} [doneMessage] - success toast once deleted (Discard)
 * @returns {Promise<void>}
 */
export async function deleteDraft(draftId, doneMessage = null) {
  let gone = true;
  try {
    await draftsApi.delete(draftId);
  } catch (error) {
    gone = error instanceof ApiError && error.status === 404;
    if (!gone) console.error('Failed to delete draft:', error);
  }
  if (!gone) showToast('Could not delete the draft; it is still listed in Drafts.', WARN);
  else if (doneMessage) showToast(doneMessage, { kind: 'success' });
  setState({ draftsRev: getState().draftsRev + 1 });
}

/** The parent entry id of an open follow-up, submitted as `metadata.follows_up`. */
/** @type {string|null} */
let followsUp = null;

const FOLLOW_UP_PREFIX = 'Follow-up: ';

/** Error codes that name one declared entry field the operator must correct. */
const ENTRY_FIELD_ERROR_CODES = ['invalid_entry_field', 'entry_field_options_unavailable'];

/**
 * The metadata to submit: the draft's metadata with the declared entry-field
 * values on top, plus `follows_up` for a follow-up, or null when there is none.
 * @returns {Record<string, any>|null}
 */
function submissionMetadata() {
  const declared = collectEntryFieldValues();
  if (!draftMetadata && !followsUp && Object.keys(declared).length === 0) return null;
  const metadata = { ...(draftMetadata || {}), ...declared };
  if (followsUp) metadata.follows_up = followsUp;
  return metadata;
}

/**
 * Show the "Follows up" row naming `label`, or hide and empty it.
 * @param {string|null} label
 */
function setParentRow(label) {
  const row = document.getElementById('ariel-compose-parent');
  const ref = document.getElementById('ariel-compose-parent-ref');
  if (ref) ref.textContent = label ?? '';
  if (row) row.hidden = label === null;
}

/**
 * Empty the create form: its inputs, tags, file previews (including staged
 * draft attachments), everything a loaded draft added, and any entry-field
 * error marks.
 */
export function resetCreateForm() {
  /** @type {HTMLFormElement|null} */ (document.getElementById('create-entry-form'))?.reset();
  setTags([]);
  const preview = document.getElementById('file-preview');
  if (preview) preview.innerHTML = '';
  draftFiles = [];
  loadedDraft = null;
  draftMetadata = null;
  followsUp = null;
  setParentRow(null);
  document.getElementById('session-info-panel')?.remove();
  document.getElementById('draft-banner')?.remove();
  clearEntryFieldErrors();
}

/**
 * Handle entry creation form submission.
 * @param {Event} e - Submit event
 */
export async function handleCreateEntry(e) {
  e.preventDefault();

  const form = /** @type {HTMLFormElement} */ (e.target);
  const submitBtn = /** @type {HTMLButtonElement|null} */ (
    form.querySelector('button[type="submit"]')
  );
  const originalText = submitBtn ? submitBtn.textContent : null;

  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Saving...';
  }

  try {
    const formData = new FormData(form);
    const entryData = {
      subject: formData.get('subject'),
      details: formData.get('details'),
      author: formData.get('author'),
      logbook: formData.get('logbook'),
      shift: formData.get('shift'),
      tags: readTags(),
      metadata: submissionMetadata(),
      auth_user: formData.get('auth_user') || null,
      auth_password: formData.get('auth_password') || null,
    };

    // Picked files plus the draft's staged files, already in memory.
    const files = previewFiles();

    let result;
    if (files.length > 0) {
      result = await entriesApi.createWithAttachments(entryData, files);
    } else {
      result = await entriesApi.create(entryData);
    }

    // Finish the sheet (no snapshot survives), close it and show the new
    // entry in the reader; then delete the draft it came from, if it loaded.
    const draftId = loadedDraft;
    finishCompose();
    resetCreateForm();
    rememberAuthor(String(entryData.author ?? ''));
    setState({ compose: null, sel: result.entry_id, narrowView: 'reader' });
    if (result.attachments_published === false) {
      showToast('Entry saved, but its attachments stayed local: they were not published.', WARN);
    }
    if (draftId) await deleteDraft(draftId);
  } catch (error) {
    console.error('Failed to create entry:', error);
    if (error instanceof ApiError && error.code === 'auth_required') {
      // The logbook needs credentials to publish. Keep the form populated (it is
      // only reset on success) and focus the username field so the operator can
      // type and resubmit.
      showToast('Logbook credentials required to publish. Enter your username and password.', {
        kind: 'error',
      });
      document.getElementById('entry-auth-user')?.focus();
    } else if (
      error instanceof ApiError &&
      ENTRY_FIELD_ERROR_CODES.includes(error.code ?? '') &&
      error.field &&
      markEntryFieldError(error.field, error.message)
    ) {
      // The named entry field was rejected: it is marked and focused, and the
      // form stays populated so the operator can correct it and resubmit.
    } else {
      showToast(`Failed to create entry: ${messageOf(error)}`, { kind: 'error' });
    }
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = originalText;
    }
  }
}

/**
 * Handle tag input keydown.
 * @param {KeyboardEvent} e - Keydown event
 */
export function handleTagInput(e) {
  if (e.key === 'Enter' || e.key === ',') {
    e.preventDefault();
    const input = /** @type {HTMLInputElement} */ (e.target);
    const value = input.value.trim();

    if (value) {
      addTag(value);
      input.value = '';
      tagsChanged();
    }
  }
}

/**
 * Tell the sheet that the operator added or removed a tag: a bubbling
 * `change` from `#entry-tags`, which the compose lifecycle saves on.
 * Programmatic tag fills (a draft, a snapshot) do not signal.
 */
function tagsChanged() {
  document.getElementById('entry-tags')?.dispatchEvent(new Event('change', { bubbles: true }));
}

/**
 * Add a tag to the tags list.
 * @param {string} value - Tag value
 */
function addTag(value) {
  const container = document.getElementById('entry-tags');
  if (!container) return;

  // Check for duplicates by comparing values, not by building a selector:
  // a tag may hold `"`, `\` or `]`, which would break `[data-value="…"]`.
  const tags = /** @type {NodeListOf<HTMLElement>} */ (container.querySelectorAll('.tag'));
  if (Array.from(tags).some((t) => t.dataset.value === value)) return;

  const tag = document.createElement('span');
  tag.className = 'tag tag-accent';
  tag.dataset.value = value;
  tag.innerHTML = `
    ${escapeHtml(value)}
    <button type="button" data-tag-remove style="background: none; border: none; cursor: pointer; color: inherit; margin-left: 4px;">&times;</button>
  `;
  container.appendChild(tag);
}

/**
 * Replace the tag list with `values` (a programmatic fill: no `change`).
 * @param {unknown[]} values
 */
function setTags(values) {
  const container = document.getElementById('entry-tags');
  if (container) container.innerHTML = '';
  for (const value of values) if (typeof value === 'string' && value) addTag(value);
}

/**
 * The tag list's values, in order.
 * @returns {string[]}
 */
function readTags() {
  const tagEls = /** @type {NodeListOf<HTMLElement>} */ (
    document.querySelectorAll('#entry-tags .tag')
  );
  return Array.from(tagEls, (t) => t.dataset.value || '').filter(Boolean);
}

/**
 * Wire up delegated click handling for tag removal.
 *
 * #entry-tags gets new `.tag` chips appended/cleared throughout the form's
 * lifetime (addTag, loadDraft), so the remove button's listener is
 * delegated on the stable tags container (attached once, at init) rather
 * than bound per-chip.
 */
export function initEntryTags() {
  const container = document.getElementById('entry-tags');
  container?.addEventListener('click', (e) => {
    const btn = /** @type {HTMLElement} */ (e.target).closest('[data-tag-remove]');
    const tag = btn?.closest('.tag');
    if (!tag) return;
    tag.remove();
    tagsChanged();
  });
}

/**
 * Build the fixed-size icon shown in place of a thumbnail for non-image
 * attachments.
 * @returns {HTMLDivElement}
 */
function createAttachmentIcon() {
  const icon = document.createElement('div');
  icon.style.cssText = 'font-size: var(--text-4xl); margin: 8px 0;';
  icon.textContent = '\u{1F4CE}';
  return icon;
}

/**
 * Build a preview card: thumbnail, filename and size.
 * @param {string} filename - Display filename
 * @param {Node} mediaEl - Thumbnail element (an <img> or createAttachmentIcon())
 * @param {string} sizeText - Human-readable size
 * @returns {HTMLDivElement}
 */
function createPreviewCard(filename, mediaEl, sizeText) {
  const card = document.createElement('div');
  card.className = 'card';
  card.style.cssText = 'width: 120px; text-align: center; padding: 8px;';
  card.appendChild(mediaEl);

  const name = document.createElement('div');
  name.className = 'truncate text-xs';
  name.textContent = filename;
  card.appendChild(name);

  const size = document.createElement('div');
  size.className = 'text-xs text-muted';
  size.textContent = sizeText;
  card.appendChild(size);

  return card;
}

/**
 * The files the next submit sends: the picked files, then the in-memory draft files.
 * @returns {File[]}
 */
function previewFiles() {
  const input = /** @type {HTMLInputElement|null} */ (document.getElementById('entry-files'));
  return [...Array.from(input?.files || []), ...draftFiles];
}

/** Handle file input change: preview the picked files and the draft's files together. */
export function handleFilePreview() {
  renderFilePreview();
}

/** Render a preview card for every file the next submit sends. */
function renderFilePreview() {
  const container = document.getElementById('file-preview');
  if (!container) return;
  container.innerHTML = '';

  for (const file of previewFiles()) {
    let media;
    if (file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.style.cssText =
        'width: 100px; height: 80px; object-fit: cover; border-radius: var(--radius-md);';
      img.src = URL.createObjectURL(file);
      img.onload = () => URL.revokeObjectURL(img.src);
      media = img;
    } else {
      media = createAttachmentIcon();
    }

    container.appendChild(createPreviewCard(file.name, media, formatFileSize(file.size)));
  }
}

/**
 * Render a read-only panel showing session metadata from a draft.
 * @param {any} meta - session_metadata object
 */
function renderSessionInfoPanel(meta) {
  const existing = document.getElementById('session-info-panel');
  if (existing) existing.remove();

  const fields = [
    ['Operator', meta.operator],
    ['Branch', meta.git_branch],
    ['Session', meta.session_id],
    ['Model', meta.model],
    ['Source', meta.created_via],
  ].filter(([, v]) => v);

  if (fields.length === 0) return;

  const panel = document.createElement('div');
  panel.id = 'session-info-panel';
  panel.className = 'session-info-panel';
  panel.innerHTML = `
    <div class="session-info-header">Session Context</div>
    <div class="session-info-fields">
      ${fields
        .map(
          ([label, value]) =>
            `<span><span class="session-info-label">${escapeHtml(label)}:</span>${escapeHtml(String(value))}</span>`
        )
        .join('')}
    </div>
  `;

  const banner = document.getElementById('draft-banner');
  if (banner) {
    banner.after(panel);
  }
}

/**
 * The create form's control named `name`, or null.
 * @param {string} name
 * @returns {HTMLInputElement|HTMLSelectElement|null}
 */
function formControl(name) {
  return document.querySelector(`#create-entry-form [name="${name}"]`);
}

/**
 * The create form's declared entry-field controls, in document order.
 * @returns {(HTMLInputElement|HTMLSelectElement)[]}
 */
function entryFieldControls() {
  return Array.from(document.querySelectorAll('#create-entry-form [data-entry-field]'));
}

/**
 * Pre-fill the rendered declared inputs from a draft's `fields`, and drop every
 * declared name from the forwarded draft metadata so the inputs alone decide
 * what is submitted for them.
 * @param {any} draft - Draft as returned by the drafts API
 */
function applyDraftFields(draft) {
  const form = document.getElementById('create-entry-form');
  if (!form) return;

  // A declared logbook/shift override replaces the built-in input, possibly
  // after the built-in one was filled, so fill it again.
  for (const name of ['logbook', 'shift']) {
    const input = formControl(name);
    if (input && draft[name]) input.value = draft[name];
  }

  const controls = entryFieldControls();
  const values = draft.fields && typeof draft.fields === 'object' ? draft.fields : {};
  // Document order puts a parent before the inputs whose choices depend on it.
  for (const control of controls) {
    const name = control.dataset.entryField || '';
    if (!Object.prototype.hasOwnProperty.call(values, name)) continue;
    const value = values[name];
    if (control instanceof HTMLInputElement && control.type === 'checkbox') {
      control.checked = value === true;
    } else {
      control.value = value === null || value === undefined ? '' : String(value);
    }
    control.dispatchEvent(new Event('change', { bubbles: true }));
  }

  if (draftMetadata && typeof draftMetadata === 'object') {
    const kept = { ...draftMetadata };
    for (const control of controls) delete kept[control.dataset.entryField || ''];
    draftMetadata = kept;
  }
}

/**
 * The form as a compose snapshot (C-AUTOSAVE): the built-in fields, the tags,
 * and every declared entry field, blank ones included, so a cleared field
 * stays cleared on restore. Credentials and attachments are never read.
 * @returns {import('./compose-autosave.js').ComposeSnapshot}
 */
export function readSnapshot() {
  const value = (/** @type {string} */ name) => formControl(name)?.value ?? '';
  /** @type {Record<string, string|boolean>} */
  const fields = {};
  for (const control of entryFieldControls()) {
    const name = control.dataset.entryField || '';
    if (!name) continue;
    fields[name] =
      control instanceof HTMLInputElement && control.type === 'checkbox'
        ? control.checked
        : control.value;
  }
  return {
    subject: value('subject'),
    details: value('details'),
    author: value('author'),
    fields,
    tags: readTags(),
    logbook: value('logbook'),
    shift: value('shift'),
    savedAt: Date.now(),
  };
}

/**
 * Restore a compose snapshot into the form. Declared fields, logbook and
 * shift go through the draft-fields apply path, so dependent selects reload
 * their choices.
 * @param {import('./compose-autosave.js').ComposeSnapshot} snapshot
 */
export function applySnapshot(snapshot) {
  if (!document.getElementById('create-entry-form')) return;
  for (const name of ['subject', 'details', 'author', 'logbook', 'shift']) {
    const input = formControl(name);
    const stored = /** @type {any} */ (snapshot)[name];
    if (input && typeof stored === 'string') input.value = stored;
  }
  setTags(snapshot.tags || []);
  applyDraftFields(snapshot);
}

/**
 * Fetch a draft's staged attachments into memory (Decision 8). Each file that
 * cannot be fetched gets an error toast naming it, unless the open is stale.
 * @param {string} draftId
 * @param {unknown} paths - The draft's `attachment_paths`
 * @param {() => boolean} isCurrent
 * @returns {Promise<File[]>}
 */
async function fetchDraftFiles(draftId, paths, isCurrent) {
  /** @type {File[]} */
  const files = [];
  for (const fullPath of Array.isArray(paths) ? paths : []) {
    const filename = String(fullPath).split('/').pop() || '';
    const url = `/api/drafts/${draftId}/attachments/${encodeURIComponent(filename)}`;
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      files.push(new File([blob], filename, { type: blob.type }));
    } catch (error) {
      console.error(`Failed to fetch draft attachment ${filename}:`, error);
      if (isCurrent()) showToast(`Could not attach ${filename} from the draft`, { kind: 'error' });
    }
  }
  return files;
}

/**
 * A draft that would not load (Decision 8): toast, and leave the draft marked
 * not loaded. A gone draft (404/422) with no `draft:<id>` snapshot turns the
 * sheet into a new entry; with one, the lifecycle restores that snapshot.
 * @param {string} draftId
 * @param {unknown} error
 */
function draftLoadFailed(draftId, error) {
  console.error('Failed to load draft:', error);
  const gone = error instanceof ApiError && (error.status === 404 || error.status === 422);
  showToast(
    gone
      ? 'This draft is gone: it was submitted, discarded or has expired.'
      : `Could not load the draft: ${messageOf(error)}`,
    { kind: 'error' }
  );
  if (gone && !loadCompose(`draft:${draftId}`)) setState({ compose: { draft: null } });
}

/**
 * Show the provenance banner and session panel of a draft that records
 * `session_metadata`.
 * @param {string} draftId
 * @param {any} meta - The draft's `session_metadata`
 */
function showProvenance(draftId, meta) {
  const form = document.getElementById('create-entry-form');
  if (!form) return;
  document.getElementById('draft-banner')?.remove();
  const banner = document.createElement('div');
  banner.id = 'draft-banner';
  banner.className = 'ariel-compose-banner';
  banner.dataset.draftId = draftId;
  banner.textContent = 'Draft loaded from the OSPREY agent — review and submit';
  form.prepend(banner);
  renderSessionInfoPanel(meta);
}

/**
 * The lifecycle's draft filler (C-SHEET-LIFECYCLE step 3): fill the form from
 * the draft and hold its staged attachments in memory. Nothing (field, banner,
 * metadata, attachment) is written once `isCurrent()` turns false; only a
 * draft that filled completely counts as loaded.
 * @param {string} draftId - Draft ID to load
 * @param {() => boolean} [isCurrent]
 */
export async function loadDraft(draftId, isCurrent = () => true) {
  /** @type {any} */
  let draft;
  try {
    draft = await draftsApi.get(draftId);
  } catch (error) {
    if (isCurrent()) draftLoadFailed(draftId, error);
    return;
  }
  if (!isCurrent()) return;
  const files = await fetchDraftFiles(draftId, draft?.attachment_paths, isCurrent);
  // The declared inputs exist only once publish-info has been rendered.
  await entryFieldsRendered();
  if (!isCurrent()) return;

  for (const name of ['subject', 'details', 'author']) {
    const input = formControl(name);
    if (input) input.value = draft[name] || '';
  }
  // Forwarded on submit; applyDraftFields prunes the declared names.
  draftMetadata = draft.metadata || null;
  setTags(Array.isArray(draft.tags) ? draft.tags : []);
  draftFiles = files;
  renderFilePreview();
  if (draftMetadata?.session_metadata) showProvenance(draftId, draftMetadata.session_metadata);
  applyDraftFields(draft);
  loadedDraft = draftId;
}

/**
 * The lifecycle's follow filler (Decision 12): link the sheet to `parentId`
 * and prefill `Follow-up: <subject>` and the parent's logbook and tags. The
 * row names the parent by id until (and unless) the parent loads; nothing is
 * written once `isCurrent()` turns false, and nothing submits.
 * @param {string} parentId
 * @param {() => boolean} isCurrent
 */
export async function loadFollowUp(parentId, isCurrent) {
  if (!isCurrent()) return;
  followsUp = parentId;
  setParentRow(parentId);

  /** @type {any} */
  let parent;
  try {
    parent = await entriesApi.get(parentId);
  } catch (error) {
    console.error('Failed to load the parent entry:', error);
    return;
  }
  if (!isCurrent()) return;

  const subject = typeof parent?.subject === 'string' ? parent.subject : '';
  const subjectInput = formControl('subject');
  if (subjectInput && subject) {
    subjectInput.value = subject.startsWith(FOLLOW_UP_PREFIX)
      ? subject
      : `${FOLLOW_UP_PREFIX}${subject}`;
  }
  if (subject) setParentRow(subject);

  const metadata = parent?.metadata || {};
  const logbookSelect = formControl('logbook');
  if (logbookSelect && typeof metadata.logbook === 'string' && metadata.logbook) {
    logbookSelect.value = metadata.logbook;
  }
  if (Array.isArray(metadata.tags)) setTags(metadata.tags);
}
