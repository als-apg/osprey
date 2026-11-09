// @ts-check
/**
 * ARIEL rail: the mailbox rows (All entries, Drafts, Mine), the Mine name and
 * the saved searches, rendered into `#ariel-rail`.
 *
 * Every row is an `.osprey-nav-row` whose icon stays visible when layout.js
 * collapses the rail to its icon strip; the row's `aria-label`/`title` carry
 * the name there. Selecting a row writes `mailbox` (with `rerank:'now'`).
 *
 * Mine needs a name: until one is known, selecting Mine expands the rail and
 * asks for it inline. `mineAuthor` is written only here — loaded from browser
 * storage at init and updated by `rememberAuthor` (which the compose host also
 * calls after a successful create). The name field suggests every author
 * (the Filters author pick-list's suggestions), because Mine matches its name
 * exactly.
 *
 * Saved searches (saved-searches.js) keep query, mode, filter and knobs under
 * a name, in browser storage only (ruling e-0001 R3). Running one restores all
 * of them in one write. A `box=saved:<name>` arriving through the hash is
 * restored by router.js, not here.
 *
 * Selecting Mine clears `filter.author` in the same write: Mine scopes the
 * author itself.
 *
 * Below the saved searches sits `.ariel-rail-facets`, the mount facets.js
 * renders the value counts and the day histogram into.
 */

import { icon } from '/design-system/js/icons.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { getState, setState, subscribe, mailboxKey, savedName } from './store.js';
import { mergeFilter, normalizeFilter, requestFilter } from './filter-state.js';
import { readSaved, findSaved, saveSearch, removeSearch } from './saved-searches.js';
import { attachAuthorSuggestions } from './advanced-options.js';

/** @typedef {import('./store.js').State} State */
/** @typedef {import('./store.js').Mailbox} Mailbox */

/** @typedef {import('./saved-searches.js').SavedSearch} SavedSearch */

/** Base storage key of the Mine name; scoped, shared by both frames. */
const MINE_KEY = 'ariel-mine-author';

/** @type {ReadonlyArray<{box: 'all'|'drafts'|'mine', label: string, icon: string}>} */
const MAILBOXES = Object.freeze([
  { box: 'all', label: 'All entries', icon: 'notebook' },
  { box: 'drafts', label: 'Drafts', icon: 'file-text' },
  { box: 'mine', label: 'Mine', icon: 'pen' },
]);

/**
 * The mounted rail, or null before `initRail`.
 * @type {{root: HTMLElement, mineForm: HTMLFormElement, mineInput: HTMLInputElement,
 *   saveForm: HTMLFormElement, saveInput: HTMLInputElement, savedList: HTMLElement,
 *   unsubscribe: () => void} | null}
 */
let mounted = null;

// ---------------------------------------------------------------------------
// Storage (may be blocked in a sandboxed iframe: never throw from here)

/**
 * @param {string} base
 * @returns {string|null}
 */
function readStored(base) {
  try {
    return localStorage.getItem(scopedStorageKey(base));
  } catch {
    return null;
  }
}

/**
 * @param {string} base
 * @param {string} value
 */
function writeStored(base, value) {
  try {
    localStorage.setItem(scopedStorageKey(base), value);
  } catch {
    // Storage blocked: the value lives for this page only.
  }
}

/**
 * A usable Mine name, or null for blank / `Anonymous` / non-strings.
 * @param {unknown} name
 * @returns {string|null}
 */
function cleanAuthor(name) {
  if (typeof name !== 'string') return null;
  const trimmed = name.trim();
  if (!trimmed || trimmed.toLowerCase() === 'anonymous') return null;
  return trimmed;
}

// ---------------------------------------------------------------------------
// Public API

/**
 * Remember the Mine name: ignores blank and `Anonymous`; otherwise persists it
 * and writes `mineAuthor`. Safe to call before `initRail`.
 * @param {string} name
 */
export function rememberAuthor(name) {
  const author = cleanAuthor(name);
  if (!author) return;
  writeStored(MINE_KEY, author);
  setState({ mineAuthor: author });
}

/**
 * Save the current query, mode, filter and knobs under `name` (blank is
 * ignored). The filter is the one the search sends (`requestFilter`), so a
 * search saved under Mine keeps the Mine name as its author. Saving an
 * existing name replaces that entry in place. Changes nothing shown besides
 * the rail's list.
 * @param {string} name
 */
export function saveCurrentSearch(name) {
  const state = getState();
  const stored = saveSearch({
    name,
    query: state.query,
    mode: state.mode,
    filter: requestFilter(state) ?? normalizeFilter(state.filter),
    filters: state.filters ?? {},
  });
  if (stored) renderSaved();
}

/**
 * Expand the rail and focus its inline save-search name field (the ⋯ menu's
 * save action). No-op before `initRail`.
 */
export function promptSaveSearch() {
  if (!mounted) return;
  setState({ railCollapsed: false });
  openField(mounted.saveForm, mounted.saveInput);
}

/**
 * Render the rail into `root` and keep it in step with the store.
 * @param {HTMLElement} root
 */
export function initRail(root) {
  if (mounted) mounted.unsubscribe();
  root.replaceChildren();

  const boxes = document.createElement('div');
  boxes.className = 'ariel-rail-boxes';
  for (const def of MAILBOXES) {
    const btn = buildRow(def.box, def.label, def.icon);
    btn.addEventListener('click', () => selectMailbox(def.box));
    boxes.append(btn);
  }

  const mine = buildField('mine', 'Your name for Mine', 'Your name', (value) => {
    rememberAuthor(value);
    if (!getState().mineAuthor) return false;
    setState(minePatch());
    return true;
  });
  const mineSuggestions = document.createElement('datalist');
  mineSuggestions.id = 'ariel-mine-suggestions';
  mine.input.setAttribute('list', mineSuggestions.id);
  mine.form.append(mineSuggestions);
  attachAuthorSuggestions(mine.input, mineSuggestions);
  boxes.append(mine.form);

  const head = document.createElement('div');
  head.className = 'ariel-rail-saved-head';
  const caption = document.createElement('div');
  caption.className = 'osprey-section-label';
  caption.textContent = 'Saved searches';
  const saveBtn = document.createElement('button');
  saveBtn.type = 'button';
  saveBtn.className = 'osprey-icon-btn';
  saveBtn.setAttribute('data-rail-action', 'save-search');
  saveBtn.setAttribute('aria-label', 'Save current search');
  saveBtn.title = 'Save current search';
  saveBtn.append(icon('save'));
  saveBtn.addEventListener('click', () => promptSaveSearch());
  head.append(caption, saveBtn);

  const save = buildField('save', 'Name for this search', 'Search name', (value) => {
    if (!value.trim()) return false;
    saveCurrentSearch(value);
    return true;
  });

  const savedList = document.createElement('div');
  savedList.className = 'ariel-rail-saved';

  // The facets mount (facets.js renders into it); it always exists.
  const facets = document.createElement('div');
  facets.className = 'ariel-rail-facets';
  facets.setAttribute('aria-busy', 'false');

  root.append(boxes, head, save.form, savedList, facets);

  mounted = {
    root,
    mineForm: mine.form,
    mineInput: mine.input,
    saveForm: save.form,
    saveInput: save.input,
    savedList,
    unsubscribe: () => {},
  };

  const stored = cleanAuthor(readStored(MINE_KEY));
  if (stored && stored !== getState().mineAuthor) setState({ mineAuthor: stored });

  renderSaved();
  mounted.unsubscribe = subscribe(onStateChange);
  // The mailbox may already be in the store (applied before the rail).
  followMailbox(getState());
}

// ---------------------------------------------------------------------------
// Internals

/**
 * @param {State} state
 * @param {State} _prev
 * @param {Set<string>} changed
 */
function onStateChange(state, _prev, changed) {
  if (!changed.has('mailbox')) return;
  markCurrent();
  followMailbox(state);
}

/**
 * What a mailbox selected from elsewhere asks of the rail: the Mine name.
 * @param {State} state
 */
function followMailbox(state) {
  if (state.mailbox === 'mine' && !state.mineAuthor) askForMineName();
}

/**
 * The write that selects Mine: its author scope replaces any author selection.
 * @returns {Partial<State>}
 */
function minePatch() {
  return {
    mailbox: 'mine',
    filter: mergeFilter(getState().filter, { author: null }),
    rerank: 'now',
  };
}

/** @param {'all'|'drafts'|'mine'} box */
function selectMailbox(box) {
  if (box === 'mine' && !getState().mineAuthor) {
    askForMineName();
    return;
  }
  if (mounted) closeField(mounted.mineForm);
  setState(box === 'mine' ? minePatch() : { mailbox: box, rerank: 'now' });
}

function askForMineName() {
  if (!mounted) return;
  setState({ railCollapsed: false });
  openField(mounted.mineForm, mounted.mineInput);
}

/** @param {SavedSearch} saved */
function runSaved(saved) {
  setState({
    mailbox: { saved: saved.name },
    query: saved.query,
    mode: saved.mode,
    filter: saved.filter,
    filters: saved.filters,
    rerank: 'now',
  });
}

/** @param {string} name */
function removeSaved(name) {
  removeSearch(name);
  renderSaved();
  if (savedName(getState().mailbox) === name) setState({ mailbox: 'all', rerank: 'now' });
}

function renderSaved() {
  if (!mounted) return;
  const rows = readSaved().map((saved) => {
    const item = document.createElement('div');
    item.className = 'ariel-rail-saved-item';
    const btn = buildRow(mailboxKey({ saved: saved.name }), saved.name, 'search');
    btn.addEventListener('click', () => {
      const current = findSaved(saved.name);
      if (current) runSaved(current);
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'osprey-icon-btn';
    remove.setAttribute('data-remove-saved', saved.name);
    remove.setAttribute('aria-label', `Remove saved search ${saved.name}`);
    remove.title = `Remove ${saved.name}`;
    remove.append(icon('x'));
    remove.addEventListener('click', () => removeSaved(saved.name));
    item.append(btn, remove);
    return item;
  });
  mounted.savedList.replaceChildren(...rows);
  markCurrent();
}

/** Mark the selected mailbox row with aria-current="page", only it. */
function markCurrent() {
  if (!mounted) return;
  const current = mailboxKey(getState().mailbox);
  for (const btn of mounted.root.querySelectorAll('button[data-box]')) {
    if (btn.getAttribute('data-box') === current) btn.setAttribute('aria-current', 'page');
    else btn.removeAttribute('aria-current');
  }
}

/**
 * One rail row. Built with DOM APIs only: saved-search names are user text.
 * @param {string} box value for `data-box`
 * @param {string} label
 * @param {string} iconName
 * @returns {HTMLButtonElement}
 */
function buildRow(box, label, iconName) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'osprey-nav-row';
  btn.setAttribute('data-box', box);
  btn.setAttribute('aria-label', label);
  btn.title = label;
  const glyph = icon(iconName);
  glyph.classList.add('osprey-nav-icon');
  const text = document.createElement('span');
  text.className = 'osprey-nav-label';
  text.textContent = label;
  btn.append(glyph, text);
  return btn;
}

/**
 * An inline, initially hidden name field. `onSubmit` returns true when the
 * value was accepted (the field then closes) and false to keep asking.
 * @param {string} field value for `data-rail-field`
 * @param {string} ariaLabel
 * @param {string} placeholder
 * @param {(value: string) => boolean} onSubmit
 * @returns {{form: HTMLFormElement, input: HTMLInputElement}}
 */
function buildField(field, ariaLabel, placeholder, onSubmit) {
  const form = document.createElement('form');
  form.className = 'ariel-rail-field';
  form.hidden = true;
  const input = document.createElement('input');
  input.type = 'text';
  input.autocomplete = 'off';
  input.setAttribute('data-rail-field', field);
  input.setAttribute('aria-label', ariaLabel);
  input.placeholder = placeholder;
  form.append(input);
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (onSubmit(input.value)) closeField(form);
    else input.focus();
  });
  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    // One Esc closes one layer: this field, not the panel's next layer.
    event.preventDefault();
    event.stopPropagation();
    closeField(form);
  });
  return { form, input };
}

/**
 * @param {HTMLFormElement} form
 * @param {HTMLInputElement} input
 */
function openField(form, input) {
  form.hidden = false;
  input.focus();
}

/** @param {HTMLFormElement} form */
function closeField(form) {
  form.hidden = true;
  const input = form.querySelector('input');
  if (input) input.value = '';
}
