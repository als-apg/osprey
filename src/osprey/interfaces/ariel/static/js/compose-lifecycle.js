// @ts-check
/**
 * ARIEL compose-sheet lifecycle (C-SHEET-LIFECYCLE): the one place that
 * sequences an open, saves a snapshot, flushes it and clears it.
 *
 * `open(compose)` flushes the previous context's pending save, takes a fresh
 * token and runs, in order:
 *   1. `reset()`;
 *   2. `await renderFields()`;
 *   3. the kind fill: `fillers.follow(followsUp)` or `fillers.draft(draft)` (a
 *      new entry has none);
 *   4. restore: the context's stored snapshot through `applySnapshot`.
 * Every await is followed by a token check, so a step of a superseded open
 * does nothing; a filler receives `isCurrent()` for its own awaits.
 *
 * Until step 4 has finished, the gated controls (Submit, Discard) are disabled
 * and the form is inert, and no change saves. After it, a user `input` or
 * `change` on a saved field schedules one debounced save of `readSnapshot()`;
 * `onSaved(ok, savedAt)` reports whether the write landed. Credential and
 * attachment inputs never save. `close()` and `pagehide` flush the pending
 * save; `finish()` cancels it and forgets the context's snapshot. Aborting
 * `options.signal` removes the listeners and drops the pending save.
 *
 * Focus: `open` records the opener, the element focused when a closed sheet
 * opens (a reopen from inside the sheet keeps the first one). When an open
 * sheet then closes or finishes (Close, Discard, Esc, a submit), focus returns
 * to the opener, or to `fallbackFocus()` (the selected row) when the opener
 * is gone; it is left alone when it already sits outside `scope`.
 */

import {
  SECRET_FIELD,
  clearCompose,
  contextKey,
  loadCompose,
  saveCompose,
} from './compose-autosave.js';

/** @typedef {import('./compose-autosave.js').ComposeSnapshot} ComposeSnapshot */
/** @typedef {import('./compose-autosave.js').ComposeKey} ComposeKey */
/** @typedef {{draft: string|null, followsUp?: string}} Compose */
/** @typedef {(id: string, isCurrent: () => boolean) => (void|Promise<void>)} Filler */

/**
 * @typedef {Object} LifecycleHooks
 * @property {() => void} reset - step 1
 * @property {() => (void|Promise<void>)} renderFields - step 2
 * @property {{follow?: Filler, draft?: Filler}} fillers - step 3, by kind; each
 *   receives its kind's id (the parent, or the draft)
 * @property {() => Partial<ComposeSnapshot>} readSnapshot - the form as a snapshot
 * @property {(snapshot: ComposeSnapshot) => (void|Promise<void>)} applySnapshot - step 4
 * @property {(ok: boolean, savedAt: number) => void} onSaved - after every save
 */

/**
 * @typedef {Object} LifecycleOptions
 * @property {HTMLElement|null} [form] - where user edits are heard; made inert while opening
 * @property {Iterable<HTMLButtonElement|null>} [gated] - disabled until step 4 finishes
 * @property {AbortSignal} [signal] - removes the lifecycle's listeners
 * @property {number} [delay] - the save debounce, in ms
 * @property {HTMLElement|null} [scope] - the sheet: focus inside it returns to
 *   the opener on close (defaults to `form`)
 * @property {() => (HTMLElement|null)} [fallbackFocus] - where focus returns
 *   when the opener is gone (the selected row)
 */

/** The built-in fields a snapshot holds, by `name`. */
const SAVED_NAMES = new Set(['subject', 'details', 'author', 'logbook', 'shift']);

/** The default save debounce. */
export const SAVE_DELAY_MS = 800;

/**
 * Whether a user edit on `target` is an edit of a saved field: a built-in
 * field, a declared entry field, or the tag list (which signals its own adds
 * and removes with a `change`).
 * @param {EventTarget|null} target
 * @returns {boolean}
 */
function isSavedTarget(target) {
  if (!(target instanceof Element)) return false;
  if (target.id === 'entry-tags') return true;
  const type = target.getAttribute('type');
  if (type === 'file' || type === 'password') return false;
  const name = target.getAttribute('name') || '';
  if (SECRET_FIELD.test(name)) return false;
  if (target.hasAttribute('data-entry-field')) {
    return !SECRET_FIELD.test(target.getAttribute('data-entry-field') || '');
  }
  return SAVED_NAMES.has(name);
}

/**
 * Run one lifecycle step, logging (not propagating) its failure, so a broken
 * hook never leaves the sheet locked.
 * @param {string} step
 * @param {() => any} fn
 */
async function runStep(step, fn) {
  try {
    await fn();
  } catch (err) {
    console.error(`Compose ${step} failed:`, err);
  }
}

/**
 * Whether `el` can still take focus back: in the document and not inside a
 * hidden or inert subtree.
 * @param {HTMLElement|null} el
 * @returns {el is HTMLElement}
 */
function canFocus(el) {
  return !!el && el.isConnected && !el.closest('[hidden], [inert]');
}

/**
 * Create the compose sheet's lifecycle.
 * @param {LifecycleHooks} hooks
 * @param {LifecycleOptions} [options]
 * @returns {{open(compose: Compose): Promise<void>, close(): void, finish(): void, isOpen(): boolean}}
 */
export function createComposeLifecycle(hooks, options = {}) {
  const { form = null, signal, delay = SAVE_DELAY_MS, fallbackFocus } = options;
  const scope = options.scope ?? form;
  const gated = Array.from(options.gated || []).filter(
    /** @returns {b is HTMLButtonElement} */ (b) => !!b
  );

  let token = 0;
  /** @type {ComposeKey|null} the open context's key */
  let key = null;
  /** Whether step 4 has finished for the open context. */
  let ready = false;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let timer = null;
  /** @type {HTMLElement|null} the element focused when the sheet opened */
  let opener = null;

  /** @returns {Element|null} the focused element, or null for none or the body */
  function focused() {
    if (typeof document === 'undefined') return null;
    const active = document.activeElement;
    return active && active !== document.body ? active : null;
  }

  /** Take the opener of a closed sheet; a reopen from inside the sheet keeps it. */
  function recordOpener() {
    const active = focused();
    if (active && scope?.contains(active)) return;
    opener = active instanceof HTMLElement ? active : null;
  }

  /** Return focus from a closing sheet to the opener, else the fallback. */
  function restoreFocus() {
    const target = opener;
    opener = null;
    const active = focused();
    if (active && !scope?.contains(active)) return;
    const next = canFocus(target) ? target : (fallbackFocus?.() ?? null);
    next?.focus();
  }

  /** @param {boolean} on */
  function setReady(on) {
    ready = on;
    for (const btn of gated) btn.disabled = !on;
    if (!form) return;
    form.toggleAttribute('inert', !on);
    form.setAttribute('aria-busy', String(!on));
  }

  function cancel() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  }

  function save() {
    cancel();
    if (!key) return;
    const savedAt = Date.now();
    const ok = saveCompose(key, { ...hooks.readSnapshot(), savedAt });
    hooks.onSaved(ok, savedAt);
  }

  /** Write the pending save now, if there is one. */
  function flush() {
    if (timer !== null) save();
  }

  /** @param {Event} e */
  function onEdit(e) {
    if (!ready || !key || !isSavedTarget(e.target)) return;
    cancel();
    timer = setTimeout(save, delay);
  }

  function invalidate() {
    token += 1;
    key = null;
    setReady(false);
  }

  form?.addEventListener('input', onEdit, { signal });
  form?.addEventListener('change', onEdit, { signal });
  if (typeof window !== 'undefined') window.addEventListener('pagehide', flush, { signal });
  // Teardown (the host's destroy) drops the pending save rather than writing it.
  signal?.addEventListener('abort', () => {
    cancel();
    token += 1;
    key = null;
    opener = null;
  });

  return {
    async open(compose) {
      flush();
      cancel();
      recordOpener();
      const mine = ++token;
      const isCurrent = () => mine === token;
      key = contextKey(compose);
      setReady(false);

      await runStep('reset', () => hooks.reset());
      if (!isCurrent()) return;
      await runStep('render', () => hooks.renderFields());
      if (!isCurrent()) return;
      const [filler, id] = compose.draft
        ? [hooks.fillers.draft, compose.draft]
        : [hooks.fillers.follow, compose.followsUp];
      if (filler && id) {
        await runStep('fill', () => filler(id, isCurrent));
        if (!isCurrent()) return;
      }
      const snapshot = loadCompose(/** @type {ComposeKey} */ (key));
      if (snapshot) {
        await runStep('restore', () => hooks.applySnapshot(snapshot));
        if (!isCurrent()) return;
      }
      setReady(true);
    },

    close() {
      flush();
      const wasOpen = key !== null;
      invalidate();
      if (wasOpen) restoreFocus();
    },

    finish() {
      cancel();
      const wasOpen = key !== null;
      if (key) clearCompose(key);
      invalidate();
      if (wasOpen) restoreFocus();
    },

    isOpen() {
      return key !== null;
    },
  };
}
