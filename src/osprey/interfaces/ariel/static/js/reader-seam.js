// @ts-check
/**
 * ARIEL reader seam: the registry behind `reader.js`'s extension API.
 *
 * Features extend the reader here instead of editing it. `reader.js`
 * re-exports the four registrations and owns everything that touches a shown
 * entry (the slots, the tab strip's state, click and key routing); this
 * module holds what needs no reader state: the registry, the markup of the
 * seam's buttons, and `place`, which drops a stale render. The thread section
 * is the first section, seeded here.
 */

import { escapeHtml } from './components.js';
import { threadSection } from './reader-thread.js';

/**
 * What a seam feature is handed for the entry it works on.
 * @typedef {object} ReaderContext
 * @property {HTMLElement} container - The reader's container
 * @property {any} entry - The shown entry
 * @property {string} entryId - Its id
 * @property {import('./reader.js').ReaderCallbacks} callbacks - The newest
 *   `showInReader` call's callbacks
 * @property {() => boolean} isCurrent - Whether this entry is still the one
 *   shown; false once `showInReader`/`clearReader` is called again
 */

/**
 * A seam render: escaped HTML or a node (with its own listeners), or nothing.
 * The result is placed only while `ctx.isCurrent()`, so a stale entry's work
 * is discarded; a thrown error is logged and renders nothing.
 * @typedef {(ctx: ReaderContext) => string|Node|null|undefined|Promise<string|Node|null|undefined>} ReaderRender
 */

/** @typedef {{id: string, render: ReaderRender}} ReaderSection - After the body, in registration order */
/** @typedef {{id: string, label: string, key?: string, render: ReaderRender}} ReaderTab - Beside "Entry"; rendered on every open, `key` opens it */
/** @typedef {{id: string, label: string, key?: string, title?: string, run: (ctx: ReaderContext) => void}} ReaderAction - A header button; `key` runs it */

/**
 * The registry. Ids are lowercase slugs (`a-z`, digits, `-`), safe in an
 * attribute and a selector; registering an id or a key again replaces the
 * earlier registration.
 */
export const seam = {
  /** @type {ReaderSection[]} */ sections: [threadSection],
  /** @type {ReaderTab[]} */ tabs: [],
  /** @type {ReaderAction[]} */ actions: [],
  /** @type {Map<string, (ctx: ReaderContext) => void>} */ keys: new Map(),
};

/**
 * @template {{id: string}} T
 * @param {T[]} list
 * @param {T} item
 */
function put(list, item) {
  if (!/^[a-z][a-z0-9-]*$/.test(item.id)) throw new Error(`Bad reader feature id: ${item.id}`);
  const at = list.findIndex((x) => x.id === item.id);
  list.splice(at < 0 ? list.length : at, 1, item);
}

/** @param {ReaderSection} section - Rendered after the entry body, below earlier sections */
export function registerReaderSection(section) {
  put(seam.sections, section);
}

/** @param {ReaderTab} tab */
export function registerReaderTab(tab) {
  put(seam.tabs, tab);
}

/** @param {ReaderAction} action */
export function registerReaderAction(action) {
  put(seam.actions, action);
}

/**
 * A key that acts on the shown entry. The caller's keydown listener routes
 * keys through `reader.js`'s `runReaderKey`; a tab's or action's own `key`
 * takes precedence.
 * @param {string} key - `KeyboardEvent.key`
 * @param {(ctx: ReaderContext) => void} run
 */
export function registerReaderKey(key, run) {
  seam.keys.set(key, run);
}

/**
 * Render into `slot`, placing the result only while the entry (and
 * `current`, when given) is still current.
 * @param {ReaderContext} ctx
 * @param {ReaderRender} render
 * @param {Element|null} slot
 * @param {() => boolean} [current] - An extra check, such as "this tab is still open"
 * @returns {Promise<void>}
 */
export async function place(ctx, render, slot, current = () => true) {
  if (!slot) return;
  /** @type {string|Node|null|undefined} */
  let out = null;
  try {
    out = await render(ctx);
  } catch (error) {
    console.error('Reader feature failed:', error);
  }
  if (!ctx.isCurrent() || !current()) return;
  if (typeof out === 'string') slot.innerHTML = out;
  else slot.replaceChildren(...(out ? [out] : []));
}

/**
 * The registered actions' header buttons.
 * @returns {string} HTML string
 */
export function renderActionButtons() {
  return seam.actions
    .map((a) => {
      const title = a.title ? ` title="${escapeHtml(a.title)}"` : '';
      return `<button type="button" class="btn btn-secondary btn-sm" data-reader-action="${a.id}"${title}>${escapeHtml(a.label)}</button>`;
    })
    .join('');
}

/**
 * The tab strip ("Entry", selected, then each registered tab), or nothing
 * without registered tabs.
 * @returns {string} HTML string
 */
export function renderTabStrip() {
  if (seam.tabs.length === 0) return '';
  const tab = (/** @type {string} */ id, /** @type {string} */ label, selected = false) =>
    `<button type="button" class="btn btn-secondary btn-sm" role="tab" data-reader-tab="${id}" aria-selected="${selected}">${escapeHtml(label)}</button>`;
  const tabs = seam.tabs.map((t) => tab(t.id, t.label)).join('');
  return `<div class="entry-detail-tabs" role="tablist">${tab('', 'Entry', true)}${tabs}</div>`;
}
