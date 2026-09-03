/* OSPREY Web Terminal — bar edit mode.
 *
 * The operator's way into the two bars: an edit mode that outlines them and
 * drops the item sheet below the header, and the single funnel every change
 * passes through on its way to the server.
 *
 * Two rules govern this module.
 *
 * 1. ONE WRITE PATH. Every edit — a tile click, the status-bar toggle, and the
 *    drag and options gestures that land on top of this — becomes
 *    `saveLayout(next, {edit: true})` and nothing else. The `{edit: true}` flag
 *    is the caller stating a human did this, which is the mechanism behind "a
 *    PUT only ever originates from an explicit user edit". The revision is the
 *    sync layer's, never this module's.
 *
 * 2. A REFUSED EDIT IS NAMED, NOT DROPPED. `normalize()` discards an item over
 *    the per-host cap, one a host may not hold, and one this deployment cannot
 *    render — silently, because it is repairing a stored document rather than
 *    answering a person. An edit the operator just made is different: it is
 *    refused where it was made, with the reason on the tile, and issues no PUT.
 *    `refusalFor()` is that judgment, asked once and answered the same way by
 *    the sheet's tiles and by anything that later drops an item into a bar.
 *
 * Neither rule reads the ui mode. The bars are the operator's chrome, not the
 * mode's, so Simple mode edits them exactly as Expert does; what Simple
 * simplifies is the workspace around them.
 *
 * The DOM of the sheet itself lives in bar-customize-sheet.js; this file is the
 * state machine and the rules.
 */

import { BAR_CATALOG, BAR_HOSTS, barItemType, defaultOptions } from './bar-catalog.js';
import { MAX_ITEMS_PER_HOST, normalize } from './bar-layout.js';
import { closeBarPopovers, docOf } from './bar-host.js';
import {
  BarSyncError,
  currentLayout,
  deploymentContext,
  isLayoutReadonly,
  onSyncNotice,
  resetLayout as discardStoredLayout,
  saveLayout,
} from './bar-sync.js';
import {
  closeSheet,
  destroySheet,
  openSheet,
  renderSheet,
  sheetNotice,
} from './bar-customize-sheet.js';
import { armDrag, disarmDrag } from './bar-customize-drag.js';
import { barSurfaceOpen, closeBarSurfaces } from './bar-customize-menus.js';
import { initEntryPoints, stopEntryPoints } from './bar-customize-entry.js';

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */
/** @typedef {import('./bar-layout.js').BarLayout} BarLayout */
/** @typedef {import('./bar-layout.js').BarLayoutItem} BarLayoutItem */
/** @typedef {import('./bar-host.js').BarRoot} BarRoot */

/**
 * Where one placed item sits, and what it is.
 * @typedef {{host: BarHost, index: number, type: string}} BarItemPlace
 */

/**
 * Everything the sheet, the drag gesture, the options popover and the entry
 * points are allowed to ask and to do. They are handed this object once and
 * decide nothing themselves, which is what keeps the refusal rules and the one
 * write path in this module: a drop target and a tile ask the same function and
 * get the same sentence.
 * @typedef {object} BarEditController
 * @property {(type: string, host: BarHost) => string | null} refusalFor
 * @property {(type: string, host: BarHost, from: BarHost | null) => string | null} dropRefusal
 * @property {() => BarHost} defaultHostFor
 * @property {(type: string, host: BarHost, index?: number) => Promise<boolean>} addItem
 * @property {(shell: Element) => BarItemPlace | null} locate
 * @property {(from: BarHost, fromIndex: number, to: BarHost, toIndex: number)
 *            => Promise<boolean>} moveItem
 * @property {(host: BarHost, index: number) => Promise<boolean>} removeAt
 * @property {(host: BarHost, index: number, key: string, value: string | number | boolean)
 *            => Promise<boolean>} setOption
 * @property {(host: BarHost, index: number) => BarLayoutItem | null} itemAt
 * @property {(host: BarHost) => readonly BarLayoutItem[]} hostItems
 * @property {(type: string) => BarHost | null} placedIn
 * @property {() => boolean} readonly
 * @property {() => Promise<boolean>} resetToDefault
 * @property {(host: BarHost, visible: boolean) => Promise<boolean>} setBarVisible
 * @property {(host: BarHost) => boolean} barVisible
 * @property {(text: string) => void} notice
 * @property {() => void} enterEditMode
 * @property {() => void} exitEditMode
 * @property {() => boolean} isEditing
 * @property {(listener: (editing: boolean) => void) => () => void} onEditModeChange
 */

/** How each host is named to the operator. */
const HOST_LABEL = /** @type {Readonly<Record<BarHost, string>>} */ (
  Object.freeze({ header: 'Header', status: 'Status bar' })
);

/** @type {BarRoot} */
let activeRoot = typeof document === 'undefined' ? /** @type {any} */ (null) : document;
let editing = false;
/** @type {Set<(editing: boolean) => void>} */
const listeners = new Set();
/** @type {(() => void) | null} */
let unsubscribeNotice = null;

/**
 * What this deployment offers, as the catalog's `available()` asks it. THE SAME
 * context the normalizer uses, deliberately: a sheet that offered a tile the
 * normalizer then dropped would latch the document read-only and kill saving
 * for the session with nothing said. The server stamps the facts; nobody here
 * infers them.
 * @param {BarRoot} [root]
 * @returns {import('./bar-layout.js').BarLayoutContext}
 */
export function editContext(root = activeRoot) {
  return deploymentContext(root);
}

/* ---- the refusal rules ---- */

/**
 * Where a type already sits in the held document, or null when it is placed
 * nowhere. Asked of the whole document, header first — the answer a
 * single-node item's tile dims itself on.
 * @param {string} type
 * @returns {BarHost | null}
 */
export function placedIn(type) {
  const layout = currentLayout();
  if (!layout) return null;
  for (const host of BAR_HOSTS) {
    if (layout[host].some((item) => item.type === type)) return host;
  }
  return null;
}

/**
 * Why `type` cannot be added to `host` right now, or null when it can.
 *
 * The order matters: the read-only latch answers first because it makes every
 * other question moot, and the cap answers last so a type that could never go
 * there is told the honest reason rather than "full".
 * @param {string} type
 * @param {BarHost} host
 * @param {BarRoot} [root]
 * @param {{moving?: boolean}} [gesture] - `moving` when the item being judged
 *   is already placed and is on its way to `host`, so its own presence is not
 *   a duplicate
 * @returns {string | null}
 */
export function refusalFor(type, host, root = activeRoot, gesture = {}) {
  if (isLayoutReadonly()) return 'Layout not editable';
  const entry = barItemType(type);
  if (!entry) return 'Not available in this build';
  if (!entry.available({ ...editContext(root), host })) return 'Not in this deployment';
  // A single-node item exists once: its body is one server-rendered node or
  // one id-owning dot, and a second shell for it could only ever be empty.
  const sitting = entry.multi || gesture.moving ? null : placedIn(type);
  if (sitting) return `Already in the ${HOST_LABEL[sitting].toLowerCase()}`;
  const layout = currentLayout();
  if (layout && layout[host].length >= MAX_ITEMS_PER_HOST) return `${HOST_LABEL[host]} is full`;
  return null;
}

/**
 * Why a DROP cannot land, or null when it can.
 *
 * A reorder inside one bar is a permutation: the cap and the host rules already
 * hold, and asking `refusalFor` would refuse a bar that is legitimately full of
 * the items being reordered. Only the read-only latch still applies. Moving to
 * the OTHER bar, or dragging a new item in from the sheet, is an addition and
 * gets the full judgment — except that a moving item is not its own duplicate.
 * @param {string} type
 * @param {BarHost} host - where the pointer is
 * @param {BarHost | null} from - where the item is now; null when it is new
 * @param {BarRoot} [root]
 * @returns {string | null}
 */
export function dropRefusal(type, host, from, root = activeRoot) {
  if (from === host) return isLayoutReadonly() ? 'Layout not editable' : null;
  return refusalFor(type, host, root, { moving: from !== null });
}

/**
 * The host a tile adds to when the operator just clicks it. Every type may
 * live in either bar, so a click lands in the header; dropping an item into
 * the status bar is the drag gesture's job.
 * @returns {BarHost}
 */
export function defaultHostFor() {
  return 'header';
}

/**
 * Where a live shell sits in the held document, or null when it does not
 * (a parked shell, a document that has not arrived yet).
 *
 * The key is the host layer's identity for an item — `type` for the first of a
 * type, `type#n` after that, counted across the whole document — so this walks
 * the layout the same way `planHost()` does rather than counting DOM nodes,
 * which a reconcile in flight would have moved.
 * @param {Element} shell
 * @param {BarLayout | null} [layout]
 * @returns {BarItemPlace | null}
 */
export function locate(shell, layout = currentLayout()) {
  const key = /** @type {HTMLElement} */ (shell).dataset.barKey;
  if (!key || !layout) return null;
  /** @type {Map<string, number>} */
  const counts = new Map();
  for (const host of BAR_HOSTS) {
    const list = layout[host] ?? [];
    for (let index = 0; index < list.length; index += 1) {
      const type = list[index].type;
      if (!barItemType(type)) continue;
      const seen = counts.get(type) ?? 0;
      counts.set(type, seen + 1);
      if ((seen === 0 ? type : `${type}#${seen}`) === key) return { host, index, type };
    }
  }
  return null;
}

/**
 * The item at one place in the held document, or null.
 * @param {BarHost} host
 * @param {number} index
 * @returns {BarLayoutItem | null}
 */
export function itemAt(host, index) {
  return currentLayout()?.[host]?.[index] ?? null;
}

/**
 * One host's items, in layout order. The list a drop index is stated against —
 * it is not the same list as the host's DOM children, because the overflow
 * ladder parks a folded item's shell in the pool.
 * @param {BarHost} host
 * @returns {readonly BarLayoutItem[]}
 */
export function hostItems(host) {
  return currentLayout()?.[host] ?? [];
}

/**
 * What to tell the operator about an entry `normalize()` would have discarded.
 * @param {{host: string, reason: string}} drop
 * @returns {string}
 */
function reasonText(drop) {
  const host = /** @type {BarHost} */ (drop.host);
  if (drop.reason === 'overflow') return `${HOST_LABEL[host] ?? 'Bar'} is full`;
  if (drop.reason === 'unavailable') return 'Not in this deployment';
  if (drop.reason === 'duplicate') return `Already in the ${(HOST_LABEL[host] ?? 'bar').toLowerCase()}`;
  return 'Not available in this build';
}

/* ---- the one write path ---- */

/**
 * A mutable copy of the held document. `normalize()` deep-freezes its output,
 * so an edit builds a new document rather than touching the current one.
 * @param {BarLayout} layout
 * @returns {any}
 */
function draftOf(layout) {
  return {
    version: layout.version,
    rev: layout.rev,
    header: layout.header.map((item) => ({ type: item.type, options: { ...item.options } })),
    status: layout.status.map((item) => ({ type: item.type, options: { ...item.options } })),
    header_visible: layout.header_visible,
    status_visible: layout.status_visible,
  };
}

/**
 * Run one edit: build the next document, refuse it if normalization would lose
 * anything, otherwise save it. Resolves to whether the document was stored.
 *
 * Nothing here retries and nothing here chooses a revision — a refusal that the
 * sync layer already announced ("Layout not saved") is left to its notice,
 * which this module routes into the sheet.
 * @param {(draft: any) => any} mutate
 * @param {BarRoot} [root]
 * @returns {Promise<boolean>}
 */
export async function applyEdit(mutate, root = activeRoot) {
  if (isLayoutReadonly()) {
    sheetNotice('Layout not editable', root);
    return false;
  }
  const base = currentLayout();
  if (!base) {
    sheetNotice('Layout not loaded', root);
    return false;
  }
  const result = normalize(mutate(draftOf(base)), BAR_CATALOG, editContext(root));
  if (result.dropped.length > 0) {
    sheetNotice(reasonText(result.dropped[0]), root);
    return false;
  }
  if (result.readonly) {
    sheetNotice('Change refused', root);
    return false;
  }
  try {
    await saveLayout(result.layout, { edit: true, root });
  } catch (error) {
    if (error instanceof BarSyncError && error.reason === 'readonly') {
      sheetNotice('Layout not editable', root);
    }
    return false;
  }
  sheetNotice('', root);
  renderSheet(root);
  return true;
}

/**
 * Add one item of `type` to `host`, at `index` or at the end.
 * @param {string} type
 * @param {BarHost} host
 * @param {number} [index]
 * @param {BarRoot} [root]
 * @returns {Promise<boolean>}
 */
export async function addItem(type, host, index, root = activeRoot) {
  const refusal = refusalFor(type, host, root);
  if (refusal) {
    sheetNotice(refusal, root);
    return false;
  }
  const options = defaultOptions(type);
  return applyEdit((draft) => {
    draft[host].splice(index ?? draft[host].length, 0, { type, options });
    return draft;
  }, root);
}

/**
 * Move one placed item — to another index in its own bar, or into the other
 * bar. Moving an item into the status bar SHOWS the status bar: an item the
 * operator just put somewhere they cannot see is a lost item.
 * @param {BarHost} from
 * @param {number} fromIndex
 * @param {BarHost} to
 * @param {number} toIndex
 * @param {BarRoot} [root]
 * @returns {Promise<boolean>}
 */
export async function moveItem(from, fromIndex, to, toIndex, root = activeRoot) {
  const item = itemAt(from, fromIndex);
  if (!item) return false;
  const refusal = dropRefusal(item.type, to, from, root);
  if (refusal) {
    sheetNotice(refusal, root);
    return false;
  }
  return applyEdit((draft) => {
    const [moved] = draft[from].splice(fromIndex, 1);
    if (!moved) return draft;
    // The index was read while the item was still in the list, so a move to the
    // right of its own position has shifted by one.
    const at = from === to && fromIndex < toIndex ? toIndex - 1 : toIndex;
    draft[to].splice(Math.max(0, Math.min(at, draft[to].length)), 0, moved);
    // Dropping into a withdrawn bar brings it back: the operator put something
    // there to see it.
    draft[`${to}_visible`] = true;
    return draft;
  }, root);
}

/**
 * Remove one placed item.
 * @param {BarHost} host
 * @param {number} index
 * @param {BarRoot} [root]
 * @returns {Promise<boolean>}
 */
export async function removeAt(host, index, root = activeRoot) {
  const item = itemAt(host, index);
  if (!item) return false;
  return applyEdit((draft) => {
    draft[host].splice(index, 1);
    return draft;
  }, root);
}

/**
 * The value an option may actually take, from the type's own spec. CLAMPED
 * here rather than at the surface that collected it: the store answers 422 to
 * an out-of-spec value and the client has nothing useful to say about that, so
 * a value the catalog would refuse must never reach a PUT.
 * @param {string} type
 * @param {string} key
 * @param {string | number | boolean} value
 * @returns {string | number | boolean | null} null when the type has no such option
 */
export function clampOption(type, key, value) {
  const spec = barItemType(type)?.options[key];
  if (!spec) return null;
  if (spec.kind === 'boolean') return value === true || value === 'true';
  if (spec.kind === 'enum') {
    const text = String(value);
    return spec.values.includes(text) ? text : spec.default;
  }
  // An emptied field is unreadable, not zero: `Number('')` is 0, which would
  // snap a cleared input to the option's MINIMUM instead of restoring what the
  // type says it should be. A number input also blanks itself on any text the
  // browser cannot parse, so this is the path typing "wide" arrives on.
  if (typeof value === 'string' && value.trim() === '') return spec.default;
  const raw = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(raw)) return spec.default;
  const stepped = spec.step > 0 ? Math.round(raw / spec.step) * spec.step : raw;
  return Math.min(spec.max, Math.max(spec.min, stepped));
}

/**
 * Set one option on one placed item.
 * @param {BarHost} host
 * @param {number} index
 * @param {string} key
 * @param {string | number | boolean} value
 * @param {BarRoot} [root]
 * @returns {Promise<boolean>}
 */
export async function setOption(host, index, key, value, root = activeRoot) {
  const item = itemAt(host, index);
  if (!item) return false;
  const clamped = clampOption(item.type, key, value);
  if (clamped === null) return false;
  return applyEdit((draft) => {
    draft[host][index].options[key] = clamped;
    return draft;
  }, root);
}

/**
 * Give up this operator's arrangement and take back the deployment's own —
 * the sheet's one preset, "Default", and what `web.bar_items` configures.
 *
 * NOT `applyEdit`, and deliberately outside the one-write-path rule: there is no
 * document to normalize and nothing to PUT. The deployment default is not a
 * document this client can hold — only the server knows what the deployment
 * configured — so the only honest way to ask for it is to delete what the
 * operator saved and let the server answer.
 *
 * It is also the only edit offered while the layout is READ-ONLY: the latch
 * exists to stop this build writing back a document it had to drop content
 * from, and this is the gesture that throws that document away instead.
 * @param {BarRoot} [root]
 * @returns {Promise<boolean>}
 */
export async function resetToDefault(root = activeRoot) {
  try {
    await discardStoredLayout({ edit: true, root });
  } catch {
    // The sync layer already said so, through the notice this module routes
    // into the sheet.
    return false;
  }
  sheetNotice('', root);
  // The tiles are re-read here rather than left alone: a reset that lifted the
  // read-only latch has just turned every disabled tile back into a live one.
  renderSheet(root);
  return true;
}

/**
 * Show or withdraw one bar. Either bar may go: a hidden bar comes back from
 * the other bar's menu, from the Customize sheet, from the command palette's
 * Customize bars, or from any tile header's menu (panel-menu-policy.js
 * appends a Show row for each hidden bar), so hiding both is never a trap.
 * @param {BarHost} host
 * @param {boolean} visible
 * @param {BarRoot} [root]
 * @returns {Promise<boolean>}
 */
export function setBarVisible(host, visible, root = activeRoot) {
  return applyEdit((draft) => {
    draft[`${host}_visible`] = visible;
    return draft;
  }, root);
}

/** Whether the held document shows `host`. @param {BarHost} host @returns {boolean} */
export function barVisible(host) {
  return currentLayout()?.[`${host}_visible`] !== false;
}

/* ---- edit mode ---- */

/**
 * What every customize surface is allowed to ask and to do. One object, handed
 * to the sheet, the drag gesture, the popovers and the entry points — none of
 * which reaches back into this module, so the rules have exactly one home.
 * @type {BarEditController}
 */
const CONTROLLER = Object.freeze({
  refusalFor: (/** @type {string} */ type, /** @type {BarHost} */ host) =>
    refusalFor(type, host, activeRoot),
  dropRefusal: (
    /** @type {string} */ type,
    /** @type {BarHost} */ host,
    /** @type {BarHost | null} */ from
  ) => dropRefusal(type, host, from, activeRoot),
  defaultHostFor,
  addItem: (
    /** @type {string} */ type,
    /** @type {BarHost} */ host,
    /** @type {number | undefined} */ index
  ) => addItem(type, host, index, activeRoot),
  locate: (/** @type {Element} */ shell) => locate(shell),
  moveItem: (
    /** @type {BarHost} */ from,
    /** @type {number} */ fromIndex,
    /** @type {BarHost} */ to,
    /** @type {number} */ toIndex
  ) => moveItem(from, fromIndex, to, toIndex, activeRoot),
  removeAt: (/** @type {BarHost} */ host, /** @type {number} */ index) =>
    removeAt(host, index, activeRoot),
  setOption: (
    /** @type {BarHost} */ host,
    /** @type {number} */ index,
    /** @type {string} */ key,
    /** @type {string | number | boolean} */ value
  ) => setOption(host, index, key, value, activeRoot),
  itemAt,
  hostItems,
  placedIn,
  readonly: () => isLayoutReadonly(),
  resetToDefault: () => resetToDefault(activeRoot),
  setBarVisible: (/** @type {BarHost} */ host, /** @type {boolean} */ visible) =>
    setBarVisible(host, visible, activeRoot),
  barVisible,
  notice: (/** @type {string} */ text) => sheetNotice(text, activeRoot),
  enterEditMode: () => {
    enterEditMode(activeRoot);
  },
  exitEditMode: () => exitEditMode(activeRoot),
  isEditing,
  onEditModeChange,
});

/** Whether edit mode is on. @returns {boolean} */
export function isEditing() {
  return editing;
}

/**
 * Subscribe to edit-mode changes — what an entry point reflects in its own
 * label or pressed state.
 * @param {(editing: boolean) => void} listener
 * @returns {() => void} unsubscribe
 */
export function onEditModeChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/**
 * Escape, while editing. It closes what is OPEN before it closes edit mode, and
 * this is the only handler that decides that: the menus module answers the
 * question and acts only when edit mode is off, so the two never race. Holding
 * the order by listener registration instead — which is what an earlier version
 * did — broke the moment anything re-armed the menus mid-edit.
 * @param {KeyboardEvent} event
 */
function onKeydown(event) {
  if (event.key !== 'Escape' || !editing) return;
  if (barSurfaceOpen()) {
    closeBarSurfaces();
    return;
  }
  exitEditMode(activeRoot);
}

/**
 * Enter edit mode. Returns whether the page is now editing — always true
 * today; the boolean stays so a caller can keep treating a refusal as a
 * possibility should one ever be added.
 * @param {BarRoot} [root]
 * @returns {boolean}
 */
export function enterEditMode(root = activeRoot) {
  if (editing) return true;
  const owner = docOf(root);
  activeRoot = root;
  editing = true;
  closeBarPopovers();
  owner.body?.classList.add('bar-editing');
  // On <html> as well as on the body, because a withdrawn status bar is hidden
  // by an html-level rule and the height it gives back is an html-level token:
  // CSS cannot reach up from the body to undo either for the duration of the
  // edit. This is what lets the operator drop an item into a hidden bar.
  owner.documentElement?.setAttribute('data-bar-editing', 'true');
  openSheet(CONTROLLER, /** @type {any} */ (root));
  armDrag(CONTROLLER, root);
  owner.addEventListener('keydown', onKeydown);
  for (const listener of listeners) listener(true);
  return true;
}

/**
 * Leave edit mode. Safe to call when it is already off.
 * @param {BarRoot} [root]
 */
export function exitEditMode(root = activeRoot) {
  if (!editing) return;
  editing = false;
  const owner = docOf(root);
  owner.removeEventListener('keydown', onKeydown);
  disarmDrag();
  owner.body?.classList.remove('bar-editing');
  owner.documentElement?.removeAttribute('data-bar-editing');
  closeSheet(/** @type {any} */ (root));
  closeBarPopovers();
  for (const listener of listeners) listener(false);
}

/* ---- boot ---- */

/**
 * Route the sync layer's notices into the sheet. Registering a listener is
 * what retires bar-sync.js's own inline pill: every PUT this build issues comes
 * from edit mode, so the sheet is open whenever one of these arrives.
 * @param {BarRoot} [root]
 * @returns {() => void} stop
 */
export function initBarCustomize(root = activeRoot) {
  activeRoot = root;
  unsubscribeNotice?.();
  unsubscribeNotice = onSyncNotice((/** @type {string} */ text) => sheetNotice(text, activeRoot));
  initEntryPoints(CONTROLLER, root);
  return stopBarCustomize;
}

/**
 * Tear down edit mode, the entry points and the notice subscription — and the
 * sheet with them. The sheet is a mounted element holding listeners that call
 * back into this module, so leaving it in the body after a teardown leaves a
 * Done button wired to a controller nothing is listening to any more.
 */
export function stopBarCustomize() {
  if (editing) exitEditMode(activeRoot);
  stopEntryPoints();
  destroySheet(activeRoot);
  unsubscribeNotice?.();
  unsubscribeNotice = null;
  listeners.clear();
}

if (typeof document !== 'undefined') initBarCustomize(document);
