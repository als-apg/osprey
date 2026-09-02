/* OSPREY Web Terminal — the customize sheet.
 *
 * The panel that drops below the header in edit mode: one tile per catalog
 * type, the status-bar visibility toggle, Done, and the surface every "Layout
 * not saved" notice renders in.
 *
 * This module builds and re-renders DOM. It decides NOTHING: whether a type can
 * be added, and what to say when it cannot, are the controller's answers
 * (bar-customize.js), handed in once. That is what keeps the refusal rules in
 * one place — the sheet and a later drag target ask the same function and get
 * the same sentence — and what lets this file be tested through the controller
 * that owns the write path rather than around it.
 *
 * A refused tile is DISABLED AND NAMED. The rule the design of record states
 * and `normalize()` cannot honour on its own: an edit that would be dropped is
 * refused where the operator made it, with the reason on the tile, instead of
 * being accepted and then quietly discarded on the way to the server.
 */

import { BAR_CATALOG, BAR_ITEM_TYPES } from './bar-catalog.js';
import { PRESETS } from './bar-layout.js';
import { dragJustEnded } from './bar-customize-drag.js';

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */

/**
 * What the sheet is allowed to ask and to do: the whole edit controller, the
 * same object the drag gesture and the popovers are handed. Every answer is
 * read fresh at render time, so a re-render after a save reflects the saved
 * document.
 * @typedef {import('./bar-customize.js').BarEditController} SheetController
 */

/**
 * The two tile groups. The catalog declares no grouping of its own — the
 * spacing types are set apart in its source and nowhere else — so the split is
 * named here, next to the only surface that renders it.
 * @type {readonly string[]}
 */
const SPACING_TYPES = Object.freeze(['space', 'gap', 'separator']);

/** The sheet is a singleton per page; the controller arrives with it. */
/** @type {SheetController | null} */
let controller = null;

/**
 * Make an element with a class and, optionally, text. Text only — the sheet
 * never interpolates markup.
 * @param {string} tag
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function make(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * The sheet element, built on first use and reused after.
 * @param {SheetController} ctrl
 * @param {ParentNode & {querySelector: Function}} root
 * @returns {HTMLElement}
 */
export function sheetElement(ctrl, root = document) {
  controller = ctrl;
  const existing = /** @type {HTMLElement | null} */ (root.querySelector('.bar-sheet'));
  if (existing) return existing;

  const sheet = make('div', 'bar-sheet');
  sheet.setAttribute('role', 'dialog');
  sheet.setAttribute('aria-label', 'Customize bars');

  const top = make('div', 'bar-sheet-top');
  top.append(
    make('span', 'bar-sheet-title', 'Customize bars'),
    // The design of record's three facts, verbatim. The middle one is the
    // reason it cannot be shortened: dragging an item out REMOVES it, there is
    // no confirmation, and this line is the only place that says so.
    make(
      'span',
      'bar-sheet-hint',
      'Drag items into the header or the status bar. Drag an item out to remove it. ' +
        'Click an item for its options.'
    )
  );
  const done = make('button', 'bar-btn is-primary bar-sheet-done', 'Done');
  /** @type {HTMLButtonElement} */ (done).type = 'button';
  done.addEventListener('click', () => controller?.exitEditMode());
  top.append(done);

  const foot = make('div', 'bar-sheet-foot');
  // The stock arrangements, first thing in the foot. A preset is an ordinary
  // edit — one PUT, same revision — so a pill is a button like any other.
  const presets = make('div', 'bar-sheet-presets');
  presets.append(make('span', 'bar-sheet-presets-label', 'Presets'));
  for (const preset of PRESETS) {
    const pill = make('button', 'bar-pill', preset.label);
    /** @type {HTMLButtonElement} */ (pill).type = 'button';
    pill.dataset.barPreset = preset.id;
    pill.title = preset.description;
    pill.addEventListener('click', () => {
      void controller?.applyPreset(preset.id);
    });
    presets.append(pill);
  }
  const check = make('label', 'bar-sheet-check');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.className = 'bar-sheet-status-visible';
  box.addEventListener('change', () => {
    void controller?.setStatusVisible(box.checked);
  });
  check.append(box, make('span', 'bar-sheet-check-label', 'Status bar'));
  const notice = make('p', 'bar-sheet-notice');
  notice.setAttribute('role', 'status');
  foot.append(
    presets,
    check,
    make('span', 'bar-sheet-note', 'Locked items can be moved, not removed.'),
    notice
  );

  sheet.append(top, make('div', 'bar-sheet-groups'), foot);
  // A document mounts the sheet in its body; an element root mounts it in
  // itself. Duck-typed rather than `instanceof Document`, which does not hold
  // across realms (and does not hold for happy-dom's document either).
  const parent = /** @type {any} */ (root).body ?? root;
  parent.append(sheet);
  return sheet;
}

/**
 * One tile. A refused type is a disabled button carrying its reason, so the
 * operator reads why it cannot be added rather than watching a click do
 * nothing.
 * @param {string} type
 * @param {SheetController} ctrl
 * @returns {HTMLElement}
 */
function buildTile(type, ctrl) {
  const host = ctrl.defaultHostFor(type);
  const refusal = ctrl.refusalFor(type, host);
  const tile = make('button', 'bar-tile');
  const button = /** @type {HTMLButtonElement} */ (tile);
  button.type = 'button';
  tile.dataset.barTile = type;
  tile.dataset.barTileHost = host;
  tile.append(make('span', 'bar-tile-label', BAR_CATALOG[type].label));
  if (refusal) {
    button.disabled = true;
    tile.title = refusal;
    tile.append(make('span', 'bar-tile-reason', refusal));
  } else {
    tile.addEventListener('click', () => {
      // A DRAG THAT ENDED ON THIS TILE IS NOT A CLICK ON IT. The drag begins
      // with `preventDefault()` on pointerdown, which suppresses the
      // compatibility mouse events but NOT the click the browser still
      // dispatches at the capture target on release — so without this guard one
      // drag-in gesture adds the item twice: once where it was dropped, and
      // once at the type's default host, whose save then conflicts with the
      // first and wins on the retry. The options popover reads the same flag.
      if (dragJustEnded()) return;
      void ctrl.addItem(type, host);
    });
  }
  return tile;
}

/**
 * Re-render the tiles and the status-bar toggle from the current document.
 * Called on every open and after every accepted edit, because a save changes
 * what the next edit may do — a host that just filled up refuses its tiles.
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function renderSheet(root = document) {
  const ctrl = controller;
  const sheet = /** @type {HTMLElement | null} */ (root.querySelector('.bar-sheet'));
  if (!ctrl || !sheet) return;

  const groups = /** @type {HTMLElement} */ (sheet.querySelector('.bar-sheet-groups'));
  groups.replaceChildren();
  for (const [heading, types] of [
    ['Items', BAR_ITEM_TYPES.filter((type) => !SPACING_TYPES.includes(type))],
    ['Spacing', BAR_ITEM_TYPES.filter((type) => SPACING_TYPES.includes(type))],
  ]) {
    const group = make('div', 'bar-sheet-group');
    group.append(make('h2', 'bar-sheet-group-heading', /** @type {string} */ (heading)));
    const tiles = make('div', 'bar-tiles');
    for (const type of /** @type {string[]} */ (types)) tiles.append(buildTile(type, ctrl));
    group.append(tiles);
    groups.append(group);
  }

  const box = /** @type {HTMLInputElement | null} */ (
    sheet.querySelector('.bar-sheet-status-visible')
  );
  if (box) box.checked = ctrl.statusVisible();
}

/**
 * Show the sheet. Rendering happens here rather than at build time so an
 * operator who left and re-entered edit mode sees the document as it now is.
 * @param {SheetController} ctrl
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function openSheet(ctrl, root = document) {
  const sheet = sheetElement(ctrl, root);
  renderSheet(root);
  sheetNotice('', root);
  sheet.classList.add('is-open');
}

/**
 * Hide the sheet, leaving it built. Nothing is torn down: the next open is a
 * re-render, and the notice element has to survive so a message that arrives
 * while the sheet reopens is not lost.
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function closeSheet(root = document) {
  /** @type {HTMLElement | null} */ (root.querySelector('.bar-sheet'))?.classList.remove(
    'is-open'
  );
}

/**
 * Remove the sheet and let go of the controller. The teardown entry point:
 * `closeSheet` hides an element that stays wired to this module's controller,
 * which is right between two edit sessions and wrong once the module that owns
 * the controller has been torn down.
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function destroySheet(root = document) {
  /** @type {HTMLElement | null} */ (root.querySelector('.bar-sheet'))?.remove();
  controller = null;
}

/**
 * Render a transient notice in the sheet's own foot — the surface that retires
 * bar-sync.js's inline pill. Empty text clears it.
 *
 * Every PUT this build issues originates in edit mode, so a save notice always
 * arrives while the sheet is open; there is no case where a message lands with
 * nowhere to render.
 * @param {string} text
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function sheetNotice(text, root = document) {
  const notice = /** @type {HTMLElement | null} */ (root.querySelector('.bar-sheet-notice'));
  if (notice) notice.textContent = text;
}
