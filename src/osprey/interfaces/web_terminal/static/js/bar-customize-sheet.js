/* OSPREY Web Terminal — the customize sheet.
 *
 * The panel that drops below the header in edit mode: one tile per catalog
 * type, filed under the catalog's own group headings, the status-bar
 * visibility toggle, the Default preset, Done, and the surface every "Layout
 * not saved" notice renders in.
 *
 * A TILE SHOWS THE ITEM, NOT A NAME FOR IT. Each tile carries a preview of the
 * body the item renders at header density — the same builder the bar uses for
 * a JS-built item, a copy of the server-rendered node for an adopted one — so
 * the operator sees what they are about to put in the bar. The label sits
 * under it.
 *
 * This module builds and re-renders DOM. It decides NOTHING: whether a type can
 * be added, and what to say when it cannot, are the controller's answers
 * (bar-customize.js), handed in once. That is what keeps the refusal rules in
 * one place — the sheet and the drag gesture ask the same function and get
 * the same sentence — and what lets this file be tested through the controller
 * that owns the write path rather than around it.
 *
 * A refused tile is DISABLED AND NAMED. The rule the design of record states
 * and `normalize()` cannot honour on its own: an edit that would be dropped is
 * refused where the operator made it, with the reason on the tile, instead of
 * being accepted and then quietly discarded on the way to the server. The one
 * refusal that is NOT named is "already in a bar": a single-node item that is
 * placed dims its tile instead, which is how the sheet reads as an inventory
 * — what is out, what is in — rather than as a list of complaints.
 */

import { BAR_CATALOG, BAR_GROUPS, BAR_HOSTS, BAR_ITEM_TYPES, barItemType } from './bar-catalog.js';
import { shellForKey } from './bar-host.js';
import { previewBarItem } from './bar-items.js';
import { dragJustEnded } from './bar-customize-drag.js';

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */

/** The two bars as the sheet's checkboxes name them. */
const HOST_LABEL = /** @type {Readonly<Record<BarHost, string>>} */ (
  Object.freeze({ header: 'Header', status: 'Status bar' })
);

/**
 * What the sheet is allowed to ask and to do: the whole edit controller, the
 * same object the drag gesture and the popovers are handed. Every answer is
 * read fresh at render time, so a re-render after a save reflects the saved
 * document.
 * @typedef {import('./bar-customize.js').BarEditController} SheetController
 */

/**
 * The display menu is a component that renders itself on connection and binds
 * its listeners to its own children, so its preview is a second INSTANCE
 * rather than a copy: a copied trigger outside the component's tag would lose
 * the component's own styling, and a copied component would carry a card that
 * no constructor ever wired.
 */
const DISPLAY_MENU_TAG = 'osprey-display-menu';

/** What an empty activity strip previews as: the strip's idle reading. */
const ACTIVITY_IDLE = 'idle';

/** The sheet is a singleton per page; the controller arrives with it. */
/** @type {SheetController | null} */
let controller = null;

/** Disposers for the live previews the tiles currently hold. @type {(() => void)[]} */
let previewDisposers = [];

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
  // The one preset: the arrangement this deployment configured (`web.bar_items`).
  // Applying it is a DELETE of the operator's own document, never a PUT of a
  // copy — only the server knows what the deployment configured.
  const presets = make('div', 'bar-sheet-presets');
  presets.append(make('span', 'bar-sheet-presets-label', 'Preset'));
  const pill = make('button', 'bar-pill', 'Default');
  /** @type {HTMLButtonElement} */ (pill).type = 'button';
  pill.dataset.barPreset = 'default';
  pill.title = 'The arrangement this deployment ships';
  pill.addEventListener('click', () => {
    void controller?.resetToDefault();
  });
  presets.append(pill);
  // One checkbox per bar. Either may be withdrawn; with both gone the command
  // palette's Customize bars is the way back in, and edit mode shows every bar.
  const checks = make('div', 'bar-sheet-checks');
  for (const host of BAR_HOSTS) {
    const check = make('label', 'bar-sheet-check');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.className = `bar-sheet-${host}-visible`;
    box.addEventListener('change', () => {
      void controller?.setBarVisible(host, box.checked);
    });
    check.append(box, make('span', 'bar-sheet-check-label', HOST_LABEL[host]));
    checks.append(check);
  }
  const notice = make('p', 'bar-sheet-notice');
  notice.setAttribute('role', 'status');
  foot.append(presets, checks, notice);

  sheet.append(top, make('div', 'bar-sheet-groups'), foot);
  // A document mounts the sheet in its body; an element root mounts it in
  // itself. Duck-typed rather than `instanceof Document`, which does not hold
  // across realms (and does not hold for happy-dom's document either).
  const parent = /** @type {any} */ (root).body ?? root;
  parent.append(sheet);
  return sheet;
}

/* ---- previews ---- */

/**
 * Strip what a copy of a live node must not carry: the ids other modules
 * resolve, and the `hidden` an item puts on itself until it has something to
 * show — a docs link waiting for its URL still previews as the docs link.
 * @param {Element} node
 */
function neutralize(node) {
  node.removeAttribute('id');
  node.removeAttribute('hidden');
  for (const child of node.querySelectorAll('[id], [hidden]')) {
    child.removeAttribute('id');
    child.removeAttribute('hidden');
  }
}

/**
 * A copy of an adopted item's server-rendered body, or null when this page
 * holds no node for it (a type this deployment does not render).
 * @param {string} type
 * @returns {Element | null}
 */
function copyOfLive(type) {
  // A separator has no body at all — it IS its shell's styling — so its
  // preview is a shell of the same type, which bars.css draws as the line.
  if (type === 'separator') {
    const line = make('span', 'bar-item');
    line.dataset.barItem = type;
    return line;
  }
  const shell = shellForKey(type);
  if (!shell || !shell.firstElementChild) return null;
  if (type === 'display' && shell.querySelector(DISPLAY_MENU_TAG)) {
    return document.createElement(DISPLAY_MENU_TAG);
  }
  const copy = /** @type {Element} */ (shell.firstElementChild.cloneNode(true));
  neutralize(copy);
  if (type === 'activity' && !copy.textContent?.trim()) copy.textContent = ACTIVITY_IDLE;
  return copy;
}

/**
 * The preview for one type: a fresh instance from the type's builder where
 * one exists, else a copy of the live node.
 * @param {string} type
 * @returns {Node | null}
 */
function preview(type) {
  const built = previewBarItem(type, document, 'comfortable');
  if (built) {
    previewDisposers.push(built.dispose ?? (() => {}));
    return built.node;
  }
  return copyOfLive(type);
}

/** Stop every live preview the last render started. */
function disposePreviews() {
  for (const dispose of previewDisposers) {
    try {
      dispose();
    } catch (err) {
      console.error('[bar-sheet] preview disposer threw', err);
    }
  }
  previewDisposers = [];
}

/* ---- tiles ---- */

/**
 * One tile: the preview over the label. A refused type is a disabled button
 * carrying its reason, so the operator reads why it cannot be added rather
 * than watching a click do nothing; a single-node type that is already placed
 * is dimmed instead, and says where it is on hover.
 * @param {string} type
 * @param {SheetController} ctrl
 * @returns {HTMLElement}
 */
function buildTile(type, ctrl) {
  const entry = barItemType(type);
  const host = ctrl.defaultHostFor();
  const tile = make('button', 'bar-tile');
  const button = /** @type {HTMLButtonElement} */ (tile);
  button.type = 'button';
  tile.dataset.barTile = type;
  tile.dataset.barTileHost = host;

  const body = make('span', 'bar-tile-body');
  body.dataset.barDensity = 'comfortable';
  const shown = preview(type);
  if (shown) body.append(shown);
  tile.append(body, make('span', 'bar-tile-label', BAR_CATALOG[type].label));

  const sitting = entry && !entry.multi ? ctrl.placedIn(type) : null;
  if (sitting) {
    tile.classList.add('is-in-bar');
    tile.setAttribute('aria-disabled', 'true');
    tile.title = ctrl.refusalFor(type, host) ?? '';
    return tile;
  }
  const refusal = ctrl.refusalFor(type, host);
  if (refusal) {
    button.disabled = true;
    tile.title = refusal;
    tile.append(make('span', 'bar-tile-reason', refusal));
    return tile;
  }
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
  return tile;
}

/**
 * Re-render the tiles and the status-bar toggle from the current document.
 * Called on every open and after every accepted edit, because a save changes
 * what the next edit may do — a host that just filled up refuses its tiles,
 * and an item that just went in dims its own.
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function renderSheet(root = document) {
  const ctrl = controller;
  const sheet = /** @type {HTMLElement | null} */ (root.querySelector('.bar-sheet'));
  if (!ctrl || !sheet) return;

  disposePreviews();
  const groups = /** @type {HTMLElement} */ (sheet.querySelector('.bar-sheet-groups'));
  groups.replaceChildren();
  for (const heading of BAR_GROUPS) {
    const types = BAR_ITEM_TYPES.filter((type) => BAR_CATALOG[type].group === heading);
    if (types.length === 0) continue;
    const group = make('div', 'bar-sheet-group');
    group.append(make('h2', 'bar-sheet-group-heading', heading));
    const tiles = make('div', 'bar-tiles');
    for (const type of types) tiles.append(buildTile(type, ctrl));
    group.append(tiles);
    groups.append(group);
  }

  for (const host of BAR_HOSTS) {
    const box = /** @type {HTMLInputElement | null} */ (
      sheet.querySelector(`.bar-sheet-${host}-visible`)
    );
    if (box) box.checked = ctrl.barVisible(host);
  }

  // A read-only document refuses every tile with the same words. Said once,
  // with the way out beside it — the one edit the latch still allows.
  if (ctrl.readonly()) sheetNotice('Layout not editable. Default resets it.', root);
}

/**
 * Show the sheet. Rendering happens here rather than at build time so an
 * operator who left and re-entered edit mode sees the document as it now is.
 * @param {SheetController} ctrl
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function openSheet(ctrl, root = document) {
  const sheet = sheetElement(ctrl, root);
  sheetNotice('', root);
  renderSheet(root);
  sheet.classList.add('is-open');
}

/**
 * Hide the sheet, leaving it built. The previews stop — a clock ticking in a
 * hidden sheet is a timer spent on nothing — and the next open re-renders
 * them. The notice element survives so a message that arrives while the sheet
 * reopens is not lost.
 * @param {ParentNode & {querySelector: Function}} [root]
 */
export function closeSheet(root = document) {
  disposePreviews();
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
  disposePreviews();
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
