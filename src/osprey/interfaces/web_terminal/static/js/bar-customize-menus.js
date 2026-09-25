/* OSPREY Web Terminal — the item popovers.
 *
 * Two surfaces that hang off a placed item: the OPTIONS popover (click an item
 * while editing) and the CONTEXT MENU (right-click a bar or an item). Both are
 * built here and both decide nothing — every question goes to the controller,
 * so a row offered here and a tile in the sheet refuse the same edits for the
 * same stated reason.
 *
 * THE OPTION ROWS COME FROM THE CATALOG'S SPEC. There is no table of option
 * kinds in this file: a type that declares a bounded number gets a number
 * input carrying those bounds, an enum gets one button per value, a boolean
 * gets a checkbox, and a type with no options says so. That is what keeps the
 * bounds honest — the client, the route validator and this popover are reading
 * the same declaration, and the clamp that keeps an out-of-spec value off the
 * wire lives with the write path rather than in this markup.
 *
 * THE FOCUS GOES IN AND COMES BACK. Opening the popover puts the focus on its
 * first control, and closing it with Escape or one of its own buttons gives
 * the focus back to the item it was opened from. A popover that re-opens after
 * an accepted edit (an option set, Move left, Move right) puts the focus back
 * on the control that had it, so a repeated key press repeats the edit.
 *
 * ONE POPOVER AT A TIME, AND THE HOST CAN CLOSE IT. Every open registers with
 * bar-host.js, which closes an item's popover before a reconcile moves it —
 * the same invariant the adopted chrome keeps.
 *
 * ESCAPE IS SPLIT BY STATE, NOT BY LISTENER ORDER. Both this module and
 * bar-customize.js listen for Escape, and only one of them may act on a given
 * press. WHILE EDITING the decision is bar-customize.js's: it owns the key,
 * asks `barSurfaceOpen()` whether one of these two surfaces is up, and closes
 * that before it closes edit mode — so the first Escape puts a popover away
 * rather than ending the edit. This module's own handler therefore returns
 * without acting whenever `ctrl.isEditing()`, and only handles Escape outside
 * edit mode, where nothing else wants the key. The order used to be held by
 * registration order instead, which broke the moment either side was re-armed
 * mid-edit and reversed who ran first.
 */

import { BAR_CATALOG, DENSITY_BY_HOST } from './bar-catalog.js';
import { docOf, isLive, registerBarPopover, shellForKey } from './bar-host.js';
import { dragJustEnded } from './bar-customize-drag.js';

/** @typedef {import('./bar-catalog.js').BarHost} BarHost */
/** @typedef {import('./bar-catalog.js').BarOptionSpec} BarOptionSpec */
/** @typedef {import('./bar-customize.js').BarEditController} BarEditController */
/** @typedef {import('./bar-customize.js').BarItemPlace} BarItemPlace */

/** How each host is named to the operator. */
const HOST_LABEL = /** @type {Readonly<Record<BarHost, string>>} */ (
  Object.freeze({ header: 'Header', status: 'Status bar' })
);

/** How each density is named to the operator. */
const DENSITY_LABEL = Object.freeze({ comfortable: 'Comfortable', compact: 'Compact' });

/** Undo the document listeners. @type {(() => void) | null} */
let stopListening = null;

/** Close the open options popover, if there is one. @type {(() => void) | null} */
let closeOpenOptions = null;

/** The open options popover. @type {HTMLElement | null} */
let optionsNode = null;

/**
 * One control in the options popover, named by what it does rather than by
 * node, so the same control can be found again in a rebuilt popover.
 * @typedef {{option?: string, value?: string, action?: string}} PopControl
 */

/** Where the focus goes when the options popover closes. @type {HTMLElement | null} */
let optionsReturn = null;

/** The popover control that last had the focus. @type {PopControl | null} */
let optionsFocus = null;

/** The open context menu. @type {HTMLElement | null} */
let openMenuNode = null;

/** Where the focus goes when the menu closes. @type {HTMLElement | null} */
let menuReturn = null;

/**
 * Make an element with a class and, optionally, text. Text only — nothing here
 * interpolates markup.
 * @param {Document} owner
 * @param {string} tag
 * @param {string} className
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function make(owner, tag, className, text) {
  const node = owner.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * A button that does one thing.
 * @param {Document} owner
 * @param {string} className
 * @param {string} text
 * @param {string} action
 * @param {() => void} run
 * @returns {HTMLButtonElement}
 */
function button(owner, className, text, action, run) {
  const node = /** @type {HTMLButtonElement} */ (make(owner, 'button', className, text));
  node.type = 'button';
  node.dataset.barAction = action;
  node.addEventListener('click', run);
  return node;
}

/* ---- the options popover ---- */

/**
 * One option row, rendered from the type's own spec.
 * @param {Document} owner
 * @param {string} key
 * @param {BarOptionSpec} spec
 * @param {string | number | boolean} value
 * @param {(value: string | number | boolean) => void} commit
 * @returns {HTMLElement}
 */
function optionRow(owner, key, spec, value, commit) {
  const row = make(owner, 'div', 'bar-option');
  row.dataset.barOption = key;
  row.append(make(owner, 'span', 'bar-option-label', key[0].toUpperCase() + key.slice(1)));

  if (spec.kind === 'enum') {
    const seg = make(owner, 'div', 'bar-seg');
    for (const option of spec.values) {
      const pick = button(owner, 'bar-seg-option', option, 'option-value', () => commit(option));
      pick.dataset.barValue = option;
      pick.setAttribute('aria-pressed', String(String(value) === option));
      seg.append(pick);
    }
    row.append(seg);
    return row;
  }

  if (spec.kind === 'boolean') {
    const label = make(owner, 'label', 'bar-check');
    const box = owner.createElement('input');
    box.type = 'checkbox';
    box.checked = value === true;
    box.addEventListener('change', () => commit(box.checked));
    label.append(box);
    row.append(label);
    return row;
  }

  const input = owner.createElement('input');
  input.type = 'number';
  input.className = 'bar-input';
  input.min = String(spec.min);
  input.max = String(spec.max);
  input.step = String(spec.step);
  input.value = String(value);
  if (spec.unit) input.setAttribute('aria-label', `${key} in ${spec.unit}`);
  input.addEventListener('change', () => commit(input.value));
  row.append(input);
  return row;
}

/**
 * The popover's foot: where the item can go, within its bar and across, and
 * whether it can be taken away.
 * @param {Document} owner
 * @param {BarEditController} ctrl
 * @param {BarItemPlace} place
 * @returns {HTMLElement}
 */
function optionsFoot(owner, ctrl, place) {
  const foot = make(owner, 'div', 'bar-pop-foot');
  const run = ctrl.hostItems(place.host);
  const reorder = !ctrl.dropRefusal(place.type, place.host, place.host);
  if (reorder && place.index > 0) {
    foot.append(
      button(owner, 'bar-btn', 'Move left', 'move-left', () => {
        void moveAndFollow(ctrl, place, place.index - 1, place.index - 1, 'move-left');
      })
    );
  }
  if (reorder && place.index < run.length - 1) {
    foot.append(
      button(owner, 'bar-btn', 'Move right', 'move-right', () => {
        // moveItem reads the target index before it removes the item, so the
        // slot after the right-hand neighbour is two past this one.
        void moveAndFollow(ctrl, place, place.index + 2, place.index + 1, 'move-right');
      })
    );
  }
  const other = /** @type {BarHost} */ (place.host === 'header' ? 'status' : 'header');
  if (!ctrl.dropRefusal(place.type, other, place.host)) {
    foot.append(
      button(owner, 'bar-btn', `Move to ${HOST_LABEL[other].toLowerCase()}`, 'move', () => {
        void editThenFocus(
          ctrl,
          () => ctrl.moveItem(place.host, place.index, other, Number.MAX_SAFE_INTEGER),
          () => ({ host: other, index: ctrl.hostItems(other).length - 1 })
        );
      })
    );
  }
  foot.append(
    button(owner, 'bar-btn is-danger', 'Remove', 'remove', () => {
      // The item is gone, so the focus goes to the one that took its place,
      // or to the one before it when it was last.
      void editThenFocus(
        ctrl,
        () => ctrl.removeAt(place.host, place.index),
        () => ({
          host: place.host,
          index: Math.min(place.index, ctrl.hostItems(place.host).length - 1),
        })
      );
    })
  );
  return foot;
}

/**
 * Open one item's options.
 *
 * An accepted edit re-opens the popover on the item's shell: the save
 * reconciles, which rebuilds a body whose options changed, and an operator
 * setting two options in a row should not have to click the item again.
 *
 * The focus moves to `how.focus` when the new popover has that control, and to
 * its first control otherwise. It goes back to `how.returnTo` on close when
 * that is still inside this item, and otherwise to what had the focus inside
 * the item when it opened, or to the item's own first control.
 * @param {HTMLElement} shell
 * @param {BarEditController} ctrl
 * @param {{focus?: PopControl | null, returnTo?: HTMLElement | null}} [how]
 */
export function openOptions(shell, ctrl, how = {}) {
  const owner = shell.ownerDocument;
  const returnTo = returnPointIn(shell, how.returnTo ?? null, owner.activeElement);
  closeOptions({ restoreFocus: false });
  optionsFocus = null;
  const place = ctrl.locate(shell);
  const item = place && ctrl.itemAt(place.host, place.index);
  if (!place || !item) return;
  const key = shell.dataset.barKey ?? '';
  const entry = BAR_CATALOG[place.type];

  const pop = make(owner, 'div', 'bar-pop bar-options');
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', `${entry.label} options`);
  // Anchored to whichever side keeps it on screen: an item in the left half of
  // the window opens rightward, one in the right half opens leftward.
  const box = shell.getBoundingClientRect();
  if (box.left < (owner.defaultView?.innerWidth ?? 0) / 2) pop.classList.add('is-left');

  pop.append(make(owner, 'div', 'bar-pop-eyebrow', entry.label));
  pop.append(
    make(
      owner,
      'div',
      'bar-pop-density',
      `Density · ${DENSITY_LABEL[DENSITY_BY_HOST[place.host]]}`
    )
  );

  const keys = Object.keys(entry.options);
  for (const optionKey of keys) {
    pop.append(
      optionRow(owner, optionKey, entry.options[optionKey], item.options[optionKey], (value) => {
        void commitOption(ctrl, place, optionKey, value, key);
      })
    );
  }
  if (keys.length === 0) pop.append(make(owner, 'div', 'bar-pop-note', 'No options'));
  // The width field's one convention, stated beside it: zero is the flexible
  // space. The grip on the item itself is the other way to set this.
  if (place.type === 'space') {
    pop.append(make(owner, 'div', 'bar-pop-note', '0 fills the remaining room'));
  }
  pop.append(optionsFoot(owner, ctrl, place));

  shell.append(pop);
  pop.addEventListener('focusin', (event) => {
    optionsFocus = controlOf(/** @type {Element} */ (event.target));
  });
  // The host closes the popover while a reconcile moves or rebuilds the item,
  // and the focus is the reconcile's to keep then, not this popover's.
  const unregister = registerBarPopover(shell, () => closeOptions({ restoreFocus: false }));
  optionsReturn = returnTo;
  optionsNode = pop;
  closeOpenOptions = () => {
    unregister();
    pop.remove();
    optionsNode = null;
    closeOpenOptions = null;
  };
  (findControl(pop, how.focus ?? null) ?? controlsIn(pop)[0])?.focus({ preventScroll: true });
}

/**
 * Where the focus goes back to when a popover over `shell` closes: `preferred`
 * when it is still inside the item, else whatever had the focus inside the
 * item, else the item's first control. Null when the item has none.
 * @param {HTMLElement} shell
 * @param {HTMLElement | null} preferred
 * @param {Element | null} active
 * @returns {HTMLElement | null}
 */
function returnPointIn(shell, preferred, active) {
  /** @param {unknown} node */
  const ownControl = (node) =>
    node instanceof HTMLElement && shell.contains(node) && !node.closest('.bar-pop');
  for (const node of [preferred, active]) {
    if (ownControl(node)) return /** @type {HTMLElement} */ (node);
  }
  const first = Array.from(shell.querySelectorAll(FOCUSABLE)).find(ownControl);
  return /** @type {HTMLElement | undefined} */ (first) ?? null;
}

/** What in an item or a popover can take the focus from the keyboard. */
const FOCUSABLE =
  'button:not([disabled]), a[href], input:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** The popover's controls, in order. @param {HTMLElement} pop @returns {HTMLElement[]} */
function controlsIn(pop) {
  return /** @type {HTMLElement[]} */ (Array.from(pop.querySelectorAll(FOCUSABLE)));
}

/**
 * Name a popover control by what it does.
 * @param {Element} node
 * @returns {PopControl | null}
 */
function controlOf(node) {
  const row = /** @type {HTMLElement | null} */ (node.closest('[data-bar-option]'));
  const own = /** @type {HTMLElement} */ (node);
  if (row) return { option: row.dataset.barOption, value: own.dataset?.barValue };
  if (own.dataset?.barAction) return { action: own.dataset.barAction };
  return null;
}

/**
 * The control in `pop` that `want` names, or null.
 * @param {HTMLElement} pop
 * @param {PopControl | null} want
 * @returns {HTMLElement | null}
 */
function findControl(pop, want) {
  if (!want) return null;
  for (const node of controlsIn(pop)) {
    const have = controlOf(node);
    if (!have) continue;
    const same = want.action
      ? have.action === want.action
      : have.option === want.option && (want.value === undefined || have.value === want.value);
    if (same) return node;
  }
  return null;
}

/**
 * Run one of the foot's edits that closes the popover, then give the focus to
 * the item where the stored layout now has it. The item's body can be rebuilt
 * on the way (a move to the other bar changes its density), so the control that
 * had the focus is looked for inside the item at its new place, and the item's
 * first control stands in when it is gone. An operator who has put the focus
 * somewhere else in the meantime keeps it there.
 * @param {BarEditController} ctrl
 * @param {() => Promise<boolean>} edit
 * @param {() => {host: BarHost, index: number}} landing - read after the edit
 * @returns {Promise<void>}
 */
async function editThenFocus(ctrl, edit, landing) {
  const back = optionsReturn;
  closeOptions();
  if (!(await edit())) return;
  const where = landing();
  const key = where.index >= 0 ? ctrl.keyAt(where.host, where.index) : null;
  const shell = key ? shellForKey(key) : null;
  if (!shell || !isLive(shell)) return;
  const doc = shell.ownerDocument;
  const active = doc.activeElement;
  const free =
    !active ||
    active === doc.body ||
    !active.isConnected ||
    !isLive(active) ||
    shell.contains(active);
  if (free) returnPointIn(shell, back, active)?.focus({ preventScroll: true });
}

/**
 * Move an item within its bar, then re-open the popover on it at its new place
 * with the same button focused, so the next press moves it one further. The
 * key is looked up at the place the item landed: an item that passes another
 * of its own type takes that one's key.
 * @param {BarEditController} ctrl
 * @param {BarItemPlace} place
 * @param {number} toIndex - the index handed to `moveItem`
 * @param {number} landsAt - where the item sits once the move is stored
 * @param {string} action
 * @returns {Promise<void>}
 */
async function moveAndFollow(ctrl, place, toIndex, landsAt, action) {
  const back = optionsReturn;
  closeOptions();
  if (!(await ctrl.moveItem(place.host, place.index, place.host, toIndex))) return;
  const key = ctrl.keyAt(place.host, landsAt);
  const shell = key ? shellForKey(key) : null;
  if (shell && isLive(shell)) openOptions(shell, ctrl, { focus: { action }, returnTo: back });
}

/**
 * Write one option, then put the popover back over the item it belongs to.
 * @param {BarEditController} ctrl
 * @param {BarItemPlace} place
 * @param {string} optionKey
 * @param {string | number | boolean} value
 * @param {string} itemKey
 * @returns {Promise<void>}
 */
async function commitOption(ctrl, place, optionKey, value, itemKey) {
  const back = optionsReturn;
  const stored = await ctrl.setOption(place.host, place.index, optionKey, value);
  if (!stored) return;
  const shell = shellForKey(itemKey);
  const focus = optionsFocus ?? { option: optionKey };
  if (shell && isLive(shell)) openOptions(shell, ctrl, { focus, returnTo: back });
}

/**
 * Close the options popover, if one is open. The focus goes back to the item
 * the popover was opened from when it was inside the popover or nowhere; an
 * operator who has already put it somewhere else keeps it there.
 * @param {{restoreFocus?: boolean}} [how] - false when the caller owns the
 *   focus: the pointer put it somewhere, or a reconcile is moving the item
 */
export function closeOptions({ restoreFocus = true } = {}) {
  const back = optionsReturn;
  const doc = optionsNode?.ownerDocument;
  const active = doc?.activeElement ?? null;
  const wasInside = !!active && !!optionsNode?.contains(active);
  optionsReturn = null;
  closeOpenOptions?.();
  if (!restoreFocus || !back || !doc || !back.isConnected || !isLive(back)) return;
  if (wasInside || !active || active === doc.body) back.focus({ preventScroll: true });
}

/* ---- the context menu ---- */

/**
 * Open the context menu at a point, for an item or for the bar itself.
 *
 * Customize (or Done) on either bar, and Hide/Show for THE BAR UNDER THE
 * POINTER: the row names the bar it acts on, and offered from the other bar it
 * read as "hide this one". Hiding needs no sheet to answer into (it cannot be
 * refused), so a bar offers it before edit mode as well as during it. A hidden
 * bar is not stranded: every tile header's menu offers Show for it
 * (panel-menu-policy.js), as does the command palette's Customize bars, and
 * edit mode shows every bar. An item under the pointer while editing adds its
 * own two rows above: its options, and Remove.
 * @param {Document} owner
 * @param {BarEditController} ctrl
 * @param {number} x
 * @param {number} y
 * @param {HTMLElement | null} shell
 * @param {BarHost | null} host - the bar under the pointer
 */
function openMenu(owner, ctrl, x, y, shell, host) {
  closeMenu();
  closeOptions();
  const editing = ctrl.isEditing();
  const menu = make(owner, 'div', 'bar-context-menu');
  menu.setAttribute('role', 'menu');

  const place = editing && shell ? ctrl.locate(shell) : null;
  if (place && shell) {
    const label = BAR_CATALOG[place.type].label;
    menu.append(
      // No `enterEditMode()` here: this row only exists when `place` is set,
      // and `place` is only computed while editing.
      button(owner, 'bar-context-row', `${label} options…`, 'options', () => {
        closeMenu();
        const live = shellForKey(shell.dataset.barKey ?? '');
        if (live) openOptions(live, ctrl);
      })
    );
    menu.append(
      button(owner, 'bar-context-row', `Remove ${label}`, 'remove', () => {
        closeMenu();
        void ctrl.removeAt(place.host, place.index);
      })
    );
    menu.append(make(owner, 'hr', 'bar-context-rule'));
  }

  menu.append(
    button(
      owner,
      'bar-context-row',
      editing ? 'Done customizing' : 'Customize bars…',
      'customize',
      () => {
        closeMenu();
        if (editing) ctrl.exitEditMode();
        else ctrl.enterEditMode();
      }
    )
  );
  if (host) {
    const visible = ctrl.barVisible(host);
    const name = HOST_LABEL[host].toLowerCase();
    menu.append(
      button(owner, 'bar-context-row', visible ? `Hide ${name}` : `Show ${name}`, host, () => {
        closeMenu();
        void ctrl.setBarVisible(host, !visible);
      })
    );
  }

  // ARIA containment, stamped in one place so a row added later cannot forget
  // it: everything inside a `role="menu"` has to declare what it is.
  for (const row of menu.querySelectorAll('.bar-context-row')) {
    row.setAttribute('role', 'menuitem');
  }
  for (const rule of menu.querySelectorAll('.bar-context-rule')) {
    rule.setAttribute('role', 'separator');
  }
  for (const note of menu.querySelectorAll('.bar-context-note')) {
    note.setAttribute('role', 'presentation');
  }

  owner.body?.append(menu);
  // Clamped to the window: a right-click near the edge must not open a menu
  // half of which is off screen.
  const view = owner.defaultView;
  const right = (view?.innerWidth ?? 0) - menu.offsetWidth - 8;
  const bottom = (view?.innerHeight ?? 0) - menu.offsetHeight - 8;
  menu.style.left = `${Math.max(0, Math.min(x, right))}px`;
  menu.style.top = `${Math.max(0, Math.min(y, bottom))}px`;
  openMenuNode = menu;
  // A menu that opens under the pointer must also be reachable from the
  // keyboard: focus moves in, Up/Down walk the rows (armMenus), and the focus
  // goes back where it came from when the menu closes.
  menuReturn = /** @type {HTMLElement | null} */ (owner.activeElement);
  menuRows(menu)[0]?.focus();
}

/** The focusable rows of a menu. @param {HTMLElement} menu */
function menuRows(menu) {
  return /** @type {HTMLElement[]} */ (Array.from(menu.querySelectorAll('[role="menuitem"]')));
}

/**
 * Move the focus one row on, wrapping at both ends.
 * @param {HTMLElement} menu
 * @param {number} step
 */
function focusRow(menu, step) {
  const rows = menuRows(menu);
  if (rows.length === 0) return;
  const at = rows.indexOf(/** @type {HTMLElement} */ (menu.ownerDocument.activeElement));
  const next = at < 0 ? 0 : (at + step + rows.length) % rows.length;
  rows[next].focus();
}

/** Close the context menu, if one is open, and give the focus back. */
export function closeMenu() {
  if (!openMenuNode) return;
  openMenuNode.remove();
  openMenuNode = null;
  menuReturn?.focus?.();
  menuReturn = null;
}

/* ---- arming ---- */

/**
 * Listen for the gestures that open these surfaces. Armed by the mode gate,
 * not by edit mode: right-clicking a bar is one of the ways INTO edit mode, so
 * it has to work before there is anything to edit. What edit mode gates is the
 * plain click — outside it, a click on an item belongs to the item.
 * @param {BarEditController} ctrl
 * @param {Document | Element} root
 */
export function armMenus(ctrl, root) {
  disarmMenus();
  const owner = docOf(root);

  /** @param {Event} event */
  const onClick = (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest('.bar-pop, .bar-context-menu')) return;
    closeMenu();
    const shell = /** @type {HTMLElement | null} */ (
      target?.closest('.bar-item[data-bar-item]') ?? null
    );
    if (!shell || !isLive(shell) || !ctrl.isEditing() || dragJustEnded()) {
      closeOptions({ restoreFocus: false });
      return;
    }
    openOptions(shell, ctrl);
  };

  /** @param {Event} event */
  const onContextMenu = (event) => {
    const pointer = /** @type {MouseEvent} */ (event);
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;
    const shell = /** @type {HTMLElement | null} */ (
      target.closest('.bar-item[data-bar-item]') ?? null
    );
    const bar = /** @type {HTMLElement | null} */ (target.closest('[data-bar-host]'));
    if (!bar && !(shell && isLive(shell))) return;
    event.preventDefault();
    const host = /** @type {BarHost | null} */ (bar?.dataset.barHost ?? null);
    openMenu(
      owner,
      ctrl,
      pointer.clientX,
      pointer.clientY,
      shell && isLive(shell) ? shell : null,
      host
    );
  };

  /** @param {Event} event */
  const onKeydown = (event) => {
    const key = /** @type {KeyboardEvent} */ (event).key;
    if (openMenuNode && (key === 'ArrowDown' || key === 'ArrowUp')) {
      event.preventDefault();
      focusRow(openMenuNode, key === 'ArrowDown' ? 1 : -1);
      return;
    }
    if (key !== 'Escape' || !barSurfaceOpen()) return;
    // Escape closes what is open before it closes edit mode — but WHILE editing
    // that decision belongs to bar-customize.js, which owns the same key and
    // asks this module whether anything is open. Splitting it by state rather
    // than by listener order is what stops the two racing when either side is
    // re-armed.
    if (ctrl.isEditing()) return;
    closeBarSurfaces();
  };

  owner.addEventListener('click', onClick);
  owner.addEventListener('contextmenu', onContextMenu);
  owner.addEventListener('keydown', onKeydown);
  stopListening = () => {
    owner.removeEventListener('click', onClick);
    owner.removeEventListener('contextmenu', onContextMenu);
    owner.removeEventListener('keydown', onKeydown);
  };
}

/**
 * Whether either of this module's surfaces is open. Asked by whoever owns the
 * Escape key at that moment.
 * @returns {boolean}
 */
export function barSurfaceOpen() {
  return !!openMenuNode || !!closeOpenOptions;
}

/** Put both surfaces away, leaving the listeners armed. */
export function closeBarSurfaces() {
  closeMenu();
  closeOptions();
}

/** Stop listening and put both surfaces away. */
export function disarmMenus() {
  closeBarSurfaces();
  stopListening?.();
  stopListening = null;
}
