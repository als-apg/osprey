/* OSPREY Web Terminal — the ways into edit mode.
 *
 * Three of them: right-click either bar, the row this module projects into the
 * display menu beside the View toggle, and the palette action (wired where the
 * palette's actions are assembled, palette-boot.js). All three are EXPERT ONLY.
 *
 * THE MODE AXIS IS NOT THE LAYOUT AXIS. The saved arrangement renders in both
 * modes — the bars are the operator's chrome, not the mode's — and Simple mode
 * takes away only the ways to REARRANGE it. So nothing here touches a layout,
 * a document or a bar: it mounts and unmounts entry points, and edit mode's own
 * guard stays the backstop underneath them.
 *
 * ABSENT, NOT INERT. A control that is visible and does nothing is worse than
 * no control, so Simple mode removes the row and unbinds the gestures rather
 * than disabling them. The mode is read from `html[data-ui-mode]` — the same
 * authoritative attribute mode-boot.js stamps pre-paint and the display menu's
 * View row flips at runtime — and watched, because a flip must take the entry
 * points with it.
 */

import { docOf } from './bar-host.js';
import { armMenus, disarmMenus } from './bar-customize-menus.js';

/** @typedef {import('./bar-customize.js').BarEditController} BarEditController */

/** The class that marks the row this module owns inside the display menu. */
const ROW_CLASS = 'bar-customize-entry';

/** @type {MutationObserver | null} */
let watcher = null;

/** The mode the entry points are currently built for. @type {string | null} */
let lastMode = null;

/**
 * Which generation of entry points is current. Bumped by BOTH `initEntryPoints`
 * and `stopEntryPoints`, so anything still in flight from a previous one can
 * tell that it is stale.
 *
 * The only thing that outlives a teardown is `mountWhenReady`'s `whenDefined`
 * promise: it is settled by the browser, not by us, and there is nothing to
 * cancel. Without this, a teardown between the arm and the upgrade would still
 * mount the row — into a page the module has already unmounted, where nothing
 * will ever take it away again, because the next `unmountRow` has already run.
 */
let generation = 0;

/** Whether this page is in Expert mode. @param {Document} owner */
function isExpert(owner) {
  return owner.documentElement.getAttribute('data-ui-mode') !== 'simple';
}

/**
 * Put the Customize row in the display menu's action row — the component's own
 * projection point, beside the View toggle that renders in both modes. Built
 * here rather than declared in the template because the row must come and go
 * with the mode, and the template is rendered once.
 * @param {Document} owner
 * @param {BarEditController} ctrl
 * @returns {boolean} whether the row is now mounted
 */
function mountRow(owner, ctrl) {
  const menu = /** @type {any} */ (owner.querySelector('osprey-display-menu'));
  const actions = menu?.querySelector('.display-menu-actions');
  if (!menu) return false;
  if (actions?.querySelector(`.${ROW_CLASS}`)) return true;
  if (!actions) return false;
  const row = owner.createElement('button');
  // `display-menu-settings` is the component's skin for an action-row button,
  // the same one its own Settings and Log out rows wear.
  row.className = `display-menu-settings ${ROW_CLASS}`;
  row.type = 'button';
  row.textContent = 'Customize bars';
  row.title = 'Rearrange the header and status bar';
  row.addEventListener('click', () => {
    menu?.closeMenu?.();
    ctrl.enterEditMode();
  });
  actions.prepend(row);
  return true;
}

/**
 * Mount the row now if the display menu is ready, and otherwise as soon as it
 * is.
 *
 * The projection point does not exist until `osprey-display-menu.js` has run
 * `customElements.define()` and the element has upgraded — and NOTHING
 * guarantees that module is evaluated before this one. ES modules evaluate
 * depth-first in import order, so any file that imports the bar stack early
 * (palette-boot.js does, for the palette's Customize action) pulls this
 * module's boot ahead of the component's definition. A row that mounted only on
 * the synchronous first try would then be silently missing for the life of the
 * page, since nothing re-stamps `data-ui-mode` on an ordinary load.
 *
 * `whenDefined` is the fix rather than an import reorder because it cannot be
 * undone by a later refactor: the upgrade runs inside `define()`, so by the
 * time this callback is reached the card and its action row exist.
 * @param {Document} owner
 * @param {BarEditController} ctrl
 */
function mountWhenReady(owner, ctrl) {
  if (mountRow(owner, ctrl)) return;
  const registry = owner.defaultView?.customElements;
  if (!registry?.whenDefined) return;
  const mine = generation;
  registry
    .whenDefined('osprey-display-menu')
    .then(() => {
      if (mine !== generation) return;
      if (isExpert(owner)) mountRow(owner, ctrl);
    })
    .catch(() => {
      // No display menu on this page; there is nothing to project into.
    });
}

/** Take the Customize row away. @param {Document} owner */
function unmountRow(owner) {
  for (const row of owner.querySelectorAll(`.${ROW_CLASS}`)) row.remove();
}

/**
 * Bring the entry points into line with the current mode.
 *
 * A mutation that did not actually change the mode is ignored: re-arming would
 * tear down and rebuild the listeners, closing whatever popover the operator
 * has open, for a stamp that said nothing new.
 * @param {BarEditController} ctrl
 * @param {Document | Element} root
 * @param {boolean} [force] - run even when the mode has not changed (first sync)
 */
function sync(ctrl, root, force = false) {
  const owner = docOf(root);
  const mode = isExpert(owner) ? 'expert' : 'simple';
  if (!force && mode === lastMode) return;
  lastMode = mode;
  if (mode === 'expert') {
    armMenus(ctrl, root);
    mountWhenReady(owner, ctrl);
    return;
  }
  // Leaving Expert with the sheet open would strand the operator in an edit
  // mode they can no longer reach.
  if (ctrl.isEditing()) ctrl.exitEditMode();
  disarmMenus();
  unmountRow(owner);
}

/**
 * Mount the entry points and follow the mode from here on.
 * @param {BarEditController} ctrl
 * @param {Document | Element} root
 */
export function initEntryPoints(ctrl, root) {
  stopEntryPoints();
  generation += 1;
  const owner = docOf(root);
  sync(ctrl, root, true);
  if (typeof MutationObserver === 'undefined') return;
  watcher = new MutationObserver(() => sync(ctrl, root));
  watcher.observe(owner.documentElement, {
    attributes: true,
    attributeFilter: ['data-ui-mode'],
  });
}

/** Take every entry point away and stop watching the mode. */
export function stopEntryPoints() {
  generation += 1;
  watcher?.disconnect();
  watcher = null;
  lastMode = null;
  disarmMenus();
  if (typeof document !== 'undefined') unmountRow(document);
}
