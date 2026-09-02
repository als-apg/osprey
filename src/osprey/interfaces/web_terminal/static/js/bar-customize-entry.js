/* OSPREY Web Terminal — the ways into edit mode.
 *
 * Three of them: right-click either bar, the row this module projects into the
 * display menu beside the View toggle, and the palette action (wired where the
 * palette's actions are assembled, palette-boot.js). All three are present in
 * BOTH ui modes.
 *
 * THE MODE AXIS IS NOT THE LAYOUT AXIS. The saved arrangement renders in both
 * modes, and so do the ways to rearrange it: the bars are the operator's
 * chrome, not the mode's. Simple mode simplifies the workspace — one service
 * tile, the operator console in place of the terminal — and leaves the
 * operator's own chrome exactly as editable as it is in Expert. So nothing here
 * reads `data-ui-mode` at all; it mounts the entry points once and takes them
 * away on teardown.
 */

import { docOf } from './bar-host.js';
import { armMenus, disarmMenus } from './bar-customize-menus.js';

/** @typedef {import('./bar-customize.js').BarEditController} BarEditController */

/** The class that marks the row this module owns inside the display menu. */
const ROW_CLASS = 'bar-customize-entry';

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

/**
 * Put the Customize row in the display menu's action row — the component's own
 * projection point, beside the View toggle. Built here rather than declared in
 * the template so the row lives and dies with the bar stack that owns it.
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
 * page, since nothing calls back in here on an ordinary load.
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
      mountRow(owner, ctrl);
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
 * Mount the entry points: arm the right-click menus on both bars and project
 * the Customize row into the display menu.
 * @param {BarEditController} ctrl
 * @param {Document | Element} root
 */
export function initEntryPoints(ctrl, root) {
  stopEntryPoints();
  generation += 1;
  armMenus(ctrl, root);
  mountWhenReady(docOf(root), ctrl);
}

/** Take every entry point away. */
export function stopEntryPoints() {
  generation += 1;
  disarmMenus();
  if (typeof document !== 'undefined') unmountRow(document);
}
