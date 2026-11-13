// @ts-check
/* OSPREY Web Terminal — Rail Context Menu
 *
 * The rail entry's and tile bar's popover, listing a panel's verbs in words
 * — the legible home for the actions that used to be hover-revealed corner
 * glyphs ("open in a new tile" ⊞, "pop out" ↗). Opened by panel-menu-policy
 * from a rail entry's or tile bar's contextmenu event and from the tile bar's
 * ⋯ button, and by panel-rail from the menu key and Shift+F10.
 *
 * The behaviour — popover DOM, dismissal, focus hand-back, viewport clamping —
 * is the design system's shared context menu (/design-system/js/context-menu.js),
 * which every OSPREY panel opens. This wrapper only keeps the hub's own class
 * names, so terminal.css's `.rail-context-*` skin and the selectors the browser
 * suite uses stay as they were:
 *
 *   <div class="rail-context-menu" role="menu" aria-label="Logbook actions">
 *     <button class="rail-context-item" role="menuitem" type="button">
 *       <span class="rail-context-label">Open in new window</span>
 *       <span class="rail-context-glyph" aria-hidden="true">↗</span>
 *     </button>
 *     <div class="rail-context-divider" role="separator"></div>
 *     <button class="rail-context-item danger" ...>
 *   </div>
 */

import { openContextMenu as openShared } from '/design-system/js/context-menu.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/**
 * Open the context menu at a viewport position, replacing any open menu.
 *
 * @param {{
 *   x: number,
 *   y: number,
 *   anchorEl?: HTMLElement | null,
 *   triggerEl?: HTMLElement | null,
 *   side?: 'below' | 'right',
 *   ariaLabel: string,
 *   items: MenuItem[],
 * }} opts  - `anchorEl` is the rail entry or tile bar the menu belongs to;
 *   it scopes the scroll dismissal (omitting it leaves only document scrolls).
 *   `triggerEl` is the bar's ⋯ button when that opened the menu: the menu
 *   hangs under it and it carries `aria-expanded` (context-menu.js).
 */
export function openContextMenu(opts) {
  openShared({ ...opts, classPrefix: 'rail-context' });
}
