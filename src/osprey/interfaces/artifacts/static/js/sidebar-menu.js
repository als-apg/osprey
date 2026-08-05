// @ts-check
/**
 * OSPREY Artifact Gallery — browser-column overflow menu (⋯).
 *
 * A minimal disclosure menu for the browser card's rare actions (scope /
 * refresh / layout). This module owns only the open/close mechanics —
 * aria-expanded on the trigger, [hidden] on the popover, dismiss on
 * outside pointerdown, Escape, or any menu-item click. What the items DO
 * stays with their existing owners (gallery.js wires #all-sessions-btn and
 * #refresh-btn, browse-layout.js wires #orient-toggle-btn), keyed off the
 * same element ids those handlers have always used.
 *
 * @module sidebar-menu
 */

/**
 * Wire the overflow menu. No-ops when either element is absent. Safe to
 * call once at boot.
 *
 * @param {object} els
 * @param {HTMLElement|null} els.button  the ⋯ trigger (aria-expanded owner)
 * @param {HTMLElement|null} els.menu    the popover ([hidden] owner)
 * @returns {{ close: () => void }}
 */
export function initSidebarMenu({ button, menu }) {
  if (!button || !menu) return { close: () => {} };

  const setOpen = (/** @type {boolean} */ open) => {
    menu.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
  };

  const close = () => setOpen(false);

  button.addEventListener("click", () => setOpen(menu.hidden));

  // Any item click performs its action (via the item's own listener) and
  // dismisses the menu — the scope pill / spinner give the lasting feedback.
  menu.addEventListener("click", (e) => {
    if (/** @type {HTMLElement} */ (e.target).closest(".sidebar-menu-item")) close();
  });

  document.addEventListener("pointerdown", (e) => {
    if (menu.hidden) return;
    const target = /** @type {Node} */ (e.target);
    if (!menu.contains(target) && !button.contains(target)) close();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !menu.hidden) {
      close();
      button.focus();
    }
  });

  return { close };
}
