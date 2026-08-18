// @ts-check
/* OSPREY Web Terminal — Rail Context Menu
 *
 * A small cursor-anchored popover listing a rail entry's verbs in words —
 * the legible home for the panel actions that used to be hover-revealed
 * corner glyphs ("open in a new tile" ⊞, "open in a new window" ↗). Opened
 * by panel-manager from a rail entry's or tile header's contextmenu event.
 *
 * The keyboard path is NOT the browser's: macOS fires no `contextmenu` event
 * from the menu key or Shift+F10, so panel-rail handles those keydowns itself
 * and calls the same opener with the entry's rect. (Windows/Firefox DO fire a
 * native `contextmenu` from that keydown, which is why the rail cancels the
 * keydown's default action — otherwise one keypress would open two menus.)
 *
 * Purely presentational, like panel-add-menu: it owns the popover DOM,
 * open/close, dismissal, focus bookkeeping, and viewport clamping. It holds
 * no panel state and issues no fetches — the caller passes fully-bound `run`
 * closures, so this module never knows what the verbs do. One menu at a time:
 * opening replaces any open menu.
 *
 * Dismissal contract — the menu closes on:
 *   - pointerdown outside it (capture phase, so the press dismisses first),
 *   - Escape or Tab,
 *   - running an item (the menu closes BEFORE the action),
 *   - window `blur` — a click into a panel iframe never reaches this document,
 *   - window `resize`,
 *   - a scroll that actually moves the anchor: capture-phase `scroll` whose
 *     target is the document or an ancestor of `anchorEl`. Scoping matters —
 *     an unscoped listener would dismiss on xterm's own viewport scrolls, i.e.
 *     every time the terminal streams output.
 *
 * Focus contract — `openContextMenu` records `document.activeElement` as
 * `previousFocus` and focuses the first enabled row. That focus is handed back
 * only on the KEYBOARD dismissals (Escape/Tab) and on close-before-run, where
 * it is restored before the action runs so an action that moves focus itself
 * (window.open, docking a tile) still wins. Pointer/blur/scroll/resize
 * dismissals never move focus: blur in particular fires AFTER focus entered
 * the clicked iframe, and restoring would yank it straight back out.
 * `previousFocus` rather than "the invoking element" because tile headers are
 * non-focusable divs — there is nothing else to hand focus back to.
 *
 * DOM contract (the browser suite selects on it):
 *
 *   <div class="rail-context-menu" role="menu" aria-label="ARIEL actions">
 *     <button class="rail-context-item" role="menuitem" type="button">
 *       <span class="rail-context-label">Open in a new window</span>
 *       <span class="rail-context-glyph" aria-hidden="true">↗</span>
 *     </button>
 *     <div class="rail-context-divider" role="separator"></div>
 *     <button class="rail-context-item danger" ...>
 *   </div>
 */

/**
 * One menu row. A `divider: true` entry renders a separator instead of a
 * button (its other fields are ignored).
 * @typedef {object} MenuItem
 * @property {boolean} [divider]  - render a separator, not a button
 * @property {string} [label]     - the verb, in words
 * @property {string} [glyph]     - trailing glyph tying the row to rail iconography
 * @property {boolean} [disabled] - render the row inert (aria-disabled, no run)
 * @property {boolean} [danger]   - destructive verb — hover styles error-red
 * @property {() => void} [run]   - the caller-bound action; the menu closes first
 */

/** @type {HTMLElement | null} the single open menu, or null */
let menuEl = null;

/** @type {HTMLElement | null} the surface the menu belongs to (scroll scoping) */
let anchorEl = null;

/** @type {HTMLElement | null} what had focus when the menu opened */
let previousFocus = null;

/** @param {MouseEvent} e */
function onDocPointerDown(e) {
  if (menuEl && e.target instanceof Node && !menuEl.contains(e.target)) dismiss(false);
}

/** @param {KeyboardEvent} e */
function onDocKeydown(e) {
  if (!menuEl) return;
  // Escape and Tab both leave the menu by keyboard, so both hand focus back.
  // Tab's default is cancelled: the operator resumes tabbing from where they
  // were, not from wherever a removed menu row left the sequence.
  if (e.key === 'Escape' || e.key === 'Tab') {
    e.preventDefault();
    e.stopPropagation();
    dismiss(true);
    return;
  }
  // Arrow navigation over the enabled rows — the menu's only focusables.
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    e.stopPropagation();
    const rows = /** @type {HTMLElement[]} */ (
      [...menuEl.querySelectorAll('.rail-context-item:not([aria-disabled="true"])')]
    );
    if (!rows.length) return;
    const idx = rows.indexOf(/** @type {HTMLElement} */ (document.activeElement));
    const next = e.key === 'ArrowDown'
      ? rows[(idx + 1 + rows.length) % rows.length]
      : rows[(idx - 1 + rows.length) % rows.length];
    next.focus();
  }
}

/** The window lost focus — usually a click landing inside a panel iframe. */
function onWindowBlur() {
  dismiss(false);
}

/** A resize moves every anchor; a cursor-anchored popover cannot follow. */
function onWindowResize() {
  dismiss(false);
}

/**
 * Close only when the scroll actually moved the menu's anchor: the document
 * scrolled, or a container holding the anchor did. A panel's own scroller —
 * the terminal viewport above all — leaves the menu alone.
 * @param {Event} e
 */
function onDocScroll(e) {
  if (!menuEl) return;
  if (e.target === document) {
    dismiss(false);
    return;
  }
  if (anchorEl && e.target instanceof Node && e.target.contains(anchorEl)) dismiss(false);
}

/**
 * Tear down the open menu, optionally handing focus back to whatever held it
 * when the menu opened.
 * @param {boolean} restoreFocus
 */
function dismiss(restoreFocus) {
  if (!menuEl) return;
  menuEl.remove();
  menuEl = null;
  anchorEl = null;
  const restoreTo = previousFocus;
  previousFocus = null;
  document.removeEventListener('pointerdown', onDocPointerDown, true);
  document.removeEventListener('keydown', onDocKeydown, true);
  document.removeEventListener('scroll', onDocScroll, true);
  window.removeEventListener('blur', onWindowBlur);
  window.removeEventListener('resize', onWindowResize);
  if (restoreFocus && restoreTo && restoreTo.isConnected) restoreTo.focus();
}

/**
 * Close the open context menu, if any, leaving focus where it is. Safe to
 * call when none is open.
 */
export function closeContextMenu() {
  dismiss(false);
}

/**
 * Open the context menu at a viewport position, replacing any open menu.
 *
 * All children are assembled via DOM APIs (never innerHTML) because labels can
 * carry server-supplied panel labels. After insertion the menu is clamped so
 * it never overflows the viewport (opened near an edge it flips inward), and
 * the first enabled row takes focus for keyboard users.
 *
 * @param {{
 *   x: number,
 *   y: number,
 *   anchorEl?: HTMLElement | null,
 *   ariaLabel: string,
 *   items: MenuItem[],
 * }} opts  - `anchorEl` is the rail entry or tile header the menu belongs to;
 *   it scopes the scroll dismissal (omitting it leaves only document scrolls).
 */
export function openContextMenu(opts) {
  const { x, y, ariaLabel, items } = opts;
  // Capture the focus to hand back BEFORE tearing down any open menu — when
  // one menu replaces another the active element is the old menu's row, so
  // the earlier capture is the one worth keeping.
  const active = document.activeElement;
  const held = menuEl && active instanceof Node && menuEl.contains(active) ? previousFocus : active;
  dismiss(false);
  previousFocus = held instanceof HTMLElement ? held : null;
  anchorEl = opts.anchorEl ?? null;

  const menu = document.createElement('div');
  menu.className = 'rail-context-menu';
  menu.setAttribute('role', 'menu');
  menu.setAttribute('aria-label', ariaLabel);
  // A right-click on the popover itself (or the menu key on a focused row)
  // must not stack the browser's native menu on top of this one.
  menu.addEventListener('contextmenu', (e) => e.preventDefault());

  for (const item of items) {
    if (item.divider) {
      const hr = document.createElement('div');
      hr.className = 'rail-context-divider';
      hr.setAttribute('role', 'separator');
      menu.appendChild(hr);
      continue;
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'rail-context-item' + (item.danger ? ' danger' : '');
    btn.setAttribute('role', 'menuitem');
    const label = document.createElement('span');
    label.className = 'rail-context-label';
    label.textContent = item.label ?? '';
    btn.appendChild(label);
    if (item.glyph) {
      const glyph = document.createElement('span');
      glyph.className = 'rail-context-glyph';
      glyph.setAttribute('aria-hidden', 'true');
      glyph.textContent = item.glyph;
      btn.appendChild(glyph);
    }
    if (item.disabled) {
      btn.setAttribute('aria-disabled', 'true');
    } else {
      const run = item.run;
      // Close BEFORE running, matching the palette's Enter path — an action
      // that moves focus (window.open, dock changes) must not race dismissal.
      // Focus goes back first for the same reason: whatever the action does
      // with focus lands last and therefore wins.
      btn.addEventListener('click', () => {
        dismiss(true);
        run?.();
      });
    }
    menu.appendChild(btn);
  }

  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  document.body.appendChild(menu);
  menuEl = menu;

  // Clamp into the viewport: near the right/bottom edge the menu flips to
  // sit on the cursor's other side rather than overflowing.
  const rect = menu.getBoundingClientRect();
  if (rect.right > window.innerWidth) menu.style.left = `${Math.max(0, x - rect.width)}px`;
  if (rect.bottom > window.innerHeight) menu.style.top = `${Math.max(0, y - rect.height)}px`;

  // Capture-phase pointerdown (not click) so a drag started outside also
  // dismisses, and before the outside target acts on the press. Scroll is
  // capture-phase because element scrolls do not bubble.
  document.addEventListener('pointerdown', onDocPointerDown, true);
  document.addEventListener('keydown', onDocKeydown, true);
  document.addEventListener('scroll', onDocScroll, true);
  window.addEventListener('blur', onWindowBlur);
  window.addEventListener('resize', onWindowResize);

  const first = /** @type {HTMLElement | null} */ (
    menu.querySelector('.rail-context-item:not([aria-disabled="true"])')
  );
  first?.focus();
}

/** @returns {boolean} whether a context menu is currently open (test hook). */
export function isContextMenuOpen() {
  return menuEl !== null;
}
