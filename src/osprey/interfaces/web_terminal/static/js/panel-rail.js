// @ts-check
/* OSPREY Web Terminal — Vertical Panel Rail (DOM renderer)
 *
 * A pure DOM-rendering module for the 74px left icon rail that replaces the
 * horizontal header tab strip. It builds one entry per rail MEMBER (icon,
 * label, active accent, close corner, drag source) and exposes small
 * imperative mutators — set active, enable, attention, append, remove — so
 * `panel-manager.js`'s state machine can drive the rail without owning any DOM
 * specifics. (The `＋` add control is NOT the rail's: the template supplies it
 * as the sibling `#panel-add-btn`.)
 *
 * The rail deliberately carries NO per-panel health readout. Backend liveness
 * is reported in one place — the SYSTEM panel's `web_panels` health category —
 * so the rail stays a navigation surface rather than a status board. Health
 * still reaches the rail, but only as the coarse `.disabled` state: an entry
 * whose backend has never answered is dimmed and inert.
 *
 * The rail is a curated MEMBERSHIP list, not a tab strip with open/closed
 * state: an entry exists iff its panel is in the rail, at full brightness.
 * Which member currently holds the workspace tile is per-client layout state
 * and is reflected only by the `.active` accent — never by dimming. Panels
 * removed from the rail ("×", agent remove_panel_from_rail) lose their entry
 * live in the "+" catalog until re-added.
 *
 * This module holds NO panel state and issues NO fetches or POSTs. The caller
 * passes closures for the interactions (activate, close, add); everything
 * else is a one-shot mutation of the DOM the caller already owns. Styling is
 * entirely class/attribute driven (`.panel-rail-*`, `data-panel-id`,
 * `data-icon`) — the rail's CSS lands separately, this module only produces
 * the markup it hooks.
 *
 * DOM contract (stable — the browser suite selects on it):
 *
 *   <nav class="panel-rail" role="tablist">
 *     <button class="panel-rail-button disabled" data-panel-id="artifacts"
 *             type="button" role="tab" aria-selected="false" title="WORKSPACE"
 *             aria-label="WORKSPACE"                (title gains " · hint" when panel.hint is given)
 *             draggable="true">                                       (only when onDragStart given)
 *       <span class="panel-rail-icon" data-icon="artifacts" aria-hidden="true"></span>
 *       <span class="panel-rail-label">WORKSPACE</span>
 *       <span class="panel-rail-close" aria-hidden="true">×</span>    (only when onClose given)
 *     </button>
 *   </nav>
 *
 * Beyond activate/close, an entry's verbs (open beside, open standalone,
 * remove) live in the caller's right-click context menu — the rail only
 * forwards the contextmenu event through `onContextMenu` and suppresses the
 * browser menu when the caller reports it handled. A caller-supplied
 * `panel.hint` is appended to the tooltip so the menu is advertised where the
 * corner glyphs used to be.
 *
 * That menu is reachable from the keyboard, and explicitly so: a focused entry
 * handles the `ContextMenu` key and Shift+F10 itself and asks the caller to
 * open the menu centred on the entry's own rect. macOS fires no `contextmenu`
 * event from the keyboard at all, so leaving this to the platform would strand
 * those operators; on the platforms that DO fire one (Windows menu key,
 * Firefox Shift+F10) it is that keydown's default action, and suppressing the
 * default is what stops a single keypress opening two menus. The suppression
 * follows the same boolean contract as the right-click path, so an entry the
 * caller declines keeps its native behaviour on every input route.
 *
 * State classes on an entry: `.active` (surfaced panel), `.disabled` (backend
 * not healthy yet), `.agent-attention` (badge). A badged entry also carries a
 * transient `data-title-base` holding the tooltip text the badge borrowed;
 * clearing the badge restores it and removes the attribute.
 */

import { flashElement } from '/design-system/js/highlight.js';

// ---- Types ----

/**
 * The minimal panel descriptor the rail renders — a structural subset of the
 * `Panel` records `panel-manager.js` holds and of the `/api/panels` payload
 * (`{ enabled, custom, labels, ... }`). Only id + label are needed to draw an
 * entry; health and active state are applied by the mutators.
 * @typedef {object} RailPanel
 * @property {string} id
 * @property {string} label
 * @property {string} [hint] - tooltip suffix advertising the context menu
 *   (rendered as "LABEL · hint"; aria-label stays the bare label)
 */

/**
 * Interaction closures injected by the caller so the rail stays a dumb view.
 * Every callback is optional: omit `onClose` to render no close corner, omit
 * `onDragStart` to render non-draggable entries. `onDragStart` may return
 * false to cancel the drag for that entry, and `onContextMenu` false to let
 * the browser's native menu through (the caller's policy — e.g. the terminal
 * entry, simple mode); this module stays policy-free.
 * @typedef {object} RailOptions
 * @property {(id: string) => void} [onActivate]   - an entry was clicked
 * @property {(id: string) => void} [onClose]      - the entry's close "×" was clicked
 * @property {(id: string, e: MouseEvent) => boolean} [onContextMenu]
 *   - an entry was right-clicked, or asked for its menu from the keyboard
 *     (`ContextMenu` / Shift+F10, where `e` is a synthetic contextmenu event
 *     positioned at the entry's rect); return true if a menu was opened
 *     (suppresses the browser menu), false to decline and let the native menu
 *     show
 * @property {(id: string, dataTransfer: DataTransfer | null) => boolean} [onDragStart]
 *   - a drag left an entry; return false to cancel it
 * @property {(id: string) => void} [onDragEnd]    - the entry's drag gesture ended (any outcome)
 */

const BUTTON_SELECTOR = '.panel-rail-button';

/**
 * Where an entry's pre-suffix tooltip is parked while the agent-attention
 * badge owns the `title`. Present only for the badge's lifetime — see
 * {@link applyTouchedTooltip}.
 */
const TITLE_BASE_ATTR = 'data-title-base';

const TOUCHED_SEPARATOR = ' · agent touched ';

// ---- Rendering ----

/**
 * Build one rail entry button for a panel. Children are assembled via DOM APIs
 * (never innerHTML) because `panel.label` originates from server JSON / SSE.
 *
 * Entries start `.disabled` (matching the tab strip's cold state); the caller
 * clears it via {@link setEntryEnabled} once the panel's backend is healthy.
 * `data-icon` carries the panel id so the rail CSS can map known ids to glyphs
 * and fall back generically for custom panels.
 * @param {RailPanel} panel
 * @param {RailOptions} [options]
 * @returns {HTMLButtonElement}
 */
function buildRailButton(panel, options = {}) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'panel-rail-button disabled';
  btn.setAttribute('data-panel-id', panel.id);
  btn.setAttribute('role', 'tab');
  btn.setAttribute('aria-selected', 'false');
  // The tooltip advertises the context menu (the entry's non-corner verbs have
  // no visible affordance); aria-label stays the bare label — assistive tech
  // reaches the menu itself via the menu key / Shift+F10.
  btn.title = panel.hint ? `${panel.label} · ${panel.hint}` : panel.label;
  btn.setAttribute('aria-label', panel.label);

  const icon = document.createElement('span');
  icon.className = 'panel-rail-icon';
  icon.setAttribute('data-icon', panel.id);
  icon.setAttribute('aria-hidden', 'true');
  btn.appendChild(icon);

  const label = document.createElement('span');
  label.className = 'panel-rail-label';
  label.textContent = panel.label;
  btn.appendChild(label);

  if (options.onActivate) {
    const onActivate = options.onActivate;
    btn.addEventListener('click', () => onActivate(panel.id));
  }

  // The one corner affordance — close "×", the frequent membership verb that
  // must stay reachable even on a disabled (offline) entry. The rest of the
  // entry's verbs (open beside, open standalone, remove) live in the caller's
  // context menu. (Closing an OPEN tile is the tile header's own "×" — a
  // layout gesture, not a membership one.)
  if (options.onClose) {
    btn.appendChild(
      buildCornerAffordance('panel-rail-close', '×', `Close ${panel.label}`, panel.id, options.onClose)
    );
  }

  // Right-click → the caller's context menu. The caller decides per-entry
  // whether a menu exists (returns true); declining (false — e.g. the
  // terminal entry) leaves the event alone so the browser menu shows.
  if (options.onContextMenu) {
    const onContextMenu = options.onContextMenu;
    btn.addEventListener('contextmenu', (e) => {
      if (onContextMenu(panel.id, e)) e.preventDefault();
    });

    // The keyboard route to the same menu, opened by this module rather than
    // by the platform: macOS synthesises no `contextmenu` event from the menu
    // key, so a native-only path would have no keyboard route at all there.
    // The synthetic event is handed straight to the caller and never
    // dispatched, so the listener above cannot also fire for one keypress.
    btn.addEventListener('keydown', (e) => {
      if (e.key !== 'ContextMenu' && !(e.key === 'F10' && e.shiftKey)) return;
      const rect = btn.getBoundingClientRect();
      const synthetic = new MouseEvent('contextmenu', {
        bubbles: true,
        cancelable: true,
        clientX: rect.left + rect.width / 2,
        clientY: rect.top + rect.height / 2,
      });
      // Preventing the default is what keeps Windows/Firefox — where the
      // native menu IS this keydown's default action — from stacking a second
      // menu on the one just opened; a declined entry keeps that default.
      if (onContextMenu(panel.id, synthetic)) e.preventDefault();
    });
  }

  // Drag source: dragging an entry into the workspace opens the panel as a
  // NEW tile at the drop position (the precise half of the open-beside verb).
  // The caller's onDragStart stamps the dataTransfer payload and may cancel
  // (return false) — e.g. for the terminal entry or a locked simple layout.
  if (options.onDragStart) {
    const onDragStart = options.onDragStart;
    btn.setAttribute('draggable', 'true');
    btn.addEventListener('dragstart', (e) => {
      if (!onDragStart(panel.id, e.dataTransfer)) {
        e.preventDefault();
        return;
      }
      btn.classList.add('dragging');
    });
    const onDragEnd = options.onDragEnd;
    btn.addEventListener('dragend', () => {
      btn.classList.remove('dragging');
      onDragEnd?.(panel.id);
    });
  }

  return btn;
}

/**
 * The entry's one hover-revealed corner glyph — the close "×". Decorative
 * (aria-hidden) so it is not a control nested inside the entry button;
 * assistive tech reaches the same action through the entry's context menu and
 * the command palette. The click stops propagation so acting on the corner
 * does not also activate the entry underneath it.
 *
 * @param {string} className
 * @param {string} glyph
 * @param {string} title
 * @param {string} panelId
 * @param {(id: string) => void} handler
 * @returns {HTMLSpanElement}
 */
function buildCornerAffordance(className, glyph, title, panelId, handler) {
  const el = document.createElement('span');
  el.className = className;
  el.setAttribute('aria-hidden', 'true');
  el.title = title;
  el.textContent = glyph;
  el.addEventListener('click', (e) => {
    e.stopPropagation();
    handler(panelId);
  });
  return el;
}

/**
 * Render the full rail into `railEl`, replacing any existing content. Entries
 * are appended in `panels` order. Marks `railEl` as `role="tablist"` for
 * assistive tech. (The `＋` add control is the template's sibling
 * `#panel-add-btn`, outside this nav — never rendered here.)
 *
 * This is the destructive full render — the twin of the tab strip's
 * `renderTabs()`. Use {@link addEntry} / {@link removeEntry} for
 * non-destructive runtime membership changes so live entries keep their
 * active / health / enabled state.
 * @param {HTMLElement} railEl
 * @param {RailPanel[]} panels
 * @param {RailOptions} [options]
 */
export function createRail(railEl, panels, options = {}) {
  railEl.classList.add('panel-rail');
  railEl.setAttribute('role', 'tablist');
  railEl.replaceChildren();
  for (const panel of panels) {
    railEl.appendChild(buildRailButton(panel, options));
  }
}

/**
 * Append a single entry without disturbing existing ones — the non-destructive
 * path for membership additions (agent add_panel_to_rail, "+" menu pick, runtime
 * registration).
 *
 * Idempotent by id: if an entry for `panel.id` already exists it is returned
 * unchanged (no duplicate node), mirroring the tab strip's re-register guard.
 * @param {HTMLElement} railEl
 * @param {RailPanel} panel
 * @param {RailOptions} [options]
 * @returns {HTMLButtonElement}
 */
export function addEntry(railEl, panel, options = {}) {
  const existing = getEntry(railEl, panel.id);
  if (existing) return existing;

  const btn = buildRailButton(panel, options);
  railEl.appendChild(btn);
  return btn;
}

/**
 * Remove a panel's entry from the rail — the membership-removal twin of
 * {@link addEntry} ("×", agent remove_panel_from_rail). The node is taken out of the DOM
 * entirely; the panel lives on in the caller's catalog and a later addEntry
 * rebuilds a fresh entry. No-op for an unknown id.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 */
export function removeEntry(railEl, panelId) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return;
  // A detached drag source can never fire dragend (HTML5 delivers it to the
  // source element only), so a mid-drag removal would strand the caller's
  // drag-end cleanup — the iframe pointer shields raised in onDragStart.
  // End the gesture through the entry's own dragend listener before detaching.
  if (entry.classList.contains('dragging')) {
    entry.dispatchEvent(new Event('dragend'));
  }
  entry.remove();
}

// ---- Mutators ----

/**
 * Return the entry button for a panel id, or null when absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @returns {HTMLButtonElement | null}
 */
export function getEntry(railEl, panelId) {
  return /** @type {HTMLButtonElement | null} */ (
    railEl.querySelector(`${BUTTON_SELECTOR}[data-panel-id="${panelId}"]`)
  );
}

/**
 * Mark exactly one entry active (the secondary-accent left edge lives on
 * `.panel-rail-button.active` in CSS) and clear it from every other entry.
 * Passing an id with no matching entry clears the active state on all.
 * @param {HTMLElement} railEl
 * @param {string | null} panelId
 */
export function setActive(railEl, panelId) {
  for (const el of railEl.querySelectorAll(BUTTON_SELECTOR)) {
    const isActive = el.getAttribute('data-panel-id') === panelId;
    el.classList.toggle('active', isActive);
    el.setAttribute('aria-selected', isActive ? 'true' : 'false');
  }
}

/**
 * Enable or disable an entry by toggling `.disabled`. Entries render disabled;
 * the caller enables one once its backend first reports healthy. This is the
 * rail's ONLY health-derived state — the per-entry LED was retired in favour of
 * the SYSTEM panel's `web_panels` category. No-op when the entry is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} enabled
 */
export function setEntryEnabled(railEl, panelId, enabled) {
  getEntry(railEl, panelId)?.classList.toggle('disabled', !enabled);
}

/**
 * Point an entry's tooltip at the moment the agent touched its panel, or put
 * the tooltip back the way it was.
 *
 * The pre-suffix text is stashed on the entry for the badge's lifetime rather
 * than recomputed, so the restore is exact even if the caller retitled the
 * entry, and so a second event REPLACES the time instead of appending a second
 * suffix. Restoring is keyed on the stash, which makes a clear on an unbadged
 * entry a true no-op.
 * @param {HTMLElement} entry
 * @param {number | null} ts - server epoch seconds, or null to restore the base
 */
function applyTouchedTooltip(entry, ts) {
  const base = entry.getAttribute(TITLE_BASE_ATTR) ?? entry.title;
  if (ts === null) {
    if (entry.hasAttribute(TITLE_BASE_ATTR)) {
      entry.title = base;
      entry.removeAttribute(TITLE_BASE_ATTR);
    }
    return;
  }
  const touchedAt = new Date(ts * 1000).toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit',
  });
  entry.setAttribute(TITLE_BASE_ATTR, base);
  entry.title = `${base}${TOUCHED_SEPARATOR}${touchedAt}`;
}

/**
 * Set or clear the agent-attention affordance on an entry. Turning it on
 * toggles the persistent `agent-attention` badge class (the design system's
 * highlight.css draws an absolutely-positioned `::after` accent dot — class
 * only, no child nodes, no layout shift), fires the one-shot `agent-flash`
 * glow via {@link flashElement}, and scrolls the entry into view: a rail
 * taller than its viewport can otherwise take a badge entirely off-screen,
 * which is the one case where the affordance reports nothing to the operator.
 * `block: 'nearest'` leaves an already-visible entry exactly where it is.
 *
 * Turning it off removes the badge class and restores the tooltip; an
 * in-flight flash is left to finish on its own `animationend`.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} on
 * @param {number} [ts] - the originating event's SERVER timestamp (epoch
 *   seconds, as the `agent_activity` SSE frames carry it), appended to the
 *   entry's tooltip as "· agent touched <time>". Omit it — or pass a
 *   non-finite value — for a badge with no time claim; never substitute a
 *   client clock, which would report when the browser rendered rather than
 *   when the agent acted.
 * @returns {boolean} true when the entry existed and was updated; false for an
 *   unknown id (safe no-op, so callers can fall back)
 */
export function setEntryAttention(railEl, panelId, on, ts) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return false;
  entry.classList.toggle('agent-attention', on);
  const touchedAt = on && typeof ts === 'number' && Number.isFinite(ts) ? ts : null;
  applyTouchedTooltip(entry, touchedAt);
  if (on) {
    flashElement(entry);
    entry.scrollIntoView({ block: 'nearest' });
  }
  return true;
}
