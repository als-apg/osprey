// @ts-check
/**
 * Panel side of the tile-bar contribution contract: panel-chrome protocol v3
 * on the unchanged embed-contract transport.
 *
 * An embedded panel gets exactly ONE header: the web-terminal hub's tile bar
 * (`--wt-pane-head-height`, 40px). The bar is OSPREY chrome: the tile's name,
 * a subtitle, at most three panel icons, the deployment's shared controls, and
 * the fixed ⋯ / Pop out / × cluster. A panel describes what it wants there
 * with {@link contributeHeader}; user interaction round-trips back as a
 * `canopy-host-header-action` message (see {@link onHeaderAction}); the panel
 * reacts internally and re-sends the WHOLE contribution with its new state.
 * The hub renders only what the last contribution says — it never mutates
 * item state locally, so a contribution is an idempotent replace, not a diff.
 *
 * v3 vocabulary (what a first-party panel sends):
 *
 * - `{kind:'text', id, text}` — the tile SUBTITLE, beside the name. Identity
 *   only: the shown artifact's name, the open concept's id, the lattice
 *   model. Never a status, a count or a control.
 * - `{kind:'icon', id, icon, title, disabled?, pressed?, priority?}` — one
 *   icon-only button in the panel-icon slot, left of the shared controls.
 *   `icon` is an `icons.js` name (an unknown name drops the item); `title` is
 *   the tooltip and the accessible name. At most three render in the bar; in
 *   a narrow tile they fold into the tile ⋯ ("This panel") first, lowest
 *   `priority` first. A click round-trips as `canopy-host-header-action`
 *   with the item's `id`.
 * - `{kind:'menu', id, items:[{id, label, checked?, disabled?, danger?,
 *   separator?}]}` — the panel's entries in the tile ⋯, section "This panel".
 *   The hub renders no separate button for it. A panel sends at most one.
 *   `checked` renders a checkmark, `danger` the danger colour (destructive
 *   actions go last), `separator` a divider row before the entry. The
 *   action's `value` is the clicked entry's id. A menu click is only an
 *   action: a destructive entry keeps its own confirm step inside the panel.
 *
 * Everything else a panel needs — a view switcher, a search field, text
 * buttons, a status chip — lives in the panel's own toolbar
 * (`.osprey-panel-toolbar`), not in the bar. See DESIGN.md, "Panel chrome
 * standard".
 *
 * Legacy kinds (deprecated for first-party panels; still rendered exactly as
 * before for CANOPY screens and any panel not yet migrated):
 *
 * - `{kind:'nav', id, items:[{id, label, active}]}` — view switcher in the
 *   bar; the action's `value` is the clicked entry's id.
 * - `{kind:'button', id, label?, title?, tone?:'default'|'accent',
 *   disabled?, icon?}` — text or icon button in the bar. With `icon` and no
 *   `label` it renders icon-only.
 * - `{kind:'search', id, placeholder?, value?}` — filter box in the bar. The
 *   action fires debounced with the current text as `value`. `value` seeds
 *   the field at creation ONLY; re-contributions leave the live field alone.
 * - `{kind:'crumb', id, root, leaf?}` — breadcrumb in place of the name.
 * - `{kind:'sep', id}` — a vertical hairline between controls.
 * - A legacy `menu` may carry a `label`; the hub ignores it. Every `menu`,
 *   legacy or v3, feeds the tile ⋯: a tile has exactly one ⋯ button.
 *
 * The set is closed and the hub ignores kinds it does not know, which is what
 * keeps a panel and a hub of different vintages compatible — there is no
 * negotiation beyond that. The message `version` stays 1: the transport
 * (`postMessage`, same origin, whole-replace) did not change with v3.
 *
 * ORDER IS THE LAYOUT for legacy items: the hub lays the interactive cluster
 * out in contribution order, pinned against the right edge, so order
 * conditional and width-changing items first and the repeatedly aimed-at
 * control last. v3 `icon` items also keep contribution order, inside their
 * slot. `priority` (number, default 0) is a separate axis: in a narrow tile
 * the hub folds lowest-priority items into ⋯ first; `text` truncates before
 * anything folds.
 *
 * Both helpers are strict no-ops outside an embedded frame, so panels call
 * them unconditionally.
 *
 * @module header-contrib
 */

/** Version stamped on every contribution message. @type {number} */
export const HEADER_CONTRACT_VERSION = 1;

/**
 * @typedef {object} HeaderNavEntry   legacy `nav` entry
 * @property {string} id
 * @property {string} label
 * @property {boolean} [active]
 */

/**
 * @typedef {object} HeaderMenuEntry
 * @property {string} id
 * @property {string} label
 * @property {boolean} [checked]    renders a checkmark (a toggling entry)
 * @property {boolean} [disabled]
 * @property {boolean} [danger]     destructive: danger colour, listed last
 * @property {boolean} [separator]  a divider row before this entry
 */

/**
 * @typedef {object} HeaderItem
 * @property {'text'|'icon'|'menu'|'nav'|'button'|'search'|'crumb'|'sep'} kind
 *   v3: `text`, `icon`, `menu`. Legacy (CANOPY, unmigrated panels): the rest.
 * @property {string} id
 * @property {number} [priority]    fold order in a narrow tile (lowest folds first)
 * @property {string} [text]        text items: the subtitle
 * @property {string} [icon]        icon items: an icons.js name; legacy button items too
 * @property {string} [title]       icon items: tooltip + accessible name; legacy button items
 * @property {boolean} [disabled]   icon items; legacy button items
 * @property {boolean} [pressed]    icon items: `aria-pressed` for a toggle
 * @property {HeaderMenuEntry[] | HeaderNavEntry[]} [items]  menu / legacy nav entries
 * @property {string} [label]       legacy button / legacy menu (gives it its own ⋯)
 * @property {'default'|'accent'} [tone]  legacy button items
 * @property {string} [placeholder] legacy search items
 * @property {string} [value]       legacy search items (seed only — see the module docstring)
 * @property {string} [root]        legacy crumb items
 * @property {string} [leaf]        legacy crumb items
 */

/** @returns {boolean} true when running inside the web-terminal hub. */
function isEmbedded() {
  return document.body.classList.contains('embedded') && window.parent !== window;
}

/**
 * Send (replace) this panel's tile-bar contribution. Call once after init and
 * again WHOLE whenever any item's state changes (active nav entry, button
 * label/disabled, live text). No-op standalone.
 * @param {HeaderItem[]} items
 */
export function contributeHeader(items) {
  if (!isEmbedded()) return;
  try {
    window.parent.postMessage(
      { type: 'canopy-host-header-contribution', version: HEADER_CONTRACT_VERSION, items },
      window.location.origin
    );
  } catch {
    /* cross-origin host — not our hub, drop */
  }
}

/**
 * Subscribe to the hub's header-action round-trip. The callback receives the
 * contributed item's id and, for nav items, the clicked entry's id. Origin-
 * checked; never fires standalone (the hub only messages its own iframes).
 * @param {(id: string, value?: string) => void} callback
 */
export function onHeaderAction(callback) {
  window.addEventListener('message', (e) => {
    if (e.origin !== window.location.origin) return;
    const data = e.data;
    if (!data || data.type !== 'canopy-host-header-action' || typeof data.id !== 'string') return;
    callback(data.id, typeof data.value === 'string' ? data.value : undefined);
  });
}
