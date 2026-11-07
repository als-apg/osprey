// @ts-check
/**
 * Panel side of the tile header-bar contribution contract: panel-chrome
 * protocol v3 on the unchanged embed-contract transport.
 *
 * An embedded panel gets exactly ONE header: the web-terminal hub's 36px tile
 * bar. A panel whose standalone top bar carries real controls does not render
 * a second bar when embedded — it describes those controls to the hub with
 * {@link contributeHeader} and the hub renders them in the tile bar, between
 * the tile's name and its close button. User interaction round-trips back as
 * a `canopy-host-header-action` message (see {@link onHeaderAction}); the panel
 * reacts internally and re-sends the WHOLE contribution with its new state.
 * The hub renders only what the last contribution says — it never mutates
 * item state locally, so a contribution is an idempotent replace, not a diff.
 *
 * Item vocabulary (closed set; the hub ignores unknown kinds). Every item is
 * a control: the bar shows the tile's name and nothing else that is inert, so
 * there is no text item — a status sentence or a loaded-file name belongs in
 * the panel's body. (The hub drops the retired `{kind:'text'}` like any other
 * unknown kind, so a panel that still sends one keeps working.)
 *
 * - `{kind:'nav', id, items:[{id, label, active}]}` — view switcher; the
 *   action's `value` is the clicked entry's id.
 * - `{kind:'button', id, label, title?, tone?:'default'|'accent', disabled?}`
 *   — workflow action.
 * - `{kind:'search', id, placeholder?, value?}` — filter box, rendered as the
 *   hub's own magnifier pill. The action fires debounced with the current
 *   text as `value`. `value` seeds the field at creation ONLY: once the
 *   element is live the operator owns it, and re-contributions leave it
 *   alone, so a panel may re-send freely while someone is typing.
 * - `{kind:'menu', id, label?, items:[{id, label, checked?, disabled?, danger?,
 *   separator?}]}` — the panel's entries in the tile ⋯, section "This panel": a
 *   tile has one ⋯, and every `menu` feeds it. The action's `value` is the
 *   clicked entry's id. `checked` renders a checkmark, for entries that toggle;
 *   `danger` the danger colour (destructive entries go last, and keep their own
 *   confirm step inside the panel); `separator` a divider row before the entry.
 * - `{kind:'icon', id, icon, title, disabled?, pressed?, priority?}` — one
 *   icon-only button in the bar's panel-icon slot. `icon` is an `icons.js` name;
 *   `title` is the tooltip and the accessible name; `pressed` sets `aria-pressed`
 *   for a toggle. At most three show in the bar and the rest fold into the tile
 *   ⋯, lowest `priority` first. A click round-trips with the item's `id`.
 *
 * `icon` and `menu` are what a panel should send. `nav`, `button` and `search`
 * are the legacy kinds: the hub keeps rendering them for CANOPY screens and for
 * a panel not yet moved to its own toolbar.
 *
 * The set is additive and the hub ignores kinds it does not know, which is
 * what keeps a panel and a hub of different vintages compatible — there is no
 * negotiation beyond that. The message `version` stays 1, because the transport
 * (`postMessage`, same origin, whole replace) did not change.
 *
 * ORDER IS THE LAYOUT, AND THE RIGHT EDGE IS THE ANCHOR. The hub gathers every
 * INTERACTIVE item into one cluster pinned against the tile's close button — a
 * flexible spacer takes all the slack to its left — and lays that cluster out
 * in contribution order. Right-anchoring makes change ASYMMETRIC: when an item
 * appears, disappears, or changes width, everything contributed BEFORE it
 * slides sideways and everything after it does not move at all.
 *
 * So order the array by how STABLE each item is, least stable first:
 *
 *   1. items contributed CONDITIONALLY — only in one view, only while
 *      something is running;
 *   2. items whose label CHANGES WIDTH at runtime — an arming button's
 *      "Confirm?", a count, an activity marker;
 *   3. the control the operator aims at repeatedly — the view switcher, the
 *      primary action — LAST, hard against the close button.
 *
 * Put a conditional item after a persistent one and the persistent one moves
 * every time the conditional comes and goes — usually as a direct result of
 * clicking that persistent control, which is the worst case there is: the
 * target slides out from under the cursor that just hit it. Reordering the
 * array is a panel's ONLY control over this; there is no side or slot to ask
 * the hub for.
 *
 * Every item may carry `priority` (number, default 0): in a narrow tile the
 * hub hides lowest-priority items first. That is a SEPARATE axis from order — it decides what survives a
 * narrow tile, not where anything sits — and the two may legitimately
 * disagree. Note the coupling, though: a re-contribution that widens a label
 * can itself make the bar crowded, so an item that both grows and carries the
 * lowest priority can fold ITSELF into the ⋯ menu at the moment it grows.
 * Both helpers are strict no-ops outside an embedded frame, so panels call
 * them unconditionally.
 *
 * @module header-contrib
 */

/** Version stamped on every contribution message. @type {number} */
export const HEADER_CONTRACT_VERSION = 1;

/**
 * @typedef {object} HeaderNavEntry
 * @property {string} id
 * @property {string} label
 * @property {boolean} [active]
 */

/**
 * @typedef {object} HeaderMenuEntry
 * @property {string} id
 * @property {string} label
 * @property {boolean} [checked]
 * @property {boolean} [disabled]
 * @property {boolean} [danger]     destructive: danger colour, listed last
 * @property {boolean} [separator]  a divider row before this entry
 */

/**
 * @typedef {object} HeaderItem
 * @property {'text'|'icon'|'nav'|'button'|'search'|'menu'} kind  `text` is retired; see above
 * @property {string} id
 * @property {number} [priority]
 * @property {string} [text]        the retired text items
 * @property {HeaderNavEntry[] | HeaderMenuEntry[]} [items]  nav / menu items
 * @property {string} [label]       button / menu items
 * @property {string} [icon]        icon items: an icons.js name
 * @property {boolean} [pressed]    icon items: `aria-pressed`, for a toggle
 * @property {string} [title]       icon / button items
 * @property {'default'|'accent'} [tone]  button items
 * @property {boolean} [disabled]   icon / button items
 * @property {string} [placeholder] search items
 * @property {string} [value]       search items (seed only — see the module docstring)
 */

/** @returns {boolean} true when running inside the web-terminal hub. */
function isEmbedded() {
  return document.body.classList.contains('embedded') && window.parent !== window;
}

/**
 * Send (replace) this panel's tile-bar contribution. Call once after init and
 * again WHOLE whenever any item's state changes (active nav entry, button
 * label/disabled). No-op standalone.
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
