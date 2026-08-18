// @ts-check
/**
 * Panel side of the tile header-bar contribution contract (embed contract v2).
 *
 * An embedded panel gets exactly ONE header: the web-terminal hub's 36px tile
 * bar. A panel whose standalone top bar carries real controls does not render
 * a second bar when embedded — it describes those controls to the hub with
 * {@link contributeHeader} and the hub renders them in the tile bar, between
 * the tile's name and its close button. User interaction round-trips back as
 * an `osprey-header-action` message (see {@link onHeaderAction}); the panel
 * reacts internally and re-sends the WHOLE contribution with its new state.
 * The hub renders only what the last contribution says — it never mutates
 * item state locally, so a contribution is an idempotent replace, not a diff.
 *
 * Item vocabulary (closed set; the hub ignores unknown kinds):
 *
 * - `{kind:'text', id, text}` — inert label (e.g. a loaded-file name). The
 *   hub renders it as the tile's SUBTITLE, beside the tile name — it is
 *   identity, not a control — while interactive items right-anchor.
 * - `{kind:'nav', id, items:[{id, label, active}]}` — view switcher; the
 *   action's `value` is the clicked entry's id.
 * - `{kind:'button', id, label, title?, tone?:'default'|'accent', disabled?}`
 *   — workflow action.
 * - `{kind:'search', id, placeholder?, value?}` — filter box, rendered as the
 *   hub's own magnifier pill. The action fires debounced with the current
 *   text as `value`. `value` seeds the field at creation ONLY: once the
 *   element is live the operator owns it, and re-contributions leave it
 *   alone, so a panel may re-send freely while someone is typing.
 * - `{kind:'menu', id, label?, items:[{id, label, checked?, disabled?}]}` —
 *   overflow menu behind a ⋯ button; the action's `value` is the clicked
 *   entry's id. `checked` renders a checkmark, for entries that toggle.
 *
 * The set is additive and the hub ignores kinds it does not know, which is
 * what keeps a panel and a hub of different vintages compatible — there is no
 * negotiation beyond that.
 *
 * ORDER IS THE LAYOUT, AND THE RIGHT EDGE IS THE ANCHOR. The hub gathers every
 * INTERACTIVE item into one cluster pinned against the tile's close button — a
 * flexible spacer takes all the slack to its left — and lays that cluster out
 * in contribution order. (`text` items are outside it: they are the tile's
 * subtitle, left-anchored beside the name, and their width costs the spacer,
 * not a neighbour.) Right-anchoring makes change ASYMMETRIC: when an item
 * appears, disappears, or changes width, everything contributed BEFORE it
 * slides sideways and everything after it does not move at all.
 *
 * So order the array by how STABLE each item is, least stable first:
 *
 *   1. items contributed CONDITIONALLY — only in one view, only outside
 *      Simple, only while something is running;
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
 * hub hides lowest-priority items first (text truncates before anything
 * hides). That is a SEPARATE axis from order — it decides what survives a
 * narrow tile, not where anything sits — and the two may legitimately
 * disagree. Note the coupling, though: a re-contribution that widens a label
 * can itself make the bar crowded, so an item that both grows and carries the
 * lowest priority can fold ITSELF into the ⋯ menu at the moment it grows.
 * Both helpers are strict no-ops outside an embedded frame, so panels call
 * them unconditionally.
 *
 * SIMPLE MODE: the hub collapses a service tile's whole bar to zero height in
 * Simple mode, so anything contributed is invisible there. That suits chrome
 * Simple already drops (view switchers), but NOT a search box, which Simple
 * deliberately enlarges — nor anything safety-bearing. Gate those on
 * {@link isSimpleMode} and keep the in-body control for Simple; re-contribute
 * from the panel's existing `osprey-mode-change` handler.
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
 */

/**
 * @typedef {object} HeaderItem
 * @property {'text'|'nav'|'button'|'search'|'menu'} kind
 * @property {string} id
 * @property {number} [priority]
 * @property {string} [text]        text items
 * @property {HeaderNavEntry[] | HeaderMenuEntry[]} [items]  nav / menu items
 * @property {string} [label]       button / menu items
 * @property {string} [title]       button items
 * @property {'default'|'accent'} [tone]  button items
 * @property {boolean} [disabled]   button items
 * @property {string} [placeholder] search items
 * @property {string} [value]       search items (seed only — see the module docstring)
 */

/** @returns {boolean} true when running inside the web-terminal hub. */
function isEmbedded() {
  return document.body.classList.contains('embedded') && window.parent !== window;
}

/**
 * True when the shell is in Simple mode. Panels use this to decide whether a
 * control belongs in the bar at all: the hub collapses a service tile's bar
 * in Simple, so a search box (or anything safety-bearing) must stay in the
 * body there. Reads the same `data-ui-mode` attribute mode-boot.js stamps
 * pre-paint and the hub's `osprey-mode-change` handler keeps current, so it
 * is correct from first paint onward.
 * @returns {boolean}
 */
export function isSimpleMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple';
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
      { type: 'osprey-header-contribution', version: HEADER_CONTRACT_VERSION, items },
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
    if (!data || data.type !== 'osprey-header-action' || typeof data.id !== 'string') return;
    callback(data.id, typeof data.value === 'string' ? data.value : undefined);
  });
}
