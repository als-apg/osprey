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
 * - `{kind:'text', id, text}` — inert label (e.g. a loaded-file name).
 * - `{kind:'nav', id, items:[{id, label, active}]}` — view switcher; the
 *   action's `value` is the clicked entry's id.
 * - `{kind:'button', id, label, title?, tone?:'default'|'accent', disabled?}`
 *   — workflow action.
 *
 * Every item may carry `priority` (number, default 0): in a narrow tile the
 * hub hides lowest-priority items first (text truncates before anything
 * hides). Both helpers are strict no-ops outside an embedded frame, so panels
 * call them unconditionally.
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
 * @typedef {object} HeaderItem
 * @property {'text'|'nav'|'button'} kind
 * @property {string} id
 * @property {number} [priority]
 * @property {string} [text]        text items
 * @property {HeaderNavEntry[]} [items]  nav items
 * @property {string} [label]       button items
 * @property {string} [title]       button items
 * @property {'default'|'accent'} [tone]  button items
 * @property {boolean} [disabled]   button items
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
