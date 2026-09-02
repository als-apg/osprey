// @ts-check
/* OSPREY Web Terminal — Status-bar health readout.
 *
 * The one DOM consequence of a health poll settling that lives outside the
 * rail: panels with a `statusBarId` mirror their healthy flag onto the status
 * bar's dot. Kept out of panel-health.js on purpose — that module owns timing
 * and fetch only, with no DOM knowledge.
 */

/** @typedef {import('./panel-catalog.js').Panel} Panel */

/**
 * Reflect a panel's health on its status-bar item, if it has one. The dot ships
 * `hidden` and is revealed once the panel's config has loaded (`state.url`
 * set), then carries the dot class live/error.
 *
 * Revealing through the `hidden` ATTRIBUTE rather than through `style.display`
 * is what lets bar-host.js mirror it onto the surrounding `.bar-item` shell: a
 * dot that stays hidden must cost the bar no shell and no gap. The same
 * contract the docs link follows, and the reason `.status-item[hidden]` in
 * files.css sits beside that class's `display: flex`.
 *
 * The node is resolved BY ID on every call and never held. The dots are the
 * panel-health bar item's body, and that body is rebuilt whenever the item
 * changes host — a cached reference would keep painting a node the bar has
 * already thrown away. A declared id with no node in this deployment is a
 * no-op for the same reason it is not an error: the item renders a dot only
 * for a panel this deployment actually serves.
 * @param {Panel} panel
 * @param {{url: string | null, healthy: boolean}} state
 */
export function updateStatusBar(panel, state) {
  if (!panel.statusBarId) return;

  const statusItem = document.getElementById(panel.statusBarId);
  if (!statusItem) return;

  if (state.url) {
    statusItem.hidden = false;
    const dot = statusItem.querySelector('.status-dot');
    if (dot) {
      dot.className = 'status-dot' + (state.healthy ? ' live' : ' error');
    }
  }
}
