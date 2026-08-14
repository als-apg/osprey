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
 * Reflect a panel's health on its status-bar item, if it has one. Hidden until
 * the panel's config has loaded (`state.url` set), then dot class live/error.
 * @param {Panel} panel
 * @param {{url: string | null, healthy: boolean}} state
 */
export function updateStatusBar(panel, state) {
  if (!panel.statusBarId) return;

  const statusItem = document.getElementById(panel.statusBarId);
  if (!statusItem) return;

  if (state.url) {
    statusItem.style.display = '';
    const dot = statusItem.querySelector('.status-dot');
    if (dot) {
      dot.className = 'status-dot' + (state.healthy ? ' live' : ' error');
    }
  }
}
