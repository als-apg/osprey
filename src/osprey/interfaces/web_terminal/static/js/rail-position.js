// @ts-check
/* OSPREY Web Terminal — Rail-Position Runtime Axis (left | top)
 *
 * Chrome-only sibling of the ui-mode axis: the pre-paint rung lives in the
 * design system's rail-boot.js; this module owns the explicit runtime flip.
 * Never forwarded to panel iframes — embedded panels don't care where the
 * host rail sits.
 *
 * The resize event is the relayout seam: dockview watches its container and
 * terminal.js re-fits xterm on window resize, so one dispatch covers both
 * after the CSS reflows the shell around the moved rail.
 */

const STORAGE_KEY = 'osprey-rail-position';
const VALID_POSITIONS = ['left', 'top'];
const DEFAULT_POSITION = 'left';

/** Current rail position, read off <html>. @returns {"left"|"top"} */
export function getRailPosition() {
  const value = document.documentElement.getAttribute('data-rail-position');
  return VALID_POSITIONS.includes(value) ? /** @type {"left"|"top"} */ (value) : DEFAULT_POSITION;
}

/**
 * Flip the rail: stamp the attribute (the CSS gate), persist the explicit
 * choice, drop a leftover one-shot ?rail=, and nudge the layout engines.
 * Invalid input is rejected wholesale — no write, no persistence.
 * @param {"left"|"top"} position
 */
export function setRailPosition(position) {
  if (!VALID_POSITIONS.includes(position)) return;
  document.documentElement.setAttribute('data-rail-position', position);
  try {
    localStorage.setItem(STORAGE_KEY, position);
  } catch { /* storage blocked — the flip still applies for this session */ }
  stripQueryRail();
  window.dispatchEvent(new Event('resize'));
}

/**
 * Strip a one-shot `rail` param from the URL's query string, if present,
 * without adding a history entry — the rail-axis twin of app.js's
 * stripQueryMode(). Once a choice is made explicit, a leftover `?rail=`
 * must not out-rank it (or localStorage) on the next reload.
 */
function stripQueryRail() {
  try {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('rail')) return;
    params.delete('rail');
    const query = params.toString();
    const url = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
    window.history.replaceState(window.history.state, '', url);
  } catch { /* non-browser environment or a blocked history API — non-fatal */ }
}
