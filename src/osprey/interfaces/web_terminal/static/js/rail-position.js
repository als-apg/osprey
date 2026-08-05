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
 *
 * Two ways the rail moves, and they do not persist alike:
 *   - setRailPosition() is the explicit flip. It PINS the choice in
 *     localStorage, and a pinned choice outranks everything below forever.
 *   - followThemeFamily() is the theme coupling: the retro family restores
 *     the pre-redesign look, and the horizontal tab strip is part of that
 *     look, so switching family moves the rail with it. It never persists —
 *     switching back to another family moves the rail back. The coupling
 *     itself is served by GET /api/panels (app.FAMILY_RAIL_DEFAULTS), so
 *     this module holds no opinion about which family implies which rail.
 */

const STORAGE_KEY = 'osprey-rail-position';
const VALID_POSITIONS = ['left', 'top'];
const DEFAULT_POSITION = 'left';

/** @type {Record<string, string>} */
let _familyDefaults = {};
// True when web.rail_position named a position: a deployment that states one
// keeps it in every theme, so the family coupling stays out of the way.
let _configPinned = false;

/** Current rail position, read off <html>. @returns {"left"|"top"} */
export function getRailPosition() {
  const value = document.documentElement.getAttribute('data-rail-position');
  return value != null && VALID_POSITIONS.includes(value)
    ? /** @type {"left"|"top"} */ (value)
    : DEFAULT_POSITION;
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
 * Seed the theme coupling from the `GET /api/panels` payload. Safe to call
 * with a partial or absent payload — anything missing leaves the coupling
 * inert (no family implies a move), which is exactly the pre-coupling
 * behavior.
 * @param {{family_rail_defaults?: Record<string, string>, rail_position_configured?: boolean}|null|undefined} payload
 */
export function initRailThemeCoupling(payload) {
  const defaults = payload?.family_rail_defaults;
  _familyDefaults =
    defaults && typeof defaults === 'object' ? /** @type {Record<string, string>} */ (defaults) : {};
  _configPinned = payload?.rail_position_configured === true;
}

/**
 * Whether the operator has pinned a rail position by flipping it themselves.
 * A pinned choice is never overridden by a theme switch.
 * @returns {boolean}
 */
function hasUserPin() {
  try {
    return VALID_POSITIONS.includes(String(localStorage.getItem(STORAGE_KEY)));
  } catch {
    return false; // storage blocked — treat as unpinned
  }
}

/**
 * Move the rail to whatever `family` implies, unless the operator or the
 * deployment has stated a position of their own. Deliberately does NOT
 * persist: the rail follows the family for as long as that family is active,
 * and follows it back on the way out.
 * @param {string|null} family
 */
export function followThemeFamily(family) {
  if (_configPinned || hasUserPin()) return;
  const wanted = _familyDefaults[String(family)] || DEFAULT_POSITION;
  if (!VALID_POSITIONS.includes(wanted) || wanted === getRailPosition()) return;
  document.documentElement.setAttribute('data-rail-position', wanted);
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
