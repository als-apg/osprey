// @ts-check
/* OSPREY Design System — Theme Manager
 *
 * Hand-written (not generated). Single runtime shared by every interface,
 * replacing web-terminal's old theme.js and each follower's private
 * postMessage handler. No framework, no build step — a plain ES module.
 *
 * Two roles, chosen by the page at init time via initTheme({role}):
 *   'hub'      — web-terminal only. Persists the user's preference to
 *                localStorage['osprey-theme'], live-follows the OS
 *                color-scheme preference while that preference is 'auto',
 *                and broadcasts every resolved change to every same-origin
 *                <iframe> panel via postMessage.
 *   'follower' — every embedded interface. Never persists or broadcasts;
 *                applies its own '?theme=' query param if present (panel-
 *                manager.js appends a validated one to each panel's iframe
 *                URL), otherwise trusts whatever theme-boot.js already set
 *                pre-paint, and applies whatever the hub broadcasts.
 *
 * Colors are never imported as JS data (tokens.js intentionally carries
 * none — see its own header comment). xtermPalette()/chartTheme()/
 * chartSeries() are *computed-style bridges*: every call does a fresh
 * getComputedStyle(document.documentElement) read of the --ansi-* and
 * --chart-* custom properties tokens.css defines for the theme currently
 * applied. There is no cache to invalidate, which is itself the fix for
 * the hidden-iframe empty-read problem below: a repeated read always has
 * another chance to succeed once the iframe is visible again.
 *
 * Hidden-iframe robustness protocol: a hidden (`display: none`) iframe's
 * getComputedStyle() can return empty strings for every custom property
 * in Firefox, even though the value is set correctly. This module never
 * treats an unchanged theme id as a no-op — every apply() (init, message,
 * explicit setTheme/toggleTheme, or a live OS preference change)
 * re-notifies every subscribe() callback unconditionally, even when the
 * id is identical to what's already applied. That is what makes
 * panel-manager.js's "re-send the theme on tab activation, even if
 * unchanged" repair path work: a panel that was hidden (and so may have
 * read empty colors) on the last apply gets a completely fresh read the
 * next time it becomes visible and the hub re-sends. The bridge functions
 * additionally validate one sentinel token (--bg-primary, present in
 * every theme) before trusting the rest of a read; an empty sentinel logs
 * a console.error (the behavioral test suite asserts this never fires
 * across the documented user flows) and marks the read dirty rather than
 * silently returning colors it can't vouch for.
 */

import { DEFAULTS, THEMES } from './tokens.js';

const STORAGE_KEY = 'osprey-theme';
const MESSAGE_TYPE = 'osprey-theme-change';

// The computed-style bridges validate every read against this token
// first. --bg-primary is defined by every theme (WCAG-gated, never an
// alpha composite), so an empty read for it means the read as a whole
// can't be trusted -- not that this particular theme has no background.
const SENTINEL_VAR = '--bg-primary';

// Generous headroom above the 6 categorical chart.series.* steps the
// design system ships today; chartSeries() just stops collecting past
// this bound rather than treating the first empty slot as "the end" (an
// empty slot could just as easily be a transient hidden-iframe read).
const MAX_CHART_SERIES = 12;

const _themesById = new Map(THEMES.map((theme) => [theme.id, theme]));
const _validIds = THEMES.map((theme) => theme.id);

// ---- Module state ----

let _role = 'follower';
// The user's actual preference: 'auto' or a concrete theme id. Only ever
// meaningful (and only ever persisted) for the hub; a follower has no
// preference of its own; it just applies whatever it's told.
let _preference = 'auto';
// The currently applied, always-concrete theme id. Never 'auto' -- that
// is resolved before anything touches data-theme.
/** @type {string|null} */
let _currentId = null;
/** @type {Set<(id: string) => void>} */
const _subscribers = new Set();
let _messageListenerAttached = false;
let _mediaQueryListenerAttached = false;
let _toggleButtonWired = false;

// ---- id / mode resolution ----

/**
 * @param {unknown} value
 * @returns {value is string}
 */
function _isKnownId(value) {
  return typeof value === 'string' && _validIds.includes(value);
}

/**
 * @param {string} id
 * @returns {string}
 */
function _modeOf(id) {
  const theme = _themesById.get(id);
  return theme ? theme.mode : 'dark';
}

function _prefersDarkOS() {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  } catch (error) {
    return true;
  }
}

function _resolveAuto() {
  const mode = _prefersDarkOS() ? 'dark' : 'light';
  return DEFAULTS[mode] || DEFAULTS.dark || DEFAULTS.light || _validIds[0];
}

/**
 * Resolve a preference ('auto' or a candidate id) to a concrete, valid theme id.
 * @param {unknown} preference
 * @returns {string}
 */
function _resolve(preference) {
  if (preference === 'auto') return _resolveAuto();
  if (_isKnownId(preference)) return preference;
  return _resolveAuto();
}

// ---- Reading the initial preference (?theme=, localStorage, data-theme) ----

function _readQueryTheme() {
  try {
    return new URLSearchParams(window.location.search).get('theme');
  } catch (error) {
    return null;
  }
}

function _readStoredPreference() {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

/** @param {string} preference */
function _persistPreference(preference) {
  try {
    window.localStorage.setItem(STORAGE_KEY, preference);
  } catch (error) {
    // Storage unavailable (private browsing, quota) -- non-fatal; the
    // preference just won't survive a reload this session.
  }
}

// ---- Applying a theme (data-theme, hljs swap, toggle UI, subscribers) ----

/** @param {string} mode */
function _swapHljsHref(mode) {
  const link = /** @type {HTMLLinkElement|null} */ (document.getElementById('hljs-theme'));
  if (!link) return; // pages without a highlight.js stylesheet (most)
  const attr = mode === 'light' ? 'data-href-light' : 'data-href-dark';
  const href = link.getAttribute(attr);
  if (href) link.href = href;
}

/** @param {string} id */
function _updateToggleUI(id) {
  const button = document.getElementById('theme-toggle');
  const sunIcon = document.getElementById('theme-icon-sun');
  const moonIcon = document.getElementById('theme-icon-moon');
  const mode = _modeOf(id);

  if (sunIcon && moonIcon) {
    sunIcon.style.display = mode === 'dark' ? 'block' : 'none';
    moonIcon.style.display = mode === 'light' ? 'block' : 'none';
  }
  if (button) {
    const targetMode = mode === 'dark' ? 'light' : 'dark';
    button.setAttribute('aria-label', `Switch to ${targetMode} theme`);
    button.setAttribute('aria-pressed', mode === 'light' ? 'true' : 'false');
  }
}

/** @param {() => void} fn */
function _withTransition(fn) {
  const root = document.documentElement;
  root.classList.add('theme-transitioning');
  fn();
  window.setTimeout(() => root.classList.remove('theme-transitioning'), 300);
}

/** @param {string} id */
function _notifySubscribers(id) {
  for (const callback of _subscribers) {
    try {
      callback(id);
    } catch (error) {
      console.error('osprey theme-manager: a subscribe() callback threw', error);
    }
  }
}

// ---- Clearing the one-shot ?theme= query param ----

/**
 * Strip a `theme` param from the URL's query string, if present, without
 * adding a history entry. Called from setTheme() -- the explicit-choice
 * path -- for both roles: once the user has made an explicit choice, a
 * leftover `?theme=` must not out-rank it (or the OS/localStorage
 * resolution a follower falls back to) on the next reload. initTheme()'s
 * one-time read of `?theme=` still happens first, so an incoming panel
 * URL still applies its param on first load -- this only clears it after
 * that initial resolution so it doesn't linger.
 */
function _stripQueryTheme() {
  try {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('theme')) return;
    params.delete('theme');
    const query = params.toString();
    const url = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
    window.history.replaceState(window.history.state, '', url);
  } catch (error) {
    // Non-browser environment or a blocked history API -- non-fatal.
  }
}

/** @param {string} id */
function _broadcast(id) {
  const iframes = document.querySelectorAll('iframe');
  for (const iframe of iframes) {
    try {
      iframe.contentWindow?.postMessage({ type: MESSAGE_TYPE, theme: id }, window.location.origin);
    } catch (error) {
      // Cross-origin -- expected for some iframes; nothing to do.
    }
  }
}

/**
 * Apply a concrete theme id: set data-theme, swap the hljs stylesheet,
 * update the toggle button, and notify every subscriber. NEVER deduped on
 * an unchanged id -- see the module docstring's hidden-iframe protocol.
 *
 * @param {string} id
 * @param {{broadcast?: boolean, transition?: boolean}} [options]
 */
function _applyTheme(id, { broadcast = false, transition = false } = {}) {
  _currentId = id;

  const apply = () => {
    document.documentElement.setAttribute('data-theme', id);
    _swapHljsHref(_modeOf(id));
    _updateToggleUI(id);
  };
  if (transition) {
    _withTransition(apply);
  } else {
    apply();
  }

  _notifySubscribers(id);
  if (broadcast) _broadcast(id);
}

// ---- Follower: obey hub broadcasts ----

/** @param {MessageEvent} event */
function _handleMessage(event) {
  if (event.origin !== window.location.origin) return;
  const data = event.data;
  if (!data || data.type !== MESSAGE_TYPE) return;
  // Never applies an arbitrary string to data-theme: _resolve() falls
  // back to 'auto' for anything not in the baked-in id list.
  _applyTheme(_resolve(data.theme));
}

function _attachMessageListener() {
  if (_messageListenerAttached) return;
  _messageListenerAttached = true;
  window.addEventListener('message', _handleMessage);
}

// ---- Hub: live-follow the OS preference while in 'auto' ----

function _attachMediaQueryListener() {
  if (_mediaQueryListenerAttached) return;
  _mediaQueryListenerAttached = true;

  let media;
  try {
    media = window.matchMedia('(prefers-color-scheme: dark)');
  } catch (error) {
    return;
  }

  const handleChange = () => {
    if (_preference !== 'auto') return; // an explicit choice does not auto-follow the OS
    _applyTheme(_resolveAuto(), { broadcast: true });
  };

  if (typeof media.addEventListener === 'function') {
    media.addEventListener('change', handleChange);
  } else if (typeof media.addListener === 'function') {
    // Safari < 14 / older WebKit.
    media.addListener(handleChange);
  }
}

// ---- Toggle button wiring ----

function _wireToggleButton() {
  if (_toggleButtonWired) return;
  const button = document.getElementById('theme-toggle');
  if (!button) return;
  _toggleButtonWired = true;
  button.addEventListener('click', toggleTheme);
}

// ---- Public API ----

/**
 * Initialize the theme runtime. Call once per page, before any code that
 * depends on the applied theme (e.g. before constructing an xterm
 * Terminal or a Plotly chart) -- theme-boot.js has already set
 * data-theme pre-paint, but subscribers only exist after this runs.
 *
 * @param {{role?: 'hub'|'follower'}} [options]
 */
export function initTheme({ role = 'follower' } = {}) {
  _role = role === 'hub' ? 'hub' : 'follower';

  const queryTheme = _readQueryTheme();
  const attrTheme = document.documentElement.getAttribute('data-theme');

  if (_role === 'hub') {
    const stored = _readStoredPreference();
    const queryPreference = queryTheme === 'auto' || _isKnownId(queryTheme) ? queryTheme : null;
    const storedPreference = stored === 'auto' || _isKnownId(stored) ? stored : null;
    _preference = queryPreference || storedPreference || 'auto';
    _applyTheme(_resolve(_preference));
    _attachMediaQueryListener();
  } else {
    // Followers have no persisted preference of their own: apply
    // ?theme= if present and valid, else trust theme-boot.js's already-
    // applied data-theme (same-origin panels share the hub's
    // localStorage, so it already resolved the same preference).
    const initial = _isKnownId(queryTheme)
      ? queryTheme
      : _isKnownId(attrTheme)
        ? attrTheme
        : _resolveAuto();
    _applyTheme(initial);
    _attachMessageListener();
  }

  _wireToggleButton();
}

/**
 * The currently applied, concrete theme id (never 'auto').
 * @returns {string|null}
 */
export function getTheme() {
  return _currentId;
}

/**
 * Set the theme. `id` may be a concrete theme id or 'auto'; anything else
 * resolves to 'auto'. Only the hub role persists (to
 * localStorage['osprey-theme']) and broadcasts to embedded panels. Both
 * roles strip a `theme` query param from the URL (D15): this is the
 * explicit-choice path (reached only via the toggle button), so a
 * leftover `?theme=` must not out-rank it on the next reload.
 *
 * @param {string} id
 */
export function setTheme(id) {
  const preference = id === 'auto' || _isKnownId(id) ? id : 'auto';
  if (_role === 'hub') {
    _preference = preference;
    _persistPreference(preference);
  }
  _applyTheme(_resolve(preference), { broadcast: _role === 'hub', transition: true });
  _stripQueryTheme();
}

/** Cycle between the resolved dark and light defaults (never sets 'auto'). */
export function toggleTheme() {
  const nextMode = _modeOf(/** @type {string} */ (_currentId)) === 'dark' ? 'light' : 'dark';
  setTheme(DEFAULTS[nextMode] || /** @type {string} */ (_currentId));
}

/**
 * Register a callback invoked with the applied theme id on every apply --
 * including applies where the id is unchanged (see the module docstring).
 * Returns an unsubscribe function.
 *
 * @param {(id: string) => void} callback
 * @returns {() => void}
 */
export function subscribe(callback) {
  _subscribers.add(callback);
  return () => _subscribers.delete(callback);
}

// ---- Computed-style bridges (the only runtime color source) ----

function _computedStyles() {
  return getComputedStyle(document.documentElement);
}

/**
 * @param {CSSStyleDeclaration} styles
 * @param {string} name
 * @returns {string}
 */
function _readVar(styles, name) {
  return styles.getPropertyValue(name).trim();
}

/**
 * Validate the sentinel token; logs (never throws) on an empty read.
 * @param {CSSStyleDeclaration} styles
 * @returns {boolean}
 */
function _checkSentinel(styles) {
  if (_readVar(styles, SENTINEL_VAR)) {
    return true;
  }
  console.error(
    `osprey theme-manager: computed style read for ${SENTINEL_VAR} was empty ` +
      '(a hidden iframe on Firefox reads empty custom properties); the next ' +
      'apply() re-fires every subscriber, giving this bridge another chance.'
  );
  return false;
}

/** xterm.js `theme` option, built from --ansi-* custom properties. */
export function xtermPalette() {
  const styles = _computedStyles();
  _checkSentinel(styles);
  return {
    background: _readVar(styles, '--ansi-background'),
    foreground: _readVar(styles, '--ansi-foreground'),
    cursor: _readVar(styles, '--ansi-cursor'),
    cursorAccent: _readVar(styles, '--ansi-cursor-accent'),
    selectionBackground: _readVar(styles, '--ansi-selection'),
    black: _readVar(styles, '--ansi-black'),
    red: _readVar(styles, '--ansi-red'),
    green: _readVar(styles, '--ansi-green'),
    yellow: _readVar(styles, '--ansi-yellow'),
    blue: _readVar(styles, '--ansi-blue'),
    magenta: _readVar(styles, '--ansi-magenta'),
    cyan: _readVar(styles, '--ansi-cyan'),
    white: _readVar(styles, '--ansi-white'),
    brightBlack: _readVar(styles, '--ansi-bright-black'),
    brightRed: _readVar(styles, '--ansi-bright-red'),
    brightGreen: _readVar(styles, '--ansi-bright-green'),
    brightYellow: _readVar(styles, '--ansi-bright-yellow'),
    brightBlue: _readVar(styles, '--ansi-bright-blue'),
    brightMagenta: _readVar(styles, '--ansi-bright-magenta'),
    brightCyan: _readVar(styles, '--ansi-bright-cyan'),
    brightWhite: _readVar(styles, '--ansi-bright-white'),
  };
}

/**
 * A Plotly `layout` fragment, built from --chart-* custom properties.
 * Spread directly into a layout object, e.g.
 * `Plotly.relayout(el, chartTheme())` or `{...layout, ...chartTheme()}`.
 */
export function chartTheme() {
  const styles = _computedStyles();
  _checkSentinel(styles);
  const gridcolor = _readVar(styles, '--chart-grid');
  return {
    paper_bgcolor: _readVar(styles, '--chart-paper-bg'),
    plot_bgcolor: _readVar(styles, '--chart-plot-bg'),
    font: { color: _readVar(styles, '--chart-axis-text') },
    xaxis: { gridcolor },
    yaxis: { gridcolor },
  };
}

/** The categorical chart color palette, built from --chart-series-N. */
export function chartSeries() {
  const styles = _computedStyles();
  _checkSentinel(styles);
  const series = [];
  for (let index = 1; index <= MAX_CHART_SERIES; index++) {
    const value = _readVar(styles, `--chart-series-${index}`);
    if (value) series.push(value);
  }
  return series;
}
