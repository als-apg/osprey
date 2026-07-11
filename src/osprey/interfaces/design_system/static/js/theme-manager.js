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
 * Family model: a THEMES entry is `{id, label, mode, family}` -- a family
 * is a `{light, dark}` pair (e.g. the built-in 'osprey' family, or the
 * WCAG-AAA 'high-contrast' family). DEFAULTS is keyed by family, then mode:
 * `DEFAULTS[family][mode] -> concrete id`. The hub's preference is a
 * (family, mode|auto) pair, not a single id -- picking a family and
 * toggling mode never lose each other: toggling flips mode within the
 * active family, and 'auto' resolves the OS preference within the active
 * family. See setTheme()/setFamily()/toggleTheme() below for the exact
 * contract.
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

import { DEFAULT_FAMILY as _EMITTED_DEFAULT_FAMILY, DEFAULTS, THEMES } from './tokens.js';

/** @typedef {{id: string, label: string, mode: string, family: string}} ThemeEntry */
/** @typedef {'auto'|'dark'|'light'} ModePreference */
/** @typedef {{family: string, mode: ModePreference}} Preference */

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

// tokens.js is plain (unchecked) generated JS: cast its exports to the
// shapes documented above rather than relying on tsc's own inference of
// the literal object it emits. This also means theme-manager.js type-checks
// against the *documented* family contract even if tokens.js momentarily
// lags a build regeneration.
const _themes = /** @type {ThemeEntry[]} */ (THEMES);
/** @type {Record<string, Record<string, string>>} */
const _defaults = /** @type {Record<string, Record<string, string>>} */ (DEFAULTS);

const _themesById = new Map(_themes.map((theme) => [theme.id, theme]));
const _validIds = _themes.map((theme) => theme.id);

// The single authoritative fallback family (first family declared in the
// manifest, insertion/filename order -- never re-sorted), emitted by
// emit_js.py's render_tokens_js and shared verbatim with theme-boot.js's
// own baked copy (see that generator's docstring) -- this module never
// re-derives it from DEFAULTS, so the two generated-consuming runtimes
// can't drift on a future regeneration. Falls back to 'osprey' only in
// the pathological case of an empty manifest (no families declared at
// all), which build validation never allows in practice.
const DEFAULT_FAMILY = _isKnownFamily(_EMITTED_DEFAULT_FAMILY) ? _EMITTED_DEFAULT_FAMILY : 'osprey';

// ---- Module state ----

let _role = 'follower';
// The user's actual preference: a (family, mode) pair, where mode may be
// 'auto'. Only ever meaningful (and only ever persisted) for the hub; a
// follower has no preference of its own -- it just applies whatever it's
// told.
let _preferenceFamily = DEFAULT_FAMILY;
/** @type {ModePreference} */
let _preferenceMode = 'auto';
// The family of the currently applied, concrete theme id. Kept in sync on
// every _applyTheme() call (both roles) so toggleTheme()/setFamily() and
// the follower's own _resolve() always have a family to stay within, even
// though only the hub tracks an explicit preference.
let _activeFamily = DEFAULT_FAMILY;
// The currently applied, always-concrete theme id. Never 'auto' -- that
// is resolved before anything touches data-theme.
/** @type {string|null} */
let _currentId = null;
/** @type {Set<(id: string) => void>} */
const _subscribers = new Set();
let _messageListenerAttached = false;
let _mediaQueryListenerAttached = false;

// ---- id / family / mode helpers ----

/**
 * @param {unknown} value
 * @returns {value is string}
 */
function _isKnownId(value) {
  return typeof value === 'string' && _validIds.includes(value);
}

/**
 * @param {unknown} value
 * @returns {value is string}
 */
function _isKnownFamily(value) {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(_defaults, value);
}

/**
 * @param {unknown} value
 * @returns {value is ModePreference}
 */
function _isKnownMode(value) {
  return value === 'auto' || value === 'dark' || value === 'light';
}

/**
 * @param {string} id
 * @returns {string}
 */
function _modeOf(id) {
  const theme = _themesById.get(id);
  return theme ? theme.mode : 'dark';
}

/**
 * @param {string} id
 * @returns {string}
 */
function _familyOf(id) {
  const theme = _themesById.get(id);
  return theme ? theme.family : DEFAULT_FAMILY;
}

function _prefersDarkOS() {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  } catch {
    return true;
  }
}

/**
 * Resolve a family's default id for an explicit dark/light mode.
 * @param {string} family
 * @param {'dark'|'light'} mode
 * @returns {string}
 */
function _resolveInFamily(family, mode) {
  const familyDefaults = _defaults[family] || _defaults[DEFAULT_FAMILY];
  if (!familyDefaults) return _validIds[0];
  return familyDefaults[mode] || familyDefaults.dark || familyDefaults.light || _validIds[0];
}

/**
 * 'auto' resolves the OS light/dark preference WITHIN the given family --
 * never the global first-family default.
 * @param {string} family
 * @returns {string}
 */
function _resolveAuto(family) {
  const mode = _prefersDarkOS() ? 'dark' : 'light';
  return _resolveInFamily(family, mode);
}

/**
 * Resolve a full (family, mode|auto) preference to a concrete, valid id.
 * @param {string} family
 * @param {ModePreference} mode
 * @returns {string}
 */
function _resolvePreference(family, mode) {
  const validFamily = _isKnownFamily(family) ? family : DEFAULT_FAMILY;
  if (mode === 'dark' || mode === 'light') return _resolveInFamily(validFamily, mode);
  return _resolveAuto(validFamily);
}

/**
 * Resolve a preference ('auto' or a candidate id) to a concrete, valid
 * theme id, WITHIN the currently active family (never jumps family). Used
 * by the follower's message handler -- "preference" there is really
 * whatever raw token the hub broadcast.
 * @param {unknown} preference
 * @returns {string}
 */
function _resolve(preference) {
  if (preference === 'auto') return _resolveAuto(_activeFamily);
  if (_isKnownId(preference)) return preference;
  return _resolveAuto(_activeFamily);
}

// ---- Reading the initial preference (?theme=, localStorage, data-theme) ----

function _readQueryTheme() {
  try {
    return new URLSearchParams(window.location.search).get('theme');
  } catch {
    return null;
  }
}

/**
 * Parse a bare preference token -- 'auto' or a concrete theme id, the
 * legacy pre-family-model shape used by both `?theme=` query params and
 * (before this task) the single-value localStorage format -- into a
 * `{family, mode}` pair. Never throws; an unrecognized token yields null
 * so the caller can fail safe to auto within DEFAULT_FAMILY.
 * @param {unknown} token
 * @returns {Preference|null}
 */
function _parsePreferenceToken(token) {
  if (token === 'auto') return { family: DEFAULT_FAMILY, mode: 'auto' };
  if (_isKnownId(token)) {
    return { family: _familyOf(token), mode: /** @type {'dark'|'light'} */ (_modeOf(token)) };
  }
  return null;
}

/**
 * Read the persisted (family, mode) preference from localStorage.
 * Understands the current structured JSON format (`{"family":...,
 * "mode":...}`) and migrates the pre-family-model bare-string format
 * ('auto' or a concrete id like 'dark') forward via _parsePreferenceToken.
 * Never throws -- storage errors, malformed JSON, and unrecognized values
 * all resolve to null (caller falls back to auto within DEFAULT_FAMILY).
 * @returns {Preference|null}
 */
function _readStoredPreference() {
  /** @type {string|null} */
  let raw;
  try {
    raw = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
  if (raw === null) return null;

  try {
    const parsed = JSON.parse(raw);
    if (
      parsed &&
      typeof parsed === 'object' &&
      _isKnownFamily(parsed.family) &&
      _isKnownMode(parsed.mode)
    ) {
      return { family: parsed.family, mode: parsed.mode };
    }
  } catch {
    // Not JSON -- fall through to the legacy bare-token format below.
  }

  return _parsePreferenceToken(raw);
}

/**
 * Persist the (family, mode) preference as `{"family":..., "mode":...}`
 * JSON under STORAGE_KEY.
 * @param {string} family
 * @param {ModePreference} mode
 */
function _persistPreference(family, mode) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ family, mode }));
  } catch {
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
 * adding a history entry. Called from setTheme()/setFamily() -- the
 * explicit-choice path -- for both roles: once the user has made an
 * explicit choice, a leftover `?theme=` must not out-rank it (or the
 * OS/localStorage resolution a follower falls back to) on the next
 * reload. initTheme()'s one-time read of `?theme=` still happens first, so
 * an incoming panel URL still applies its param on first load -- this only
 * clears it after that initial resolution so it doesn't linger.
 */
function _stripQueryTheme() {
  try {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('theme')) return;
    params.delete('theme');
    const query = params.toString();
    const url = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
    window.history.replaceState(window.history.state, '', url);
  } catch {
    // Non-browser environment or a blocked history API -- non-fatal.
  }
}

/** @param {string} id */
function _broadcast(id) {
  const iframes = document.querySelectorAll('iframe');
  for (const iframe of iframes) {
    try {
      iframe.contentWindow?.postMessage({ type: MESSAGE_TYPE, theme: id }, window.location.origin);
    } catch {
      // Cross-origin -- expected for some iframes; nothing to do.
    }
  }
}

/**
 * Apply a concrete theme id: set data-theme, swap the hljs stylesheet, and
 * notify every subscriber. NEVER deduped on an unchanged id -- see the
 * module docstring's hidden-iframe protocol.
 *
 * @param {string} id
 * @param {{broadcast?: boolean, transition?: boolean}} [options]
 */
function _applyTheme(id, { broadcast = false, transition = false } = {}) {
  _currentId = id;
  _activeFamily = _familyOf(id);

  const apply = () => {
    document.documentElement.setAttribute('data-theme', id);
    _swapHljsHref(_modeOf(id));
  };
  if (transition) {
    _withTransition(apply);
  } else {
    apply();
  }

  _notifySubscribers(id);
  if (broadcast) _broadcast(id);
}

/**
 * Resolve and apply a (family, mode|auto) preference -- the shared core
 * behind setTheme() and setFamily(). Only the hub role persists it (and
 * only the hub broadcasts the resolved concrete id to embedded panels).
 * Unknown family/mode values fail safe to DEFAULT_FAMILY/'auto' rather
 * than throwing.
 *
 * @param {string} family
 * @param {ModePreference} mode
 */
function _applyPreference(family, mode) {
  const validFamily = _isKnownFamily(family) ? family : DEFAULT_FAMILY;
  const validMode = _isKnownMode(mode) ? mode : 'auto';
  if (_role === 'hub') {
    _preferenceFamily = validFamily;
    _preferenceMode = validMode;
    _persistPreference(validFamily, validMode);
  }
  _applyTheme(_resolvePreference(validFamily, validMode), {
    broadcast: _role === 'hub',
    transition: true,
  });
  _stripQueryTheme();
}

// ---- Follower: obey hub broadcasts ----

/** @param {MessageEvent} event */
function _handleMessage(event) {
  if (event.origin !== window.location.origin) return;
  const data = event.data;
  if (!data || data.type !== MESSAGE_TYPE) return;
  // Never applies an arbitrary string to data-theme: _resolve() falls
  // back to 'auto' (within the active family) for anything not in the
  // baked-in id list.
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
  } catch {
    return;
  }

  const handleChange = () => {
    if (_preferenceMode !== 'auto') return; // an explicit choice does not auto-follow the OS
    _applyTheme(_resolveAuto(_preferenceFamily), { broadcast: true });
  };

  if (typeof media.addEventListener === 'function') {
    media.addEventListener('change', handleChange);
  } else if (typeof media.addListener === 'function') {
    // Safari < 14 / older WebKit.
    media.addListener(handleChange);
  }
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
    const queryPreference = _parsePreferenceToken(queryTheme);
    const storedPreference = _readStoredPreference();
    const preference = queryPreference || storedPreference || {
      family: DEFAULT_FAMILY,
      mode: /** @type {ModePreference} */ ('auto'),
    };
    _preferenceFamily = preference.family;
    _preferenceMode = preference.mode;
    _applyTheme(_resolvePreference(_preferenceFamily, _preferenceMode));
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
        : _resolveAuto(DEFAULT_FAMILY);
    _applyTheme(initial);
    _attachMessageListener();
  }
}

/**
 * The currently applied, concrete theme id (never 'auto').
 * @returns {string|null}
 */
export function getTheme() {
  return _currentId;
}

/**
 * The family of the currently applied, concrete theme id.
 * @returns {string|null}
 */
export function getFamily() {
  return _currentId ? _familyOf(_currentId) : null;
}

/**
 * Set the theme. `id` may be a concrete theme id (its family is derived
 * from THEMES) or 'auto' (resolves the OS preference WITHIN the currently
 * active family -- it never jumps family). Anything else fails safe to
 * 'auto'. Only the hub persists (to localStorage['osprey-theme'], as a
 * `{"family":..., "mode":...}` JSON pair -- see the module docstring) and
 * broadcasts the resolved concrete id to embedded panels. Both roles strip
 * a `theme` query param from the URL (D15): this is the explicit-choice
 * path, so a leftover `?theme=` must not out-rank it on the next reload.
 *
 * To switch family while PRESERVING the current mode preference (what a
 * family-picker switcher needs), use setFamily() instead -- setTheme()
 * with a concrete id always sets mode to that id's own mode.
 *
 * @param {string} id
 */
export function setTheme(id) {
  if (id === 'auto') {
    _applyPreference(_activeFamily, 'auto');
    return;
  }
  if (_isKnownId(id)) {
    const theme = /** @type {ThemeEntry} */ (_themesById.get(id));
    _applyPreference(theme.family, /** @type {'dark'|'light'} */ (theme.mode));
    return;
  }
  _applyPreference(_activeFamily, 'auto');
}

/**
 * Switch to `family`, PRESERVING the current mode preference: if the mode
 * preference is 'auto' it stays 'auto' (still OS-resolved, now within the
 * new family); if it's an explicit dark/light it stays that mode. Falls
 * back to DEFAULT_FAMILY on an unrecognized family id (fail-safe, never
 * throws). Same persistence/broadcast/query-strip behavior as setTheme().
 *
 * Contract for the family-picker switcher (Task 1.9):
 *   `setFamily(family: string): void`
 * Call it with one of the family ids that key `DEFAULTS` / appear as
 * `THEMES[].family` (e.g. `'osprey'`, `'high-contrast'`). Use
 * `toggleTheme()` for the mode control and `getFamily()`/`getTheme()` to
 * read back current state (e.g. to mark the active family selected).
 *
 * @param {string} family
 */
export function setFamily(family) {
  const mode =
    _role === 'hub'
      ? _preferenceMode
      : /** @type {ModePreference} */ (_modeOf(/** @type {string} */ (_currentId)));
  _applyPreference(family, mode);
}

/**
 * Cycle between dark and light WITHIN the currently active family (never
 * jumps family, never sets 'auto').
 */
export function toggleTheme() {
  const nextMode = _modeOf(/** @type {string} */ (_currentId)) === 'dark' ? 'light' : 'dark';
  setTheme(_resolveInFamily(_activeFamily, nextMode));
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
