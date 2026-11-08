// @ts-check
/* The one definition of the shortcuts a menu row shows.
 *
 * A menu row that prints a chord promises the chord works. So the chord a row
 * prints and the code that answers to it read the same entry here: a hint
 * that no matcher claims cannot drift in unnoticed.
 *
 * Each binding is described per platform, as either
 *   - `{hint, matcher}` — this page claims the chord itself, or
 *   - `{hint, native: true}` — the browser already does the job (⌘C on macOS
 *     is the browser's own copy), so the hint names it and nothing is bound.
 *
 * A binding's `matcher` is what its handler installs, and it can claim a
 * chord on more platforms than the hint advertises: `terminal-copy` claims
 * Ctrl+Shift+C everywhere — on macOS too, where xterm would otherwise send
 * it as ^C — while the macOS hint names ⌘C.
 *
 * This module is a leaf: it imports only `platform.js`, read at call time
 * (never cached at import) so a test can flip the platform per case.
 */

import { isMacPlatform } from '/design-system/js/platform.js';

/** @typedef {(event: KeyboardEvent) => boolean} KeyMatcher */
/** @typedef {{hint: string, matcher: KeyMatcher} | {hint: string, native: true}} PlatformBinding */
/**
 * @typedef {object} KeyBinding
 * @property {KeyMatcher} matcher  what the binding's handler claims, on every platform
 * @property {string} [key]  the bare `event.key`, for a binding registered by key
 * @property {PlatformBinding} mac
 * @property {PlatformBinding} other
 */

/**
 * Ctrl+Shift+C, and no other modifier, on keydown.
 * @type {KeyMatcher}
 */
function isTerminalCopy(event) {
  if (event.type !== 'keydown') return false;
  if (!(event.ctrlKey && event.shiftKey && !event.metaKey && !event.altKey)) return false;
  return event.key === 'c' || event.key === 'C';
}

/**
 * Cmd+\ or Ctrl+\ (either, on every platform — as collapsible.js binds it),
 * without Shift or Alt.
 * @type {KeyMatcher}
 */
function isSidebarToggle(event) {
  if (event.type !== 'keydown') return false;
  if (!(event.metaKey || event.ctrlKey) || event.shiftKey || event.altKey) return false;
  return event.key === '\\';
}

/** @type {Readonly<Record<string, KeyBinding>>} */
export const KEY_BINDINGS = Object.freeze({
  'terminal-copy': {
    matcher: isTerminalCopy,
    mac: { hint: '⌘C', native: true },
    other: { hint: 'Ctrl+Shift+C', matcher: isTerminalCopy },
  },
  'sidebar-toggle': {
    matcher: isSidebarToggle,
    key: '\\',
    mac: { hint: '⌘\\', matcher: isSidebarToggle },
    other: { hint: 'Ctrl+\\', matcher: isSidebarToggle },
  },
});

/**
 * @param {string} id
 * @returns {KeyBinding}
 */
function entry(id) {
  const binding = KEY_BINDINGS[id];
  if (!binding) throw new Error(`Unknown key binding: ${id}`);
  return binding;
}

/**
 * The binding as this viewer's platform describes it.
 * @param {string} id
 * @param {boolean} [mac]  defaults to the current platform
 * @returns {PlatformBinding}
 */
export function bindingFor(id, mac = isMacPlatform()) {
  const binding = entry(id);
  return mac ? binding.mac : binding.other;
}

/**
 * The chord a menu row prints, as this viewer's platform spells it.
 * @param {string} id
 * @returns {string}
 */
export function hintFor(id) {
  return bindingFor(id).hint;
}

/**
 * The matcher a binding's handler installs (platform-independent).
 * @param {string} id
 * @returns {KeyMatcher}
 */
export function matcherFor(id) {
  return entry(id).matcher;
}

/**
 * The bare `event.key` of a binding registered by key (collapsible.js's
 * `shortcut` option).
 * @param {string} id
 * @returns {string}
 */
export function keyFor(id) {
  const key = entry(id).key;
  if (key === undefined) throw new Error(`Key binding ${id} has no bare key`);
  return key;
}
