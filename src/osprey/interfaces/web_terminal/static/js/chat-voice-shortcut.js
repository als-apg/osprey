// @ts-check
/**
 * OSPREY Web Terminal — the key held to talk in voice input's `key` mode
 * (chat-voice.js): `web.voice.key` parsed into a combination, and matched
 * against key events.
 *
 * @module chat-voice-shortcut
 */

/**
 * A key combination held to talk.
 * @typedef {{ key: string, ctrl: boolean, alt: boolean, shift: boolean,
 *   meta: boolean, label: string, aria: string }} Shortcut
 */

/** Modifier names `web.voice.key` may use, by the flag each sets. */
const MODIFIERS = /** @type {Record<string, 'ctrl' | 'alt' | 'shift' | 'meta'>} */ ({
  ctrl: 'ctrl',
  control: 'ctrl',
  alt: 'alt',
  option: 'alt',
  shift: 'shift',
  meta: 'meta',
  cmd: 'meta',
  command: 'meta',
  win: 'meta',
  super: 'meta',
});
const ARIA_MODIFIERS = { ctrl: 'Control', alt: 'Alt', shift: 'Shift', meta: 'Meta' };
const LABEL_MODIFIERS = { ctrl: 'Ctrl', alt: 'Alt', shift: 'Shift', meta: 'Meta' };
/** Keys that already do something on the chat, so never a talk key. */
const RESERVED_KEYS = new Set(['escape', 'enter', 'tab']);

/**
 * Parse a key combination such as "Ctrl+Space", "Alt+M" or "F8".
 *
 * Null when it is not one the page can use: empty, two keys, a key the chat
 * already uses (Esc, Enter, Tab), or a single character with no modifier,
 * which would stop that character typing in the box.
 * @param {unknown} text
 * @returns {Shortcut | null}
 */
export function parseShortcut(text) {
  const parts = String(text ?? '')
    .split('+')
    .map((part) => part.trim())
    .filter(Boolean);
  const flags = { ctrl: false, alt: false, shift: false, meta: false };
  /** @type {string | null} */
  let key = null;
  for (const part of parts) {
    const modifier = MODIFIERS[part.toLowerCase()];
    if (modifier) flags[modifier] = true;
    else if (key === null) key = part;
    else return null;
  }
  if (key === null) return null;
  const value = /^space(bar)?$/i.test(key) ? ' ' : key;
  const held = /** @type {Array<'ctrl' | 'alt' | 'shift' | 'meta'>} */ (
    ['ctrl', 'alt', 'shift', 'meta'].filter((m) => flags[/** @type {'ctrl'} */ (m)])
  );
  if (RESERVED_KEYS.has(value.toLowerCase())) return null;
  if (value.length === 1 && value !== ' ' && held.length === 0) return null;
  const name = value === ' ' ? 'Space' : value.length === 1 ? value.toUpperCase() : value;
  return Object.freeze({
    key: value,
    ...flags,
    label: [...held.map((m) => LABEL_MODIFIERS[m]), name].join(' + '),
    aria: [...held.map((m) => ARIA_MODIFIERS[m]), name].join('+'),
  });
}

/** The talk key when `web.voice.key` says nothing: Ctrl + Space. */
export const DEFAULT_SHORTCUT = /** @type {Shortcut} */ (parseShortcut('Ctrl+Space'));

/**
 * Whether a key event is the shortcut: its key, with exactly its modifiers.
 * @param {KeyboardEvent} e
 * @param {Shortcut} shortcut
 * @returns {boolean}
 */
export function matchesShortcut(e, shortcut) {
  return (
    String(e.key).toLowerCase() === shortcut.key.toLowerCase() &&
    e.ctrlKey === shortcut.ctrl &&
    e.altKey === shortcut.alt &&
    e.shiftKey === shortcut.shift &&
    e.metaKey === shortcut.meta
  );
}

/**
 * Whether letting go of *key* ends a hold of *shortcut*: its key, or one of
 * its modifiers.
 * @param {string} key
 * @param {Shortcut} shortcut
 */
export function releasesShortcut(key, shortcut) {
  const lower = String(key).toLowerCase();
  return (
    lower === shortcut.key.toLowerCase() ||
    (shortcut.ctrl && lower === 'control') ||
    (shortcut.alt && lower === 'alt') ||
    (shortcut.shift && lower === 'shift') ||
    (shortcut.meta && lower === 'meta')
  );
}
