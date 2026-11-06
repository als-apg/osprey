// @ts-check
/**
 * ARIEL display preferences: whether list rows show the enhancement badge
 * strip. Stored per frame kind under `scopedStorageKey('ariel-strip-<frame>')`.
 * With nothing stored the strip is on standalone and, embedded, on unless the
 * panel is narrow (store `stacked`).
 *
 * Store keys: reads `stacked` (the embedded default); writes nothing.
 *
 * `onStripPrefChange(cb)` fires on every effective change: a `setStripPref`,
 * or a `stacked` flip while the default applies. `bindStripSwitch(button)`
 * wires the standalone `[role=switch]` projected into `<osprey-display-menu>`.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { getState, subscribe } from './store.js';

/** @returns {'embedded'|'standalone'} */
function frameKind() {
  return isEmbedded() ? 'embedded' : 'standalone';
}

/** @returns {string} */
export function stripStorageKey() {
  return scopedStorageKey(`ariel-strip-${frameKind()}`);
}

/** @returns {boolean|null} The stored choice, or null when none is stored. */
function readStored() {
  try {
    const raw = localStorage.getItem(stripStorageKey());
    if (raw === 'true') return true;
    if (raw === 'false') return false;
  } catch {
    // Storage unavailable: fall through to the default.
  }
  return null;
}

/** @returns {boolean} The default with nothing stored. */
function defaultPref() {
  return !isEmbedded() || !getState().stacked;
}

/**
 * Whether list rows show the badge strip.
 * @returns {boolean}
 */
export function getStripPref() {
  return readStored() ?? defaultPref();
}

/** @type {Set<(on: boolean) => void>} */
const listeners = new Set();

function emit() {
  const on = getStripPref();
  for (const cb of [...listeners]) cb(on);
}

/**
 * Store the choice for this frame kind and notify subscribers.
 * @param {boolean} on
 */
export function setStripPref(on) {
  try {
    localStorage.setItem(stripStorageKey(), on ? 'true' : 'false');
  } catch {
    // Storage unavailable: the choice lasts until reload.
  }
  emit();
}

/**
 * Follow the effective preference.
 * @param {(on: boolean) => void} cb
 * @returns {() => void} Unsubscribe
 */
export function onStripPrefChange(cb) {
  listeners.add(cb);
  const offStacked = subscribe((next, _prev, changed) => {
    if (changed.has('stacked') && isEmbedded() && readStored() === null) cb(getStripPref());
  });
  return () => {
    listeners.delete(cb);
    offStacked();
  };
}

/**
 * Wire the standalone strip switch: `aria-checked` follows the preference and
 * a click flips it.
 * @param {HTMLElement|null} button - A `[role=switch]` element
 * @returns {() => void} Unbind
 */
export function bindStripSwitch(button) {
  if (!button) return () => {};
  const paint = () => button.setAttribute('aria-checked', String(getStripPref()));
  const onClick = () => setStripPref(!getStripPref());
  button.addEventListener('click', onClick);
  const off = onStripPrefChange(paint);
  paint();
  return () => {
    button.removeEventListener('click', onClick);
    off();
  };
}
