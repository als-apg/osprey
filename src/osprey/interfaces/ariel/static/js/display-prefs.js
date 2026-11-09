// @ts-check
/**
 * ARIEL display preferences: whether list rows show the enhancement badge
 * strip. Stored under `scopedStorageKey('ariel-strip-embedded')`; the key keeps
 * its suffix so an operator's stored choice survives. With nothing stored the
 * strip is on unless the panel is narrow (store `stacked`).
 *
 * Store keys: reads `stacked` (the default); writes nothing.
 *
 * `onStripPrefChange(cb)` fires on every effective change: a `setStripPref`,
 * or a `stacked` flip while the default applies.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { getState, subscribe } from './store.js';

/** @returns {string} */
export function stripStorageKey() {
  return scopedStorageKey('ariel-strip-embedded');
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
  return !getState().stacked;
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
    if (changed.has('stacked') && readStored() === null) cb(getStripPref());
  });
  return () => {
    listeners.delete(cb);
    offStacked();
  };
}
