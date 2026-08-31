// @ts-check
/* OSPREY Web Terminal — "Don't ask again" storage
 *
 * The two localStorage moves behind every waivable confirm: read whether one
 * was waived, and record or clear the waiver as the checkbox was left when
 * the operator confirmed. Key bases are the caller's (one per dialog, plus
 * whatever narrows it — per machine for the control-target confirms); every
 * key resolves through scopedStorageKey(), so on a multi-user mount one
 * operator's waiver never silences the dialog for everyone else.
 *
 * Storage that throws (blocked, private window) reads as "never waived" —
 * the dialog then shows, which is the safe direction to fail in.
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';

/**
 * Whether this confirm was waived with "don't ask again".
 * @param {string} keyBase  unscoped key base, one per dialog
 * @returns {boolean}
 */
export function confirmSkipped(keyBase) {
  try {
    return localStorage.getItem(scopedStorageKey(keyBase)) === '1';
  } catch {
    return false;
  }
}

/**
 * Record or clear one confirm's waiver. Only a CONFIRM should record it — a
 * cancel with the box ticked must not mute a dialog whose question was just
 * answered "no".
 * @param {string} keyBase
 * @param {boolean} skip
 */
export function rememberConfirmSkip(keyBase, skip) {
  try {
    if (skip) localStorage.setItem(scopedStorageKey(keyBase), '1');
    else localStorage.removeItem(scopedStorageKey(keyBase));
  } catch {
    /* storage blocked — the dialog simply keeps asking, harmless */
  }
}
