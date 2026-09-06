// @ts-check
/* OSPREY Web Terminal — the browser's session pointer
 *
 * One slot, `osprey-pty-session`, holding the session key this tab is on. Both
 * views read it: the terminal resumes it on page load, and the chat sends it
 * with every turn, so the two surfaces address the same session rather than
 * each starting one of their own.
 *
 * Per persona, not per origin. localStorage is origin-scoped, so on a
 * multi-user mount (`/u/alice/`, `/u/bob/`) a bare key would be one shared
 * pointer — and a session key is not a preference that can be shared:
 * replaying one persona's key would attach another persona's surface to a
 * process that is not theirs. Every read, write and clear resolves the key
 * through pointerKey() below.
 *
 * @module session-pointer
 */

import { scopedStorageKey } from '/design-system/js/storage-scope.js';

const POINTER_KEY_BASE = 'osprey-pty-session';

/**
 * This document's pointer key. Resolved per call rather than once at module
 * load: the scope lives on the served document, and reading it at the point of
 * use is what keeps the key honest.
 * @returns {string}
 */
function pointerKey() {
  return scopedStorageKey(POINTER_KEY_BASE);
}

/**
 * What this module last wrote. Storage can be unavailable (private browsing,
 * storage disabled), and a pointer that reads back as null there would make
 * every write look like a change and every read look like "no session". The
 * mirror keeps both honest for the life of the page.
 * @type {string|null}
 */
let mirror = null;

/**
 * Listeners for pointer changes, notified with the new value (null on clear).
 * @type {((sessionId: string|null) => void)[]}
 */
const listeners = [];

/**
 * The session key this tab is on, or null if there is none.
 * @returns {string|null}
 */
export function getPointer() {
  try {
    return localStorage.getItem(pointerKey());
  } catch {
    return mirror;
  }
}

/**
 * Notify listeners that the pointer settled on `sessionId`. One listener
 * throwing must not cost the others their notification.
 * @param {string|null} sessionId
 */
function notify(sessionId) {
  for (const fn of listeners) {
    try {
      fn(sessionId);
    } catch (err) {
      console.error('osprey web_terminal: a session-pointer listener threw', err);
    }
  }
}

/**
 * Point this tab at `sessionId`. A write of the key already stored is not a
 * change and notifies nobody — the same id arrives from several places (a
 * connect, a confirmation, a chat turn) and each of those is the same fact.
 * @param {string} sessionId
 */
export function setPointer(sessionId) {
  if (getPointer() === sessionId) return;
  mirror = sessionId;
  try {
    localStorage.setItem(pointerKey(), sessionId);
  } catch {
    // Ignore — private browsing / storage disabled. Persistence is a
    // convenience; both views still work without it, for this page load.
  }
  notify(sessionId);
}

/**
 * Forget the session key (a dead id, or logout). A clear of an already-empty
 * pointer notifies nobody, for the same reason a repeated set does not.
 */
export function clearPointer() {
  if (getPointer() === null) return;
  mirror = null;
  try {
    localStorage.removeItem(pointerKey());
  } catch {
    // Ignore — see setPointer().
  }
  notify(null);
}

/**
 * Subscribe to pointer changes. The callback fires for every key the tab
 * settles on, and with null when the pointer is cleared, but not for the
 * current value at subscribe time; callers that need it now read
 * {@link getPointer}. Returns a function that removes the subscription.
 * @param {(sessionId: string|null) => void} fn
 * @returns {() => void}
 */
export function subscribe(fn) {
  listeners.push(fn);
  return () => {
    const at = listeners.indexOf(fn);
    if (at !== -1) listeners.splice(at, 1);
  };
}
