// @ts-check
/**
 * OSPREY ARIEL — Shared Utilities
 *
 * Common helpers used across multiple view modules.
 */

/**
 * Normalize an unknown thrown value into a human-readable message string.
 *
 * Shared by every error-path sink so each `catch` binding — typed `unknown`
 * under strict checkJs — reads its message uniformly. `Error`/`ApiError`
 * instances yield their `.message`; anything else is stringified.
 * @param {unknown} e
 * @returns {string}
 */
export function messageOf(e) {
  return e instanceof Error ? e.message : String(e);
}
