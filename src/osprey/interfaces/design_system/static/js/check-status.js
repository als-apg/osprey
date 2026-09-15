// @ts-check
/* OSPREY Design System — health-check status severity
 *
 * A health check reports a status, and every surface that lists checks reduces
 * a set of them to the single worst one present: the System Health dashboard's
 * own bundle, the web terminal's health bar item. Two copies of a severity
 * order have no mechanism to stay in step — a status added to the sidecar's
 * vocabulary lands in one table and not the other, and the surfaces then
 * disagree about which check is the loudest. The design system is the only
 * mount both pages hold, so the order lives here, with the reducer that reads
 * it, and both surfaces import them.
 */

/**
 * Severity of a check's status, worst last. A status this table does not
 * name ranks below every status it does, so an unfamiliar one neither wins
 * a comparison nor displaces a status that is genuinely worse.
 * @type {Readonly<Record<string, number>>}
 */
export const STATUS_RANK = Object.freeze({ ok: 0, skip: 1, warning: 2, error: 3 });

/**
 * The worst status among a list of checks, `"ok"` for an empty list.
 *
 * @param {readonly {status: string}[]} checks
 * @returns {string}
 */
export function worstStatus(checks) {
  let worst = "ok";
  for (const check of checks) {
    if ((STATUS_RANK[check.status] ?? 0) > (STATUS_RANK[worst] ?? 0)) worst = check.status;
  }
  return worst;
}
