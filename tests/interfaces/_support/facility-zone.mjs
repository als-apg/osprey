// @ts-check
/**
 * Facility-zone fixtures for interface tests.
 *
 * A page carries the facility zone on `<html data-facility-timezone>`, and
 * `/design-system/js/facility-time.js` renders every time in it. A test that
 * proves a site reads that clock must stamp a zone whose wall clock differs
 * from the runner's, or a viewer-zone formatter would pass it too.
 * `FACILITY_ZONE` is such a zone in any runner zone.
 *
 * Expected strings are computed with `Intl.DateTimeFormat` directly, never
 * with the module under test.
 */

const ATTR = 'data-facility-timezone';

/** The runner's own zone, as Intl resolves it. */
export const VIEWER_ZONE = new Intl.DateTimeFormat().resolvedOptions().timeZone;

/**
 * The `HH:MM` wall clock of a fixed instant in *zone*.
 * @param {string} [zone]
 * @returns {string}
 */
function wallClock(zone) {
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    timeZone: zone,
  }).format(Date.UTC(2026, 0, 15, 20, 4));
}

/**
 * A whole-hour-offset zone whose wall clock differs from the runner's. Chosen
 * by wall clock, never by id: ICU reports some zones under an alias
 * (`Asia/Kathmandu` resolves to `Asia/Katmandu`).
 */
export const FACILITY_ZONE = /** @type {string} */ (
  ['Asia/Tokyo', 'Europe/Berlin', 'America/New_York'].find(
    (zone) => wallClock(zone) !== wallClock(VIEWER_ZONE),
  )
);

/**
 * Stamp *zone* on `<html>` as the facility zone, or remove the stamp.
 * @param {string | null} zone
 */
export function stampFacilityZone(zone) {
  if (zone === null) document.documentElement.removeAttribute(ATTR);
  else document.documentElement.setAttribute(ATTR, zone);
}

/**
 * The short zone name Intl gives *zone* at *at*, in the runner's locale.
 * @param {string} zone
 * @param {number} [at]
 * @returns {string}
 */
export function zoneName(zone, at = Date.now()) {
  const part = new Intl.DateTimeFormat(undefined, { timeZone: zone, timeZoneName: 'short' })
    .formatToParts(at)
    .find((p) => p.type === 'timeZoneName');
  return part ? part.value : zone;
}
