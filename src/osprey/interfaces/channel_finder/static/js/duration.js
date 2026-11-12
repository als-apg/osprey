// @ts-check
/**
 * OSPREY Channel Finder — the panel's one way of writing a span of time
 *
 * {@link durationText} writes a number of seconds in the largest two units that
 * matter; {@link sinceText} writes how long ago a time was, the same way. Every
 * age and uptime the panel shows is one of these.
 */

/**
 * A number of seconds in the largest two units that matter: `N s`, `N min`,
 * `H h M min`, `D d H h`.
 * @param {number} seconds
 * @returns {string}
 */
export function durationText(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  if (whole < 60) return `${whole} s`;
  const minutes = Math.floor(whole / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return minutes % 60 ? `${hours} h ${minutes % 60} min` : `${hours} h`;
  const days = Math.floor(hours / 24);
  return hours % 24 ? `${days} d ${hours % 24} h` : `${days} d`;
}

/**
 * How long ago `time` was, in the largest two units that matter; never negative.
 * @param {string|null|undefined} time
 * @param {number} now - ms since the epoch
 * @returns {string} '' when `time` is not a time
 */
export function sinceText(time, now) {
  const at = time ? Date.parse(time) : NaN;
  if (Number.isNaN(at)) return '';
  return durationText((now - at) / 1000);
}
