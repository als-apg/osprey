// @ts-check
/**
 * Formatters for the Activity page: durations, sizes, a call's arguments on
 * one line, and times on the facility clock.
 *
 * @module activity-log-format
 */

import { facilityDayKey, formatFacilityTime } from '/design-system/js/facility-time.js';

/** Characters one argument value keeps in a call's one-line summary. */
const ARG_VALUE_MAX = 44;

const MS_PER_DAY = 86_400_000;

/**
 * A duration in the largest unit that reads well: `840 ms`, `17.4 s`,
 * `2 min 5 s`, `1 h 5 min`.
 * @param {number | null | undefined} ms
 * @returns {string} `''` when nothing was measured
 */
export function formatDuration(ms) {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '';
  // Round before choosing the unit, so no unit shows its own upper bound.
  if (Math.round(ms) < 1000) return `${Math.round(ms)} ms`;
  if (ms < 59_950) return `${(ms / 1000).toFixed(1)} s`;
  const seconds = Math.round(ms / 1000);
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
  const minutes = Math.round(ms / 60_000);
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/**
 * A byte count as `512 B`, `2.1 KB` or `3.0 MB`.
 * @param {number | null | undefined} n
 * @returns {string}
 */
export function formatBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return `${n} B`;
  if (n < 1_048_576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1_048_576).toFixed(1)} MB`;
}

/**
 * A call's arguments on one line: `key=value` pairs two spaces apart, each
 * value cut at {@link ARG_VALUE_MAX} characters.
 * @param {unknown} input the call's arguments
 * @returns {string} `''` when there are none
 */
export function argSummary(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return '';
  const parts = [];
  for (const [key, value] of Object.entries(input)) {
    const raw = typeof value === 'string' ? value : (JSON.stringify(value) ?? String(value));
    const text = raw.replace(/\s+/g, ' ');
    const shown = text.length > ARG_VALUE_MAX ? `${text.slice(0, ARG_VALUE_MAX - 1)}…` : text;
    parts.push(`${key}=${shown}`);
  }
  return parts.join('  ');
}

/**
 * A time of day on the facility clock, 24-hour with seconds.
 * @param {string | number | Date | null | undefined} value
 * @param {{ ms?: boolean, seconds?: boolean }} [options] `ms` adds the
 *   milliseconds; `seconds: false` leaves the hour and the minute
 * @returns {string} `''` for empty or unparsable input
 */
export function clock(value, { ms = false, seconds = true } = {}) {
  return formatFacilityTime(value, {
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
    ...(seconds ? { second: '2-digit' } : {}),
    ...(seconds && ms ? { fractionalSecondDigits: 3 } : {}),
  });
}

/**
 * The facility day of *value* as a list heading: `Today`, `Yesterday`, or the
 * date in the viewer's locale (with the year when it is not this year's).
 * @param {string | number | Date | null | undefined} value
 * @param {number | Date} [now]
 * @returns {string} `''` for empty or unparsable input
 */
export function dayLabel(value, now = Date.now()) {
  const day = facilityDayKey(value);
  if (!day) return '';
  const today = facilityDayKey(now);
  if (day === today) return 'Today';
  // Day keys are calendar dates, so the day before is exact in UTC arithmetic.
  const before = new Date(Date.parse(`${today}T00:00:00Z`) - MS_PER_DAY);
  if (day === before.toISOString().slice(0, 10)) return 'Yesterday';
  return formatFacilityTime(value, {
    day: 'numeric',
    month: 'short',
    ...(day.slice(0, 4) === today.slice(0, 4) ? {} : { year: 'numeric' }),
  });
}
