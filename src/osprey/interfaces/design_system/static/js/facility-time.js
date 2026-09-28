// @ts-check
/* OSPREY Design System — facility time
 *
 * Every served page that renders times carries the facility zone on `<html>`
 * as `data-facility-timezone="<IANA id>"`. The server stamps the zone
 * `system.timezone` names, resolved by the same call that tells the agent its
 * zone, so a time on the page and a time in the agent's reply read one clock.
 * This module is the one place a browser renders an instant in that zone.
 *
 * THE LOCALE IS THE VIEWER'S. Every display formatter passes `undefined` as
 * the locale, so digits, month names and ordering follow the browser. There is
 * no locale parameter and no config key for one.
 *
 * ABSENT OR UNKNOWN MEANS THE VIEWER'S ZONE, NAMED. With no stamp, or a stamp
 * this browser's Intl data rejects, times render in the viewer's own zone, and
 * a result that shows a time of day names that zone so it is never mistaken
 * for facility time.
 *
 * The attribute is read on every call, not once at module load: a page may
 * stamp it after boot (ARIEL learns the zone from its capabilities payload).
 *
 * @module facility-time
 */

/** The `<html>` attribute the server stamps. */
const ZONE_ATTRIBUTE = 'data-facility-timezone';

/**
 * Display options a caller may pass. The zone is this module's; the date and
 * time styles are excluded because Intl rejects them alongside the
 * `timeZoneName` the fallback adds.
 * @typedef {Omit<Intl.DateTimeFormatOptions, 'timeZone' | 'dateStyle' | 'timeStyle'>} FacilityTimeOptions
 */

/**
 * @typedef {Object} FacilityZone
 * @property {string} id  IANA zone id times render in.
 * @property {boolean} facility  True when `id` is the stamped facility zone,
 *   false when it is the viewer's own zone taken as the fallback.
 */

/** Stamped ids already reported as unknown, so each warns once. @type {Set<string>} */
const warnedIds = new Set();

/** Formatters by resolved options. @type {Map<string, Intl.DateTimeFormat>} */
const formatters = new Map();

/** @returns {string} */
function viewerZone() {
  return new Intl.DateTimeFormat().resolvedOptions().timeZone;
}

/**
 * @param {string|undefined} locale
 * @param {Intl.DateTimeFormatOptions} options
 * @returns {Intl.DateTimeFormat}
 */
function formatter(locale, options) {
  const key = JSON.stringify([locale ?? null, options]);
  let fmt = formatters.get(key);
  if (!fmt) {
    fmt = new Intl.DateTimeFormat(locale, options);
    formatters.set(key, fmt);
  }
  return fmt;
}

/**
 * The instant *value* names, or `null` for empty or unparsable input.
 * @param {Date|string|number|null|undefined} value  A Date, an ISO string or epoch ms.
 * @returns {Date|null}
 */
function toDate(value) {
  if (value === null || value === undefined || value === '') return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * The zone times render in: the stamped facility zone, or the viewer's own
 * when the stamp is absent, empty or unknown to this browser.
 * @returns {FacilityZone}
 */
export function facilityZone() {
  let id = null;
  try {
    id = document.documentElement.getAttribute(ZONE_ATTRIBUTE) || null;
  } catch {
    id = null;
  }
  if (id) {
    try {
      new Intl.DateTimeFormat(undefined, { timeZone: id });
      return { id, facility: true };
    } catch (err) {
      if (!(err instanceof RangeError)) throw err;
      if (!warnedIds.has(id)) {
        warnedIds.add(id);
        console.warn(
          `Facility time zone "${id}" is unknown to this browser; times show in ${viewerZone()}.`,
        );
      }
    }
  }
  return { id: viewerZone(), facility: false };
}

/**
 * Render *value* in the facility zone, in the viewer's locale.
 *
 * A fallback result that shows a time of day names the viewer's zone; a
 * date-only result is never labelled.
 *
 * @param {Date|string|number|null|undefined} value  A Date, an ISO string or epoch ms.
 * @param {FacilityTimeOptions} [options]  Intl component options.
 * @returns {string} The rendering, or `''` for empty or unparsable input.
 */
export function formatFacilityTime(value, options = {}) {
  const date = toDate(value);
  if (!date) return '';
  const zone = facilityZone();
  /** @type {Intl.DateTimeFormatOptions} */
  const resolved = { ...options, timeZone: zone.id };
  const showsTime = 'hour' in options || 'minute' in options || 'second' in options;
  if (!zone.facility && showsTime && options.timeZoneName === undefined) {
    resolved.timeZoneName = 'short';
  }
  return formatter(undefined, resolved).format(date);
}

/**
 * The short name of the resolved zone at *value*, as the viewer's locale
 * spells it (`UTC`, `PST`, `GMT+9`), for chrome that names a zone apart from
 * a time.
 * @param {Date|string|number} [value]  The instant; defaults to now.
 * @returns {string} The short name, or the zone id when Intl gives none.
 */
export function facilityZoneLabel(value = Date.now()) {
  const zone = facilityZone();
  const date = toDate(value) ?? new Date();
  const part = formatter(undefined, { timeZone: zone.id, timeZoneName: 'short' })
    .formatToParts(date)
    .find((p) => p.type === 'timeZoneName');
  return part ? part.value : zone.id;
}

/**
 * The calendar date `YYYY-MM-DD` of *value* in the resolved zone. A comparison
 * key, not display text, so its locale is fixed to keep its digits Latin.
 * @param {Date|string|number|null|undefined} value  A Date, an ISO string or epoch ms.
 * @returns {string} The day key, or `''` for empty or unparsable input.
 */
export function facilityDayKey(value) {
  const date = toDate(value);
  if (!date) return '';
  const parts = formatter('en-US', {
    timeZone: facilityZone().id,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);
  /** @param {string} type */
  const get = (type) => parts.find((p) => p.type === type)?.value ?? '';
  return `${get('year')}-${get('month')}-${get('day')}`;
}

/**
 * The wall-clock reading `YYYY-MM-DD HH:MM:SS.mmm` of *value* in the resolved
 * zone: the form a chart's date axis takes. Plotly drops any offset in a date
 * string and draws the digits as written, so an instant has to become the
 * facility's digits before it reaches the axis. Machine text, so its locale is
 * fixed to keep its digits Latin.
 * @param {Date|string|number|null|undefined} value  A Date, an ISO string or epoch ms.
 * @returns {string} The wall clock, or `''` for empty or unparsable input.
 */
export function facilityWallClock(value) {
  const date = toDate(value);
  if (!date) return '';
  const parts = formatter('en-US', {
    timeZone: facilityZone().id,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3,
    hourCycle: 'h23',
  }).formatToParts(date);
  /** @param {string} type */
  const get = (type) => parts.find((p) => p.type === type)?.value ?? '';
  return (
    `${get('year')}-${get('month')}-${get('day')} ` +
    `${get('hour')}:${get('minute')}:${get('second')}.${get('fractionalSecond')}`
  );
}

/**
 * Whether the viewer's own zone shows the same wall time as the resolved zone
 * at *value*. Aliases and distinct zones at the same offset count as the same
 * clock; with no usable stamp the answer is always true.
 * @param {Date|string|number} [value]  The instant; defaults to now.
 * @returns {boolean}
 */
export function viewerSharesFacilityClock(value = Date.now()) {
  const zone = facilityZone();
  if (!zone.facility) return true;
  const date = toDate(value) ?? new Date();
  /** @param {string} timeZone */
  const wall = (timeZone) =>
    formatter('en-US', {
      timeZone,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  return wall(zone.id) === wall(viewerZone());
}
