// @ts-check
/* OSPREY Web Terminal — the sidebar's control-system block
 *
 * The block under the OSPREY mark that answers, at one place in every view,
 * "where does a write land, would it be refused, and is anything wrong": the
 * control-target chip (control-target-chip.js, mounted into
 * `#sidebar-target`), dressed as the block — a crosshair glyph carrying the
 * chip's state dot, the `Control system` eyebrow and a ⇕ caret above the
 * chip's own name and write state — and, under them, a meta line with the
 * facility's service health and its clock.
 *
 * Everything the chip says stays the chip's: this module never writes its
 * name, its state, its attributes or its accessible name, it only adds
 * decoration (aria-hidden) and reads {@link targetReading} for the one place
 * the words are repeated, the tooltip of the collapsed sidebar
 * (`Simulator · writes asked · 15:11 Los Angeles`). The meta line is part of the
 * chip button, so the whole block is one control that opens the popover; the
 * chip's `aria-describedby` points at it so the health and the time are read
 * out with it.
 *
 * Where the deployment has no control-target picker (`html[data-control-target
 * ="off"]`), no chip mounts and the meta line stands alone in the block.
 *
 * HEALTH is the worst outcome across the SYSTEM panel's checks, read from the
 * panel's proxy (`GET /panel/system-health/checks`) on the sidecar's own
 * cadence: one dot, with its word on the dot's label. A deployment that does
 * not serve the SYSTEM panel answers that route 404; the dot then leaves the
 * line and nothing is polled again.
 *
 * THE CLOCK is the facility's (facility-time.js), hour and minute, followed by
 * the facility zone's city (`15:11 Los Angeles`). Where the page names no zone
 * the browser knows, facility-time.js falls back to the viewer's own zone and
 * names it itself (`15:11 PST`). It ticks on the minute.
 */

import { worstStatus } from '/design-system/js/check-status.js';
import { facilityZone, formatFacilityTime } from '/design-system/js/facility-time.js';
import { icon } from '/design-system/js/icons.js';
import { hideTooltip } from '/design-system/js/tooltip.js';
import { withPrefix } from './api.js';
import { getChipElement, subscribe, targetReading } from './control-target-chip.js';

/** The SYSTEM panel's check report, through the terminal's panel proxy. */
export const HEALTH_API = '/panel/system-health/checks';

/** Poll cadence when the report names none, seconds — the sidecar's default. */
const HEALTH_DEFAULT_INTERVAL_S = 60;
/** Poll cadence while the sidecar's first suite is still running, seconds. */
const HEALTH_WARMING_INTERVAL_S = 3;
/** Floor on the cadence, so a misconfigured interval cannot become a busy loop. */
const HEALTH_MIN_INTERVAL_S = 5;

const META_ID = 'sidebar-target-meta';

/**
 * One reading of the health report: a tone for the dot and a word for it.
 * @typedef {object} HealthReading
 * @property {'off' | 'ok' | 'warn' | 'err'} tone
 * @property {string} word
 */

/**
 * The health report as a dot: `checking` before the first answer,
 * `unreachable` when the route fails, `warming` while the first suite runs,
 * else the worst outcome among the checks that ran.
 * @param {unknown} report - the parsed envelope, `null` before any answer,
 *   `undefined` after a failed read
 * @returns {HealthReading}
 */
export function readHealth(report) {
  if (report === null) return { tone: 'off', word: 'checking' };
  if (report === undefined) return { tone: 'err', word: 'unreachable' };
  const frame = typeof report === 'object' ? /** @type {Record<string, any>} */ (report) : {};
  if (frame.warming === true) return { tone: 'off', word: 'warming' };
  const results = Array.isArray(frame.results) ? frame.results : [];
  const counted = results.filter((r) => r && typeof r.status === 'string' && r.status !== 'skip');
  if (counted.length === 0) return { tone: 'off', word: 'no checks' };
  const worst = worstStatus(counted);
  const tally = (/** @type {string} */ status) => counted.filter((r) => r.status === status).length;
  if (worst === 'error') {
    const n = tally('error');
    return { tone: 'err', word: `${n} error${n === 1 ? '' : 's'}` };
  }
  if (worst === 'warning') {
    const n = tally('warning');
    return { tone: 'warn', word: `${n} warning${n === 1 ? '' : 's'}` };
  }
  return { tone: 'ok', word: 'ok' };
}

/**
 * The facility clock as the block shows it: hour and minute on the facility
 * clock and the facility zone's city, or facility-time.js's own zone name when
 * it fell back to the viewer's zone.
 * @param {Date|number} [now]
 * @returns {string}
 */
export function clockText(now = Date.now()) {
  const time = formatFacilityTime(now, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
  const zone = facilityZone();
  if (!zone.facility) return time;
  const city = (zone.id.split('/').pop() ?? zone.id).replaceAll('_', ' ');
  return `${time} ${city}`;
}

/** @type {HTMLElement | null} */
let meta = null;
/** @type {HTMLElement | null} */
let healthDot = null;
/** @type {HTMLElement | null} */
let clockEl = null;
/** The last health envelope: null before any answer, undefined after a failed read. */
/** @type {unknown} */
let healthReport = null;
let healthServed = true;
/** @type {ReturnType<typeof setTimeout> | null} */
let healthTimer = null;
/** @type {ReturnType<typeof setTimeout> | null} */
let clockTimer = null;
/** Bumped by teardown, so a read still in flight then lands nowhere. */
let healthEpoch = 0;
/** @type {(() => void) | null} */
let unsubscribeChip = null;
/** @type {MutationObserver | null} */
let expandedObserver = null;

/** @returns {HTMLElement} the meta line: health dot and clock */
function buildMeta() {
  const line = document.createElement('span');
  line.className = 'sidebar-target-meta';
  line.id = META_ID;
  healthDot = document.createElement('span');
  healthDot.className = 'osprey-dot sidebar-health';
  healthDot.setAttribute('role', 'img');
  clockEl = document.createElement('span');
  clockEl.className = 'sidebar-target-clock';
  line.append(healthDot, clockEl);
  return line;
}

/**
 * Dress the chip as the block: glyph (carrying the chip's own state dot,
 * placed over it by the stylesheet), eyebrow and caret, all decorative, and
 * the meta line as the chip's description.
 * @param {HTMLElement} chip
 * @param {HTMLElement} line
 */
function decorateChip(chip, line) {
  const glyph = icon('crosshair');
  glyph.classList.add('sidebar-target-glyph');
  const eyebrow = document.createElement('span');
  eyebrow.className = 'sidebar-target-eyebrow';
  eyebrow.setAttribute('aria-hidden', 'true');
  eyebrow.textContent = 'Control system';
  const caret = icon('chevrons-ud');
  caret.classList.add('sidebar-target-caret');
  chip.prepend(glyph, eyebrow, caret);
  chip.append(line);
  chip.setAttribute('aria-describedby', META_ID);
}

/** Repaint the health dot from the last report. */
function paintHealth() {
  if (!healthDot) return;
  healthDot.hidden = !healthServed;
  const reading = readHealth(healthReport);
  healthDot.classList.toggle('ok', reading.tone === 'ok');
  healthDot.classList.toggle('err', reading.tone === 'err');
  healthDot.classList.toggle('warn', reading.tone === 'warn');
  healthDot.setAttribute('aria-label', `Service health: ${reading.word}`);
  healthDot.title = `Service health: ${reading.word}`;
}

/**
 * Keep the collapsed sidebar's tooltip on the chip current, and off while the
 * chip's popover is open (the popover then stands where the tooltip would).
 */
function paintTip() {
  const chip = getChipElement();
  if (!chip) return;
  const reading = targetReading();
  if (!reading || chip.getAttribute('aria-expanded') === 'true') {
    delete chip.dataset.tip;
    return;
  }
  chip.dataset.tip = [reading.name, reading.state, clockEl?.textContent]
    .filter(Boolean)
    .join(' · ');
}

/** Show the time now, then again at the top of every minute. */
function tickClock() {
  if (clockEl) clockEl.textContent = clockText();
  paintTip();
  clockTimer = setTimeout(tickClock, 60_000 - (Date.now() % 60_000));
}

/** @returns {number} seconds until the next health read */
function nextHealthDelayS() {
  const frame =
    healthReport && typeof healthReport === 'object'
      ? /** @type {Record<string, any>} */ (healthReport)
      : null;
  if (!frame) return HEALTH_DEFAULT_INTERVAL_S;
  if (frame.warming === true) return HEALTH_WARMING_INTERVAL_S;
  const interval = Number(frame.interval_s);
  return Number.isFinite(interval) && interval > 0
    ? Math.max(HEALTH_MIN_INTERVAL_S, Math.round(interval))
    : HEALTH_DEFAULT_INTERVAL_S;
}

/** Read the report once, repaint, and schedule the next read. */
async function pollHealth() {
  const epoch = healthEpoch;
  healthTimer = null;
  /** @type {unknown} */
  let report;
  try {
    const response = await fetch(withPrefix(HEALTH_API), { cache: 'no-store' });
    if (epoch !== healthEpoch) return;
    if (response.status === 404) {
      // No SYSTEM panel on this deployment: nothing to show, nothing to poll.
      healthServed = false;
      paintHealth();
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    report = await response.json();
  } catch {
    report = undefined;
  }
  if (epoch !== healthEpoch) return;
  healthReport = report;
  paintHealth();
  healthTimer = setTimeout(pollHealth, nextHealthDelayS() * 1000);
}

/**
 * Lay out the control-system block around the chip mounted in
 * `#sidebar-target` (or, with no picker, the meta line alone) and start the
 * health reads and the clock. Call once, after the chip's init. No-op without
 * the block.
 * @returns {void}
 */
export function initSidebarTarget() {
  const host = document.getElementById('sidebar-target');
  if (!host || meta) return;
  meta = buildMeta();
  const chip = getChipElement();
  if (chip && host.contains(chip)) {
    decorateChip(chip, meta);
    unsubscribeChip = subscribe(paintTip);
    // The chip's aria-expanded follows its popover; the tooltip steps aside
    // while the popover is open.
    expandedObserver = new MutationObserver(() => {
      if (chip.getAttribute('aria-expanded') === 'true') hideTooltip();
      paintTip();
    });
    expandedObserver.observe(chip, { attributes: true, attributeFilter: ['aria-expanded'] });
    host.dataset.target = 'chip';
  } else {
    host.append(meta);
    host.dataset.target = 'none';
  }
  paintHealth();
  tickClock();
  void pollHealth();
}

/**
 * Stop the reads and the clock and remove the meta line, for a page that
 * tears the block down (tests).
 * @returns {void}
 */
export function teardownSidebarTarget() {
  healthEpoch += 1;
  clearTimeout(healthTimer ?? undefined);
  clearTimeout(clockTimer ?? undefined);
  healthTimer = null;
  clockTimer = null;
  unsubscribeChip?.();
  unsubscribeChip = null;
  expandedObserver?.disconnect();
  expandedObserver = null;
  meta?.remove();
  meta = null;
  healthDot = null;
  clockEl = null;
  healthReport = null;
  healthServed = true;
}
