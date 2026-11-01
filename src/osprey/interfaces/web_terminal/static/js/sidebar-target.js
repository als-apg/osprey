// @ts-check
/* OSPREY Web Terminal — the sidebar's control-system block
 *
 * The block under the OSPREY mark that answers, at one place in every view,
 * "where does a write land, would it be refused, and is anything wrong": the
 * control-target chip (control-target-chip.js, mounted into
 * `#sidebar-target`), dressed as one row — a crosshair glyph carrying the
 * chip's state dot, the chip's own name and writes pill, the facility's
 * service health dot and a ⇕ caret.
 *
 * Everything the chip says stays the chip's: this module never writes its
 * name, its state or its accessible name. It adds decoration (aria-hidden),
 * the health dot, and the chip's tooltip text, which repeats
 * {@link targetReading}: `data-tip` is the machine's name and
 * `data-tip-detail` says the rest (`Control system: Simulator, writes asked`).
 * The design system's tooltip layer shows the detail alone while the sidebar
 * shows its labels, both when it is collapsed, and nothing while the chip's
 * popover is open. The health dot is part of the chip button, so the whole
 * row is one control that opens the popover; the chip's `aria-describedby`
 * points at the dot so the health is read out with it.
 *
 * Where the deployment has no control-target picker (`html[data-control-target
 * ="off"]`), no chip mounts and the health dot stands alone in the block.
 *
 * HEALTH is the worst outcome across the SYSTEM panel's checks, read from the
 * panel's proxy (`GET /panel/system-health/checks`) on the sidecar's own
 * cadence: one dot, with its word on the dot's label. The dot shows and the
 * reads run exactly while the SYSTEM panel is served (panel-availability.js).
 */

import { worstStatus } from '/design-system/js/check-status.js';
import { icon } from '/design-system/js/icons.js';
import { withPrefix } from './api.js';
import { whilePanelServed } from './panel-availability.js';
import { getChipElement, subscribe, targetReading } from './control-target-chip.js';

/** The panel the dot reports on. */
const SYSTEM_PANEL_ID = 'system-health';

/** The SYSTEM panel's check report, through the terminal's panel proxy. */
export const HEALTH_API = '/panel/system-health/checks';

/** Poll cadence when the report names none, seconds — the sidecar's default. */
const HEALTH_DEFAULT_INTERVAL_S = 60;
/** Poll cadence while the sidecar's first suite is still running, seconds. */
const HEALTH_WARMING_INTERVAL_S = 3;
/** Floor on the cadence, so a misconfigured interval cannot become a busy loop. */
const HEALTH_MIN_INTERVAL_S = 5;

const HEALTH_ID = 'sidebar-target-health';

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

/** @type {HTMLElement | null} */
let healthDot = null;
/** The glyph and caret this module put on the chip, so teardown can take them off. */
/** @type {Element[]} */
let decorations = [];
/** The last health envelope: null before any answer, undefined after a failed read. */
/** @type {unknown} */
let healthReport = null;
let healthServed = false;
/** @type {ReturnType<typeof setTimeout> | null} */
let healthTimer = null;
/** Bumped by teardown, so a read still in flight then lands nowhere. */
let healthEpoch = 0;
/** @type {(() => void) | null} */
let unsubscribeChip = null;
/** @type {(() => void) | null} */
let unfollowPanel = null;

/** @returns {HTMLElement} the health dot */
function buildHealthDot() {
  const dot = document.createElement('span');
  dot.className = 'osprey-dot sidebar-health';
  dot.id = HEALTH_ID;
  dot.setAttribute('role', 'img');
  return dot;
}

/**
 * Dress the chip as the block's one row: the glyph (carrying the chip's own
 * state dot, placed over it by the stylesheet) first, the health dot and the
 * caret last, and the health dot as the chip's description.
 * @param {HTMLElement} chip
 * @param {HTMLElement} dot
 */
function decorateChip(chip, dot) {
  const glyph = icon('crosshair');
  glyph.classList.add('sidebar-target-glyph');
  const caret = icon('chevrons-ud');
  caret.classList.add('sidebar-target-caret');
  chip.prepend(glyph);
  chip.append(dot, caret);
  chip.setAttribute('aria-describedby', HEALTH_ID);
  decorations = [glyph, caret];
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

/** Keep the chip's tooltip text in step with the chip's reading. */
function paintTip() {
  const chip = getChipElement();
  if (!chip) return;
  const reading = targetReading();
  if (!reading) {
    delete chip.dataset.tip;
    delete chip.dataset.tipDetail;
    return;
  }
  chip.dataset.tip = reading.name;
  chip.dataset.tipDetail = `Control system: ${reading.name}, ${reading.state}`;
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

/** The SYSTEM panel is served: show the dot and start reading. */
function startHealth() {
  healthServed = true;
  paintHealth();
  void pollHealth();
}

/** The SYSTEM panel is not served: hide the dot and stop reading. */
function stopHealth() {
  healthEpoch += 1;
  clearTimeout(healthTimer ?? undefined);
  healthTimer = null;
  healthReport = null;
  healthServed = false;
  paintHealth();
}

/**
 * Lay out the control-system block around the chip mounted in
 * `#sidebar-target` (or, with no picker, the health dot alone) and start the
 * health reads while the SYSTEM panel is served. Call once, after the chip's
 * init. No-op without the block.
 * @returns {void}
 */
export function initSidebarTarget() {
  const host = document.getElementById('sidebar-target');
  if (!host || healthDot) return;
  healthDot = buildHealthDot();
  const chip = getChipElement();
  if (chip && host.contains(chip)) {
    decorateChip(chip, healthDot);
    unsubscribeChip = subscribe(paintTip);
    paintTip();
    host.dataset.target = 'chip';
  } else {
    host.append(healthDot);
    host.dataset.target = 'none';
  }
  paintHealth();
  unfollowPanel = whilePanelServed(SYSTEM_PANEL_ID, startHealth, stopHealth);
}

/**
 * Stop the reads and take the block's additions back off the page, for a page
 * that tears the block down (tests).
 * @returns {void}
 */
export function teardownSidebarTarget() {
  unfollowPanel?.();
  unfollowPanel = null;
  stopHealth();
  unsubscribeChip?.();
  unsubscribeChip = null;
  for (const node of decorations) node.remove();
  decorations = [];
  healthDot?.remove();
  healthDot = null;
}
