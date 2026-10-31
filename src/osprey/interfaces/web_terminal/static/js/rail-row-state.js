// @ts-check
/* OSPREY Web Terminal — a sidebar row's live state
 *
 * The lookup of a panel's sidebar row and the state the row shows while the
 * page runs: active, enabled, the agent's attention badge, a sidecar's start
 * status, reachability, and the popped-out mark. panel-rail.js renders and
 * places the rows and re-exports every function here, so the rest of the
 * shell keeps importing them from panel-rail.js.
 *
 * The row's tooltips are shared between those states. While a badge time, an
 * unreachable notice or a status message owns the native `title`, the row
 * carries a transient `data-title-base` holding the title it borrowed;
 * clearing all of them restores it and removes the attribute. While the row
 * is popped out, its explanatory line (`data-tip-detail`) names the window,
 * and the panel's own sentence waits in `data-tip-detail-base`.
 * {@link writeRowTips} is how a render writes the row's own tooltips under
 * both rules.
 *
 * A page-owned row parked out of view (its panel is not a member) counts as
 * absent for every lookup here.
 */

import { flashElement } from '/design-system/js/highlight.js';
import { icon } from '/design-system/js/icons.js';
import { formatFacilityTime, viewerSharesFacilityClock } from '/design-system/js/facility-time.js';
import { readTitle, writeTitle } from '/design-system/js/tooltip.js';

/** A panel's row button. */
export const BUTTON_SELECTOR = '.panel-rail-button';

/** The tooltip layer's explanatory line (tooltip.js). */
const DETAIL_ATTR = 'data-tip-detail';

/** Where a popped-out row parks its own tooltip detail while the window's
 *  sentence stands in for it. */
const DETAIL_BASE_ATTR = 'data-tip-detail-base';

/** The state class of a row whose panel runs in its own window. */
const POPPED_OUT_CLASS = 'popped-out';

/** Marks the trailing glyph a popped-out row carries. */
const POPOUT_TRAIL_ATTR = 'data-popout-trail';

/** A popped-out row's tooltip detail: the click's new meaning. */
const POPPED_OUT_DETAIL = 'Popped out — click to bring the window to the front';

/**
 * Where a row's own tooltip is parked while a badge time, an unreachable
 * notice or a status message owns the `title`. Present only while one of them
 * does — see {@link renderTitle}.
 */
const TITLE_BASE_ATTR = 'data-title-base';

/** The badge's "agent touched" clock time, present while the badge carries one. */
const TOUCHED_ATTR = 'data-title-touched';

/** The clock time the backend stopped answering, present while `.unreachable`. */
const UNREACHABLE_ATTR = 'data-title-unreachable';

const TOUCHED_SEPARATOR = ' · agent touched ';

/** A server-reported start status sentence, present while the row shows one. */
const STATUS_ATTR = 'data-title-status';

/** Marks the attention dot a row carries while it is `.agent-attention`. */
const ATTENTION_BADGE_ATTR = 'data-attention-badge';

/**
 * The row element that holds `entry`: the entry itself, or the page-owned
 * `.sidebar-row` wrapping it.
 * @param {HTMLElement} entry
 * @returns {HTMLElement}
 */
export function rowOf(entry) {
  const wrapper = entry.parentElement;
  return wrapper && wrapper.classList.contains('sidebar-row') ? wrapper : entry;
}

/**
 * Whether an entry is a page-owned one parked out of view (not a member).
 * @param {Element} entry
 * @returns {boolean}
 */
export function parked(entry) {
  return rowOf(/** @type {HTMLElement} */ (entry)).hidden === true;
}

/**
 * Write a row's own tooltips: the native `title` and the explanatory line.
 * A title borrowed by a notice is written to its stash, and a detail that a
 * popped-out mark stands in for is written to its parking place, so the
 * notice and the mark keep showing and give back the new text when they
 * clear.
 * @param {HTMLElement} btn
 * @param {string} title
 * @param {string | null} detail - none when null
 */
export function writeRowTips(btn, title, detail) {
  const detailAttr = btn.classList.contains(POPPED_OUT_CLASS) ? DETAIL_BASE_ATTR : DETAIL_ATTR;
  if (detail) btn.setAttribute(detailAttr, detail);
  else btn.removeAttribute(detailAttr);
  if (btn.hasAttribute(TITLE_BASE_ATTR)) btn.setAttribute(TITLE_BASE_ATTR, title);
  else writeTitle(btn, title);
}

// ---- Mutators ----

/**
 * Return the row button for a panel id, or null when the panel has no row in
 * view (absent, or a page-owned row parked out of view).
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @returns {HTMLButtonElement | null}
 */
export function getEntry(railEl, panelId) {
  const entry = /** @type {HTMLButtonElement | null} */ (
    railEl.querySelector(`${BUTTON_SELECTOR}[data-panel-id="${CSS.escape(panelId)}"]`)
  );
  return entry && !parked(entry) ? entry : null;
}

/**
 * Mark exactly one row active (`.active` plus `aria-current="page"`, which
 * draws the selected wash and the accent bar) and clear it from every other
 * row. Passing an id with no matching row clears the active state on all.
 * @param {HTMLElement} railEl
 * @param {string | null} panelId
 */
export function setActive(railEl, panelId) {
  for (const el of railEl.querySelectorAll(BUTTON_SELECTOR)) {
    const isActive = el.getAttribute('data-panel-id') === panelId;
    el.classList.toggle('active', isActive);
    if (isActive) el.setAttribute('aria-current', 'page');
    else el.removeAttribute('aria-current');
  }
}

/**
 * Enable or disable a row by toggling `.disabled`. Rows render disabled; the
 * caller enables one once its backend first reports healthy. This is the
 * sidebar's ONLY health-derived state besides reachability. No-op when the
 * row is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} enabled
 */
export function setEntryEnabled(railEl, panelId, enabled) {
  getEntry(railEl, panelId)?.classList.toggle('disabled', !enabled);
}

/**
 * A server or poll instant as the sidebar's tooltips show it: hour and minute
 * on the facility clock, naming its zone when the viewer's clock reads
 * differently. It is the one formatter for every "since" time a panel surface
 * shows, so the sidebar and the tile never disagree.
 * @param {number} ms - epoch milliseconds
 * @returns {string}
 */
export function clockTime(ms) {
  return formatFacilityTime(ms, {
    hour: 'numeric',
    minute: '2-digit',
    ...(viewerSharesFacilityClock(ms) ? {} : { timeZoneName: 'short' }),
  });
}

/**
 * Compose a row's tooltip from its own text and the notices borrowing it.
 *
 * The own text is stashed on the row while any notice is up rather than
 * recomputed, so the restore is exact even if the caller retitled the row,
 * and a repeated notice REPLACES its time instead of appending a second one.
 * A status message stands in for the own text, so a badge time is appended to
 * it. An unreachable notice outranks both: "not answering" is the fact an
 * operator needs before clicking. With no notice the stash is restored and
 * removed, which makes a clear on an untouched row a true no-op. The title
 * goes through tooltip.js's readTitle/writeTitle: while the collapsed
 * sidebar's tip shows, the native `title` is parked there.
 * @param {HTMLElement} entry
 */
function renderTitle(entry) {
  const base = entry.getAttribute(TITLE_BASE_ATTR) ?? readTitle(entry);
  const touched = entry.getAttribute(TOUCHED_ATTR);
  const unreachable = entry.getAttribute(UNREACHABLE_ATTR);
  const status = entry.getAttribute(STATUS_ATTR);
  if (touched === null && unreachable === null && status === null) {
    if (entry.hasAttribute(TITLE_BASE_ATTR)) {
      writeTitle(entry, base);
      entry.removeAttribute(TITLE_BASE_ATTR);
    }
    return;
  }
  entry.setAttribute(TITLE_BASE_ATTR, base);
  const own = status ?? base;
  if (unreachable !== null) writeTitle(entry, `not answering since ${unreachable}`);
  else writeTitle(entry, touched !== null ? `${own}${TOUCHED_SEPARATOR}${touched}` : own);
}

/**
 * Show or remove a row's attention dot: an `.osprey-nav-badge` beside the
 * label (before the disclosure chevron or the popped-out glyph, where the
 * row has one, so that glyph stays last).
 * @param {HTMLElement} entry
 * @param {boolean} on
 */
function renderAttentionBadge(entry, on) {
  const badge = entry.querySelector(`:scope > [${ATTENTION_BADGE_ATTR}]`);
  if (!on) {
    badge?.remove();
    return;
  }
  if (badge) return;
  const dot = document.createElement('span');
  dot.className = 'osprey-nav-badge sidebar-attention';
  dot.setAttribute(ATTENTION_BADGE_ATTR, '');
  dot.setAttribute('aria-hidden', 'true');
  entry.insertBefore(
    dot,
    entry.querySelector(`:scope > .osprey-nav-disc, :scope > [${POPOUT_TRAIL_ATTR}]`)
  );
}

/**
 * Set or clear the agent-attention affordance on a row. Turning it on toggles
 * the persistent `agent-attention` state class, draws the row's attention
 * dot, fires the one-shot `agent-flash` glow via {@link flashElement}, and
 * scrolls the row into view: the group that holds the sidebar's lists
 * scrolls when they are open, and a badge scrolled out of sight reports
 * nothing to the operator. `block: 'nearest'` leaves a visible row exactly
 * where it is.
 *
 * Turning it off removes the dot and restores the tooltip; an in-flight flash
 * is left to finish on its own `animationend`.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} on
 * @param {number} [ts] - the originating event's SERVER timestamp (epoch
 *   seconds, as the `agent_activity` SSE frames carry it), appended to the
 *   row's tooltip as "· agent touched <time>". Omit it — or pass a non-finite
 *   value — for a badge with no time claim; never substitute a client clock,
 *   which would report when the browser rendered rather than when the agent
 *   acted.
 * @returns {boolean} true when the row existed and was updated; false for an
 *   unknown id (safe no-op, so callers can fall back)
 */
export function setEntryAttention(railEl, panelId, on, ts) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return false;
  entry.classList.toggle('agent-attention', on);
  renderAttentionBadge(entry, on);
  if (on && typeof ts === 'number' && Number.isFinite(ts)) {
    entry.setAttribute(TOUCHED_ATTR, clockTime(ts * 1000));
  } else {
    entry.removeAttribute(TOUCHED_ATTR);
  }
  renderTitle(entry);
  if (on) {
    flashElement(entry);
    entry.scrollIntoView({ block: 'nearest' });
  }
  return true;
}

/**
 * Show or clear a server-reported start status on a row. `failed` toggles
 * `.failed`, which replaces `.disabled`: the row stays dimmed but takes a
 * click, the caller's retry. While `message` is set, the tooltip and
 * `aria-description` read exactly that string (with a badge time appended
 * while the row is badged); `message: null` puts the row's own tooltip back.
 * No-op when the row is absent.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {{ failed: boolean, message: string | null }} status
 */
export function setEntryStatus(railEl, panelId, { failed, message }) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return;
  entry.classList.toggle('failed', failed);
  if (failed) entry.classList.remove('disabled');
  if (message) {
    entry.setAttribute(STATUS_ATTR, message);
    entry.setAttribute('aria-description', message);
  } else {
    entry.removeAttribute(STATUS_ATTR);
    entry.removeAttribute('aria-description');
  }
  renderTitle(entry);
}

/**
 * Mark a row whose backend answered before but has stopped, or bring it
 * back. Unreachable dims the row like `.disabled` but leaves it clickable —
 * its menu and popout stay live — and points its tooltip at when the backend
 * stopped answering. Reachable restores the row and its tooltip. `.disabled`
 * (never answered) is untouched either way.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} reachable
 * @param {number} [sinceMs] - epoch ms of the first unanswered poll, shown in
 *   the tooltip while unreachable
 * @returns {boolean} true when the row existed and was updated
 */
export function setEntryReachable(railEl, panelId, reachable, sinceMs) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return false;
  entry.classList.toggle('unreachable', !reachable);
  if (reachable) {
    entry.removeAttribute(UNREACHABLE_ATTR);
  } else {
    entry.setAttribute(UNREACHABLE_ATTR, clockTime(sinceMs ?? Date.now()));
  }
  renderTitle(entry);
  return true;
}

/**
 * Mark a row whose panel runs in its own popped-out window, or clear the
 * mark: `.popped-out` on the row, a trailing `span.osprey-nav-trail`
 * (`data-popout-trail`, `aria-hidden="true"`) holding the `maximize` glyph as
 * the row's last child, and the tooltip detail "Popped out — click to bring
 * the window to the front", which is what a click on the row now does. The
 * row's own detail is parked in `data-tip-detail-base` while the mark is up,
 * and is restored (or removed, when there was none) when it clears. Marking
 * a marked row, or clearing a clear one, changes nothing.
 * @param {HTMLElement} railEl
 * @param {string} panelId
 * @param {boolean} on
 * @returns {boolean} true when the row existed and was updated; false for an
 *   unknown id
 */
export function setEntryPoppedOut(railEl, panelId, on) {
  const entry = getEntry(railEl, panelId);
  if (!entry) return false;
  const popped = entry.classList.contains(POPPED_OUT_CLASS);
  if (on && !popped) {
    const own = entry.getAttribute(DETAIL_ATTR);
    if (own !== null) entry.setAttribute(DETAIL_BASE_ATTR, own);
    entry.setAttribute(DETAIL_ATTR, POPPED_OUT_DETAIL);
    entry.classList.add(POPPED_OUT_CLASS);
    const trail = document.createElement('span');
    trail.className = 'osprey-nav-trail';
    trail.setAttribute(POPOUT_TRAIL_ATTR, '');
    trail.setAttribute('aria-hidden', 'true');
    trail.append(icon('popout'));
    entry.append(trail);
  } else if (!on && popped) {
    const own = entry.getAttribute(DETAIL_BASE_ATTR);
    if (own !== null) entry.setAttribute(DETAIL_ATTR, own);
    else entry.removeAttribute(DETAIL_ATTR);
    entry.removeAttribute(DETAIL_BASE_ATTR);
    entry.classList.remove(POPPED_OUT_CLASS);
    entry.querySelector(`:scope > [${POPOUT_TRAIL_ATTR}]`)?.remove();
  }
  return true;
}
