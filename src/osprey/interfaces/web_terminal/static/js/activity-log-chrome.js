// @ts-check
/**
 * What frames a session on the Activity page: its header, the strip of counts
 * under it, the session list beside it, and the state of the agent record the
 * reading was made from. Every renderer returns an HTML string.
 *
 * @module activity-log-chrome
 */

import { escapeHtml } from '/design-system/js/dom.js';
import {
  facilityDayKey,
  facilityZone,
  facilityZoneLabel,
  formatFacilityTime,
} from '/design-system/js/facility-time.js';

import { clock, dayLabel, formatDuration } from './activity-log-format.js';
import { chip, iconHtml, stateBlock } from './activity-log-rows.js';

/** @typedef {import('/design-system/js/info-popover.js').InfoContent} InfoContent */
/** @typedef {import('/design-system/js/info-popover.js').InfoRow} InfoRow */
/** @typedef {import('./activity-log-model.js').ActivityPayload} ActivityPayload */
/** @typedef {import('./activity-log-model.js').ActivitySession} ActivitySession */
/** @typedef {import('./activity-log-model.js').ActivitySummary} ActivitySummary */

/**
 * @typedef {Object} SessionRow  One `/api/sessions` row, as far as the list reads it.
 * @property {string} transcript_id
 * @property {string} title
 * @property {string} [state]  `live` or `idle`.
 * @property {string | null} [first_ts]
 * @property {string | null} [last_ts]
 */

/** @typedef {ActivityPayload['record']} RecordState */

const MS_PER_DAY = 86_400_000;

/** The record chip of each state. @type {Record<RecordState, { tone: 'ok' | 'warn' | 'err', label: string, icon: string }>} */
const RECORD_CHIPS = {
  complete: { tone: 'ok', label: 'Record complete', icon: 'database' },
  incomplete: { tone: 'warn', label: 'Record incomplete', icon: 'triangle-alert' },
  unavailable: { tone: 'err', label: 'Record unavailable', icon: 'triangle-alert' },
};

/** What each record state means for the reader. @type {Record<RecordState, string>} */
const RECORD_TEXT = {
  complete: 'Everything this session wrote has reached the agent record.',
  incomplete: 'The record is behind this session. Some entries have not arrived yet.',
  unavailable: 'The record service did not answer.',
};

/**
 * The toolbar chip of a record state.
 * @param {RecordState} record
 * @returns {{ tone: 'ok' | 'warn' | 'err', label: string, icon: string }}
 */
export function recordChip(record) {
  return { ...(RECORD_CHIPS[record] ?? RECORD_CHIPS.unavailable) };
}

/**
 * What the record chip opens: what the record holds of the session, and until
 * when.
 * @param {ActivityPayload} payload
 * @returns {InfoContent}
 */
export function recordPopover(payload) {
  const session = payload.session;
  /** @type {InfoRow[]} */
  const rows = [];
  if (session) {
    rows.push({ label: 'Entries', value: session.entries });
    const kinds = Object.entries(session.kinds)
      .filter(([, count]) => count > 0)
      .map(([kind, count]) => `${count} ${kind.replace(/_/g, ' ')}`);
    if (kinds.length) rows.push({ label: 'Holds', value: kinds.join(', ') });
    const start = Date.parse(session.start ?? '');
    if (!Number.isNaN(start)) {
      const until = facilityDayKey(start + session.retention_days * MS_PER_DAY);
      rows.push({ label: 'Kept until', value: `${until}, then in the archive` });
    }
    rows.push({ label: 'Session', value: session.id, copy: true });
  }
  return {
    title: 'Agent record',
    text: RECORD_TEXT[payload.record] ?? RECORD_TEXT.unavailable,
    rows,
  };
}

/**
 * The warning above the views while the record is behind the session.
 * @param {RecordState} record
 * @returns {string} `''` unless the record is incomplete
 */
export function recordBanner(record) {
  if (record !== 'incomplete') return '';
  return (
    `<div class="act-banner" role="status">${iconHtml('triangle-alert')}` +
    `<span>Entries may be missing. The record has not yet received everything this ` +
    `session wrote; the page fills in as it arrives.</span></div>`
  );
}

/**
 * The counts on the view switcher.
 * @param {ActivityPayload} payload
 * @param {ActivitySummary} model
 * @returns {{ tools: number, safety: number, artifacts: number }}
 */
export function viewCounts(payload, model) {
  return {
    tools: model.tools.length,
    safety: model.approvals.length,
    artifacts: payload.artifacts.length,
  };
}

/**
 * Whether the session is still being written, or when it last was.
 * @param {ActivitySession | null} session
 * @returns {string} `''` without a session or an entry
 */
export function renderLiveState(session) {
  if (!session) return '';
  if (session.state === 'live') {
    return (
      `<span class="osprey-chip osprey-chip--info"><span class="act-live-dot"></span>` +
      `<span class="osprey-chip-label">Live</span></span>`
    );
  }
  const last = clock(session.end, { seconds: false });
  return last ? `<span class="osprey-panel-toolbar__text">Last entry ${last}</span>` : '';
}

/**
 * The day and the hours of a session, on the facility clock.
 * @param {ActivitySession} session
 * @returns {string} `''` for a session with no entry yet
 */
function timeRange(session) {
  const date = formatFacilityTime(session.start, {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
  const start = clock(session.start, { seconds: false });
  if (!date || !start) return '';
  const end = session.state === 'live' ? '' : clock(session.end, { seconds: false });
  const hours = end ? `${start} to ${end}` : `since ${start}`;
  // Without a facility zone every time already names the viewer's own.
  const zone = facilityZone().facility ? ` ${facilityZoneLabel(session.start ?? undefined)}` : '';
  return `${date}, ${hours}${zone}`;
}

/**
 * The session's header: its title, whose it is, when it ran and for how long,
 * what it was pointed at, and whether it could write.
 * @param {ActivityPayload} payload
 * @param {ActivitySummary} model
 * @returns {string} `''` without a session
 */
export function renderHeader(payload, model) {
  const session = payload.session;
  if (!session) return '';
  const facts = [];
  if (session.identity) facts.push(`<span><b>${escapeHtml(session.identity)}</b></span>`);
  const range = timeRange(session);
  if (range) facts.push(`<span>${escapeHtml(range)}</span>`);
  const length = session.state === 'live' ? '' : formatDuration(model.ms);
  if (length) facts.push(`<span>${length}</span>`);
  if (session.target) facts.push(`<span>Target <b>${escapeHtml(session.target)}</b></span>`);
  const posture = session.posture
    ? session.posture === 'writes'
      ? chip('warn', 'Writes enabled', 'The session could write to its control target')
      : chip('', session.posture)
    : '';
  const copy = 'Copy link to this session';
  return (
    `<div class="act-head"><div class="act-head-text"><h1>${escapeHtml(session.title)}</h1>` +
    `<div class="act-meta">${facts.join('<span class="sep">·</span>')}${posture}</div></div>` +
    `<div class="act-head-actions"><button class="osprey-icon-btn" type="button"` +
    ` data-act="copy-link" title="${copy}" aria-label="${copy}">${iconHtml('link')}</button>` +
    `</div></div>`
  );
}

/**
 * The strip of counts under the header; each cell opens the view that lists
 * what it counts.
 * @param {ActivityPayload} payload
 * @param {ActivitySummary} model
 * @returns {string}
 */
export function renderStats(payload, model) {
  /** @param {string | number} n @param {string} label @param {string} go @param {string} [tone] */
  const cell = (n, label, go, tone = '') =>
    `<button class="act-stat" type="button" data-go="${go}" data-tone="${tone}">` +
    `<span class="act-stat-n">${n}</span><span class="act-stat-l">${label}</span></button>`;
  const approvals = model.approvals.length;
  const failed = model.errors.length;
  return (
    `<div class="act-stats">` +
    cell(model.prompts, 'Prompts', 'timeline:conversation') +
    cell(model.tools.length, 'Tool calls', 'tools:all') +
    cell(approvals, 'Approvals', 'safety:all', approvals ? 'warn' : '') +
    cell(failed, 'Failed calls', 'timeline:errors', failed ? 'err' : '') +
    cell(model.subs.length, 'Subagents', 'timeline:all') +
    cell(payload.artifacts.length, 'Artifacts', 'artifacts:all') +
    cell(model.modelMs ? formatDuration(model.modelMs) : '–', 'Model time', 'timeline:all') +
    cell(model.cost ? `$${model.cost.toFixed(2)}` : '–', 'Model cost', 'timeline:all') +
    `</div>`
  );
}

/**
 * The session list: the sessions of the record in the order the server sends
 * them, under the facility day of their last entry.
 * @param {SessionRow[]} rows
 * @param {string | null} selectedId  the transcript shown
 * @param {number | Date} [now]
 * @returns {string}
 */
export function renderSidebar(rows, selectedId, now = Date.now()) {
  if (rows.length === 0) return stateBlock('message', 'No sessions in the record');
  let html = '';
  let last = '';
  for (const row of rows) {
    // A session the record holds no entry of yet has just begun.
    const day = dayLabel(row.last_ts ?? row.first_ts ?? now, now);
    if (day !== last) {
      html += `<div class="osprey-section-label">${escapeHtml(day)}</div>`;
      last = day;
    }
    const title = escapeHtml(row.title);
    const mark = row.state === 'live' ? '<span class="act-live-dot" title="Live"></span>' : '';
    html +=
      `<button class="osprey-list-row" type="button"` +
      ` data-session="${escapeHtml(row.transcript_id)}"` +
      ` aria-selected="${row.transcript_id === selectedId}" title="${title}">` +
      `${iconHtml('message')}<span class="osprey-list-title">${title}</span>${mark}` +
      `<time>${clock(row.last_ts, { seconds: false })}</time></button>`;
  }
  return html;
}
